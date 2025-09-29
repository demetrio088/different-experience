# импорт библиотек
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime
from visualization_data import VisualizationData

class SSA(object):
    __supported_types = (pd.Series, np.ndarray, list)
    def __init__(self, tseries, L, save_mem=True):
        """
        Класс выполняет декомпозицию временного ряда с помощью анализа сингулярного спектра.
        Предполагается, что значения временного ряда записываются через равные промежутки времени.
        Принимает параметры:
        - tseries: DataFrame, np.array, list.
            Получение исходного временного ряда.
        - L: int.
            Длина окна. 2 <= L <= N/2, N - длина временного ряда.
            Значение >= 2.
        - save_mem: bool.
            Позволяет экономить память, не сохраняя сгенерированные матрицы. Только для длинных временных рядов со значениями более 1000.
            По умолчанию значение True.
        """
        # Проверка типа входных данных
        if not isinstance(tseries, self.__supported_types):
            raise TypeError('Неподдерживаемый объект, попробуйте формат DataFrame, np.array, list')

        # Проверка значения величины окна L
        self.N = len(tseries)
        if not 2 <= L <= self.N/2:
            raise ValueError('Величина окна должна быть в интервале между (2, N/2).')

        # Задаем значение величины окна
        self.L = L
        self.orig_TS = pd.Series(tseries) # оригинальный временной ряд, как эталон для сравнения реконструкции
        self.K = self.N - self.L + 1 # количество векторов-строк траекторной матрицы

        # Вставляем временной ряд в матрицу траекторий,
        # транспонирование матрицы траекторий, сформированной из временного ряда
        self.X = np.array([self.orig_TS.values[i:L+i] for i in range(0, self.K)]).T

        # Разложение матрицы траекторий на компоненты:
        # - U - матрица левых сингулярных векторов(L*L)(форма базисных функций)
        # - E - диагональная матрица сингулярных значений(L*K)(важность каждой компоненты)
        # - V^T - транспонированная матрица правых сингулярных векторов(K*K)(коэффициенты разложения)
        self.U, self.Sigma, VT = np.linalg.svd(self.X, full_matrices=False) # full_matrices=False - добавил для экономии ресурса
        # Ранг матрицы
        self.d = np.linalg.matrix_rank(self.X)
        # Для экономии памяти вместо L компонентов - d компонентов
        self.TS_comps = np.zeros((self.N, self.d))

        # Если не экономим память
        if not save_mem:
            # Построим и сохраним все элементарные матрицы
            self.X_elem = np.array([ self.Sigma[i]*np.outer(self.U[:,i], VT[i,:]) for i in range(self.d) ])

            # Усреднение элементарных матриц по диагонали, сохранение их в виде столбцов массива           
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            # Матрица правых сингулярных векторов
            self.V = VT.T
        else:
            # Реконструкция элементарных матриц не сохраняя их
            for i in range(self.d):
                X_elem = self.Sigma[i]*np.outer(self.U[:,i], VT[i,:])
                X_rev = X_elem[::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.X_elem = 'Перезапустите функцию с save_mem=False, чтобы сохранить элементарные матрицы.'
            
            # Массив может быть очень большим, не сохраняем его
            self.V = 'Перезапустите функцию с save_mem=False, чтобы сохранить матрицу V.'
        
        # Расчет взвешенной корреляции
        self.calc_wcorr()
            
    def components_to_df(self, n=0):
        """
        Компановка всех компонентов в DataFrame.
        Принимает параметры:
        - n: int.
        Возвращает:
        - DataFrame.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        
        # Создание списка столбцов с названиями F0, F1, F2, ...
        cols = ['F{}'.format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)
            
    
    def reconstruct(self, indices):
        """
        Восстановление временного ряда из его элементарных компонентов, используя заданные индексы.
        Принимает параметры:
        - indices: int, list, slice(n,m) object.
        Возвращает:
        - series, DataFrame.
        """
        if isinstance(indices, int): indices = [indices]
        
        ts_vals = self.TS_comps[:,indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)
    
    def calc_wcorr(self):
        """
        Вычисляет взвешенную матрицу корреляций временного ряда.
        """
        # Расчет весов для компенсации разного количества диагоналей
        w = np.array(list(np.arange(self.L)+1) + [self.L]*(self.K-self.L-1) + list(np.arange(self.L)+1)[::-1])
        # Взвешенное скалярное произведение
        def w_inner(F_i, F_j):
            return w.dot(F_i*F_j)
        
        # Вычисление взвешенных норм, ||F_i||_w, затем инвертирование.
        F_wnorms = np.array([w_inner(self.TS_comps[:,i], self.TS_comps[:,i]) for i in range(self.d)])
        F_wnorms = F_wnorms**-0.5
        
        # Расчет взвешенной корреляции
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i+1,self.d):
                self.Wcorr[i,j] = abs(w_inner(self.TS_comps[:,i], self.TS_comps[:,j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j,i] = self.Wcorr[i,j]
    
    def plot_wcorr(self,
                   min: int=None,
                   max: int=None,
                   matrix_width: int=800,
                   matrix_height: int=800,
                   sign: pd.DataFrame=None,
                   upload_path: str=None) -> None:
        """
        Строит взвешенную матрицу корреляций для разложенного временного ряда.
        Принимает параметры:
        - min: int.
            Минимальное количество компонентов.
        - max: int.
            Максимальное количество компонентов.            
        Возвращает:
        - графические элементы.
        """
        # Задаем параметры мин и макс для больших матриц корреляций
        if min is None:
            min = 0
        if max is None:
            max = self.d
        # Вычисление взвешенной корреляции
        if self.Wcorr is None:
            self.calc_wcorr()

        # адаптация тепловой карты
        component_names = [f'Component_{i}' for i in range(min, max)]
        wcorr_df = pd.DataFrame(self.Wcorr[min:max,min:max], index=component_names, columns=component_names)
        fig = px.imshow(wcorr_df, color_continuous_scale='Plasma', text_auto='.2f', title=f'Взвешенная матрица корреляций')
        fig.update_layout(
            width=matrix_width,
            height=matrix_height,
            xaxis_title='i',
            yaxis_title='j')
        if upload_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f') # именует файл в милисекундах %f
            filename = f'Взвешенная корреляционная матрица признака_{sign.name}_{timestamp}.html'
            filepath = os.path.join(upload_path, filename)
            fig.write_html(filepath)
        fig.show()
