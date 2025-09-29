# импорт библиотек
import pandas as pd
import numpy as np
from visualization_data import VisualizationData
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy.fft import fft, fftfreq
import scipy.stats as st
from scipy.stats import linregress
from statsmodels.stats.diagnostic import het_breuschpagan

class ResearchComponentsOfSigns:
    def __init__(self, 
                 data: pd.DataFrame,
                 sign: str=''):
        self.data = data
        self.sign = sign

    # проведение проверки на стационарность тестом Дики-Фуллера
    def checking_stationarity(self):
        """
        Проводит проверку на стационарность тестом Дики-Фуллера.
        Принимает параметры:
        - data: DataFrame
             Выборка.
        Возвращает параметры:
        - adf_test_column_p_value: float
            P-значение.
        """
        adf_test_column_p_value = adfuller(self.data)
        print(f'P-value Дики-Фуллера для признака составило {adf_test_column_p_value[1]}\n')
        #print(f'P-value Дики-Фуллера для признака {self.data.name} составило {adf_test_column_p_value[1]}\n')
        print('СПРАВОЧНО: при значении Дики-Фуллера p-value < 0.05 ряд стационарен, если p-value >= 0.05, то ряд нестационарен.\n' + 
              'Если нестационарность подтверждается, то характеристики рядов нестабильны, данные не предсказуемы,\n' + 
              'а модели на их основе могут быть нестабильнми и неточными.\n')
        return adf_test_column_p_value[1]

    def acf_analysis(self,
                     lag: int=50,
                     xlabel: str='Лаг',
                     ylabel: str='Автокорреляция',
                     visual_data_upload_path: str=''):
        """
        Проводит автокорреляционный анализ.
        Принимает параметры:
        - data: DataFrame
             Выборка.
        - sign: str
             Признак.
        - lag: int
             Величина лага.
        - xlabel: str
             Название оси X.
        - ylabel: str
             Название оси Y.
        - visual_data_upload_path: str
             Путь к директории для сохранения графиков/гистограмм/прочей визуализации.
        Возвращает параметры:
        - график функции автокорреляции: plot
            Построенный график.
        """
        self.lag = lag
        self.xlabel = xlabel
        self.ylabel = ylabel
        acf_data = acf(self.data, nlags=self.lag, fft=True)
        VisualizationData(pd.DataFrame({
            'index': np.arange(len(acf_data)),
            'acf': acf_data})).plot_data(titles=['acf'],
                                         sub_title=f'Функция автокорреляции (ACF) признака {self.sign.name}', 
                                         x_title=self.xlabel, 
                                         y_title=self.ylabel, 
                                         upload_path=visual_data_upload_path)
        print('СПРАВОЧНО: по графикам определяются признаки силы тренда, признаки нестационарности.')
    
    def pacf_analysis(self,
                     lag: int=50,
                     xlabel: str='Лаг',
                     ylabel: str='Частичная автокорреляция',
                     visual_data_upload_path: str=''):
        """
        Проводит анализ частичной автокорреляции.
        Принимает параметры:
        - data: DataFrame
             Выборка.
        - sign: str
             Признак.
        - lag: int
             Величина лага.
        - xlabel: str
             Название оси X.
        - ylabel: str
             Название оси Y.
        - visual_data_upload_path: str
             Путь к директории для сохранения графиков/гистограмм/прочей визуализации.
        Возвращает параметры:
        - график функции частичной автокорреляции: plot
            Построенный график.
        """
        self.lag = lag
        self.xlabel = xlabel
        self.ylabel = ylabel
        pacf_data = pacf(self.data, nlags=self.lag, method='yw')
        VisualizationData(pd.DataFrame({
            'index': np.arange(len(pacf_data)),
            'pacf': pacf_data})).plot_data(titles=['pacf'],
                                           sub_title=f'Функция частичной автокорреляции (PACF) признака {self.sign.name}', 
                                           x_title=self.xlabel, 
                                           y_title=self.ylabel, 
                                           upload_path=visual_data_upload_path)

        print('СПРАВОЧНО: по графикам определяются признаки силы тренда, признаки нестационарности.')
    
    def ff_analysis(self,
                     xlabel: str='Частота',
                     ylabel: str='Амплитуда',
                     visual_limit: int=None,
                     visual_data_upload_path: str=''):
        """
        Проводит визуальное представление-частотного спектра Фурье.
        Принимает параметры:
        - data: DataFrame
             Выборка.
        - sign: str
             Признак.
        - xlabel: str
             Название оси X.
        - ylabel: str
             Название оси Y.
        - visual_limit: int
             Лимит выборки для детализации визуализации.
        - visual_data_upload_path: str
             Путь к директории для сохранения графиков/гистограмм/прочей визуализации.
        Возвращает параметры:
        - график частотной функции Фурье: plot
            Построенный график.
        """
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.visual_limit = visual_limit
        fft_values = fft(np.array(self.data))
        fft_frequencies = fftfreq(len(self.data))
        amplitude_spectrum = np.abs(fft_values)
        VisualizationData(pd.DataFrame({
            'index': np.arange(len(fft_frequencies[:self.visual_limit])),
            'amplitude_spectrum': amplitude_spectrum[:self.visual_limit]})).plot_data(titles=['amplitude_spectrum'], 
                                                                                      sub_title=f'Частотный анализ Фурье признака {self.sign.name}', 
                                                                                      x_title=self.xlabel,
                                                                                      y_title=self.ylabel,
                                                                                      upload_path=visual_data_upload_path)
        print('СПРАВОЧНО: по графику частотного спектра Фурье можно определить пики и интерпритировать величину и частоту пик и колебаний в качестве характера цикличности и трендовости ряда данных.')

    def shapiro_uilk(self, alpha: float=0.05):
        """
        Критерий Шапиро-Уилка подтверждает или опровергает нормальность распределения.
        Принимает параметры:
        - data: DataFrame
            Выборка данных.
        - alpha: float
            P-значение для заданной гипотезы.
        Возвращает параметры:
        - stat: float
            Статистика теста.
        - p: float
            P-значение рассчитанное для заданной гипотезы.
        Ограничения по размеру выборки: возможны искажения в выборках при n < 3 и n > 5000.
        """
        self.alpha = alpha
        stat, p = st.shapiro(self.data)
        if p < alpha:
            print(f'Статистика Шапиро-Уилка по выборке {self.data.name} - {stat}. P-value - {p}. Отклонить гипотезу о нормальности распределения.')
        else:
            print(f'Статистика Шапиро-Уилка по выборке {self.data.name} - {stat}. P-value - {p}. Принять гипотезу о нормальности распределения.')
        return stat, p
    
    def test_leven(self, alpha: float=0.05):
        """
        Тест Левена подтверждает или опровергает гипотезу о гомогенности(схожесть) дисперсии выборок.
        Принимает параметры:
        - data: DataFrame
            Выборки данных.
        - alpha: float
            P-значение для заданной гипотезы.
        Возвращает параметры:
        - stat: float
            Статистика теста.
        - p: float
            P-значение рассчитанное для заданной гипотезы.
        Ограничения по размеру выборки: n >= 2.
        """
        self.alpha = alpha
        stat, p = st.levene(*[np.asarray(s).flatten() for s in self.data]) # DataFrame сворачивается в одномерный массив
        if p < alpha:
            print(f'Статистика Левена по выборкам {", ".join([s.name for s in self.data])} - {stat}. P-value - {p}. Отвергаем гипотезу о гомогенности дисперсий выборок.')
        else:
            print(f'Статистика Левена по выборкам {", ".join([s.name for s in self.data])} - {stat}. P-value - {p}. Принимаем гипотезу о гомогенности дисперсий выборок.')
        return stat, p
    
    def rolling_trend(self,
                      window: int=10000,
                      visual_data_upload_path: str=''):
        """
        Скользящее среднее для визуального контроля тренда и скользящее стандартное отклонение для визуального определения изменчивости.
        Принимает параметры:
        - data: DataFrame
            Выборки данных.
        - sign: str
            Признак.
        - window: int
            Ширина окна.
        - visual_data_upload_path: str
            Путь к директории для сохранения графиков/гистограмм/прочей визуализации.
        Возвращает параметры:
        - график скользящего среднего и стандартного отклонения с заданной шириной окна: plot
            Построенный график.
        """
        self.window = window
        rolling_mean = self.data.rolling(window=self.window).mean()
        VisualizationData(pd.DataFrame({
            'index': range(len(self.data)),
            'rolling_mean': rolling_mean})).plot_data(titles=['rolling_mean'], 
                                                      sub_title=f'Скользящее среднее признака {self.sign.name}', 
                                                      upload_path=visual_data_upload_path)

        rolling_std = self.data.rolling(window=self.window).std()
        VisualizationData(pd.DataFrame({
            'index': range(len(self.data)),
            'rolling_std': rolling_std})).plot_data(titles=['rolling_std'], 
                                                    sub_title=f'Скользящее стандартное отклонение признака {self.sign.name}', 
                                                    upload_path=visual_data_upload_path)
        
    def linear_regression_trend(self,
                                visual_data_upload_path: str=''):
        """
        Определение тренда линейной регрессией.
        Принимает параметры:
        - data: DataFrame
            Выборка данных.
        - sign: str
            Признак.
        - visual_data_upload_path: str
            Путь к директории для сохранения графиков/гистограмм/прочей визуализации.
        Возвращает параметры:
        - график скользящего среднего и стандартного отклонения с заданной шириной окна: plot
            Построенный график.
        - slope: float
            Угол наклона тренда.
        - intercept: float
            Сдвиг.
        - r_value: float
            Коэффициент корреляии между x и y.
        - p_value: float
            Статистическая значимость наклона.
        - std_err: float
            Стандартная ошибка наклона.
        """
        x1 = np.arange(len(self.data))
        y1 = self.data.values
        slope, intercept, r_value, p_value, std_err = linregress(x1, y1)
        print(f'Угол наклона(+восходящий тренд/-нисходящий) тренд: {slope:.5f}')
        print(f'Сдвиг(точка пересечения с осью Y): {intercept:.5f}')
        print(f'Коэффициент корреляции между x и y: {r_value:.5f}')
        print(f'Статистическая значимость наклона(гипотеза Н0 - нет тренда, если p-value < 0.05, то тренд существует): {p_value:.5f}')
        print(f'Стандартная ошибка наклона(чем меньше, тем точнее оценка тренда): {std_err:.5f}\n')
        VisualizationData(pd.DataFrame({
            'index': x1,
            'input_data': y1,
            'trend_line': (intercept + slope*x1)})).plot_data(titles=['input_data', 'trend_line'],
                                                              sub_title=f'Визуализация входных данных и линии тренда признака {self.sign.name}', 
                                                              upload_path=visual_data_upload_path)

        return slope, intercept, r_value, p_value, std_err

    # проведение проверки гетероскедастичности тестом Бройша-Пагана
    def checking_heteroscedasticity(self, 
                                    exog: pd.DataFrame):
        """
        Проводит проверку на гетероскедастичность тестом Бройша-Пагана.
        Принимает параметры:
        - data: DataFrame
             Выборка.
        - exog: DataFrame
             Индекс как регрессор, изменение дисперсии во времени.
        Возвращает параметры:
        - bp_value: float
            P-значение.
        """
        bp_stat, bp_pvalue, bp_fstat, bp_fpvalue = het_breuschpagan(self.data, exog)
        print(f'P-value Бройша-Пагана для признака составило {bp_pvalue}\n')
        #print(f'P-value Бройша-Пагана для признака {self.data.name} составило {bp_pvalue}\n')
        print('СПРАВОЧНО: при значении Бройша-Пагана p-value < 0.05 остатки не являются гомоскедастическими(применение линейной регрессии может быть ошибочно), если p-value >= 0.05, то остатки гомоскедастические.\n')
        return bp_pvalue
