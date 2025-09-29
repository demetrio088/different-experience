# импортируем необходимые для работы библиотеки
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
# убираем незначительные предупреждения
warnings.filterwarnings('ignore')

class VizualAnalyzerParams:

    def __init__(self, 
                 data: pd.DataFrame, 
                 title: str):
        """
        Инициализирует визуальное представление данных.
        Принимает параметры:
        - data: DataFrame
            Выборка.
        - title: string
            Название выборки.
        """
        self.data = data
        self.title = title

    # ПРОВЕРКА УСЛОВИЙ ПРИМЕНЕНИЯ ДЛЯ ПАРАМЕТРИЧЕСКИХ ТЕСТОВ
    # визуальное представление - гистограмма
    def hist_sample(self) -> None:
        """
        Отрисовывает гистограмму.
        Принимает параметры:
        - data: DataFrame
            Выборка.
        - title: string
            Название выборки.
        Задано ограничение по количеству столбцов: n = 30.
        """
        plt.hist(self.data, bins=30)
        plt.title(self.title)
        plt.xlabel('Значения')
        plt.ylabel('Частота')
        plt.show()
    
    # визуальное представление - Q-Q plot
    def qq_plot_sample(self) -> None:
        """
        Отрисовывает Q-Q plot.
        Принимает параметры:
        - data: DataFrame
            Выборка.
        - title: string
            Название выборки.
        Задано ограничение по линии: 'q'(тип контрольной линии, проходит через 1й и 3й квартили, показывает совпадение с теоретическим распределением).
        """
        sm.qqplot(self.data, line='q')
        plt.title(self.title)
        plt.show()