# импортируем необходимые для работы библиотеки
import numpy as np
import warnings
# убираем незначительные предупреждения
warnings.filterwarnings('ignore')

class StatisticsCalc:

    def __init__(self, 
                 sample:list=None):

        self.sample = sample

    # расчет показателей выборки
    def sample_stats(self):
        """
        Производит расчет следующих показателей выборки: стандартное отклонение, стандартная ошибка, среднее, медиана, дисперсия.
        Принимает параметры:
        - sample: DataFrame
            Выборка.
        Возвращает параметры:
        - std_dev: float
            Стандартное отклонение.
        - std_err: float
            Стандартная ошибка.
        - mean: float
            Выборочное среднее.
        - median: float
            Медиана.
        - var: float
            Дисперсия.
        Ограничения по размеру выборки: явных ограничений нет.
        """
        std_dev = np.std(self.sample, ddof=1) # число степеней свободы, оценка стандартного отклонения генеральной совокупности при работе с выборкой данных, а не со всей генеральной совокупностью
        std_err = std_dev / np.sqrt(len(self.sample))
        mean = np.mean(self.sample)
        median = np.median(self.sample)
        var = np.var(self.sample, ddof=1) # число степеней свободы, оценка стандартного отклонения генеральной совокупности при работе с выборкой данных, а не со всей генеральной совокупностью
        return std_dev, std_err, mean, median, var
