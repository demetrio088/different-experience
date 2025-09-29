# импорт библиотек
import pandas as pd
import numpy as np
from scipy.stats import trim_mean
from statsmodels.robust.scale import mad

class StatsIndicators:

    def __init__(self,
                 data: pd.DataFrame):

        self.data = data

    # расчет показателей выборки
    def data_stats(self):
        """
        Производит расчет следующих показателей выборки: среднее, усеченное среднее, медиана, медианное абсолютное отклонение, стандартное отклонение, стандартная ошибка, дисперсия.
        Принимает параметры:
        - data: DataFrame
            Выборки датафрейма.
        Возвращает параметры:
        - pivot_table: pivot_table
            Сводная таблица по показателям.
        Ограничения по размеру выборки: явных ограничений нет.
        """
        pivot_dictionary = {'Статистика':['Среднее', 'Усеченное среднее', 'Медиана', 'Медианное абсолютное отклонение', 'Стандартное отклонение', 'Стандартная ошибка', 'Дисперсия']}

        for sample in self.data.columns:
            pivot_dictionary[f'{sample}'] = [self.data[sample].mean(),
                                                    trim_mean(self.data[sample], 0.1),
                                                    self.data[sample].median(),
                                                    mad(self.data[sample]),
                                                    np.std(self.data[sample], ddof=1),
                                                    np.std(self.data[sample], ddof=1)/np.sqrt(len(self.data[sample])),
                                                    np.var(self.data[sample])]
        data_table = pd.DataFrame(pivot_dictionary)
        data_table = data_table.round(5)
        pivot_table = pd.pivot_table(data_table,
                                     columns='Статистика',
                                     values=list(pivot_dictionary.keys()),
                                    sort=False)
        return pivot_table
