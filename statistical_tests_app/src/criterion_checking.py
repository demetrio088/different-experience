# импортируем необходимые для работы библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.multicomp as mc
import scipy.stats as st
import warnings
# убираем незначительные предупреждения
warnings.filterwarnings('ignore')

class CriterionCheck:

    def __init__(self, 
                 criterion_name: str,
                 title: str=None, 
                 sample: np.ndarray=None,
                 samples: list=None,
                 alpha: float=0.05):
        
        self.alpha = alpha
        if criterion_name == 'shapiro_uilk':
            self.title = title
            self.sample = sample
        elif criterion_name == 'test_leven':
            self.samples = samples
        elif criterion_name == 'post_analysis_anova_tukey':
            self.samples = samples
            
    # проверка характера распределения критерием Шапиро-Уилка
    def shapiro_uilk(self):
        """
        Критерий Шапиро-Уилка подтверждает или опровергает нормальность распределения.
        Принимает параметры:
        - title: string
            Название выборки.
        - sample: DataFrame
            Выборка.
        - alpha: float
            P-значение для заданной гипотезы.
        Возвращает параметры:
        - stat: float
            Статистика теста.
        - p: float
            P-значение рассчитанное для заданной гипотезы.
        Ограничения по размеру выборки: возможны искажения в выборках при n < 3 и n > 5000.
        """
        stat, p = st.shapiro(self.sample)
        if p < self.alpha:
            print(f'Статистика по выборке {self.title} - {stat}. P-value - {p}. Отклонить гипотезу о нормальности распределения.')
        else:
            print(f'Статистика по выборке {self.title} - {stat}. P-value - {p}. Принять гипотезу о нормальности распределения.')
        return stat, p
   
    # проверка дисперсии на гомогенность - тест Левена
    def test_leven(self):
        """
        Тест Левена подтверждает или опровергает гипотезу о гомогенности(схожесть) дисперсии выборок.
        Принимает параметры:
        - *samples: DataFrame
            Выборки.
        - alpha: float
            P-значение для заданной гипотезы.
        Возвращает параметры:
        - stat: float
            Статистика теста.
        - p: float
            P-значение рассчитанное для заданной гипотезы.
        Ограничения по размеру выборки: n >= 2.
        """
        stat, p = st.levene(*[np.asarray(s).flatten() for s in self.samples]) # DataFrame сворачивается в одномерный массив
        if p < self.alpha:
            print(f'Статистика Левена по выборкам - {stat}. P-value - {p}. Отвергаем гипотезу о гомогенности дисперсий выборок.')
        else:
            print(f'Статистика Левена по выборкам - {stat}. P-value - {p}. Принимаем гипотезу о гомогенности дисперсий выборок.')
        return stat, p

    # постанализ для параметрического теста ANOVA - проведение теста Тьюки HSD
    def post_analysis_anova_tukey(self):
        """
        Выполняет попарное сравнение средних значений нескольких групп после получения значимого результата ANOVA(p < 0.05).
        Принимает параметры:
        - *samples: DataFrame
            Выборки, которые сравниваются друг с другом.
        - alpha: float
            P-значение для заданной гипотезы.
        Возвращает параметры:
        - tukey_post_result: table
            Таблица попарного сравнения групп между собой с указанием разницы средних,
            p-value, доверительные интервалы для каждой пары выборок, reject-флаг,
            показывающий отвергается ли Н0 при заданном уровне alpha для заданной пары выборок.
        Ограничения по количеству групп: n > 2.
        Ограничения по количеству наблюдений: n > 3.
        """
        tukey_df = pd.DataFrame()
        for sample in self.samples:
            if hasattr(sample, 'columns'):
                group_name = sample.columns[0]
                sample = sample.rename(columns={group_name: 'Значения'})
                sample['Группа'] = group_name
                tukey_df = pd.concat([tukey_df, sample], ignore_index=True)
            else:
                for i, sample in enumerate(self.samples):
                    # для каждой входящей единицы массива - преобразуем каждый в датафрейм
                    if type(sample).__name__ != 'DataFrame':
                        sample = pd.DataFrame({f'Выборка_{i+1}': sample})
                    group_name = sample.columns[0]
                    sample = sample.rename(columns={group_name: 'Значения'})
                    sample['Группа'] = group_name
                    tukey_df = pd.concat([tukey_df, sample], ignore_index=True)                         
        tukey_post_result = mc.pairwise_tukeyhsd(endog=tukey_df['Значения'], groups=tukey_df['Группа'], alpha=self.alpha)
        print(tukey_post_result.summary())
        tukey_post_result.plot_simultaneous()
        plt.show()
        return tukey_post_result
