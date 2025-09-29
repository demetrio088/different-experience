# импортируем необходимые для работы библиотеки
import numpy as np
import scipy.stats as st
import warnings
# убираем незначительные предупреждения
warnings.filterwarnings('ignore')

class ParametricTest:

    def __init__(self, 
                 test_name: str,
                 sample_1: np.ndarray=None, 
                 sample_2: np.ndarray=None, 
                 sample_mean: float=None, 
                 samples: list=None, 
                 test_type: str=None, 
                 equal_var: bool=True,
                 alpha:float=0.05):
        
        self.alpha = alpha
        if test_name == 'student_test_one_sample':
            self.sample_1 = sample_1.flatten()
            self.sample_mean = sample_mean
            self.test_type = test_type
        elif test_name == 'student_test_two_samples':
            self.samples = samples
            self.test_type = test_type
            self.equal_var = equal_var
        elif test_name == 'anova_one':
            self.samples = samples
            self.equal_var = equal_var

    # t-тест Стьюдента одновыборочный
    def student_test_one_sample(self):
        """
        Выполняет одновыборочный t-тест Стьюдента.
        Принимает параметры:
        - sample: DataFrame
            Выборка.
        - sample_mean: float, int
            Предполагаемое среднее, с которым сверяется выборочное среднее.
        - test_type: string
            Выбор типа теста: 
                1. 'two-sided' - двусторонний тест(интересует любое отклонение),
                2. 'less' - левосторонний тест(предполагается, что среднее выборки меньше заданного среднего),
                3. 'greater' - правосторонний тест(предполагается, что среднее выборки больше заданного среднего).
        - alpha: float
            P-значение для заданной гипотезы.
        Возвращает параметры:
        - stat: float
            Статистика теста.
        - p: float
            P-значение рассчитанное для заданной гипотезы.
        Ограничения по размеру выборки: n > 5.
        """
        stat, p = st.ttest_1samp(self.sample_1, self.sample_mean, alternative=self.test_type)
        if p < self.alpha:
            print(f'Статистика Стьюдента по выборке и среднему - {stat}. P-value - {p}. Отвергаем нулевую гипотезу: среднее значение выборки отличается от предполагаемого среднего.')
        else:
            print(f'Статистика Стьюдента по выборке и среднему - {stat}. P-value - {p}. Не отвергаем нулевую гипотезу: нет оснований утверждать, что среднее выборки отличается от предполагаемого среднего.')
        return stat, p
    
    # t-тест Стьюдента двухвыборочный
    def student_test_two_samples(self):
        """
        Выполняет двухвыборочный t-тест Стьюдента.
        Принимает параметры:
        - sample_1, sample_2: DataFrame
            Выборки, которые сравниваются друг с другом.
        - test_type: string
            Выбор типа теста:
                1. 'two-sided' - двусторонний тест(интересует любое отклонение),
                2. 'less' - левосторонний тест(предполагается, что среднее выборки №1 меньше среднего выборки №2),
                3. 'greater' - правосторонний тест(предполагается, что среднее выборки №1 больше среднего выборки №2).
        - equal_var: bool
            - если True - предполагается равенство дисперсий,
            - если False - дисперсии не равны/не гомогенны(тест Уэлча).
        - alpha: float
            P-значение для заданной гипотезы.
        Возвращает параметры:
        - stat: float
            Статистика теста.
        - p: float
            P-значение рассчитанное для заданной гипотезы.
        Ограничения по размеру выборки: n > 5.
        """
        stat, p = st.ttest_ind(*[np.asarray(s).flatten() for s in self.samples], alternative=self.test_type, equal_var=self.equal_var)
        if p < self.alpha:
            print(f'Статистика Стьюдента по выборкам - {stat}. \n P-value - {p}. Отвергаем нулевую гипотезу. \n')
        else:
            print(f'Статистика Стьюдента по выборкам - {stat}. \n P-value - {p}. Не отвергаем нулевую гипотезу. \n')
        return stat, p

    # однофакторный тест ANOVA
    def anova_one(self):
        """
        Выполняет однофакторный дисперсионный анализ выборок: ANOVA не сравнивает средние напрямую, а использует соотношение 
        межгрупповой и внутригрупповой дисперсии, чтобы определить, являются ли наблюдаемые различия между средними значениями
        статистически значимыми или случайными.
        Принимает параметры:
        - *samples: DataFrame
            Выборки, которые сравниваются друг с другом.
        - alpha: float
            P-значение для заданной гипотезы.
        Возвращает параметры:
        - f_stat: float
            F-статистика теста.
        - p: float
            P-значение рассчитанное для заданной гипотезы.
        Ограничения по количеству выборок: n > 2.
        """
        f_stat, p = st.f_oneway(*[np.asarray(s).flatten() for s in self.samples])
        
        if p < self.alpha:
            print(f'F-критерий ANOVA по группам выборок - {f_stat}. P-value - {p}. Отвергаем нулевую гипотезу о равенстве средних.')
        else:
            print(f'F-критерий ANOVA по группам выборок - {f_stat}. P-value - {p}. Не отвергаем нулевую гипотезу о равенстве средних.')
        return f_stat, p    
