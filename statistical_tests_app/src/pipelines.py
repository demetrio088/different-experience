# импортируем необходимые для работы библиотеки
from parametric_tests import ParametricTest
from nonparametric_tests import NonparametricTest
from utils.visual_checking import VizualAnalyzerParams
from criterion_checking import CriterionCheck
from utils.statistics_calc import StatisticsCalc
import warnings
# убираем незначительные предупреждения
warnings.filterwarnings('ignore')

class Pipelines:

    def __init__(self, alpha:float=0.05):
        
        self.alpha = alpha

    # пайплайн для параметрических тестов
    def pipeline_all_params(self, 
                            samples: list=None, 
                            names: list=None, 
                            indicator: float=None, 
                            alpha: float=0.05, 
                            type_test: str='',
                            subtype_test: str='',
                            equal_var: bool=True):
        """
        Сценарии пайплайна параметрических тестов
        Принимает параметры:
        - samples: DataFrame array
            Выборки, которые анализируются в рамках параметрических тестов.
        - names: string array
            Названия выборок, которые анализируются в рамках параметрических тестов.
        - indicator: float, int
            Предполагаемое среднее, с которым сверяется выборочное среднее(если необходимо для выбранного типа теста).
        - alpha: float
            P-значение для заданной гипотезы.
        - type_test: string
            Тип проводимого теста: student_one_sample, student_two_samples, anova_one, anova_two
        - subtype_test: string
            Выбор типа теста: 
                1. 'two-sided' - двусторонний тест(интересует любое отклонение),
                2. 'less' - левосторонний тест(предполагается, что среднее выборки меньше заданного среднего),
                3. 'greater' - правосторонний тест(предполагается, что среднее выборки больше заданного среднего).
        - equal_var: bool
            - если True - предполагается равенство дисперсий,
            - если False - дисперсии не равны/не гомогенны(тест Уэлча).
        Возвращает параметры: данные статистик и резюме по проверяемой гипотезе.
        """
        self.samples = [sample.values.flatten() for sample in samples]
        self.names = names
        self.indicator = indicator # sample_mean для проверки одновыборочным тестом Стьюдента
        self.type_test = type_test
        self.subtype_test = subtype_test
        self.equal_var = equal_var

        print(f'Анализ выборок: {", ".join(self.names)}')
        # строим гистограммы и QQ plot для визуальной оценки распределений и проверяем характер распределений для каждой выборки критерием Шапиро-Уилка
        shapiro_p = []
        for sample, name in zip(self.samples, self.names):
            VizualAnalyzerParams(sample, f'Гистограмма для выборки {name}').hist_sample()
            VizualAnalyzerParams(sample, f'QQ-plot для выборки - {name}').qq_plot_sample()
            # выводим показатели выборок
            stats_labels = [
                'Стандартное отклонение (σ)',
                'Стандартная ошибка (SE)',
                'Среднее значение (μ)',
                'Медиана',
                'Дисперсия (σ²)'
            ]
            print(f'Стат показатели выборки {name}:')
            sample_stat_result = StatisticsCalc(sample).sample_stats()
            for label, value in zip(stats_labels, sample_stat_result):
                print(f'{label}: {float(value)}')
            print('-------------------------------------------------')
            # проверяем характер распределения критерием Шапиро-Уилка
            print(f'Критерий Шапиро-Уилка по выборке {name}:')
            stat, p = CriterionCheck(criterion_name='shapiro_uilk', 
                                     title=name, 
                                     sample=sample, 
                                     alpha=alpha).shapiro_uilk()
            shapiro_p.append(p)
        print('-------------------------------------------------')
        # если по одной из выборок распределение не является нормальным -> рекомендуется использовать непараметрические тесты
        if all(i >= self.alpha for i in shapiro_p):
            # выбираем параметрический тест в зависимости от выбранного типа теста
            if type_test == 'student_one_sample':
                # проверяем итоговые статистики одновыборочного t-теста Стьюдента
                print(f'Одновыборочный тест Стьюдента по выборке {self.names[0]} для определения адекватности гипотез:')
                ParametricTest(test_name='student_test_one_sample',
                               alpha=self.alpha, 
                               sample_1=self.samples[0], 
                               sample_mean=self.indicator, 
                               test_type=self.subtype_test).student_test_one_sample()
            if type_test == 'student_two_samples':
                # проверяем дисперсии на гомогенность тестом Левена
                print(f'Тест Левена по выборкам {", ".join(self.names)}:')
                stat, p = CriterionCheck(criterion_name='test_leven',
                                         alpha=self.alpha,
                                         samples=self.samples).test_leven()
                print('-------------------------------------------------')
                if p > 0.05:
                    # проверяем итоговые статистики двухвыборочного t-теста Стьюдента
                    print(f'Двухвыборочный тест Стьюдента по выборкам {", ".join(self.names)} для определения адекватности гипотез:')
                    ParametricTest(test_name='student_test_two_samples',
                                   alpha=self.alpha,
                                   samples=self.samples,
                                   test_type=self.subtype_test,
                                   equal_var=self.equal_var).student_test_two_samples()
                else: 
                    print('Критерий гомогенности дисперсий Левена не выполняется, рекомендуется воспользоваться непараметрическими тестами.')
            if type_test == 'anova_one':
                # проверяем дисперсии на гомогенность тестом Левена
                print(f'Тест Левена по выборкам {", ".join(self.names)}:')
                stat, p = CriterionCheck(criterion_name='test_leven',
                                         alpha=self.alpha,
                                         samples=self.samples).test_leven()
                print('-------------------------------------------------')
                if p > 0.05:
                    # проверяем итоговые статистики тестом однофакторной ANOVA
                    f_stat, p = ParametricTest(test_name='anova_one',
                                               alpha=self.alpha,
                                               samples=self.samples).anova_one()
                    print('-------------------------------------------------')
                    # постанализ для параметрического теста ANOVA - проведение теста Тьюки HSD, если p < 0.05(обнаружены статистически значимые различия -> 
                    # -> применяем пост тест Тьюки для детализации статистически значимых различий между входящими группами)
                    if p < 0.05:
                        CriterionCheck(criterion_name='post_analysis_anova_tukey',
                                       alpha=self.alpha,
                                       samples=self.samples).post_analysis_anova_tukey()
                else:
                    print('Критерий гомогенности дисперсий Левена не выполняется, рекомендуется воспользоваться непараметрическими тестами.')
        else:
            print('Критерий нормальности распределений Шапиро-Уилка не выполняется, рекомендуется воспользоваться непараметрическими тестами.')

    # пайплайн для непараметрических тестов
    def pipeline_all_nonparams(self,
                              samples: list=None,
                              names: list=None,
                              alpha: float=0.05,
                              type_test: str='',
                              subtype_test: str=''):
        """
        Сценарии пайплайна непараметрических тестов
        Принимает параметры:
        - samples: DataFrame array
            Выборки, которые анализируются в рамках непараметрических тестов.
        - names: string array
            Названия выборок, которые анализируются в рамках параметрических тестов.
        - indicator: float, int
            Предполагаемое среднее, с которым сверяется выборочное среднее(если необходимо для выбранного типа теста).
        - alpha: float
            P-значение для заданной гипотезы.
        - type_test: string
            Тип проводимого теста: mann_uitni, kolmogorov_smirnov, wilcoxon
        - subtype_test: string
            Выбор типа теста: 
                1. 'two-sided' - двусторонний тест(интересует любое отклонение),
                2. 'less' - левосторонний тест(предполагается, что среднее выборки меньше заданного среднего),
                3. 'greater' - правосторонний тест(предполагается, что среднее выборки больше заданного среднего).
        Возвращает параметры: данные статистик и резюме по проверяемой гипотезе.
        """
        self.samples = [sample.values.flatten() for sample in samples]
        self.names = names
        self.type_test = type_test
        self.subtype_test = subtype_test

        print(f'Анализ выборок: {", ".join(self.names)}')
        # строим гистограммы и QQ plot для визуальной оценки распределений и проверяем характер распределений для каждой выборки
        shapiro_p = []
        for sample, name in zip(self.samples, self.names):
            VizualAnalyzerParams(sample, f'Гистограмма для выборки {name}').hist_sample()
            VizualAnalyzerParams(sample, f'QQ-plot для выборки - {name}').qq_plot_sample()
            # выводим показатели выборок
            stats_labels = [
                'Стандартное отклонение (σ)',
                'Стандартная ошибка (SE)',
                'Среднее значение (μ)',
                'Медиана',
                'Дисперсия (σ²)'
            ]
            print(f'Стат показатели выборки {name}:')
            sample_stat_result = StatisticsCalc(sample).sample_stats()
            for label, value in zip(stats_labels, sample_stat_result):
                print(f'{label}: {float(value)}')
            print('-------------------------------------------------')
            # проверяем характер распределения критерием Шапиро-Уилка, для непараметрических тестов распределение не должно быть нормальным,
            # в ином случае применяем параметрические тесты
            print(f'Критерий Шапиро-Уилка по выборке {name}:')
            stat, p = CriterionCheck(criterion_name='shapiro_uilk', 
                                     title=name, 
                                     sample=sample, 
                                     alpha=alpha).shapiro_uilk()
            shapiro_p.append(p)
        print('-------------------------------------------------')
        # если по одной из выборок распределение не является нормальным -> используем непараметрические тесты
        if any(i < self.alpha for i in shapiro_p):
            # задействуем непараметрический тест в зависимости от выбранного типа теста
            if self.type_test == 'mann-uitni':
                # проверяем дисперсии на гомогенность тестом Левена, т.е. для критерия Манна-Уитни - желательно, чтобы дисперсии были гомогенны
                print(f'Тест Левена по выборкам {", ".join(self.names)}:')
                stat, p = CriterionCheck(criterion_name='test_leven',
                                         alpha=self.alpha,
                                         samples=self.samples).test_leven()
                print('-------------------------------------------------')
                if p > 0.05:
                    # проверяем итоговые статистики непараметрического теста Манна-Уитни
                    print(f'Двухвыборочный тест Манна-Уитни по выборкам {", ".join(self.names)} для определения адекватности гипотез:')
                    NonparametricTest(alpha=self.alpha, 
                                      samples=self.samples, 
                                      test_type=self.subtype_test).mann_uitni()
                else: 
                    print('Критерий гомогенности дисперсий Левена не выполняется, рекомендуется воспользоваться другим типом теста.')
            if self.type_test == 'kolmogorov-smirnov':
                # проверяем итоговые статистики непараметрического теста Колмогорова-Смирнова
                print(f'Двухвыборочный тест Колмогорова-Смирнова по выборкам {", ".join(self.names)} для определения адекватности гипотез:')
                NonparametricTest(alpha=self.alpha, 
                                  samples=self.samples, 
                                  test_type=self.subtype_test).kolmogorov_smirnov()
            if self.type_test == 'wilcoxon':
                # проверяем итоговые статистики непараметрического теста Вилкоксона
                print(f'Двухвыборочный тест Вилкоксона по выборкам {", ".join(self.names)} для определения адекватности гипотез:')
                NonparametricTest(alpha=self.alpha, 
                                  samples=self.samples, 
                                  test_type=self.subtype_test).wilcoxon()
        else:
            print('Критерий нормальности распределений Шапиро-Уилка выполняется, рекомендуется воспользоваться параметрическими тестами.')