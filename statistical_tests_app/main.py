# импортируем необходимые для работы библиотеки
import pandas as pd
import warnings
from src.utils.generators import generate_random_normal, generate_random, generate_dependent_samples
from src.pipelines import Pipelines
# убираем незначительные предупреждения
warnings.filterwarnings('ignore')

import argparse

def main():
    #parser = argparse.ArgumentParser(description="Пример с одним аргументом")
    #parser.add_argument('--name1', default='Выборка_1', help='Выборка_1')
    #parser.add_argument('--name2', default='Выборка_2', help='Выборка_2')
    #args = parser.parse_args()
    
    #name_1 = args.name1
    #name_2 = args.name2

    # ПРОВЕРКА ПАРАМЕТРИЧЕСКИХ ТЕСТОВ
    # сгенерируем 1 тестовую выборку и среднее для одновыборочного t-теста Стьюдента
    sample_1 = pd.DataFrame(data=generate_random_normal(1000, 33, 1), columns=['Выборка_1'])
    test_mean = 43

    # одновыборочный тест Стьюдента
    pipeline = Pipelines(alpha=0.05)
    pipeline.pipeline_all_params(samples=[sample_1],
                                 names=['Выборка 1'],
                                 indicator=test_mean,
                                 type_test='student_one_sample',
                                 subtype_test='two-sided',
                                 equal_var=True)
   
    # сгенерируем 2 тестовые выборки для двухвыборочного t-теста Стьюдента
    sample_2 = pd.DataFrame(data=generate_random_normal(1000, 33, 2), columns=['Выборка_2'])
    sample_3 = pd.DataFrame(data=generate_random_normal(500, 33, 2), columns=['Выборка_3'])
    # двухвыборочный тест Стьюдента
    pipeline_1 = Pipelines(alpha=0.05)
    pipeline_1.pipeline_all_params(samples=[sample_2, sample_3],
                                 names=['Выборка 2', 'Выборка 3'],
                                 indicator=None,
                                 type_test='student_two_samples',
                                 subtype_test='two-sided',
                                 equal_var=True)
    
    # сгенерируем 3 тестовые выборки для однофакторного теста ANOVA
    sample_4 = pd.DataFrame(data=generate_random_normal(1000, 33, 2), columns=['Выборка_4'])
    sample_5 = pd.DataFrame(data=generate_random_normal(500, 33, 2), columns=['Выборка_5'])
    sample_6 = pd.DataFrame(data=generate_random_normal(500, 33, 2), columns=['Выборка_6'])
    # однофакторный тест ANOVA со сходными средними
    pipeline_2 = Pipelines(alpha=0.05)
    pipeline_2.pipeline_all_params(samples=[sample_4, sample_5, sample_6],
                                 names=['Выборка 4', 'Выборка 5', 'Выборка 6'],
                                 indicator=None,
                                 type_test='anova_one',
                                 subtype_test=None,
                                 equal_var=None) 
    
    # сгенерируем 3 тестовые выборки для однофакторного теста ANOVA
    sample_7 = pd.DataFrame(data=generate_random_normal(1000, 40, 2), columns=['Выборка_7'])
    sample_8 = pd.DataFrame(data=generate_random_normal(500, 27, 2), columns=['Выборка_8'])
    sample_9 = pd.DataFrame(data=generate_random_normal(500, 38, 2), columns=['Выборка_9'])
    # однофакторный тест ANOVA с отличающимися средними
    pipeline_3 = Pipelines(alpha=0.05)
    pipeline_3.pipeline_all_params(samples=[sample_7, sample_8, sample_9],
                                 names=['Выборка 7', 'Выборка 8', 'Выборка 9'],
                                 indicator=None,
                                 type_test='anova_one',
                                 subtype_test=None,
                                 equal_var=None)
    
    # ПРОВЕРКА НЕПАРАМЕТРИЧЕСКИХ ТЕСТОВ
    # сгенерируем 2 тестовые выборки для критерия Манна-Уитни для двух случайных независимых выборок
    sample_10 = pd.DataFrame(data=generate_random(500, 500), columns=['Выборка_10'])
    sample_11 = pd.DataFrame(data=generate_random(700, 500), columns=['Выборка_11'])
    pipeline_4 = Pipelines(alpha=0.05)
    pipeline_4.pipeline_all_nonparams(samples=[sample_10, sample_11],
                                     names=['Выборка_10', 'Выборка_11'],
                                     type_test='mann-uitni',
                                     subtype_test='two-sided')
    
    
    # сгенерируем 2 тестовые выборки для критерия Колмогорова-Смирнова для двух случайных независимых выборок
    sample_12 = pd.DataFrame(data=generate_random(500, 1500), columns=['Выборка_12'])
    sample_13 = pd.DataFrame(data=generate_random(700, 500), columns=['Выборка_13'])
    pipeline_5 = Pipelines(alpha=0.05)
    pipeline_5.pipeline_all_nonparams(samples=[sample_12, sample_13],
                                     names=['Выборка_12', 'Выборка_13'],
                                     type_test='kolmogorov-smirnov',
                                     subtype_test='two-sided')

    # генерируем две зависимые случайные выборки и проверяем работу теста Вилкоксона
    before_sample, after_sample = generate_dependent_samples(3, 100)
    before_sample = pd.DataFrame(data=before_sample, columns=['До'])
    after_sample = pd.DataFrame(data=after_sample, columns=['После'])
    pipeline_6 = Pipelines(alpha=0.05)
    pipeline_6.pipeline_all_nonparams(samples=[before_sample, after_sample],
                                     names=['До', 'После'],
                                     type_test='wilcoxon',
                                     subtype_test='two-sided')

if __name__ == "__main__":
    main()