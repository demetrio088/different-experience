import streamlit as st
import pandas as pd
import numpy as np
from src.parametric_tests_streamlit import ParametricTest
from src.nonparametric_tests_streamlit import NonparametricTest
from src.utils.visual_checking_streamlit import VizualAnalyzerParams
from src.criterion_checking_streamlit import CriterionCheck
from src.utils.statistics_calc import StatisticsCalc
from src.utils.generators import generate_random_normal, generate_random, generate_dependent_samples
st.set_option('deprecation.showPyplotGlobalUse', False)
# создание общего вида страницы
st.set_page_config(
    page_title='A/B эксперименты',
    page_icon=':1234:',
    layout='wide',
    initial_sidebar_state='auto',
    menu_items={
        'About': 'Эта страница иллюстрирует A/B эксперименты'
    }
)

# сохраняемые данные
base_vars = {
    'type_input_data': None,
    'data': [],
    'sample_size': 0,
    'sample_size_1': 0,
    'sample_size_2': 0,    
    'sample_size_3': 0,  
    'mean': 0,
    'mean_1': 0,
    'mean_2': 0,
    'mean_3': 0,
    'std': 0,
    'std_1': 0,
    'std_2': 0,
    'std_3': 0,
    'max_value_1': 0,
    'max_value_2': 0,
    'method': None,
    'p_value': None,
    'p_value_shapiro_uilk': [],
    'homogeneity_of_dispersion': None,
    'check_norm_status': None,
    'flag_norm_status_container': None,
    'check_test_status': None,
    'do_graphics': None,
    'comparing_mean': None,
    'type_of_test': None,
    'name_of_test': None
}
 
for name, value in base_vars.items():
    if name not in st.session_state:
        st.session_state[name] = value
 
# левая боковая панель меню управления
with st.sidebar:
    # вывод названия левого сайдбара
    st.sidebar.title('Командная панель эксперимента')
    # типы входных данных
    input_data_type = ['Случайная нормальная выборка',
                       'Две случайные нормальные выборки',
                       'Три случайные нормальные выборки',
                       'Две случайные выборки',
                       'Две зависимые выборки']
    # типы одновыборочных параметрических методов
    parametric_one_test_type = ['Одновыборочный t-тест Стьюдента']
    # типы двухвыборочных параметрических методов
    parametric_two_test_type = ['Двухвыборочный t-тест Стьюдента']
    # типы трехвыборочных параметрических тестов
    parametric_three_test_type = ['Однофакторный тест ANOVA']
    # типы непараметрических двухвыборочных методов для независимых выборок
    nonparametric_independent_test_type = ['Критерий Манна-Уитни',
                       'Критерий Колмогорова-Смирнова']
    # типы непараметрических двухвыборочных методов для зависимых выборок
    nonparametric_dependent_test_type = ['Критерий Вилкоксона']
    # типы тестов
    type_of_tests = ['two-sided',
                       'less',
                       'greater']    
 
    # выбор типа входных данных из ниспадающего списка
    st.sidebar.selectbox('Выберите тип входных данных',
                         input_data_type, key='type_input_data')

    # если выбран тот или иной тип входных данных - работает соответствующий сценарий
    # сценарий для одновыборочного теста Стьюдента
    if st.session_state.type_input_data == 'Случайная нормальная выборка':
        st.write('Введите параметры для генерации входных данных')
        st.session_state.sample_size = st.number_input('Размер выборки', 0, 10000)
        st.session_state.mean = st.number_input('Выборочное среднее', 0, 10000)
        st.session_state.std = st.number_input('Стандартное отклонение', 0, 10000)
        if st.button('Сгенерировать данные'):
            st.session_state.data = [pd.DataFrame(generate_random_normal(st.session_state.sample_size, st.session_state.mean, st.session_state.std), columns=['Выборка 1'])]
            st.session_state.check_norm_status = 1
        st.divider()
        if st.session_state.check_norm_status == 1:
            st.write('Введите критерий значимости P-value')
            st.session_state.p_value = st.number_input('Критерий значимости выборки', 0.01, 1.00, 0.05, 0.01)
            # проверка на нормальность входного распределения
            if st.button('Проверить распределение на нормальность'):
                st.session_state.do_graphics = 1
        st.divider()
    # сценарий для двухвыборочного теста Стьюдента
    if st.session_state.type_input_data == 'Две случайные нормальные выборки':
        st.write('Введите параметры для генерации входных данных')
        st.session_state.sample_size_1 = st.number_input('Размер выборки 1', 0, 10000)
        st.session_state.mean_1 = st.number_input('Выборочное среднее 1', 0, 10000)
        st.session_state.std_1 = st.number_input('Стандартное отклонение 1', 0, 10000)
        st.session_state.sample_size_2 = st.number_input('Размер выборки 2', 0, 10000)
        st.session_state.mean_2 = st.number_input('Выборочное среднее 2', 0, 10000)
        st.session_state.std_2 = st.number_input('Стандартное отклонение 2', 0, 10000)
        if st.button('Сгенерировать данные'):
            st.session_state.data = [pd.DataFrame(generate_random_normal(st.session_state.sample_size_1, st.session_state.mean_1, st.session_state.std_1), columns=['Выборка 1']),
                                     pd.DataFrame(generate_random_normal(st.session_state.sample_size_2, st.session_state.mean_2, st.session_state.std_2), columns=['Выборка 2'])]
            st.session_state.check_norm_status = 2
        st.divider()
        if st.session_state.check_norm_status == 2:
            st.write('Введите критерий значимости P-value')
            st.session_state.p_value = st.number_input('Критерий значимости выборки', 0.01, 1.00, 0.05, 0.01)
            # проверка на нормальность входного распределения
            if st.button('Проверить распределение на нормальность'):
                st.session_state.do_graphics = 1
        st.divider()
    # сценарий для трехвыборочного теста ANOVA
    if st.session_state.type_input_data == 'Три случайные нормальные выборки':
        st.write('Введите параметры для генерации входных данных')
        st.session_state.sample_size_1 = st.number_input('Размер выборки 1', 0, 10000)
        st.session_state.mean_1 = st.number_input('Выборочное среднее 1', 0, 10000)
        st.session_state.std_1 = st.number_input('Стандартное отклонение 1', 0, 10000)
        st.session_state.sample_size_2 = st.number_input('Размер выборки 2', 0, 10000)
        st.session_state.mean_2 = st.number_input('Выборочное среднее 2', 0, 10000)
        st.session_state.std_2 = st.number_input('Стандартное отклонение 2', 0, 10000)
        st.session_state.sample_size_3 = st.number_input('Размер выборки 3', 0, 10000)
        st.session_state.mean_3 = st.number_input('Выборочное среднее 3', 0, 10000)
        st.session_state.std_3 = st.number_input('Стандартное отклонение 3', 0, 10000)
        if st.button('Сгенерировать данные'):
            st.session_state.data = [pd.DataFrame(generate_random_normal(st.session_state.sample_size_1, st.session_state.mean_1, st.session_state.std_1), columns=['Выборка 1']),
                                     pd.DataFrame(generate_random_normal(st.session_state.sample_size_2, st.session_state.mean_2, st.session_state.std_2), columns=['Выборка 2']),
                                     pd.DataFrame(generate_random_normal(st.session_state.sample_size_2, st.session_state.mean_2, st.session_state.std_2), columns=['Выборка 3'])]
            st.session_state.check_norm_status = 3
        st.divider()
        if st.session_state.check_norm_status == 3:
            st.write('Введите критерий значимости P-value')
            st.session_state.p_value = st.number_input('Критерий значимости выборки', 0.01, 1.00, 0.05, 0.01)
            # проверка на нормальность входного распределения
            if st.button('Проверить распределение на нормальность'):
                st.session_state.do_graphics = 1
        st.divider()
    # сценарий для двухвыборочного теста Манна-Уитни или Колмогорова-Смирнова
    if st.session_state.type_input_data == 'Две случайные выборки':
        st.write('Введите параметры для генерации входных данных')
        st.session_state.sample_size_1 = st.number_input('Размер выборки 1', 0, 10000)
        st.session_state.max_value_1 = st.number_input('Максимальное значение выборки 1', 0, 10000)
        st.session_state.sample_size_2 = st.number_input('Размер выборки 2', 0, 10000)
        st.session_state.max_value_2 = st.number_input('Максимальное значение выборки 2', 0, 10000)
        if st.button('Сгенерировать данные'):
            st.session_state.data = [pd.DataFrame(generate_random(st.session_state.sample_size_1,  st.session_state.max_value_1), columns=['Выборка 1']),
                                     pd.DataFrame(generate_random(st.session_state.sample_size_2, st.session_state.max_value_2), columns=['Выборка 2'])]
            st.session_state.check_norm_status = 4
        st.divider()
        if st.session_state.check_norm_status == 4:
            st.write('Введите критерий значимости P-value')
            st.session_state.p_value = st.number_input('Критерий значимости выборки', 0.01, 1.00, 0.05, 0.01)
            # проверка на нормальность входного распределения
            if st.button('Проверить распределение на нормальность'):
                st.session_state.do_graphics = 1
        st.divider()
    # сценарий для двухвыборочного теста Вилкоксона
    if st.session_state.type_input_data == 'Две зависимые выборки':
        st.write('Введите параметры для генерации входных данных')
        st.session_state.sample_size_1 = st.number_input('Размер выборки 1', 0, 10000)
        st.session_state.scale_1 = st.number_input('Масштаб выборки 1', 0, 10000)
        if st.button('Сгенерировать данные'):
            before, after = generate_dependent_samples(st.session_state.scale_1, st.session_state.sample_size_1)
            st.session_state.data = [pd.DataFrame(before, columns=['До']),
                                     pd.DataFrame(after, columns=['После'])]
            st.session_state.check_norm_status = 5
        st.divider()
        if st.session_state.check_norm_status == 5:
            st.write('Введите критерий значимости P-value')
            st.session_state.p_value = st.number_input('Критерий значимости выборки', 0.01, 1.00, 0.05, 0.01)
            # проверка на нормальность входного распределения
            if st.button('Проверить распределение на нормальность'):
                st.session_state.do_graphics = 1
        st.divider()

# правое верхнее рабочее поле для отображения выходных данных
with st.container():
    st.markdown('**Обзор входных данных**')
    for df in st.session_state.data:
        st.dataframe(df)
 
st.divider()
 
# правое среднее рабочее поле для отображения аналитики и выводов по входным данным
with st.container():
    st.markdown('**Проверка распределения на нормальность и вывод выборочных статистик**')
    if st.session_state.do_graphics == 1:
        # строим гистограммы и QQ plot для визуальной оценки распределений и проверяем характер распределений для каждой выборки критерием Шапиро-Уилка
        for sample, name in zip([np.asarray(df).flatten() for df in st.session_state.data], [df.columns.tolist() for df in st.session_state.data]):
            st.pyplot(VizualAnalyzerParams(sample, f'Гистограмма для выборки {name[0]}').hist_sample())
            st.scatter_chart(sample)
            st.pyplot(VizualAnalyzerParams(sample, f'QQ-plot для выборки - {name[0]}').qq_plot_sample())
            # выводим показатели выборок
            stats_labels = [
                '- стандартное отклонение (σ)',
                '- стандартная ошибка (SE)',
                '- среднее значение (μ)',
                '- медиана',
                '- дисперсия (σ²)'
            ]
            st.divider()
            st.write(f'**Статистические показатели выборки "{name[0]}":**')
            sample_stat_result = StatisticsCalc(sample).sample_stats()
            for label, value in zip(stats_labels, sample_stat_result):
                st.write(f'{label}: {value:.2f}')
            st.divider()
            # проверяем характер распределения критерием Шапиро-Уилка
            st.write(f'**Критерий Шапиро-Уилка по выборке "{name[0]}"**')
            state, p = CriterionCheck(criterion_name='shapiro_uilk', 
                                      title=name[0], 
                                      sample=sample, 
                                      alpha=st.session_state.p_value).shapiro_uilk()
            st.session_state.p_value_shapiro_uilk.append(p)
        # проверяем дисперсии на гомогенность тестом Левена
        if len(st.session_state.data) >= 2:
            st.write(f'**Тест Левена по выборкам {", ".join([", ".join(df.columns) for df in st.session_state.data])}:**')
            state, p = CriterionCheck(criterion_name='test_leven', 
                                      alpha=st.session_state.p_value, 
                                      samples=st.session_state.data).test_leven()
            st.session_state.homogeneity_of_dispersion = p
        st.session_state.flag_norm_status_container = 1
 
# левая боковая панель меню управления
with st.sidebar:
    # игнор ошибки отсутствия данных в переменной
    try:
        if len(st.session_state.data) < 2:
            if ((st.session_state.flag_norm_status_container is not None) 
                and all(i is not None for i in st.session_state.p_value_shapiro_uilk)
                and st.session_state.p_value is not None
                and all(float(i) >= float(st.session_state.p_value) for i in st.session_state.p_value_shapiro_uilk)):
                st.write('Введите среднее и тип теста')
                st.session_state.comparing_mean = st.number_input('Среднее', 0, 10000)
                st.sidebar.selectbox('Тип теста',
                                     type_of_tests, key='type_of_test')
                st.sidebar.selectbox('Наименование теста',
                                     parametric_one_test_type, key='name_of_test')
                if st.button('Запустить тест'):
                    st.session_state.check_test_status = 'student_test_one_sample'
        elif len(st.session_state.data) == 2:
            if ((st.session_state.flag_norm_status_container is not None) 
                and all(i is not None for i in st.session_state.p_value_shapiro_uilk)
                and st.session_state.p_value is not None
                and all(float(i) >= float(st.session_state.p_value) for i in st.session_state.p_value_shapiro_uilk)
                and float(st.session_state.homogeneity_of_dispersion) > float(st.session_state.p_value)):
                st.sidebar.selectbox('Тип теста',
                                     type_of_tests, key='type_of_test')
                st.sidebar.selectbox('Наименование теста',
                                     parametric_two_test_type, key='name_of_test')
                if st.button('Запустить тест'):
                    st.session_state.check_test_status = 'student_test_two_samples'
            else:
                # если какой-либо критерий нормальности или гомогенности не выполняется и выборки не связаны - используем непараметрические тест
                # для независимых выборок
                if ((st.session_state.check_norm_status == 4) 
                    and (st.session_state.flag_norm_status_container is not None) 
                    and all(i is not None for i in st.session_state.p_value_shapiro_uilk)
                    and st.session_state.p_value is not None):
                    st.sidebar.selectbox('Тип теста',
                                         type_of_tests, key='type_of_test')
                    st.sidebar.selectbox('Наименование теста',
                                         nonparametric_independent_test_type, key='name_of_test')
                    if st.button('Запустить тест'):
                        if st.session_state.name_of_test == 'Критерий Манна-Уитни':
                            st.session_state.check_test_status = 'mann-uitni'
                        if st.session_state.name_of_test == 'Критерий Колмогорова-Смирнова':
                            st.session_state.check_test_status = 'kolmogorov-smirnov'
                # если какой-либо критерий нормальности или гомогенности не выполняется и выборки связаны - используем непараметрические тест
                # для зависимых
                elif ((st.session_state.check_norm_status == 5) 
                    and (st.session_state.flag_norm_status_container is not None) 
                    and all(i is not None for i in st.session_state.p_value_shapiro_uilk)
                    and st.session_state.p_value is not None):
                    st.sidebar.selectbox('Тип теста',
                                         type_of_tests, key='type_of_test')
                    st.sidebar.selectbox('Наименование теста',
                                         nonparametric_dependent_test_type, key='name_of_test')
                    if st.button('Запустить тест'):
                        st.session_state.check_test_status = 'wilcoxon'
        elif len(st.session_state.data) == 3:
            if ((st.session_state.flag_norm_status_container is not None) 
                and all(i is not None for i in st.session_state.p_value_shapiro_uilk)
                and st.session_state.p_value is not None
                and all(float(i) >= float(st.session_state.p_value) for i in st.session_state.p_value_shapiro_uilk)
                and float(st.session_state.homogeneity_of_dispersion) > float(st.session_state.p_value)):
                st.sidebar.selectbox('Наименование теста',
                                     parametric_three_test_type, key='name_of_test')
                if st.button('Запустить тест'):
                    st.session_state.check_test_status = 'anova_one'
    except:
        pass

st.divider()

# правое нижнее рабочее поле для отображения аналитики и выводов по входным данным
with st.container():
    st.markdown('**Выводы по данным**')
    if st.session_state.check_test_status == 'student_test_one_sample':
        st.write(f'**Использовался одновыборочный тест Стьюдента по выборке "{st.session_state.data[0].columns[0]}" для определения адекватности гипотез.**')
        if len(st.session_state.data) > 0:
            ParametricTest(test_name=st.session_state.check_test_status, 
                           alpha=st.session_state.p_value,
                           sample_1=np.asarray(st.session_state.data[0]).flatten(),
                           sample_mean=st.session_state.comparing_mean,
                           test_type=st.session_state.type_of_test).student_test_one_sample()
    if st.session_state.check_test_status == 'student_test_two_samples':
        st.write(f'**Использовался двухвыборочный тест Стьюдента по выборкам "{", ".join([", ".join(df.columns) for df in st.session_state.data])}" для определения адекватности гипотез.**')
        if len(st.session_state.data) > 0:
            ParametricTest(test_name=st.session_state.check_test_status, 
                           alpha=st.session_state.p_value,
                           samples=st.session_state.data,
                           test_type=st.session_state.type_of_test).student_test_two_samples()
    if st.session_state.check_test_status == 'anova_one':
        st.write(f'**Использовался трехвыборочный тест ANOVA по выборкам "{", ".join([", ".join(df.columns) for df in st.session_state.data])}" для определения адекватности гипотез.**')
        if len(st.session_state.data) > 0:
            f_stat, p = ParametricTest(test_name=st.session_state.check_test_status, 
                                       alpha=st.session_state.p_value, 
                                       samples=st.session_state.data).anova_one()
            # постанализ для параметрического теста ANOVA - проведение теста Тьюки HSD, если p < 0.05(обнаружены статистически значимые различия -> 
            # -> применяем пост тест Тьюки для детализации статистически значимых различий между входящими группами)
            if p < 0.05:
                CriterionCheck(criterion_name='post_analysis_anova_tukey',
                               alpha=st.session_state.p_value,
                               samples=st.session_state.data).post_analysis_anova_tukey()
    if st.session_state.check_test_status == 'mann-uitni':
        st.write(f'**Использовался двухвыборочный тест Манна-Уитни по выборкам "{", ".join([", ".join(df.columns) for df in st.session_state.data])}" для определения адекватности гипотез.**')
        if len(st.session_state.data) > 0:
            NonparametricTest(alpha=st.session_state.p_value, 
                              samples=st.session_state.data, 
                              test_type=st.session_state.type_of_test).mann_uitni()
    if st.session_state.check_test_status == 'kolmogorov-smirnov':
        st.write(f'**Использовался двухвыборочный тест Колмогорова-Смирнова по выборкам "{", ".join([", ".join(df.columns) for df in st.session_state.data])}" для определения адекватности гипотез.**')
        if len(st.session_state.data) > 0:
            NonparametricTest(alpha=st.session_state.p_value, 
                              samples=st.session_state.data, 
                              test_type=st.session_state.type_of_test).kolmogorov_smirnov()
    if st.session_state.check_test_status == 'wilcoxon':
        st.write(f'**Использовался двухвыборочный тест Вилкоксона по выборкам "{", ".join([", ".join(df.columns) for df in st.session_state.data])}" для определения адекватности гипотез.**')
        if len(st.session_state.data) > 0:
            NonparametricTest(alpha=st.session_state.p_value, 
                              samples=st.session_state.data, 
                              test_type=st.session_state.type_of_test).wilcoxon()
