# импортируем необходимые для работы библиотеки
import warnings
import pandas as pd
from ssa_analysis import SSA
from src.ssa_checking_reconstruction import ssa_checking_reconstruction
from src.utils import Utils
from IPython.display import display
from src.visualization_data import VisualizationData
from src.calculation_indicators import StatsIndicators
from src.research_components_of_signs import ResearchComponentsOfSigns
from src.research_signs_pipeline import research_signs_pipeline

def main():

    # сброс настроек при каждом перезапуске main
    pd.reset_option('all')

    config = Utils().load_config()

    # проверка содержания конфигурационного файла
    print(config.keys())

    # получение конфигов по категориям
    options_config = config.get('options', {})
    path_config = config.get('paths', {})
    data_signs_config = config.get('data_signs', {})
    data_ssa_analysis = config.get('ssa_analysis', {})
    data_visualization = config.get('visualization', {})
    data_p_value = config.get('p_value', {})
    data_components = config.get('component_of_signs', {})

    # убираем незначительные предупреждения
    warnings.filterwarnings(options_config.get('warnings', 'default')) # по умолчанию - показывает первое вхождение каждого предупреждения

    # вывод данных в полном объеме в консоль в ide spyder
    pd.set_option('display.max_rows', options_config.get('max_rows', 60)) # значение 60 - по умолчанию
    pd.set_option('display.max_columns', options_config.get('max_columns', 0)) # значение 0 - автоматический вывод
    pd.set_option('display.width', options_config.get('width', 80)) # значение 80 - по умолчанию
    pd.set_option('display.max_colwidth', options_config.get('max_colwidth', 50)) # значение 50 - по умолчанию

    # ШАГ 0: загрузка и ознакомление с данными

    # загрузка файла
    data = Utils.load_csv(path_config['data_dir'], path_config['data_file'])
    # обзор основной информации датасета, ищем пропуски, дубликаты
    Utils.overview_data(data)
    # переименовываем столбцы, устанавливаем индекс
    data = data.rename(columns={'Unnamed: 0': 'time_row'})
    data = data.set_index('time_row')
    display(data.head(options_config.get('head', 5)))
    # построение гистограмм по признакам
    for i, value in enumerate(data.columns):
        VisualizationData(data).hist_data(title=value,
                                          upload_path=path_config['data_dir'])
    # построение графика по признакам
    VisualizationData(data).plot_data(titles=list(data.columns),
                                      upload_path=path_config['data_dir'])
    # построение коробчатых диаграммы по признакам
    VisualizationData(data).boxplot_data(titles=list(data.columns),
                                         upload_path=path_config['data_dir'])
    # построение матрицы корреляций по признакам
    VisualizationData(data).correlation_matrix_data(titles=list(data.columns),
                                                    upload_path=path_config['data_dir'])
    # выводим таблицу корреляций признаков
    display(data.corr())
    # обзор ключевых статистик по признакам
    display(StatsIndicators(data).data_stats())

    # ШАГ 1: исследование признаков через SSA

    # исследуем ПЕРВЫЙ признак
    # строим график для первого признака
    VisualizationData(data[data_signs_config['first_sign']]).plot_data(titles=[data_signs_config['first_sign']],
                                                                       upload_path=path_config['data_dir'])
    # исследование признака с параметрами L=3, L=6, L=12
    for window in data_ssa_analysis.get('research_windows', [3, 6, 12]):
        research_signs_pipeline(data=data[data_signs_config['first_sign']],
                                l=window,
                                visual_data_upload_path=path_config['data_dir'])

    # исследуем ВТОРОЙ признак
    # строим график для второго признака
    VisualizationData(data[data_signs_config['second_sign']]).plot_data(titles=[data_signs_config['second_sign']],
                                                                        upload_path=path_config['data_dir'])
    # исследование признака с параметрами L=3, L=6, L=12
    for window in data_ssa_analysis.get('research_windows', [3, 6, 12]):
        research_signs_pipeline(data=data[data_signs_config['second_sign']],
                                l=window,
                                visual_data_upload_path=path_config['data_dir'])

    # исследуем ТРЕТИЙ признак
    # строим график для третьего признака
    VisualizationData(data[data_signs_config['third_sign']]).plot_data(titles=[data_signs_config['third_sign']],
                                                                       upload_path=path_config['data_dir'])
    # исследование признака с параметрами L=3, L=6, L=12
    for window in data_ssa_analysis.get('research_windows', [3, 6, 12]):
        research_signs_pipeline(data=data[data_signs_config['third_sign']],
                                l=window,
                                visual_data_upload_path=path_config['data_dir'])

    # ШАГ 2: разложение признаков на компоненты

    # раскладываем ПЕРВЫЙ признак
    ssa_data_column1 = SSA(data[data_signs_config['first_sign']], data_ssa_analysis.get('decomposition_window', 2),
                           save_mem=data_ssa_analysis.get('save_mem', False))
    components_data_column1 = ssa_data_column1.components_to_df()
    # раскладываем ВТОРОЙ признак
    ssa_data_column2 = SSA(data[data_signs_config['second_sign']], data_ssa_analysis.get('decomposition_window', 2),
                           save_mem=data_ssa_analysis.get('save_mem', False))
    components_data_column2 = ssa_data_column2.components_to_df()
    # раскладываем ТРЕТИЙ признак
    ssa_data_column3 = SSA(data[data_signs_config['third_sign']], data_ssa_analysis.get('decomposition_window', 2),
                           save_mem=data_ssa_analysis.get('save_mem', False))
    components_data_column3 = ssa_data_column3.components_to_df()

    # ШАГ 3: исследование компонентов признаков

    # ШАГ 3.1: исследование основных компонентов признаков
    
    # исследуем ПЕРВЫЙ признак
    # проверка на стационарность
    ResearchComponentsOfSigns(data=components_data_column1[data_components['component_1']]).checking_stationarity()
    # проведение автокорреляционного анализа
    ResearchComponentsOfSigns(data=components_data_column1[data_components['component_1']][:data_visualization.get('acf_pacf_limit', 1000)],
                              sign=data[data_signs_config['first_sign']]).acf_analysis(visual_data_upload_path=path_config['data_dir'])
    ResearchComponentsOfSigns(data=components_data_column1[data_components['component_1']][:data_visualization.get('acf_pacf_limit', 1000)],
                              sign=data[data_signs_config['first_sign']]).pacf_analysis(visual_data_upload_path=path_config['data_dir'])
    # проведение частотного анализа Фурье
    ResearchComponentsOfSigns(data=components_data_column1[data_components['component_1']],
                              sign=data[data_signs_config['first_sign']]).ff_analysis(visual_limit=data_visualization.get('ff_limit', 1000),
                                                                                      visual_data_upload_path=path_config['data_dir'])
    # проведение тестов на нормальность распределения
    ResearchComponentsOfSigns(data=components_data_column1[data_components['component_1']]).shapiro_uilk(data_p_value.get('p_value', 0.05))
    # проведение анализа трендов
    ResearchComponentsOfSigns(data=components_data_column1[data_components['component_1']],
                              sign=data[data_signs_config['first_sign']]).rolling_trend(data_visualization.get('rolling_window_size', 50000),
                                                                                        visual_data_upload_path=path_config['data_dir'])
    ResearchComponentsOfSigns(data=components_data_column1[data_components['component_1']],
                              sign=data[data_signs_config['first_sign']]).linear_regression_trend(visual_data_upload_path=path_config['data_dir'])

    # исследуем ВТОРОЙ признак
    # проверка на стационарность
    ResearchComponentsOfSigns(data=components_data_column2[data_components['component_1']]).checking_stationarity()
    # проведение автокорреляционного анализа
    ResearchComponentsOfSigns(data=components_data_column2[data_components['component_1']][:data_visualization.get('acf_pacf_limit', 1000)],
                              sign=data[data_signs_config['second_sign']]).acf_analysis(visual_data_upload_path=path_config['data_dir'])
    ResearchComponentsOfSigns(data=components_data_column2[data_components['component_1']][:data_visualization.get('acf_pacf_limit', 1000)],
                              sign=data[data_signs_config['second_sign']]).pacf_analysis(visual_data_upload_path=path_config['data_dir'])
    # проведение частотного анализа Фурье
    ResearchComponentsOfSigns(data=components_data_column2[data_components['component_1']],
                              sign=data[data_signs_config['second_sign']]).ff_analysis(visual_limit=data_visualization.get('ff_limit', 1000),
                                                                                       visual_data_upload_path=path_config['data_dir'])
    # проведение тестов на нормальность распределения
    ResearchComponentsOfSigns(data=components_data_column2[data_components['component_1']]).shapiro_uilk(data_p_value.get('p_value', 0.05))
    # проведение анализа трендов
    ResearchComponentsOfSigns(data=components_data_column2[data_components['component_1']],
                              sign=data[data_signs_config['second_sign']]).rolling_trend(data_visualization.get('rolling_window_size', 50000),
                                                                                         visual_data_upload_path=path_config['data_dir'])
    ResearchComponentsOfSigns(data=components_data_column2[data_components['component_1']],
                              sign=data[data_signs_config['second_sign']]).linear_regression_trend(visual_data_upload_path=path_config['data_dir'])

    # исследуем ТРЕТИЙ признак
    # проверка на стационарность
    ResearchComponentsOfSigns(data=components_data_column3[data_components['component_1']]).checking_stationarity()
    # проведение автокорреляционного анализа
    ResearchComponentsOfSigns(data=components_data_column3[data_components['component_1']][:data_visualization.get('acf_pacf_limit', 1000)],
                              sign=data[data_signs_config['third_sign']]).acf_analysis(visual_data_upload_path=path_config['data_dir'])
    ResearchComponentsOfSigns(data=components_data_column3[data_components['component_1']][:data_visualization.get('acf_pacf_limit', 1000)],
                              sign=data[data_signs_config['third_sign']]).pacf_analysis(visual_data_upload_path=path_config['data_dir'])
    # проведение частотного анализа Фурье
    ResearchComponentsOfSigns(data=components_data_column3[data_components['component_1']],
                              sign=data[data_signs_config['third_sign']]).ff_analysis(visual_limit=data_visualization.get('ff_limit', 1000),
                                                                                      visual_data_upload_path=path_config['data_dir'])
    # проведение тестов на нормальность распределения
    ResearchComponentsOfSigns(data=components_data_column3[data_components['component_1']]).shapiro_uilk(data_p_value.get('p_value', 0.05))
    # проведение анализа трендов
    ResearchComponentsOfSigns(data=components_data_column3[data_components['component_1']],
                              sign=data[data_signs_config['third_sign']]).rolling_trend(data_visualization.get('rolling_window_size', 50000),
                                                                                        visual_data_upload_path=path_config['data_dir'])
    ResearchComponentsOfSigns(data=components_data_column3[data_components['component_1']],
                              sign=data[data_signs_config['third_sign']]).linear_regression_trend(visual_data_upload_path=path_config['data_dir'])

    # проведение анализа дисперсий основных компонентов признаков
    ResearchComponentsOfSigns([components_data_column1[data_components['component_1']],
                               components_data_column2[data_components['component_1']],
                               components_data_column3[data_components['component_1']]]).test_leven(data_p_value.get('p_value', 0.05))

    # оценка взаимосвязи основных компонентов признаков
    # склеиваем основные компоненты признаков в один датафрейм
    main_data_ssa = pd.DataFrame({'column1_F0': components_data_column1[data_components['component_1']],
                             'column2_F0': components_data_column2[data_components['component_1']],
                             'column3_F0': components_data_column3[data_components['component_1']]})
    # построение матрицы корреляций по основным компонентам и вывод таблицы корреляций
    VisualizationData(main_data_ssa).correlation_matrix_data(titles=list(main_data_ssa.columns),
                                                             upload_path=path_config['data_dir'])
    # выводим таблицу корреляций признаков
    display(main_data_ssa.corr())
    # обзор ключевых статистик по основным компонентам
    display(StatsIndicators(main_data_ssa).data_stats())

    # ШАГ 3.2: исследование остаточных компонентов признаков

    # исследуем ПЕРВЫЙ признак
    # проверка на стационарность
    ResearchComponentsOfSigns(data=components_data_column1[data_components['component_2']]).checking_stationarity()
    # проведение автокорреляционного анализа
    ResearchComponentsOfSigns(data=components_data_column1[data_components['component_2']][:data_visualization.get('acf_pacf_limit', 1000)],
                              sign=data[data_signs_config['first_sign']]).acf_analysis(visual_data_upload_path=path_config['data_dir'])
    ResearchComponentsOfSigns(data=components_data_column1[data_components['component_2']][:data_visualization.get('acf_pacf_limit', 1000)],
                              sign=data[data_signs_config['first_sign']]).pacf_analysis(visual_data_upload_path=path_config['data_dir'])
    # проведение частотного анализа Фурье
    ResearchComponentsOfSigns(data=components_data_column1[data_components['component_2']],
                              sign=data[data_signs_config['first_sign']]).ff_analysis(visual_limit=data_visualization.get('ff_limit', 1000),
                                                                                      visual_data_upload_path=path_config['data_dir'])
    # проведение тестов на нормальность распределения
    ResearchComponentsOfSigns(data=components_data_column1[data_components['component_2']]).shapiro_uilk(data_p_value.get('p_value', 0.05))
    # проведение анализа трендов
    ResearchComponentsOfSigns(data=components_data_column1[data_components['component_2']],
                              sign=data[data_signs_config['first_sign']]).rolling_trend(data_visualization.get('rolling_window_size', 50000),
                                                                                        visual_data_upload_path=path_config['data_dir'])
    ResearchComponentsOfSigns(data=components_data_column1[data_components['component_2']],
                              sign=data[data_signs_config['first_sign']]).linear_regression_trend(visual_data_upload_path=path_config['data_dir'])

    # исследуем ВТОРОЙ признак
    # проверка на стационарность
    ResearchComponentsOfSigns(data=components_data_column2[data_components['component_2']]).checking_stationarity()
    # проведение автокорреляционного анализа
    ResearchComponentsOfSigns(data=components_data_column2[data_components['component_2']][:data_visualization.get('acf_pacf_limit', 1000)],
                              sign=data[data_signs_config['second_sign']]).acf_analysis(visual_data_upload_path=path_config['data_dir'])
    ResearchComponentsOfSigns(data=components_data_column2[data_components['component_2']][:data_visualization.get('acf_pacf_limit', 1000)],
                              sign=data[data_signs_config['second_sign']]).pacf_analysis(visual_data_upload_path=path_config['data_dir'])
    # проведение частотного анализа Фурье
    ResearchComponentsOfSigns(data=components_data_column2[data_components['component_2']],
                              sign=data[data_signs_config['second_sign']]).ff_analysis(visual_limit=data_visualization.get('ff_limit', 1000),
                                                                                       visual_data_upload_path=path_config['data_dir'])
    # проведение тестов на нормальность распределения
    ResearchComponentsOfSigns(data=components_data_column2[data_components['component_2']]).shapiro_uilk(data_p_value.get('p_value', 0.05))
    # проведение анализа трендов
    ResearchComponentsOfSigns(data=components_data_column2[data_components['component_2']],
                              sign=data[data_signs_config['second_sign']]).rolling_trend(data_visualization.get('rolling_window_size', 50000),
                                                                                         visual_data_upload_path=path_config['data_dir'])
    ResearchComponentsOfSigns(data=components_data_column2[data_components['component_2']],
                              sign=data[data_signs_config['second_sign']]).linear_regression_trend(visual_data_upload_path=path_config['data_dir'])

    # исследуем ТРЕТИЙ признак
    # проверка на стационарность
    ResearchComponentsOfSigns(data=components_data_column3[data_components['component_2']]).checking_stationarity()
    # проведение автокорреляционного анализа
    ResearchComponentsOfSigns(data=components_data_column3[data_components['component_2']][:data_visualization.get('acf_pacf_limit', 1000)],
                              sign=data[data_signs_config['third_sign']]).acf_analysis(visual_data_upload_path=path_config['data_dir'])
    ResearchComponentsOfSigns(data=components_data_column3[data_components['component_2']][:data_visualization.get('acf_pacf_limit', 1000)],
                              sign=data[data_signs_config['third_sign']]).pacf_analysis(visual_data_upload_path=path_config['data_dir'])
    # проведение частотного анализа Фурье
    ResearchComponentsOfSigns(data=components_data_column3[data_components['component_2']],
                              sign=data[data_signs_config['third_sign']]).ff_analysis(visual_limit=data_visualization.get('ff_limit', 1000),
                                                                                      visual_data_upload_path=path_config['data_dir'])
    # проведение тестов на нормальность распределения
    ResearchComponentsOfSigns(data=components_data_column3[data_components['component_2']]).shapiro_uilk(data_p_value.get('p_value', 0.05))
    # проведение анализа трендов
    ResearchComponentsOfSigns(data=components_data_column3[data_components['component_2']],
                              sign=data[data_signs_config['third_sign']]).rolling_trend(data_visualization.get('rolling_window_size', 50000),
                                                                                        visual_data_upload_path=path_config['data_dir'])
    ResearchComponentsOfSigns(data=components_data_column3[data_components['component_2']],
                              sign=data[data_signs_config['third_sign']]).linear_regression_trend(visual_data_upload_path=path_config['data_dir']) 

    # проведение анализа дисперсий остаточных компонентов признаков
    ResearchComponentsOfSigns([components_data_column1[data_components['component_2']],
                               components_data_column2[data_components['component_2']],
                               components_data_column3[data_components['component_2']]]).test_leven(data_p_value.get('p_value', 0.05))

    # оценка взаимосвязи остаточных компонентов признаков
    # склеиваем остаточные компоненты признаков в один датафрейм
    remain_data_ssa = pd.DataFrame({'column1_F1': components_data_column1[data_components['component_2']],
                                    'column2_F1': components_data_column2[data_components['component_2']],
                                    'column3_F1': components_data_column3[data_components['component_2']]})
    # построение матрицы корреляций по остаточным компонентам и вывод таблицы корреляций
    VisualizationData(remain_data_ssa).correlation_matrix_data(titles=list(remain_data_ssa.columns),
                                                               upload_path=path_config['data_dir'])
    # выводим таблицу корреляций признаков
    display(remain_data_ssa.corr())
    # обзор ключевых статистик по остаточным компонентам
    display(StatsIndicators(remain_data_ssa).data_stats())

    # ШАГ 4: проверка работы SSA и реконструкция признаков

    ssa_checking_reconstruction(data=data[data_signs_config['first_sign']],
                                l=data_ssa_analysis.get('decomposition_window', 2),
                                visual_data_upload_path=path_config['data_dir'])
    ssa_checking_reconstruction(data=data[data_signs_config['second_sign']],
                                l=data_ssa_analysis.get('decomposition_window', 2),
                                visual_data_upload_path=path_config['data_dir'])
    ssa_checking_reconstruction(data=data[data_signs_config['third_sign']],
                                l=data_ssa_analysis.get('decomposition_window', 2),
                                visual_data_upload_path=path_config['data_dir'])

if __name__ == "__main__":
    main()
