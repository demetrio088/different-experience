# импортируем необходимые для работы библиотеки
import warnings
import pandas as pd
import sys
import os
from datetime import datetime
from src.ssa_analysis import SSA
from src.utils import Utils
from src.research_trend import research_trend

# функция проверки и оценки трендов в признаке
def trend(data: pd.DataFrame,
          l_target: int=2,
          save_mem: bool=True):
    """
    Проводит SSA разложение признаков датафрейма на трендовые и остаточные компоненты.
    Принимает параметры:
    - data: DataFrame
         Выборка.
    - l_target: int
         Количество компонентов разложения признака.
    - save_mem: bool.
         Позволяет экономить память, не сохраняя сгенерированные матрицы. Только для длинных временных рядов со значениями более 1000.
         По умолчанию значение True.
    Возвращает параметры:
    - DataFrames: pd.DataFrame
         Датафрейм основных признаков и датафрейм остаточных признаков.
    """
    main_data_ssa = pd.DataFrame()
    remain_data_ssa = pd.DataFrame()
    # раскладываем признаки на компоненты
    for column in data.columns:
        ssa_data_column = SSA(data[column], l_target, save_mem=save_mem)
        components_data_column = ssa_data_column.components_to_df()
        for sign in components_data_column.columns:
            if sign == 'F0':
                main_data_ssa[f'{column}_{sign}'] = components_data_column[sign]
            else:
                remain_data_ssa[f'{column}_{sign}'] = components_data_column[sign]
    return main_data_ssa, remain_data_ssa

def main():
    
    # сброс настроек при каждом перезапуске main
    pd.reset_option('all')
    
    # загрузка конфигурации
    config = Utils().load_config()

    # получение конфигов по категориям
    options_config = config.get('options', {})
    path_config = config.get('paths', {})
    data_ssa_analysis = config.get('ssa_analysis', {})
    data_visualization = config.get('visualization', {})
    data_p_value = config.get('p_value', {})

    # убираем незначительные предупреждения
    warnings.filterwarnings(options_config.get('warnings', 'default')) # по умолчанию - показывает первое вхождение каждого предупреждения

    # вывод данных в полном объеме в ide spyder
    pd.set_option('display.max_rows', options_config.get('max_rows', 60)) # значение 60 - по умолчанию
    pd.set_option('display.max_columns', options_config.get('max_columns', 0)) # значение 0 - автоматический вывод
    pd.set_option('display.width', options_config.get('width', 80)) # значение 80 - по умолчанию
    pd.set_option('display.max_colwidth', options_config.get('max_colwidth', 50)) # значение 50 - по умолчанию

    ##########################################################################
    # инициализация записи состояний в файл
    timestamp_file_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    recorder_path = os.path.join(path_config['trend_dir'], f'recorded_{timestamp_file_name}.txt')

    orig_out = sys.stdout
    recorder_file = open(recorder_path, 'a', encoding='utf-8') 

    def time_stamp_print(message):
        if message.strip():
            timestamp = '[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f') + ']'
            final_string = timestamp + message
            recorder_file.write(final_string)
            recorder_file.flush()

    sys.stdout = type('', (object,), {
        'write': lambda self, message: time_stamp_print(message),
        'flush': lambda self: None
        })()
    ##########################################################################

    # загрузка файла
    data = Utils.load_csv(path_config['data_dir'], path_config['data_file'])
    # переименовываем столбцы, устанавливаем индекс
    try:
        data = data.rename(columns={'Unnamed: 0': 'time_row'})
        data = data.set_index('time_row')
    except KeyError:
        print('Столбец "Unnamed: 0" не найден.')

    # запуск функции по исследованию трендов для признака
    main_components, remain_components = trend(data=data,
                                               l_target=2,
                                               save_mem=True)
    # сохранение датафреймов компонентов
    Utils().save_csv(main_components,
                     path_config['trend_dir'],
                     'main_components.csv')
    Utils().save_csv(remain_components,
                     path_config['trend_dir'],
                     'remain_components.csv')
    # запуск функции исследования тренда в основных компонентах признаков
    research_trend(main_components[:5000],
                   rolling_window_size=500,
                   upload_path_trend=path_config['trend_dir'],
                   lag=500,
                   period_range=[3, 15])

    # закрываем очередь печати в файл
    sys.stdout = orig_out
    recorder_file.close()

    print('Проверьте отчет по тренду в папке "trend_research"')
    
if __name__ == "__main__":
    main()
