# импорт библиотек
import pandas as pd
from IPython.display import display
from ssa_analysis import SSA
from visualization_data import VisualizationData
from data_reconstruction_research import data_reconstruction_research

# функция исследования признаков
def research_signs_pipeline(data: pd.DataFrame, 
                            l: int=None,
                            visual_data_upload_path: str=''):
    """
    Проводит разложение признаков на компоненты и их исследование.
    Принимает параметры:
    - data: DataFrame
         Выборка.
    - l: int
         L - количество компонентов разложения признака.
    - visual_data_upload_path: str
        Путь к директории для сохранения графиков/гистограмм/прочей визуализации.
    Возвращает параметры:
    - графики признаков и компонентов: plot
         Построенные графики.
    """
    # раскладываем на компоненты, выбираем рекомендуемый для тестирования и минимальных промежутков размер окна L
    print(f'Разложение на компоненты признака {data.name}\n')
    ssa_df_column = SSA(data, l, save_mem=True)
    components_df_column = ssa_df_column.components_to_df()
    display(components_df_column.head(10))
    display(components_df_column.tail(10))
    display(components_df_column.describe())

    # строим графики для разложенных компонентов признака
    print(f'Строим графики для разложенных компонентов признака {data.name}\n')
    VisualizationData(components_df_column.reset_index(drop=True)).plot_data(titles=list(components_df_column.reset_index(drop=True).columns),
                                                                             sub_title=f'Графики для разложенных компонентов признака {data.name} c L={l}',
                                                                             upload_path=visual_data_upload_path)
    # проверка размера данных при L для контроля
    print(f'Проверка размера данных при L для контроля {data.name}\n')
    display(ssa_df_column.TS_comps.shape)

    # смотрим матрицу взвешенной корреляции по компонентам для отражения связей между компонентами модели признака
    print(f'Смотрим матрицу взвешенной корреляции по компонентам для отражения связей между компонентами модели признака {data.name}\n')
    ssa_df_column.plot_wcorr(min=0, 
                             max=l, 
                             sign=data,
                             upload_path=visual_data_upload_path)

    # ранжируем компоненты по убыванию важности
    print(f'Ранжируем компоненты по убыванию важности признака {data.name}\n')
    sigma_column = ssa_df_column.Sigma
    sigma_column = sigma_column / sigma_column.sum()

    # запуск функции исследования реконструкции в соответствии с количеством компонентов разложения признака
    data_reconstruction_research(data=data,
                                 sigma_column=sigma_column,
                                 ssa_df_column=ssa_df_column,
                                 l=l,
                                 visual_data_upload_path=visual_data_upload_path)
