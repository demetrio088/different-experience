# импорт библиотек
import pandas as pd
from IPython.display import display
from visualization_data import VisualizationData

# функция исследования реконструкции
def data_reconstruction_research(data: pd.DataFrame,
                                 sigma_column: pd.Series,
                                 ssa_df_column: pd.DataFrame,
                                 l: int=None,
                                 visual_data_upload_path: str=''):
    """
    Проводит исследование реконструкции признаков.
    Принимает параметры:
    - data: DataFrame
         Выборка.
    -  sigma_column: pd.Series
         sigma_column(E) - диагональная матрица сингулярных значений(L*K)(важность каждой компоненты).
    - ssa_df_column: pd.DataFrame
         Разложенный на компоненты признак.
    - l: int
         L - количество компонентов разложения признака.
    - visual_data_upload_path: str
         Путь к директории для сохранения графиков/гистограмм/прочей визуализации.
    Возвращает параметры:
    - графики признаков и компонентов: plot
         Построенные графики.
    """
    if l <= 3:
        # Визуализация первых 3х компонентов
        print(f'Визуализация первых 3х компонент признака {data.name}\n')
        VisualizationData(pd.DataFrame({
            'index': range(len(sigma_column[:3])),
            'first_three_components': sigma_column[:3]})).plot_data(titles=['first_three_components'],
                                                                    sub_title=f'Визуализация первых трех компонент по убыванию важности признака {data.name}', 
                                                                    upload_path=visual_data_upload_path)
    
        # Показывает какое количество компонент объясняет основной процент информации
        print(f'Показывает какое количество компонент объясняет основной процент информации\n')
        display(sigma_column[:1].sum(), sigma_column[2:].sum())
    
        # Показывает сколько компонент необходимо для заданного процента информации
        print(f'Показывает сколько компонент необходимо для заданного процента информации\n')
        VisualizationData(pd.DataFrame({
            'index': range(len(sigma_column)),
            'cumulative_components': sigma_column.cumsum()})).plot_data(titles=['cumulative_components'],
                                                                        sub_title=f'Показывает сколько компонент необходимо для заданного процента информации признака {data.name}', 
                                                                        upload_path=visual_data_upload_path)
    
        # Реконструкция данных:
        # - F0 - тренд,
        # - F1, F2 - основные парные компоненты цикличности,
        # - F3, F4 - дополнительные парные компоненты цикличености,
        # - F5, F6 - цикличность более высокого масштаба,
        # - F7, F*** - вероятно, шум и случайные колебания.
        print(f'Реконструкция данных\n')
        VisualizationData(pd.DataFrame({
            'index': range(len(ssa_df_column.reconstruct(0))),
            'reconstruction_data_main_component': ssa_df_column.reconstruct(0).values})).plot_data(titles=['reconstruction_data_main_component'], 
                                                                                                   sub_title=f'Основной компонент цикличности_тренда {data.name}',
                                                                                                   upload_path=visual_data_upload_path)
        
        VisualizationData(pd.DataFrame({
            'index': range(len(ssa_df_column.reconstruct(1))),
            'reconstruction_data_second_component': ssa_df_column.reconstruct(1).values})).plot_data(titles=['reconstruction_data_second_component'], 
                                                                                                     sub_title=f'Второй компонент, шум_остаток признака {data.name}', 
                                                                                                     upload_path=visual_data_upload_path)
        
        VisualizationData(pd.DataFrame({
            'index': range(len(ssa_df_column.reconstruct(2))),
            'reconstruction_data_third_component': ssa_df_column.reconstruct(2).values})).plot_data(titles=['reconstruction_data_third_component'], 
                                                                                                    sub_title=f'Третий компонент, шум_остаток признака {data.name}', 
                                                                                                    upload_path=visual_data_upload_path)
    
    elif 7 > l > 3:
        # Визуализация первых 6 компонентов
        print(f'Визуализация первых 6 компонент признака {data.name}\n')
    
        VisualizationData(pd.DataFrame({
            'index': range(len(sigma_column[:6])),
            'first_six_components': sigma_column[:6]})).plot_data(titles=['first_six_components'],
                                                                  sub_title=f'Визуализация первых 6 компонент по убыванию важности признака {data.name}', 
                                                                  upload_path=visual_data_upload_path)
    
        # Показывает какое количество компонент объясняет основной процент информации
        print(f'Показывает какое количество компонент объясняет основной процент информации\n')
        display(sigma_column[:1].sum(), sigma_column[2:3].sum(), sigma_column[4:].sum())
    
        # Показывает сколько компонентов необходимо для заданного процента информации
        VisualizationData(pd.DataFrame({
            'index': range(len(sigma_column)),
            'cumulative_components': sigma_column.cumsum()})).plot_data(titles=['cumulative_components'],
                                                                        sub_title=f'Показывает сколько компонент необходимо для заданного процента информации признака {data.name}', 
                                                                        upload_path=visual_data_upload_path)
    
        # Реконструкция данных:
        # - F0 - тренд,
        # - F1, F2 - основные парные компоненты цикличности,
        # - F3, F4 - дополнительные парные компоненты цикличености,
        # - F5, F6 - цикличность более высокого масштаба,
        # - F7, F*** - вероятно, шум и случайные колебания.
        print(f'Реконструкция данных\n')
        
        VisualizationData(pd.DataFrame({
            'index': range(len(ssa_df_column.reconstruct(0))),
            'reconstruction_data_main_component': ssa_df_column.reconstruct(0).values})).plot_data(titles=['reconstruction_data_main_component'], 
                                                                                                   sub_title=f'Основной компонент цикличности_тренда признака {data.name}', 
                                                                                                   upload_path=visual_data_upload_path)
        
        VisualizationData(pd.DataFrame({
            'index': range(len(ssa_df_column.reconstruct([1,2]))),
            'reconstruction_data_second_third_components': ssa_df_column.reconstruct([1,2]).values})).plot_data(titles=['reconstruction_data_second_third_components'], 
                                                                                                                sub_title=f'2-3 компоненты, шум_остаток признака {data.name}', 
                                                                                                                upload_path=visual_data_upload_path)
        
        VisualizationData(pd.DataFrame({
            'index': range(len(ssa_df_column.reconstruct([3,5]))),
            'reconstruction_data_fourth_sixth_component': ssa_df_column.reconstruct([3,5]).values})).plot_data(titles=['reconstruction_data_fourth_sixth_component'], 
                                                                                                               sub_title=f'4-6 компоненты, шум_остаток признака {data.name}', 
                                                                                                               upload_path=visual_data_upload_path)
    
    elif l > 7:
        # Визуализация первых 12 компонент
        print(f'Визуализация первых 12 компонент признака {data.name}\n')
    
        VisualizationData(pd.DataFrame({
            'index': range(len(sigma_column[:12])),
            'first_twelve_components': sigma_column[:12]})).plot_data(titles=['first_twelve_components'],
                                                                  sub_title=f'Визуализация первых 12 компонент по убыванию важности признака {data.name}', 
                                                                  upload_path=visual_data_upload_path)
    
        # Показывает какое количество компонент объясняет основной процент информации
        print(f'Показывает какое количество компонент объясняет основной процент информации\n')
        display(sigma_column[:1].sum(), sigma_column[2:3].sum(), sigma_column[4:8].sum(), sigma_column[9:12].sum())
    
        # Показывает сколько компонентов необходимо для заданного процента информации
        VisualizationData(pd.DataFrame({
            'index': range(len(sigma_column)),
            'cumulative_components': sigma_column.cumsum()})).plot_data(titles=['cumulative_components'],
                                                                        sub_title=f'Показывает сколько компонент необходимо для заданного процента информации признака {data.name}', 
                                                                        upload_path=visual_data_upload_path)
    
        # Реконструкция данных:
        # - F0 - тренд,
        # - F1, F2 - основные парные компоненты цикличности,
        # - F3, F4 - дополнительные парные компоненты цикличености,
        # - F5, F6 - цикличность более высокого масштаба,
        # - F7, F*** - вероятно, шум и случайные колебания.
        print(f'Реконструкция данных\n')
    
        VisualizationData(pd.DataFrame({
            'index': range(len(ssa_df_column.reconstruct(0))),
            'reconstruction_data_main_component': ssa_df_column.reconstruct(0).values})).plot_data(titles=['reconstruction_data_main_component'], 
                                                                                                   sub_title=f'Основной компонент цикличности_тренда признака {data.name}', 
                                                                                                   upload_path=visual_data_upload_path)
    
        VisualizationData(pd.DataFrame({
            'index': range(len(ssa_df_column.reconstruct([1,2]))),
            'reconstruction_data_second_third_components': ssa_df_column.reconstruct([1,2]).values})).plot_data(titles=['reconstruction_data_second_third_components'], 
                                                                                                                sub_title=f'2-3 компоненты, шум_остаток признака {data.name}', 
                                                                                                                upload_path=visual_data_upload_path)
        
        VisualizationData(pd.DataFrame({
            'index': range(len(ssa_df_column.reconstruct([3,4]))),
            'reconstruction_data_fourth_fifth_components': ssa_df_column.reconstruct([3,4]).values})).plot_data(titles=['reconstruction_data_fourth_fifth_components'], 
                                                                                                                sub_title=f'4-5 компоненты, шум_остаток признака {data.name}', 
                                                                                                                upload_path=visual_data_upload_path)
        
        VisualizationData(pd.DataFrame({
            'index': range(len(ssa_df_column.reconstruct([5,6]))),
            'reconstruction_data_sixth_seventh_components': ssa_df_column.reconstruct([5,6]).values})).plot_data(titles=['reconstruction_data_sixth_seventh_components'], 
                                                                                                                 sub_title=f'6-7 компоненты, шум_остаток признака {data.name}', 
                                                                                                                 upload_path=visual_data_upload_path)
        
        VisualizationData(pd.DataFrame({
            'index': range(len(ssa_df_column.reconstruct([7,11]))),
            'reconstruction_data_eighth_twelfth_components': ssa_df_column.reconstruct([7,11]).values})).plot_data(titles=['reconstruction_data_eighth_twelfth_components'], 
                                                                                                                 sub_title=f'8-12 компоненты, шум_остаток признака {data.name}', 
                                                                                                                 upload_path=visual_data_upload_path)
