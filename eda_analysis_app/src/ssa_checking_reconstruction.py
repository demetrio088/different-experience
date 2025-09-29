# импорт библиотек
import pandas as pd
from ssa_analysis import SSA
from IPython.display import display
from visualization_data import VisualizationData
from calculation_metrics import VerificationMetrics

def ssa_checking_reconstruction(data: pd.DataFrame,
                                l: int=2,
                                visual_data_upload_path: str=''):
    """
    Проводит проверку работы SSA путем реконструкции признаков.
    Принимает параметры:
    - data: DataFrame
         Выборка.
    - l: int
         L - количество компонентов разложения признака.
    - visual_data_upload_path: str
        Путь к директории для сохранения графиков/гистограмм/прочей визуализации.
    Возвращает параметры:
    - графики исходного и реконструированного датафреймов, отклонений: plot
         Построенные графики.
    - метрики: pivot_table
         Сводная таблица по метрикам проверки качества реконструкции.
    """
    # разбиваем признак методом SSA на субпризнаки при заданном L
    ssa_data = SSA(data, l, save_mem=True)
    components_data = ssa_data.components_to_df()
    # выводим график по субпризнакам при заданном L
    for i in components_data.columns:
        VisualizationData(pd.DataFrame({
            'index': range(len(components_data)),
            'component_data': components_data[i]})).plot_data(titles=['component_data'],
                                                           sub_title=f'График по субпризнаку {components_data[i].name} при заданном L={l} признака {data.name}', 
                                                           upload_path=visual_data_upload_path)

    # создаем реконструированный признак и получаем оригинальный для сравнения
    ssa_data_reconstruct = ssa_data.reconstruct(slice(0, ssa_data.d))
    ssa_data_orig = ssa_data.orig_TS

    # проверяем реконструкцию SSA признака
    # выводим реконструкцию и оригинал на один график
    VisualizationData(pd.DataFrame({
        'index': range(len(ssa_data_orig)),
        'reconstruct': ssa_data_reconstruct, 
        'original': ssa_data_orig})).plot_data(titles=['reconstruct', 'original'],
                                               sub_title=f'График по реконструкция и оригинал признака {data.name}', 
                                               upload_path=visual_data_upload_path)

    # выводим на график отклонение реконструкции от оригинала
    results = ssa_data_orig.values - ssa_data_reconstruct.values
    VisualizationData(pd.DataFrame({
        'index': range(len(results)),
        'results': results})).plot_data(titles=['results'],
                                        sub_title=f'Отклонение реконструкции от оригинала признака {data.name}', 
                                        upload_path=visual_data_upload_path)

    # выводим метрики для проверки реконструкции
    return display(VerificationMetrics(ssa_data_orig, ssa_data_reconstruct).checking_metrics())
    
