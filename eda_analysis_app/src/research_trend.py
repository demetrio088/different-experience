# импортируем необходимые для работы библиотеки
import pandas as pd
from visualization_data import VisualizationData
from research_components_of_signs import ResearchComponentsOfSigns
from statsmodels.tsa.seasonal import seasonal_decompose

def research_trend(data: pd.DataFrame,
                   lag: int=500,
                   upload_path_trend: str='',
                   rolling_window_size: int=10000,
                   period_range: list=[],
                   visual_limit: int=1000):
    for column in data.columns:
        print(f'Исследование компонентов признака {column}\n')
        print(f'ШАГ 1: построение скользящей средней - проведение визуального анализа наличия тренда компонента признака {column}\n')
        ResearchComponentsOfSigns(data=data[column],
                                  sign=data[column]).rolling_trend(window=rolling_window_size, 
                                                             visual_data_upload_path=upload_path_trend)
        print(f'ШАГ 2: выявление стационарности ряда данных компонента признака {column} тестом Дики-Фуллера\n')
        ResearchComponentsOfSigns(data=data[column]).checking_stationarity()
        print(""" Если временной ряд нестационарен, то он содержит тренд, который можно проследить в изменениях следующих статистик временного ряда:
                  среднее, дисперсия и автокорреляция.\n""")
        print(f'ШАГ 3: анализ автокорреляции(ACF) компонента признака {column}\n')
        ResearchComponentsOfSigns(data=data[column],
                                  sign=data[column]).acf_analysis(lag=lag,
                                                                  visual_data_upload_path=upload_path_trend)
        print(f'ШАГ 4: анализ частичной автокорреляции(PACF) компонента признака {column}\n')
        ResearchComponentsOfSigns(data=data[column],
                                  sign=data[column]).pacf_analysis(lag=lag,
                                                                   visual_data_upload_path=upload_path_trend)
        print('ACF и PACF показывают, есть ли линейная зависимость(через коэффициент корреляции) между текущими и прошлыми значениями.\n')
        print(f'ШАГ 5: анализ частотного спектра Фурье признака {column}\n')
        ResearchComponentsOfSigns(data=data[column],
                                  sign=data[column]).ff_analysis(visual_limit=visual_limit,
                                                                 visual_data_upload_path=upload_path_trend)
        print(f'ШАГ 6: разложение временного ряда компонента на тренд, сезонность, остаток. Признак {column}\n')
        for p in period_range:
            print(f'Разложение временного ряда компонента на тренд, сезонность, остаток. Признак {column}. Периодичность {p} мин.\n')
            decomposition = seasonal_decompose(data[column],
                                               model='additive',
                                               period=p)
            VisualizationData(pd.DataFrame({
                'index': range(len(decomposition.trend)),
                f'trend_{column}_period_{p}': decomposition.trend})).plot_data(titles=[f'trend_{column}_period_{p}'],
                                                                               upload_path=upload_path_trend)
            VisualizationData(pd.DataFrame({
                'index': range(len(decomposition.trend)),
                f'seasonal_{column}_period_{p}': decomposition.seasonal})).plot_data(titles=[f'seasonal_{column}_period_{p}'],
                                                                                     upload_path=upload_path_trend)
            VisualizationData(pd.DataFrame({
                'index': range(len(decomposition.resid)),
                f'resid_{column}_period_{p}': decomposition.resid})).plot_data(titles=[f'resid_{column}_period_{p}'],
                                                                              upload_path=upload_path_trend)
            print(f'ШАГ 6.1: выявление стационарности остатков ряда данных компонента признака {column} периода {p} тестом Дики-Фуллера\n')
            ResearchComponentsOfSigns(data=pd.DataFrame({
                f'resid_{column}_period_{p}': decomposition.resid.dropna()})).checking_stationarity()
            print("""Стационарные остатки - признак того, что модель декомпозиции подходит для данных. В ином случае рекомендуется выбрать другую модель.\n""")
            print(f'ШАГ 6.2: тестируем на гетероскедаcтичность остатков компонента признака {column} периода {p} тестом Бройша-Пагана\n')
            ResearchComponentsOfSigns(data=pd.DataFrame({f'resid_{column}_period_{p}': decomposition.resid.dropna()})).checking_heteroscedasticity(exog = pd.DataFrame({'index': range(len(decomposition.resid.dropna())),
                                                                                                                                                                        'const': 1}))
            print("""Присутствие гетероскедастичности в временных рядах создает множественные проблемы для традиционных методов
                  моделирования и прогнозирования. Наиболее серьезное последствие — это нарушение предпосылок классической линейной
                  регрессии, что приводит к неэффективности оценок параметров и искажению статистических выводов.\n""")
        print(f'ШАГ 7: построение модели линейной регрессии -  вычисление статистик тренда компонента признака {column}\n')
        ResearchComponentsOfSigns(data=data[column],
                                  sign=data[column]).linear_regression_trend(visual_data_upload_path=upload_path_trend)
