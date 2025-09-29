# импорт библиотек
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class VerificationMetrics:

    def __init__(self, 
                 ssa_data_orig: pd.Series,
                 ssa_data_reconstruct: pd.Series):

        self.ssa_data_orig = ssa_data_orig
        self.ssa_data_reconstruct = ssa_data_reconstruct

    # расчет метрик
    def checking_metrics(self):
        """
        Производит расчет следующих метрик для проверки реконструкций:
            - среднее абсолютное отклонение предсказанных значений от фактических(средняя абсолютная ошибка)(MAE),
            - среднеквадратическая ошибка(MSE),
            - корень из среднеквадратической ошибки(RMSE),
            - коэффициент детерминации(R-квадрат).
        Принимает параметры:
        - ssa_data_orig: pd.Series
            Исходная выборка.
        - ssa_data_reconstruct: pd.Series
            Реконструированная выборка.
        Возвращает параметры:
        - показатели: str
            Вывод информации по показателям.
        Ограничения по размеру выборки: явных ограничений нет.
        """
        metrics_table = pd.DataFrame({
            'Метрика': ['Среднее абсолютное отклонение предсказанных значений от фактических(средняя абсолютная ошибка)(MAE)',
                        'Среднеквадратическая ошибка(MSE)',
                        'Корень из среднеквадратической ошибки(RMSE)',
                        'Коэффициент детерминации(R-квадрат)'],
            'Значения': [float(mean_absolute_error(self.ssa_data_orig, self.ssa_data_reconstruct)),
                         float(mean_squared_error(self.ssa_data_orig, self.ssa_data_reconstruct)),
                         float(mean_squared_error(self.ssa_data_orig, self.ssa_data_reconstruct, squared=False)),
                         float(r2_score(self.ssa_data_orig, self.ssa_data_reconstruct))]})
        return metrics_table.round(5)