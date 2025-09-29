# импорт библиотек
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
#import plotly.io as pio
#pio.renderers.default = 'browser'

class VisualizationData:
    
    def __init__(self, 
                 data: pd.DataFrame):
        """
        Визуализируем данные.
        Принимает параметры:
        - data: DataFrame
            Выборка.
        """
        self.data = data

    # смотрим и анализируем гисторграммы по признакам
    def hist_data(self,
                  x_title: str='Количество попаданий',
                  y_title: str='Частота',
                  title: str=None,
                  hist_width: int=800,
                  hist_height: int=800,
                  upload_path: str=None) -> None:
        """
        Отрисовывает гистограмму.
        Принимает параметры:
        - data: DataFrame
            Выборка.
        - x_title: str
            Название по оси X.
        - y_title: str
            Название по оси Y.
        - title: string
            Название выборки.
        - hist_width: int,
            Ширина гистограммы.
        - hist_height: int
            Высота гистограммы.
        - upload_path: str
            Путь к папке сохранения.
        Выводит:
        - гистограммы: гистограмма на экране, гистограмма в формате *.html
            Вывод гистограмм на экран и в файл.
        Задано ограничение по размеру фигуры: 8 * 8.
        """
        self.x_title = x_title
        self.y_title = y_title
        self.title = title
        self.hist_width = hist_width
        self.hist_height = hist_height
        self.upload_path = upload_path
        fig = px.histogram(self.data, x=self.title, nbins=30, title=f'Данные признака {self.title}')
        fig.update_layout(
            xaxis_title=x_title,
            yaxis_title=y_title,
            width=hist_width,
            height=hist_height)
        if upload_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f') # именует файл в милисекундах %f
            filename = f'hist_{self.title}_{timestamp}.html'
            filepath = os.path.join(upload_path, filename)
            fig.write_html(filepath)
        fig.show()

    # смотрим и анализируем графики по признакам
    def plot_data(self,
                  x_title: str='Временная шкала',
                  y_title: str='Значения',
                  titles: list=None,
                  sub_title: str='',
                  line_width: int=800,
                  line_height: int=800,
                  upload_path: str=None) -> None:
        """
        Отрисовывает графики.
        Принимает параметры:
        - data: DataFrame
            Выборка.
        - x_title: str
            Название по оси X.
        - y_title: str
            Название по оси Y.
        - titles: array, list
            Список названий выборок.
        - line_width: int,
            Ширина линейного графика.
        - line_height: int
            Высота линейного графика.
        - upload_path: str
            Путь к папке сохранения.
        Задано ограничение по размеру фигуры: 8 * 8.
        """
        if titles is None:
            titles = []
        self.x_title = x_title
        self.y_title = y_title
        self.titles = titles
        self.sub_title = sub_title
        self.line_width = line_width
        self.line_height = line_height
        self.upload_path = upload_path
        fig = px.line(self.data, y=self.titles, title=f'Линейный график ' + ', '.join(self.titles) + f'<br>{self.sub_title}<br>')
        fig.update_layout(
            xaxis_title=x_title,
            yaxis_title=y_title,
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            width=line_width,
            height=line_height)
        if upload_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f') # именует файл в милисекундах %f
            filename = f'line_{", ".join(self.titles)}_{self.sub_title}_{timestamp}.html'
            filepath = os.path.join(upload_path, filename)
            fig.write_html(filepath)
        fig.show()

    # смотрим и анализируем боксплоты по признакам
    def boxplot_data(self,
                     box_width: int=800,
                     box_height: int=800,
                     x_title: str='Признак',
                     y_title: str='Значение',
                     titles: list=None,
                     upload_path: str=None,) -> None:
        """
        Отрисовывает боксплоты.
        Принимает параметры:
        - data: DataFrame
            Выборка.
        - box_width: int,
            Ширина диаграммы боксплотов.
        - box_height: int
            Высота диаграммы боксплотов.
        - x_title: str
            Название по оси X.
        - y_title: str
            Название по оси Y.
        - titles: array, list
            Список названий выборок.
        - upload_path: str
            Путь к папке сохранения.
        Выводит:
        - боксплоты: боксплоты на экране, боксплоты в формате *.html
            Вывод боксплотов на экран и в файл.
        Задано ограничение по размеру фигуры: 8 * 8.
        """
        if titles is None:
            titles = []
        self.titles = titles
        self.box_width = box_width
        self.box_height = box_height
        self.x_title = x_title
        self.y_title = y_title
        self.upload_path = upload_path
        fig = px.box(self.data, y=self.titles, title=f'Коробчатые диаграммы для признаков: '  + ', '.join(self.titles))
        fig.update_layout(
            xaxis_title=x_title,
            yaxis_title=y_title,
            width=box_width,
            height=box_height)
        if upload_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f') # именует файл в милисекундах %f
            filename = f'box_{", ".join(self.titles)}_{timestamp}.html'
            filepath = os.path.join(upload_path, filename)
            fig.write_html(filepath)
        fig.show()
    
    # строим матрицу корреляций по признакам
    def correlation_matrix_data(self,
                                matrix_width: int=800,
                                matrix_height: int=800,
                                titles: list=None,
                                upload_path: str=None) -> None:
        """
        Отрисовывает корреляционную матрицу.
        Принимает параметры:
        - data: DataFrame
            Выборка.
        - titles: array, list
            Список названий выборок.
        - matrix_width: int,
            Ширина матрицы.
        - matrix_height: int
            Высота матрицы.
        - upload_path: str
            Путь к папке сохранения.
        Выводит:
        - корреляционная матрица: матрица на экране, матрица в формате *.html
            Вывод матрицы на экран и в файл.
        Задано ограничение по размеру фигуры: 8 * 8.
        """
        if titles is None:
            titles = []
        self.titles = titles
        self.matrix_width = matrix_width
        self.matrix_height = matrix_height
        self.upload_path = upload_path
        fig = px.imshow(self.data.corr(), color_continuous_scale='Plasma', text_auto='.2f', title=f'Матрица корреляций для признаков: '  + ', '.join(self.titles))
        fig.update_layout(
            width=matrix_width,
            height=matrix_height)
        if upload_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f') # именует файл в милисекундах %f
            filename = f'correlation_matrix_{", ".join(self.titles)}_{timestamp}.html'
            filepath = os.path.join(upload_path, filename)
            fig.write_html(filepath)
        fig.show()
