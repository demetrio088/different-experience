# импорт библиотек
import pandas as pd
import yaml
import os
from pathlib import Path
from IPython.display import display

class Utils:

    """Вспомогательные утилиты для работы проекта."""
    # загрузка файла
    @staticmethod
    def load_csv(path: str='',
                 file: str='') -> pd.DataFrame:
        """
        Открывает и загружает файл формата *.csv.
        Принимает параметры:
        - file: str
            Файл формата *.csv.
        - path: str
            Путь к файлу в формате строки.
        Возвращает:
        - pd.DataFrame
            Файл pd.DataFrame для дальнейшей работы.
        """
        return pd.read_csv(f'{path}/{file}')

    # обзор основной информации датасета, ищем пропуски, дубликаты
    @staticmethod
    def overview_data(data: pd.DataFrame) -> None:
        """
        Обзор основной информации датасета, поиск пропусков, дубликатов.
        Принимает параметры:
        - data: DataFrame
            Выборка.
        Возвращает:
        - Вывод информации на экран.
            Обзорная информация на экране.
        """
        display(data.columns)
        display(data.info())
        display(data.describe())
        display(data.head(10))
        display(data.tail(10))
        display(data.isna().sum())
        display(data.isnull().sum())
        display(data.duplicated().sum())

    # загрузка конфигурации
    @staticmethod
    def load_config():
        """
        Загружает и возвращает конфигурацию *.yml файла.
        """
        # определяем путь к файлу
        path_config = Path(__file__).parent.parent / 'config' / 'parameters.yml'

        # загрузка конфигурационного файла
        with open(path_config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config

    # сохранение файла
    @staticmethod
    def save_csv(data: pd.DataFrame, 
                 path: str='',
                 file_name: str='') -> pd.DataFrame:
        """
        Сохраняет файл формата *.csv.
        Принимает параметры:
        - file_name: str
            Имя файла.
        - path: str
            Путь к файлу в формате строки.
        Возвращает:
        - pd.DataFrame
            Файл pd.DataFrame в формате *.csv для дальнейшей работы.
        - сообщение о факте сохранения файла.
        """
        os.makedirs(path, exist_ok=True)
        output_path = os.path.join(path, file_name)
        data.to_csv(output_path, index=False)

        return f'Файл сохранен под именем {file_name}'
