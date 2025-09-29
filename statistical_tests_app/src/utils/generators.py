# импортируем необходимые для работы библиотеки
import numpy as np
import warnings
# убираем незначительные предупреждения
warnings.filterwarnings('ignore')

# генератор выборки для нормального распределения
def generate_random_normal(sample_size: int, mean: int, std: int):
    """
    Генерирует нормально распределенную случайную выборку.
    Принимает параметры:
    - sample_size: int
        Размер выборки.
    - mean: int
        Среднее выборки.
    - std: int
        Стандартное отклонение выборки.
    Возвращает параметры: нормально распределенная выборка.
    Задано ограничение: отсутствует.
    """
    return [np.random.normal(loc=mean, scale=std) for _ in range(sample_size)]

# генератор выборки для случайного распределения
def generate_random(sample_size: int, max_value: int):
    """
    Генерирует случайную выборку.
    Принимает параметры:
    - sample_size: int
        Размер выборки.
    - max_value: int
        Максимальное значение.
    Возвращает параметры: случайно распределенная выборка.
    Задано ограничение: отсутствует.
    """
    return [np.random.randint(max_value) for _ in range(sample_size)]

# генератор двух зависимых выборок(база + шум) на примере экспоненциально распределенных данных
def generate_dependent_samples(scale_difference: int, sample_size: int):
    """
    Генерирует пару зависимых выборок.
    Принимает параметры:
    - scale_difference: int
        Масштаб распределения.
    - sample_size: int
        Размер выборки.
    Возвращает параметры: две зависимые выборки.
    Задано ограничение: отсутствует.
    """
    before = np.random.exponential(scale=scale_difference, size=sample_size) # До
    difference = np.random.exponential(scale=scale_difference+2, size=sample_size) # Шум
    after = before + difference # После
    return before, after