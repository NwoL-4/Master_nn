import inspect

import pandas as pd


class VariableLogger:
    """
    Класс для автоматической записи имени и значения переменной

    Attributes:
        variables (dict): Словарь для хранения переменных
    """

    def __init__(self):
        """
        Инициализация нового логгера
        """
        self.variables = {}

    def add_variable(self, variables: list) -> None:
        """
        Автоматически добавляет имя и значение переменной

        Args:
            variables: Лист со значениями
        """
        # Получаем имя переменной
        frame = inspect.currentframe().f_back
        try:
            # Ищем переменную в локальных переменных вызывающего кода
            for variable in variables:
                for name, value in frame.f_locals.items():
                    if value is variable:
                        if self.variables.get(name, None):
                            continue
                        self.variables[name] = value
                        break
        finally:
            del frame

    def save_to_csv(self,
                    overwrite: bool = False) -> None:
        """
        Сохраняет переменные в CSV файл

        Args:
            overwrite: Перезаписать файл, если он существует?
        """
        filename = 'variables.csv'
        if not filename.endswith('.csv'):
            raise ValueError("Файл должен иметь расширение .csv")

        mode = 'w' if overwrite else 'a'
        header = not (not overwrite and self._file_exists(filename))

        df = pd.DataFrame.from_dict(self.variables,
                                    orient='index',
                                    columns=['Value'])
        df.index.name = 'Variable'
        df.to_csv(filename, mode=mode, header=header)

    def _file_exists(self, filename: str) -> bool:
        """
        Проверяет существует ли файл

        Returns:
            bool: True если файл существует
        """
        try:
            with open(filename, 'r'):
                return True
        except FileNotFoundError:
            return False

    def __str__(self):
        """Строковое представление логгера"""
        return "\n".join(f"{name}: {value}" for name, value in self.variables.items())