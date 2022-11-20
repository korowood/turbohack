import pandas
from typing import List


class Dataset:
    def __init__(self):
        self.data = None
        self.col_mapping = None
        self.features = None

    def load(self, path: str, sep: str = ';', encoding: str = 'cp1251'):
        self.data = pandas.read_csv(path, sep=sep, encoding=encoding)
        return self

    def save(self, data: pandas.DataFrame, path: str, sep: str = ';', encoding: str = 'cp1251'):
        all_names_dict_answer = self.col_mapping
        all_names_dict_answer['РЕЖИМ'] = 'predict'
        all_names_dict_answer = {v: k for k, v in all_names_dict_answer.items()}
        all_names_dict_answer.pop('РЕЖИМ', None)
        result = data.copy()
        result['predict'] = result['predict'].astype(int)
        result = result.reset_index()
        result = result.rename(columns=all_names_dict_answer)
        if 'target' in result.columns:
            result = result.drop('target', axis=1)
        for col in all_names_dict_answer.values():
            if col not in result.columns:
                result[col] = None
        result = result[self.col_mapping.keys()]
        result.to_csv(path, sep=sep, encoding=encoding)

    def prepare(
            self,
            date: str,
            target: str = None,
            drop_rows: List[int] = None,
            drop_cols: List[int] = None,
            dropna: bool = False
    ):
        self.col_mapping = dict()
        for col in list(self.data.columns):
            if col in self.col_mapping.keys():
                continue
            self.col_mapping[col] = f'f{len(self.col_mapping)}'
        self.col_mapping[date] = 'date_col'
        if target is not None:
            self.col_mapping[target] = 'target'
        self.data = self.data.rename(columns=self.col_mapping)
        if drop_rows is not None:
            self.data = self.data.drop(self.data.index[drop_rows], axis=0)
        if drop_cols is not None:
            self.data = self.data.drop(self.data.columns[drop_cols], axis=1)
        if dropna:
            self.data = self.data[self.data['target'].notna()]
        self.features = [col for col in self.data.columns if col not in ['target', 'date_col']]
        self.data = self.data.set_index('date_col')
        return self