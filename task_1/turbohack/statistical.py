import pandas
import numpy
from typing import List, Union
from tqdm.notebook import tqdm


class Statistics:
    def __init__(self, data: pandas.DataFrame, features: Union[List[str], str]):
        self.data = data
        self.features = self.init_features(features)

    def init_features(self, features: Union[List[str], str]):
        if isinstance(features, str):
            features = [features]
        return features

    def rolling_mse(self, wsize: int = 10, threashold: float = 0.01, prefix: str = 'mse') -> pandas.DataFrame:
        for feature in tqdm(self.features):
            self.data[f'{prefix}_{feature}'] = self.data.loc[:, feature]\
                .pow(2)\
                .rolling(wsize)\
                .mean()\
                .apply(numpy.sqrt, raw=True)
            self.data.loc[:, f'{prefix}_{feature}'] = (self.data[f'{prefix}_{feature}'].abs() / self.data[f'{prefix}_{feature}'].mean() > threashold).astype(int)
        return self.data

    def rolling_mape(self, wsize: int = 10, threashold: float = 0.01, prefix: str = 'mape') -> pandas.DataFrame:
        for feature in tqdm(self.features):
            self.data[f'{prefix}_{feature}'] = self.data.loc[:, feature]\
                .rolling(wsize)\
                .apply(lambda x: numpy.sqrt(x.mean()))
            self.data.loc[:, f'{prefix}_{feature}'] = (self.data[f'{prefix}_{feature}'].abs() / self.data[f'{prefix}_{feature}'].mean() > threashold).astype(int)
        return self.data


