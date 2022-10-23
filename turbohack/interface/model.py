from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Union

import pandas


class Model(ABC):
    @abstractmethod
    def fit(self, data: pandas.DataFrame, features: Union[List[str], str], date_col: str) -> Model:
        pass

    @abstractmethod
    def predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        pass

    def features_to_list(self, features: str) -> List[str]:
        if isinstance(features, str):
            features = [features]
        return features


class Cleaner(ABC):
    @abstractmethod
    def clean(self, data: pandas.DataFrame) -> pandas.DataFrame:
        pass