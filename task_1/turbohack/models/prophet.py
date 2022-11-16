from turbohack.interface import Model, Cleaner
from fbprophet import Prophet
from tqdm.notebook import tqdm
import pandas
import re


class ProphetCleaner(Model, Cleaner):
    def __init__(self, **params):
        self.params = params
        self._date_col = None
        self.model = dict()
        self.features = None
        self.forecast = dict()

    def get_model(self):
        return Prophet(**self.params)

    def predict(self, data):
        data_pdf = data.copy()
        predict_cols = list()
        for feature in tqdm(data.columns):
            if feature not in self.features:
                continue
            if feature not in self.model.keys():
                continue
            data_pdf = data_pdf.rename(columns={feature: 'y'})
            data_pdf = data_pdf.rename(columns={self._date_col: 'ds'})
            forecast = self.model[feature].predict(data_pdf)
            forecast['fact'] = data_pdf['y'].reset_index(drop=True)
            #forecasted = forecast[['ds','trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
            forecast['anomaly'] = 0
            forecast.loc[forecast['fact'] > forecast['yhat_upper'], 'anomaly'] = 1
            forecast.loc[forecast['fact'] < forecast['yhat_lower'], 'anomaly'] = -1
            self.forecast[feature] = forecast
            result = forecast['anomaly']
            index = forecast['anomaly'].index
            data_pdf = data_pdf.rename(columns={'y': feature})
            data_pdf = data_pdf.rename(columns={'ds': self._date_col})
            pred_col = f'anomaly_{feature}'
            predict_cols.append(pred_col)
            data_pdf.loc[index, pred_col] = result
        #data_pdf.loc[:, 'predict'] = data_pdf[predict_cols].max(axis=1)
        return data_pdf

    def fit(self, data, features, date_col):
        data_pdf = data.copy()
        self.features = self.features_to_list(features)
        self._date_col = date_col
        for feat in tqdm(self.features):
            model = self.get_model()
            data_pdf = data_pdf.rename(columns={self._date_col: 'ds', feat: 'y'})
            self.model[feat] = model.fit(data_pdf)
            data_pdf = data_pdf.rename(columns={'ds': self._date_col, 'y': feat})
            #data_pdf = self.predict(data_pdf)
            #data_pdf = data_pdf.rename(columns={'ds': date_col, 'y': feat, 'anomaly': f'anomaly_{feat}'})
        return self

    def clean(self, data: pandas.DataFrame) -> pandas.DataFrame:
        predict_df = self.predict(data)
        anomaly_cols = [col for col in predict_df.columns if re.match(r'^anomaly_f\d+$', col)]
        #predict_df = predict_df[anomaly_cols + ['target']].copy()
        predict_df.loc[:, anomaly_cols] = predict_df[anomaly_cols].replace({-1: 1})
        predict_df.loc[:, 'predict'] = predict_df[anomaly_cols].max(axis=1)
        predict_df = predict_df[predict_df['predict'] == 0]
        predict_df = predict_df.drop(columns=anomaly_cols + ['predict'])
        return predict_df

