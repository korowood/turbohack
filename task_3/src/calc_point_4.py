import pandas as pd
import warnings
# from seed_everything import seed_everything
from src.seed_everything import seed_everything
from src.model import TimeSeriesPredictor
from scipy.signal import periodogram, detrend
from tqdm import tqdm
from catboost import CatBoostRegressor
import time

PATH_DATA = "Задача3. Датасет 3. Контрольный для участников.csv"
# increase = ['х011', 'х012', 'х013', 'х018', 'х019', 'х020', 'х021', 'х027', 'х028', 'х029', 'х057', 'х058', 'х059']
increase = ['х011', 'х012', 'х013', ]


def get_season_period(ts):
    ts = pd.Series(detrend(ts), ts.index)
    f, Pxx = periodogram(ts)
    Pxx = list(map(lambda x: x.real, Pxx))
    ziped = list(zip(f, Pxx))
    ziped.sort(key=lambda x: x[1])
    highest_freqs = [x[0] for x in ziped[-100:]]
    season_periods = [round(1 / (x + 0.001)) for x in highest_freqs]
    for period in reversed(season_periods):
        if 4 < period < 100:
            return int(period)


def forecast_timeseries(data: pd.DataFrame, cols: list, steps: int):
    output = {}

    for col in tqdm(cols):
        # ts = data[col]
        ts_train, ts_test = data.loc[:, col].dropna(), data.loc[:, col].dropna()

        lags = get_season_period(ts_train)

        predictor = TimeSeriesPredictor(
            granularity="PT1M",
            num_lags=lags,
            model=CatBoostRegressor,
            verbose=0
            # mappers=datetime_mappers,
        )

        lags_matrix = predictor.transform_into_matrix(ts_train)

        predictor.fit(ts_train)
        prediction = predictor.predict_next(ts_train, n_steps=steps)
        output[col] = prediction
    return output


def calculating():
    seed_everything()
    test_data = pd.read_csv(PATH_DATA, encoding="cp1251", sep=';', skiprows=[1])
    print("расчет точек, которые возрастают")
    test_data = test_data.rename(columns={'Unnamed: 0': "Параметр"})
    test_data['Параметр'] = pd.to_datetime(test_data['Параметр'], infer_datetime_format=True)
    test_data.set_index("Параметр", inplace=True)

    time.sleep(2)
    predictions_increase = forecast_timeseries(test_data, increase, 200)
    pd.to_pickle(predictions_increase, "data/predictions_increase.pkl")
