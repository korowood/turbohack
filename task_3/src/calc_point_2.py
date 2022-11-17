import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from src.seed_everything import seed_everything

PATH_TRAIN = "Задача3. Датасет 1. Тренировочный.csv"
PATH_VAL = "Задача3. Датасет 2. Тестовый.csv"
PATH_TEST = "Задача3. Датасет 3. Контрольный для участников.csv"

drop_feat = ['х011', 'х012', 'х013', 'х018', 'х019', 'х020', 'х021', 'х027',
             'х028', 'х029', 'х057', 'х058', 'х059', 'х008', 'х009', 'х010', 'х014',
             'х015', 'х016', 'х017', 'х023', 'х024', 'х025', 'х026', 'х030']


def feature_engineering_for_model(train_data, val_data, test_data):
    train_ = train_data.iloc[178271-2500:].drop(drop_feat, axis=1).copy()

    train_data_diff = train_.drop(['Параметр', 'Маркер'], axis=1).diff(1) #.drop(drop_feat, axis=1)
    train_data_diff.rename(columns={col: col + "_diff" for col in train_data_diff.columns}, inplace=True)
    train = pd.concat([train_, train_data_diff], axis=1)

    val_data_diff = val_data.drop(drop_feat, axis=1).drop(['Параметр', 'Маркер'], axis=1).diff(1) #.drop(drop_feat, axis=1)
    val_data_diff.rename(columns={col: col + "_diff" for col in val_data_diff.columns}, inplace=True)
    val = pd.concat([val_data.drop(drop_feat, axis=1), val_data_diff], axis=1)

    test_ = test_data.loc[:, 'х001':'х071'].drop(drop_feat, axis=1).dropna()

    test_data_diff = test_.diff(1) # .drop(drop_feat, axis=1)
    test_data_diff.rename(columns={col: col + "_diff" for col in test_data_diff.columns}, inplace=True)
    test = pd.concat([test_, test_data_diff], axis=1)

    return train, val, test


def create_x_y_pool(train, val):
    # заменим 2 на 1 и 4 на 3 в Train и Val
    X = train.drop(['Параметр', 'Маркер'], axis=1).reset_index(drop=True)
    y = train[['Маркер']].reset_index(drop=True)

    y[y['Маркер'] == 2] = 1
    y[y['Маркер'] == 4] = 3


    X_val, y_val = val.drop(['Параметр', 'Маркер'], axis=1).reset_index(drop=True), val[['Маркер']]

    y_val[y_val['Маркер'] == 2] = 1
    y_val[y_val['Маркер'] == 4] = 3

    train_pool = Pool(X, y)
    val_pool = Pool(X_val, y_val)
    return train_pool, val_pool

def modeling(train_pool, val_pool):
    model = CatBoostClassifier(verbose=0, od_type="Iter",
                               od_wait=100,
                               depth=9,
                               colsample_bylevel=.8,
                               learning_rate=.1,
                               grow_policy="Lossguide",
                               max_leaves=500,
                               min_data_in_leaf=2,
                               l2_leaf_reg=20

                               )

    model.fit(train_pool, eval_set=val_pool)

    return model


def calculating_two():
    seed_everything() # seed all
    # load train
    train_data = pd.read_csv(PATH_TRAIN, encoding="cp1251", sep=';', skiprows=[1, 2])

    #load val
    val_data = pd.read_csv(PATH_VAL, encoding="cp1251", sep=';', skiprows=[1, 2])
    val_data = val_data.rename(columns={'Unnamed: 0': "Параметр"})

    #load test
    test_data = pd.read_csv(PATH_TEST, encoding="cp1251", sep=';', skiprows=[1])
    test_data.reset_index(drop=True, inplace=True)

    # create features
    train, val, test = feature_engineering_for_model(train_data, val_data, test_data)

    # create pool for modeling
    train_pool, val_pool = create_x_y_pool(train, val)

    # train model
    model = modeling(train_pool, val_pool)

    proba = model.predict_proba(test)[:, 1]
    proba_df = pd.DataFrame(data=proba, columns=["proba"])
    return proba_df