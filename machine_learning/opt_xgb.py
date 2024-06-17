import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
import ta
import functions_ml as ml
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

data = pd.read_csv("./data/aapl_5m_train.csv").dropna()


data_clean = data.loc[:, ["Close"]]
data_clean["Y"] = data_clean.shift(-15)
data_clean["Close_t1"] = data.loc[:, ["Close"]].shift(1)
data_clean["Close_t2"] = data.loc[:, ["Close"]].shift(2)
data_clean["Close_t3"] = data.loc[:, ["Close"]].shift(3)
data_clean["Close_t4"] = data.loc[:, ["Close"]].shift(4)
data_clean["Close_t5"] = data.loc[:, ["Close"]].shift(5)

data_clean["rsi_10"] = ((ta.momentum.RSIIndicator(data["Close"], window=10)).rsi())
data_clean["rsi_20"] = ((ta.momentum.RSIIndicator(data["Close"], window=20)).rsi())
data_clean["rsi_30"] = ((ta.momentum.RSIIndicator(data["Close"], window=30)).rsi())

data_clean["macd_10_24_7"] = ((ta.trend.MACD(close=data["Close"], window_slow=24, window_fast=10, window_sign=7)).macd())
data_clean["macd_12_26_9"] = ((ta.trend.MACD(close=data_clean["Close"], window_slow=26, window_fast=12, window_sign=9)).macd())
data_clean["macd_5_35_5"] = ((ta.trend.MACD(close=data_clean["Close"], window_slow=35, window_fast=5, window_sign=5)).macd())

### bollinger bands
bollinger_20_2 = ta.volatility.BollingerBands(close=data_clean["Close"], window=20, window_dev=2)
data_clean["bollinger_20_2_hband"] = bollinger_20_2.bollinger_hband()
data_clean["bollinger_20_2_lband"] = bollinger_20_2.bollinger_lband()
data_clean["bollinger_20_2_mavg"] = bollinger_20_2.bollinger_mavg()

bollinger_10_1_5 = ta.volatility.BollingerBands(close=data_clean["Close"], window=10, window_dev=1.5)
data_clean["bollinger_10_1_5_hband"] = bollinger_10_1_5.bollinger_hband()
data_clean["bollinger_10_1_5_lband"] = bollinger_10_1_5.bollinger_lband()
data_clean["bollinger_10_1_5_mavg"] = bollinger_10_1_5.bollinger_mavg()

bollinger_50_2_5 = ta.volatility.BollingerBands(close=data_clean["Close"], window=50, window_dev=2.5)
data_clean["bollinger_50_2_5_hband"] = bollinger_50_2_5.bollinger_hband()
data_clean["bollinger_50_2_5_lband"] = bollinger_50_2_5.bollinger_lband()
data_clean["bollinger_50_2_5_mavg"] = bollinger_50_2_5.bollinger_mavg()
data_clean = data_clean.dropna()

data_clean["atr_14"] = (ta.volatility.AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14)).average_true_range()
data_clean["atr_10"] = (ta.volatility.AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=10)).average_true_range()
data_clean["atr_20"] = (ta.volatility.AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=20)).average_true_range()

data_clas = data_clean.drop("Y", axis=1).copy()

data_clas["Y"] = data_clas.Close < data_clas.Close.shift(-1)

X_train, X_test, y_train, y_test = train_test_split(data_clas.drop("Y", axis=1),
                                                    data_clas.Y,
                                                    shuffle=False, test_size=0.2)

xgb = XGBClassifier().fit(X_train,y_train)

f1_score(y_train, xgb.predict(X_train))

metrics_xgb = ml.calculate_confusion_matrix_metrics(xgb, X_train, y_train)

fpr_xgb = ml.fpr(metrics_xgb["false_positives"], metrics_xgb["true_negatives"])

import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Optimización de XGBClassifier
def objective_xgb(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    max_leaves = trial.suggest_int('max_leaves', 0, 10)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
    gamma = trial.suggest_loguniform('gamma', 1e-8, 1.0)
    reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-8, 1.0)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_leaves=max_leaves,
        learning_rate=learning_rate,
        booster=booster,
        gamma=gamma,
        reg_lambda=reg_lambda
    )
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    return score

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=10)

best_params_xgb = study_xgb.best_params
print("Best parameters for XGBClassifier: ", best_params_xgb)

