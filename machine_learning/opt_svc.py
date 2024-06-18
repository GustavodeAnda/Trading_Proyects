import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
import ta
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


data = pd.read_csv("./data/aapl_project_train.csv").dropna()

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


## Classification

data_clas = data_clean.drop("Y", axis=1).copy()
data_clas.head()

### esto que esta haciendo?
data_clas["Y"] = data_clas.Close < data_clas.Close.shift(-1)

X_train, X_test, y_train, y_test = train_test_split(data_clas.drop("Y", axis=1),
                                                    data_clas.Y,
                                                    shuffle=False, test_size=0.2)


# Definir la función objetivo
def objective(trial):
    #     # Definir el rango de valores para los hiperparámetros
    C = trial.suggest_float('C', 1e-2, 1000, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
    gamma = trial.suggest_float('gamma', 1e-2, 1e1, log=True)

    # Crear el modelo SVC
    model = SVC(C=C, kernel=kernel, gamma=gamma, max_iter=10_000)

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate FPR
    fpr = fp / (fp + tn)

    return fpr


# Crear un objeto de estudio
study = optuna.create_study(direction="minimize")

# Ejecutar el proceso de optimización
study.optimize(objective, n_trials=30)

# Mostrar los mejores parámetros
# saved_study = optuna.load_study(study_name=study, storage=storage_url)
# storage_url = "sqlite:///example.db"
print("Best trial:", study.best_trial.number)
print("Best value:", study.best_trial.value)
print("Best hyperparameters:", study.best_params)