import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import ta

# Cargar y limpiar los datos
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

# Bollinger bands
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

#data_clean["atr_14"] = (ta.volatility.AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14)).average_true_range()
#data_clean["atr_10"] = (ta.volatility.AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=10)).average_true_range()
#data_clean["atr_20"] = (ta.volatility.AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=20)).average_true_range()

# Clasificación
data_clas = data_clean.drop("Y", axis=1).copy()
data_clas["Y"] = data_clas.Close < data_clas.Close.shift(-1)

X_train, X_test, y_train, y_test = train_test_split(data_clas.drop("Y", axis=1),
                                                    data_clas.Y,
                                                    shuffle=False, test_size=0.2)

# Modelos individuales
classification_model = LogisticRegression().fit(X_train, y_train)
svc = SVC(C=500, max_iter=100_000).fit(X_train, y_train)
xgb = XGBClassifier().fit(X_train, y_train)

ensemble_model = VotingClassifier(estimators=[
    ('logistic', classification_model),
    ('svc', svc),
    ('xgb', xgb)
], voting='soft')

ensemble_model.fit(X_train, y_train)

# Calcular FPR global para el ensemble
y_pred_ensemble = ensemble_model.predict(X_train)
cm_ensemble = confusion_matrix(y_train, y_pred_ensemble)

tn, fp, fn, tp = cm_ensemble.ravel()
fpr_ensemble = fp / (fp + tn)
print(f"FPR Ensemble: {fpr_ensemble}")