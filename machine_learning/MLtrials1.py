import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
import ta

#data_AAPL_5min = pd.read_csv("./data/aapl_5m_train.csv").dropna()
data_AAPL_5min = pd.read_csv("C:/Users/PC/OneDrive - ITESO/Documentos/GitHub/official_second_proyect/data/aapl_5m_train.csv").dropna()

data_clean = data_AAPL_5min.loc[:, ["Close"]]
data_clean["Y"] = data_clean.shift(-1)
data_clean["Close_t1"] = data_AAPL_5min.loc[:, ["Close"]].shift(10)
data_clean["Close_t2"] = data_AAPL_5min.loc[:, ["Close"]].shift(20)
data_clean["Close_t3"] = data_AAPL_5min.loc[:, ["Close"]].shift(30)
data_clean = data_clean.dropna()

rsi10 = ta.momentum.RSIIndicator(data_clean["Close"], window=10)
rsi20 = ta.momentum.RSIIndicator(data_clean["Close"], window=20)
rsi30 = ta.momentum.RSIIndicator(data_clean["Close"], window=30)


