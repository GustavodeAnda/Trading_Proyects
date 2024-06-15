import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
import ta

data = pd.read_csv("./data/aapl_project_train.csv").dropna()
data_clean = data.loc[:, ["Close"]]
data_clean["Y"] = data_clean.shift(-10)
data_clean["Close_t1"] = data.loc[:, ["Close"]].shift(1)
data_clean["Close_t2"] = data.loc[:, ["Close"]].shift(2)
data_clean["Close_t3"] = data.loc[:, ["Close"]].shift(3)
data_clean["Close_t4"] = data.loc[:, ["Close"]].shift(4)
data_clean["Close_t5"] = data.loc[:, ["Close"]].shift(5)
data_clean["rsi_10"] = data.loc[:, ["Close"]].shift(6)
data_clean = data_clean.dropna()
data_clean.head()

## Classification

data_clas = data_clean.drop("Y", axis=1).copy()
data_clas.head()

### esto que esta haciendo?
data_clas["Y"] = data_clas.Close < data_clas.Close.shift(-1)

X_train, X_test, y_train, y_test = train_test_split(data_clas.drop("Y", axis=1),
                                                    data_clas.Y,
                                                    shuffle=False, test_size=0.2)

classification_model = LogisticRegression().fit(X_train, y_train)

classification_model.predict(X_train)

classification_model.score(X_train, y_train)

from sklearn.metrics import f1_score
f1_score(y_train, classification_model.predict(X_train))

### Classification V2

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

ran_forest = RandomForestClassifier().fit(X_train,y_train)
svc = SVC(C=500, max_iter=100_000).fit(X_train,y_train)
xgb = XGBClassifier().fit(X_train,y_train)

## F1 score

### Regresión Lógistica
f1_score(y_train, classification_model.predict(X_train))
f1_score(y_train, ran_forest.predict(X_train))
f1_score(y_train, svc.predict(X_train))
f1_score(y_train, xgb.predict(X_train))

import functions_ml as ml

metrics_svc = ml.calculate_confusion_matrix_metrics(ran_forest, X_train, y_train)
metrics_xgb = ml.calculate_confusion_matrix_metrics(xgb, X_train, y_train)

fpr_svc = ml.fpr(metrics_svc["false_positives"], metrics_svc["true_negatives"])
fpr_xgb = ml.fpr(metrics_xgb["false_positives"], metrics_xgb["true_negatives"])
print(fpr_xgb)