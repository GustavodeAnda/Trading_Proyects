import pandas as pd
import matplotlib.pyplot as plt
import optuna
from technical_analysis import calculate_indicators, profit

data = pd.read_csv("aapl_project_train.csv").dropna()

technical_data = calculate_indicators(data)

### Plot
fig, axs = plt.subplots(5, 1, figsize=(12, 18))
axs[0].plot(technical_data["Close"], label="Close")
axs[1].plot(technical_data["RSI"], label="RSI")
axs[2].plot(technical_data["MACD"], label="MACD")
axs[3].plot(technical_data["BOLL"], label="BOLL")
axs[4].plot(technical_data["ATR"], label="ATR")
for ax in axs:
    ax.legend()
plt.show()

### Define BUY_SIGNAL and SELL_SIGNAL based on various indicators
technical_data["BUY_SIGNAL"] = (technical_data.RSI < 31)
technical_data["BUY_SIGNAL"] &= (technical_data.MACD > 0)
technical_data["BUY_SIGNAL"] &= (technical_data.Close < technical_data["Bollinger Low"])
technical_data["BUY_SIGNAL"] &= (technical_data.ATR > technical_data.ATR.mean())

technical_data["SELL_SIGNAL"] = (technical_data.RSI > 70)
technical_data["SELL_SIGNAL"] &= (technical_data.MACD < 0)
technical_data["SELL_SIGNAL"] &= (technical_data.Close > technical_data["Bollinger High"])
technical_data["SELL_SIGNAL"] &= (technical_data.ATR < technical_data.ATR.mean())

### Optimización
study = optuna.create_study(direction='maximize')
study.optimize(func=lambda trial: profit(trial, data), n_trials=10)

print(study.best_params)
