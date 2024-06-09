import pandas as pd
import matplotlib.pyplot as plt
import optuna
import technical_analysis

data_stock = pd.read_csv("./data/aapl_project_train.csv").dropna()
data_btc = pd.read_csv("./data/btc_project_train.csv").dropna()

# Calculation of technical indicators for stocks
technical_data_stock = technical_analysis.calculate_indicators(data_stock)

# Calculation of technical indicators for BTC
technical_data_btc = technical_analysis.calculate_indicators(data_btc)

# Plot of the technical indicators for stocks
fig, axs = plt.subplots(5, 1, figsize=(12, 18))
axs[0].plot(technical_data_stock["Close"], label="Close")
axs[1].plot(technical_data_stock["RSI"], label="RSI")
axs[2].plot(technical_data_stock["MACD"], label="MACD")
axs[3].plot(technical_data_stock["BOLL"], label="BOLL")
axs[4].plot(technical_data_stock["ATR"], label="ATR")
for ax in axs:
    ax.legend()
plt.show()

# Plot of the technical indicators for BTC
fig, axs = plt.subplots(5, 1, figsize=(12, 18))
axs[0].plot(technical_data_btc["Close"], label="Close")
axs[1].plot(technical_data_btc["RSI"], label="RSI")
axs[2].plot(technical_data_btc["MACD"], label="MACD")
axs[3].plot(technical_data_btc["BOLL"], label="BOLL")
axs[4].plot(technical_data_btc["ATR"], label="ATR")
for ax in axs:
    ax.legend()
plt.show()

# definition:  buy and sell signals based on various indicators for stocks
technical_data_stock["BUY_SIGNAL"] = (technical_data_stock.RSI < 31)
technical_data_stock["BUY_SIGNAL"] &= (technical_data_stock.MACD > 0)
technical_data_stock["BUY_SIGNAL"] &= (technical_data_stock.Close < technical_data_stock["Bollinger Low"])
technical_data_stock["BUY_SIGNAL"] &= (technical_data_stock.ATR > technical_data_stock.ATR.mean())

technical_data_stock["SELL_SIGNAL"] = (technical_data_stock.RSI > 70)
technical_data_stock["SELL_SIGNAL"] &= (technical_data_stock.MACD < 0)
technical_data_stock["SELL_SIGNAL"] &= (technical_data_stock.Close > technical_data_stock["Bollinger High"])
technical_data_stock["SELL_SIGNAL"] &= (technical_data_stock.ATR < technical_data_stock.ATR.mean())

# definition:  buy and sell signals based on various indicators for BTC
technical_data_btc["BUY_SIGNAL"] = (technical_data_btc.RSI < 31)
technical_data_btc["BUY_SIGNAL"] &= (technical_data_btc.MACD > 0)
technical_data_btc["BUY_SIGNAL"] &= (technical_data_btc.Close < technical_data_btc["Bollinger Low"])
technical_data_btc["BUY_SIGNAL"] &= (technical_data_btc.ATR > technical_data_btc.ATR.mean())

technical_data_btc["SELL_SIGNAL"] = (technical_data_btc.RSI > 70)
technical_data_btc["SELL_SIGNAL"] &= (technical_data_btc.MACD < 0)
technical_data_btc["SELL_SIGNAL"] &= (technical_data_btc.Close > technical_data_btc["Bollinger High"])
technical_data_btc["SELL_SIGNAL"] &= (technical_data_btc.ATR < technical_data_btc.ATR.mean())

# Optimization stocks
study_stock = optuna.create_study(direction='maximize')
study_stock.optimize(func=lambda trial: technical_analysis.profit(trial, data_stock, "stock"), n_trials=3)

print("Best parameters for stocks:", study_stock.best_params)

# Optimization BTC
study_btc = optuna.create_study(direction='maximize')
study_btc.optimize(func=lambda trial: technical_analysis.profit(trial, data_btc, "btc"), n_trials=3)

print("Best parameters for BTC:", study_btc.best_params)
