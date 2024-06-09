import pandas as pd
import matplotlib.pyplot as plt
import optuna
from itertools import combinations
import technical_analysis

data_stock = pd.read_csv("./data/aapl_project_train.csv").dropna()
data_btc = pd.read_csv("./data/btc_project_train.csv").dropna()

# Generate all possible combinations of indicators
indicators = ["RSI", "MACD", "Bollinger Bands", "ATR"]
all_combinations = []

for i in range(1, len(indicators) + 1):
    comb = combinations(indicators, i)
    all_combinations.extend(comb)

# Evaluate all combinations of indicators for stocks
print("Evaluating combinations for Apple stocks")
best_combination_stock = None
best_value_stock = -float("inf")

for combination in all_combinations:
    study = optuna.create_study(direction='maximize')
    study.optimize(func=lambda trial: technical_analysis.profit(trial, data_stock, combination), n_trials=1)

    print(f"Best parameters for combination {combination} (Apple): {study.best_params}")
    print(f"Best value for combination {combination} (Apple): {study.best_value}")

    if study.best_value > best_value_stock:
        best_value_stock = study.best_value
        best_combination_stock = combination

print(f"The best combination of indicators for Apple stocks is: {best_combination_stock} with a value of: {best_value_stock}")

# Evaluate all combinations of indicators for BTC
print("Evaluating combinations for Bitcoin")
best_combination_btc = None
best_value_btc = -float("inf")

for combination in all_combinations:
    study = optuna.create_study(direction='maximize')
    study.optimize(func=lambda trial: technical_analysis.profit(trial, data_btc, combination), n_trials=1)

    print(f"Best parameters for combination {combination} (Bitcoin): {study.best_params}")
    print(f"Best value for combination {combination} (Bitcoin): {study.best_value}")

    if study.best_value > best_value_btc:
        best_value_btc = study.best_value
        best_combination_btc = combination

print(f"The best combination of indicators for Bitcoin is: {best_combination_btc} with a value of: {best_value_btc}")

# Plot technical indicators for the best combination for stocks
print("Plotting results for Apple stocks")
technical_data_stock = technical_analysis.create_signals(data_stock, indicators=best_combination_stock,
                                                         rsi_window=48, macd_slow_window=26, macd_fast_window=12,
                                                         macd_sign_window=9,
                                                         bollinger_window=10, atr_window=14)

fig, axs = plt.subplots(len(best_combination_stock) + 1, 1, figsize=(12, 18))
axs[0].plot(technical_data_stock["Close"], label="Close")
for i, indicator in enumerate(best_combination_stock):
    axs[i + 1].plot(technical_data_stock[indicator], label=indicator)
for ax in axs:
    ax.legend()
plt.show()

# Plot technical indicators for the best combination for BTC
print("Plotting results for Bitcoin")
technical_data_btc = technical_analysis.create_signals(data_btc, indicators=best_combination_btc,
                                                       rsi_window=48, macd_slow_window=26, macd_fast_window=12,
                                                       macd_sign_window=9,
                                                       bollinger_window=10, atr_window=14)

fig, axs = plt.subplots(len(best_combination_btc) + 1, 1, figsize=(12, 18))
axs[0].plot(technical_data_btc["Close"], label="Close")
for i, indicator in enumerate(best_combination_btc):
    axs[i + 1].plot(technical_data_btc[indicator], label=indicator)
for ax in axs:
    ax.legend()
plt.show()