import pandas as pd
import technical_analysis
import profit_calculator
import technical_indicators
from itertools import combinations
import optuna

# Load data
data_btc = pd.read_csv("./data/btc_project_train.csv").dropna()

# Calculate technical indicators
btc_technical_data = technical_analysis.calculate_technical_indicators(data_btc)

# Plot all technical indicators for BTC
technical_indicators.plot_technical_indicators(btc_technical_data, title="BTC Technical Indicators")

# Generate all possible combinations of indicators
indicators = ["RSI", "MACD", "Bollinger Bands", "ATR"]
all_combinations = []

for i in range(1, len(indicators) + 1):
    comb = combinations(indicators, i)
    all_combinations.extend(comb)

# Evaluate all combinations of indicators for BTC
best_combination_btc = None
best_value_btc = -float("inf")

for combination in all_combinations:
    study = optuna.create_study(direction='maximize')
    study.optimize(func=lambda trial: profit_calculator.profit(trial, data_btc, combination), n_trials=1)

    print(f"Best parameters for combination {combination}: {study.best_params}")
    print(f"Best value for combination {combination}: {study.best_value}")

    if study.best_value > best_value_btc:
        best_value_btc = study.best_value
        best_combination_btc = combination

print(f"The best combination of indicators for BTC is: {best_combination_btc} with a value of: {best_value_btc}")
