import pandas as pd
import technical_analysis
import profit_calculator
import technical_indicators
from itertools import combinations
import optuna

# Load data
data_aapl = pd.read_csv("./data/aapl_project_1m_train.csv").dropna()

# Calculate technical indicators
aapl_technical_data = technical_analysis.calculate_technical_indicators(data_aapl)

# Plot all technical indicators for AAPL
technical_indicators.plot_technical_indicators(aapl_technical_data, title="AAPL Technical Indicators")

# Generate all possible combinations of indicators
indicators = ["RSI", "MACD", "Bollinger Bands", "ATR"]
all_combinations = []

for i in range(1, len(indicators) + 1):
    comb = combinations(indicators, i)
    all_combinations.extend(comb)

# Evaluate all combinations of indicators for AAPL
best_combination_aapl = None
best_value_aapl = -float("inf")

for combination in all_combinations:
    study = optuna.create_study(direction='maximize')
    study.optimize(func=lambda trial: profit_calculator.profit(trial, data_aapl, combination), n_trials=1)

    print(f"Best parameters for combination {combination}: {study.best_params}")
    print(f"Best value for combination {combination}: {study.best_value}")

    if study.best_value > best_value_aapl:
        best_value_aapl = study.best_value
        best_combination_aapl = combination

print(f"The best combination of indicators for AAPL is: {best_combination_aapl} with a value of: {best_value_aapl}")
