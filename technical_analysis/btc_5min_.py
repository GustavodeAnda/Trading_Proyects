import pandas as pd
import technical_analysis
import profit_calculator
import technical_indicators
from itertools import combinations
import optuna

# Load data
data_btc = pd.read_csv("./data/btc_project_train.csv").dropna()
data_btc = data_btc.rename(columns={data_btc.columns[0]: 'btc'})

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
#
# # Evaluate all combinations of indicators for BTC
# best_combination_btc = None
# best_value_btc = -float("inf")
#
# for combination in all_combinations:
#     study = optuna.create_study(direction='maximize')
#     study.optimize(func=lambda trial: profit_calculator.profit(trial, data_btc, combination), n_trials=1)
#
#     print(f"Best parameters for combination {combination}: {study.best_params}")
#     print(f"Best value for combination {combination}: {study.best_value}")
#
#     if study.best_value > best_value_btc:
#         best_value_btc = study.best_value
#         best_combination_btc = combination
#
# print(f"The best combination of indicators for BTC is: {best_combination_btc} with a value of: {best_value_btc}")
#
# Evaluate all combinations of indicators for BTC
best_combination_btc = None
best_value_btc = -float("inf")
best_params_btc = None

# Diccionario para almacenar los mejores resultados para cada combinación
results = {}

for combination in all_combinations:
    study = optuna.create_study(direction='maximize')
    study.optimize(func=lambda trial: profit_calculator.profit(trial, data_btc, combination), n_trials=1)

    best_params = study.best_params
    best_value = study.best_value

    print(f"Best parameters for combination {combination}: {best_params}")
    print(f"Best value for combination {combination}: {best_value}")

    # Guardar los resultados en el diccionario
    results[combination] = {"params": best_params, "value": best_value}

    if best_value > best_value_btc:
        best_value_btc = best_value
        best_combination_btc = combination
        best_params_btc = best_params

print(" ")
print(" ")
print(f"The best combination of indicators for BTC is: {best_combination_btc} with a value of: {best_value_btc}")

# Crear un diccionario con el mejor resultado
best_outcome = {
    "combination": best_combination_btc,
    "value": best_value_btc,
    "params": best_params_btc
}

print(f"Best outcome: {best_outcome}")

