import pandas as pd
import matplotlib.pyplot as plt
import optuna
import technical_analysis
from itertools import combinations

# Cargar datos
data_stock = pd.read_csv("./data/aapl_project_train.csv").dropna()
data_btc = pd.read_csv("./data/btc_project_train.csv").dropna()

# Calcular indicadores técnicos
technical_data_stock = technical_analysis.calculate_indicators(data_stock)
technical_data_btc = technical_analysis.calculate_indicators(data_btc)

# Definir indicadores
indicators = ["RSI", "MACD", "Bollinger", "ATR"]


# Generar todas las combinaciones posibles de indicadores usando números binarios
def generate_combinations(indicators):
    all_combinations = []
    for i in range(1, 2 ** len(indicators)):
        combo = []
        for j in range(len(indicators)):
            if (i >> j) & 1:
                combo.append(indicators[j])
        all_combinations.append(combo)
    return all_combinations


# Simulación de trading basada en una combinación de indicadores
def simulate_trading(technical_data, combo):
    signals = pd.Series(index=technical_data.index, data=False)
    buy_signals = pd.Series(index=technical_data.index, data=True)
    sell_signals = pd.Series(index=technical_data.index, data=True)

    if 'RSI' in combo:
        buy_signals &= (technical_data.RSI < 31)
        sell_signals &= (technical_data.RSI > 70)
    if 'MACD' in combo:
        buy_signals &= (technical_data.MACD > 0)
        sell_signals &= (technical_data.MACD < 0)
    if 'Bollinger' in combo:
        buy_signals &= (technical_data.Close < technical_data["Bollinger Low"])
        sell_signals &= (technical_data.Close > technical_data["Bollinger High"])
    if 'ATR' in combo:
        buy_signals &= (technical_data.ATR > technical_data.ATR.mean())
        sell_signals &= (technical_data.ATR < technical_data.ATR.mean())

    signals[buy_signals] = True
    signals[sell_signals] = False

    capital = 100000
    shares = 0

    for date, signal in signals.items():
        if signal and capital >= technical_data.at[date, 'Close']:
            shares = capital // technical_data.at[date, 'Close']
            capital -= shares * technical_data.at[date, 'Close']
        elif not signal and shares > 0:
            capital += shares * technical_data.at[date, 'Close']
            shares = 0

    portfolio_value = capital + shares * technical_data.iloc[-1]['Close']
    return portfolio_value


# Función de Optuna para maximizar el beneficio
def objective(trial, data, asset_type):
    combo = trial.suggest_categorical('combo', generate_combinations(indicators))
    technical_data = technical_analysis.calculate_indicators(data)
    return simulate_trading(technical_data, combo)


# Optimización de acciones
study_stock = optuna.create_study(direction='maximize')
study_stock.optimize(func=lambda trial: objective(trial, data_stock, "stock"), n_trials=100)

print("Best parameters for stocks:", study_stock.best_params)

# Optimización de BTC
study_btc = optuna.create_study(direction='maximize')
study_btc.optimize(func=lambda trial: objective(trial, data_btc, "btc"), n_trials=100)

print("Best parameters for BTC:", study_btc.best_params)

# Plot de indicadores técnicos para acciones
fig, axs = plt.subplots(5, 1, figsize=(12, 18))
axs[0].plot(technical_data_stock["Close"], label="Close")
axs[1].plot(technical_data_stock["RSI"], label="RSI")
axs[2].plot(technical_data_stock["MACD"], label="MACD")
axs[3].plot(technical_data_stock["Bollinger Low"], label="Bollinger Low", linestyle='--')
axs[3].plot(technical_data_stock["Bollinger High"], label="Bollinger High", linestyle='--')
axs[4].plot(technical_data_stock["ATR"], label="ATR")
for ax in axs:
    ax.legend()
plt.show()

# Plot de indicadores técnicos para BTC
fig, axs = plt.subplots(5, 1, figsize=(12, 18))
axs[0].plot(technical_data_btc["Close"], label="Close")
axs[1].plot(technical_data_btc["RSI"], label="RSI")
axs[2].plot(technical_data_btc["MACD"], label="MACD")
axs[3].plot(technical_data_btc["Bollinger Low"], label="Bollinger Low", linestyle='--')
axs[3].plot(technical_data_btc["Bollinger High"], label="Bollinger High", linestyle='--')
axs[4].plot(technical_data_btc["ATR"], label="ATR")
for ax in axs:
    ax.legend()
plt.show()


# Calcular y mostrar resultados promedio para cada combinación
def calculate_average_results(data, asset_type):
    combinations = generate_combinations(indicators)
    results = []
    for combo in combinations:
        total_value = 0
        for _ in range(1000):  # Ajusta el número de repeticiones según sea necesario
            technical_data = technical_analysis.calculate_indicators(data)
            total_value += simulate_trading(technical_data, combo)
        average_value = total_value / 1000
        results.append((combo, average_value))
    return results


# Resultados para acciones
average_results_stock = calculate_average_results(data_stock, "stock")
for combo, value in average_results_stock:
    print(f"Combinación: {combo}, Valor Final Promedio del Portafolio: {value}")

# Resultados para BTC
average_results_btc = calculate_average_results(data_btc, "btc")
for combo, value in average_results_btc:
    print(f"Combinación: {combo}, Valor Final Promedio del Portafolio: {value}")

