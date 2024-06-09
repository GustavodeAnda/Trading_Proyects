import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import optuna

# Cargar datos
data = pd.read_csv("./data/aapl_project_train.csv").dropna()

# Calcular indicadores
rsi_indicator = ta.momentum.RSIIndicator(close=data['Close'], window=48)
bollinger = ta.volatility.BollingerBands(close=data['Close'], window=10)
stochastic_indicator = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
obv_indicator = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume'])
macd = ta.trend.MACD(data.Close, window_slow=26, window_fast=12, window_sign=9)
atr = ta.volatility.AverageTrueRange(high=data.High, low=data.Low, close=data.Close, window=14)

# Crear DataFrame para datos técnicos
technical_data = pd.DataFrame()
technical_data['Close'] = data['Close']
technical_data['RSI'] = rsi_indicator.rsi()
technical_data['%K'] = stochastic_indicator.stoch()
technical_data['OBV'] = obv_indicator.on_balance_volume()
technical_data['Bollinger Low'] = bollinger.bollinger_lband()
technical_data['Bollinger High'] = bollinger.bollinger_hband()
technical_data['MACD'] = macd.macd()
technical_data['ATR'] = atr.average_true_range()
technical_data = technical_data.dropna()

# Lista de indicadores
indicators = ['RSI', '%K', 'OBV', 'Bollinger']

# Generar todas las combinaciones posibles de indicadores usando números binarios
all_combinations = []
for i in range(1, 2 ** len(indicators)):
    combo = []
    for j in range(len(indicators)):
        if (i >> j) & 1:
            combo.append(indicators[j])
    all_combinations.append(combo)

# Función para simular trading basado en una combinación de indicadores
def simulate_trading(technical_data, combo):
    signals = pd.Series(index=technical_data.index, data=np.nan)

    if 'RSI' in combo:
        signals[technical_data['RSI'] < 30] = 'buy'
        signals[technical_data['RSI'] > 70] = 'sell'

    if '%K' in combo:
        signals[technical_data['%K'] < 20] = 'buy'
        signals[technical_data['%K'] > 80] = 'sell'

    if 'OBV' in combo:
        signals[technical_data['OBV'] > technical_data['OBV'].shift(1)] = 'buy'
        signals[technical_data['OBV'] < technical_data['OBV'].shift(1)] = 'sell'

    if 'Bollinger' in combo:
        signals[technical_data['Close'] < technical_data['Bollinger Low']] = 'buy'
        signals[technical_data['Close'] > technical_data['Bollinger High']] = 'sell'

    signals = signals.ffill().bfill()  # Forward and backward fill signals
    capital = 100000
    shares = 0

    for date, signal in signals.items():  # Cambiado a .items()
        if signal == 'buy' and capital >= technical_data.at[date, 'Close']:
            shares = capital // technical_data.at[date, 'Close']
            capital -= shares * technical_data.at[date, 'Close']
        elif signal == 'sell' and shares > 0:
            capital += shares * technical_data.at[date, 'Close']
            shares = 0

    portfolio_value = capital + shares * technical_data.iloc[-1]['Close']
    return portfolio_value

# Repetir el proceso de simulación 100 veces
all_results = []
for _ in range(100):
    results = []
    for combo in all_combinations:
        final_value = simulate_trading(technical_data, combo)
        results.append((combo, final_value))
    all_results.append(results)

# Calcular estadísticas de los resultados
average_results = []
for i in range(len(all_combinations)):
    combo_values = [all_results[j][i][1] for j in range(100)]
    average_result = np.mean(combo_values)
    average_results.append((all_combinations[i], average_result))

# Mostrar resultados promedio
for combo, value in average_results:
    print(f"Combinación: {combo}, Valor Final Promedio del Portafolio: {value}")

# Gráficos (opcional)
fig, axs = plt.subplots(5, 1, figsize=(12, 12))
axs[0].plot(technical_data['Close'], label='Close')
axs[1].plot(technical_data['RSI'], label='RSI')
axs[2].plot(technical_data['%K'], label='%K (Stochastic Oscillator)')
axs[3].plot(technical_data['OBV'], label='OBV')
axs[4].plot(technical_data['Close'], label='Close')
axs[4].plot(technical_data['Bollinger Low'], label='Bollinger Low', linestyle='--')
axs[4].plot(technical_data['Bollinger High'], label='Bollinger High', linestyle='--')

for ax in axs:
    ax.legend()
plt.tight_layout()
plt.show()

### Define BUY_SIGNAL based on various indicators
technical_data["BUY_SIGNAL"] = (technical_data.RSI < 31)
technical_data["BUY_SIGNAL"] = (technical_data.MACD > 0)  # Example: buy signal when MACD is positive
technical_data["BUY_SIGNAL"] = (technical_data.Close < technical_data["Bollinger Low"])  # Example: buy signal when price is below the lower Bollinger band
technical_data["BUY_SIGNAL"] = (technical_data.ATR > technical_data.ATR.mean())  # Example: buy signal when ATR is above its mean

### Define SELL_SHORT_SIGNAL based on various indicators
technical_data["SELL_SIGNAL"] = (technical_data.RSI > 70)
technical_data["SELL_SIGNAL"] = (technical_data.MACD < 0)  # Example: short sell signal when MACD is negative
technical_data["SELL_SIGNAL"] = (technical_data.Close > technical_data["Bollinger High"])  # Example: short sell signal when price is above the upper Bollinger band
technical_data["SELL_SIGNAL"] = (technical_data.ATR < technical_data.ATR.mean())  # Example: short sell signal when ATR is below its mean

### BACKTESTING

capital = 1_000_000
n_shares = 127
stop_loss = 0.1258142736431898
take_profit = 0.1438947002278296
COM = 0.125 / 100

active_positions = []
portfolio_value = [capital]

for i, row in technical_data.iterrows():
    # Close all positions that are above/under tp or sl
    active_pos_copy = active_positions.copy()
    for pos in active_pos_copy:
        if row.Close < pos["stop_loss"]:
            # LOSS
            capital += row.Close * pos["n_shares"] * (1 - COM)
            active_positions.remove(pos)
        if row.Close > pos["take_profit"]:
            # PROFIT
            capital += row.Close * pos["n_shares"] * (1 - COM)
            active_positions.remove(pos)

    # Check if trading signal is True
    if row.BUY_SIGNAL:
        # Check if we have enough cash
        if capital > row.Close * (1 + COM) * n_shares:
            capital -= row.Close * (1 + COM) * n_shares
            active_positions.append({
                "type": "LONG",
                "bought_at": row.Close,
                "n_shares": n_shares,
                "stop_loss": row.Close * (1 - stop_loss),
                "take_profit": row.Close * (1 + take_profit)
            })
        else:
            print("OUT OF CASH")

    # Portfolio value through time
    positions_value = len(active_positions) * n_shares * row.Close
    portfolio_value.append(capital + positions_value)

# Close all positions that are above/under tp or sl
active_pos_copy = active_positions.copy()
for pos in active_pos_copy:
    capital += row.Close * pos["n_shares"] * (1 - COM)
    active_positions.remove(pos)

portfolio_value.append(capital)

### Benchmark

capital_benchmark = 1_000_000
shares_to_buy = capital_benchmark // (technical_data.Close.values[0] * (1 + COM))
capital_benchmark -= shares_to_buy * row.Close * (1 + COM)
portfolio_value_benchmark = (shares_to_buy * technical_data.Close) + capital_benchmark

plt.title(f"Active={(portfolio_value[-1] / 1_000_000 - 1)*100}%\n" +
          f"Passive={(portfolio_value_benchmark.values[-1] / 1_000_000 - 1)*100}%")
plt.plot(portfolio_value, label="Active")
plt.plot(portfolio_value_benchmark, label="Passive")
plt.legend()
plt.show()

## Short Selling
# Señal de venta basada en el RSI
technical_data["SELL_SIGNAL"] = (technical_data.RSI > 75)

capital = 1_000_000
n_shares = 100
stop_loss = 0.4
take_profit = 0.4
COM = 0.125 / 100

active_positions = []
portfolio_value = [capital]

for i, row in technical_data.iterrows():
    # Cerrar todas las posiciones que han alcanzado el stop loss o el take profit
    active_pos_copy = active_positions.copy()
    for pos in active_pos_copy:
        if row.Close > pos["stop_loss"]:  # La posición se cierra con pérdida
            capital += (pos["sold_at"] - row.Close) * pos["n_shares"] * (1 - COM)
            active_positions.remove(pos)
        elif row.Close < pos["take_profit"]:  # La posición se cierra con ganancia
            capital += (pos["sold_at"] - row.Close) * pos["n_shares"] * (1 - COM)
            active_positions.remove(pos)

    # Verificar si hay señal de venta
    if row.SELL_SIGNAL:
        # Verificar si tenemos suficientes acciones para vender
        if (capital > row.Close * (1 + COM) * n_shares * 1.5) and len(active_positions) < 1000:
            capital -= row.Close * (COM) * n_shares
            active_positions.append({
                "type": "SHORT",
                "sold_at": row.Close,
                "n_shares": n_shares,
                "stop_loss": row.Close * (1 + stop_loss),
                "take_profit": row.Close * (1 - take_profit)
            })
        else:
            print("OUT OF CASH")

    # Valor del portafolio a través del tiempo
    positions_value = sum([(pos["sold_at"] - row.Close) * pos["n_shares"] for pos in active_positions])
    portfolio_value.append(capital + positions_value)

# Cerrar todas las posiciones restantes al final
for pos in active_positions.copy():
    capital += (pos["sold_at"] - row.Close) * pos["n_shares"] * (1 - COM)
    active_positions.remove(pos)

portfolio_value.append(capital)


capital_benchmark = 1_000_000
shares_to_buy = capital_benchmark // (technical_data.Close.values[0] * (1 + COM))
capital_benchmark -= shares_to_buy * row.Close * (1 + COM)
portfolio_value_benchmark = (shares_to_buy * technical_data.Close) + capital_benchmark

plt.title(f"Active={(portfolio_value[-1] / 1_000_000 - 1)*100}%\n" +
          f"Passive={(portfolio_value_benchmark.values[-1] / 1_000_000 - 1)*100}%")
plt.plot(portfolio_value, label="Active")
plt.plot(portfolio_value_benchmark, label="Passive")
plt.legend()
plt.show()

def create_signals(data: pd.DataFrame, **kwargs):
    data = data.copy()

    rsi_1 = ta.momentum.RSIIndicator(data.Close, kwargs["rsi_window"])
    data["RSI"] = rsi_1.rsi()

    bollinger = ta.volatility.BollingerBands(data.Close,
                                             window=kwargs["bollinger_window"],
                                             window_dev=kwargs["bollinger_std"])
    data["Bollinger Low"] = bollinger.bollinger_lband()
    data["Bollinger High"] = bollinger.bollinger_hband()

    macd = ta.trend.MACD(data.Close, window_slow=kwargs["macd_slow_window"],
                         window_fast=kwargs["macd_fast_window"],
                         window_sign=kwargs["macd_sign_window"])
    data["MACD"] = macd.macd()
    data["MACD Signal"] = macd.macd_signal()

    atr = ta.volatility.AverageTrueRange(high=data.High, low=data.Low, close=data.Close,
                                         window=kwargs["atr_window"])
    data["ATR"] = atr.average_true_range()

    # Define BUY signals (BUY_SIGNAL)
    data["BUY_SIGNAL"] = (data.RSI < kwargs["rsi_lower_threshold"])
    data["BUY_SIGNAL"] |= (
                data.Close < data["Bollinger Low"])  # Buy signal when the price is below the lower Bollinger band
    data["BUY_SIGNAL"] |= (data.MACD > 0)  # Buy signal when MACD is positive
    data["BUY_SIGNAL"] |= (data.ATR > data.ATR.mean())  # Buy signal when ATR is above its mean

    # Define SELL signals (SELL_SIGNAL)
    data["SELL_SIGNAL"] = (data.RSI > kwargs["rsi_upper_threshold"])
    data["SELL_SIGNAL"] |= (
                data.Close > data["Bollinger High"])  # Sell signal when the price is above the upper Bollinger band
    data["SELL_SIGNAL"] |= (data.MACD < 0)  # Sell signal when MACD is negative
    data["SELL_SIGNAL"] |= (data.ATR < data.ATR.mean())  # Sell signal when ATR is below its mean

    return data.dropna()

def profit(trial):
    capital = 1_000_000
    n_shares = trial.suggest_int("n_shares", 50, 150)
    stop_loss = trial.suggest_float("stop_loss", 0.05, 0.4)
    take_profit = trial.suggest_float("take_profit", 0.05, 0.4)

    max_active_operations = 1000
    COM = 0.125 / 100

    active_positions = []
    portfolio_value = [capital]

    rsi_window = trial.suggest_int("rsi_window", 5, 50)
    rsi_lower_threshold = trial.suggest_int("rsi_lower_threshold", 10, 30)
    rsi_upper_threshold = trial.suggest_int("rsi_upper_threshold", 70, 90)
    bollinger_window = trial.suggest_int("bollinger_window", 5, 50)
    bollinger_std = 2  # Fixed value for Bollinger Bands standard deviation
    macd_slow_window = trial.suggest_int("macd_slow_window", 20, 40)
    macd_fast_window = trial.suggest_int("macd_fast_window", 5, 20)
    macd_sign_window = trial.suggest_int("macd_sign_window", 5, 20)
    atr_window = trial.suggest_int("atr_window", 5, 20)

    technical_data = create_signals(data,
                                    rsi_window=rsi_window,
                                    rsi_lower_threshold=rsi_lower_threshold,
                                    rsi_upper_threshold=rsi_upper_threshold,
                                    bollinger_window=bollinger_window,
                                    bollinger_std=bollinger_std,
                                    macd_slow_window=macd_slow_window,
                                    macd_fast_window=macd_fast_window,
                                    macd_sign_window=macd_sign_window,
                                    atr_window=atr_window)

    for i, row in technical_data.iterrows():
        # Close all positions that are above/under tp or sl
        active_pos_copy = active_positions.copy()
        for pos in active_pos_copy:
            if pos["type"] == "LONG":
                if row.Close < pos["stop_loss"]:
                    capital += row.Close * pos["n_shares"] * (1 - COM)
                    active_positions.remove(pos)
                if row.Close > pos["take_profit"]:
                    capital += row.Close * pos["n_shares"] * (1 - COM)
                    active_positions.remove(pos)
            elif pos["type"] == "SHORT":
                if row.Close > pos["stop_loss"]:
                    capital += (pos["sold_at"] - row.Close) * pos["n_shares"] * (1 - COM)
                    active_positions.remove(pos)
                elif row.Close < pos["take_profit"]:
                    capital += (pos["sold_at"] - row.Close) * pos["n_shares"] * (1 - COM)
                    active_positions.remove(pos)

        # Check if trading signal is True
        if row.BUY_SIGNAL and len(active_positions) < max_active_operations:
            if capital > row.Close * (1 + COM) * n_shares:
                capital -= row.Close * (1 + COM) * n_shares
                active_positions.append({
                    "type": "LONG",
                    "bought_at": row.Close,
                    "n_shares": n_shares,
                    "stop_loss": row.Close * (1 - stop_loss),
                    "take_profit": row.Close * (1 + take_profit)
                })

        # Check if short selling signal is True
        if row.SELL_SIGNAL and len(active_positions) < max_active_operations:
            if capital > row.Close * (1 + COM) * n_shares * 1.5:
                capital -= row.Close * (COM) * n_shares
                active_positions.append({
                    "type": "SHORT",
                    "sold_at": row.Close,
                    "n_shares": n_shares,
                    "stop_loss": row.Close * (1 + stop_loss),
                    "take_profit": row.Close * (1 - take_profit)
                })

        positions_value = sum(
            [(pos["bought_at"] if pos["type"] == "LONG" else pos["sold_at"] - row.Close) * pos["n_shares"] for pos in
             active_positions])
        portfolio_value.append(capital + positions_value)

    for pos in active_positions.copy():
        if pos["type"] == "LONG":
            capital += row.Close * pos["n_shares"] * (1 - COM)
        elif pos["type"] == "SHORT":
            capital += (pos["sold_at"] - row.Close) * pos["n_shares"] * (1 - COM)
        active_positions.remove(pos)

    portfolio_value.append(capital)
    return portfolio_value[-1]

# Assuming 'data' is a pre-loaded DataFrame with your market data
study = optuna.create_study(direction='maximize')
study.optimize(func=profit, n_trials=10)

print("Best parameters:", study.best_params)
