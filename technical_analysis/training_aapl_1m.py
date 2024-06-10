import ta
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("./data/aapl_project_1m_train.csv").dropna()

n_shares = 44
stop_loss = 0.17466016271602947
take_profit = 0.35114612061554595
rsi_window = 18
rsi_lower_threshold = 26
rsi_upper_threshold = 90
capital = 1_000_000
COM = 0.125 / 100

# RSI INDICATOR
rsi_indicator = ta.momentum.RSIIndicator(close=data.Close, window=rsi_window)
technical_data = pd.DataFrame()
technical_data["Close"] = data.Close
technical_data["RSI"] = rsi_indicator.rsi()
technical_data = technical_data.dropna()

# SIGNALS
technical_data["BUY_SIGNAL"] = (technical_data.RSI < rsi_lower_threshold)
technical_data["SELL_SIGNAL"] = (technical_data.RSI > rsi_upper_threshold)

active_positions = []
portfolio_value = [capital]

for i, row in technical_data.iterrows():
    # CLOSE POSITIONS tp o sl
    active_pos_copy = active_positions.copy()
    for pos in active_pos_copy:
        if row.Close < pos["stop_loss"]:
            capital += row.Close * pos["n_shares"] * (1 - COM)
            active_positions.remove(pos)
        elif row.Close > pos["take_profit"]:
            capital += row.Close * pos["n_shares"] * (1 - COM)
            active_positions.remove(pos)

    # OPEN POSITIONS TP O SL
    if row.BUY_SIGNAL:
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

    # VALUATION OF POSITIONS
    positions_value = sum(
        [pos["n_shares"] * row.Close for pos in active_positions if pos["type"] == "LONG"]
    )
    portfolio_value.append(capital + positions_value)

# CLOSING POSITIONS
for pos in active_positions.copy():
    capital += pos["n_shares"] * technical_data.iloc[-1].Close * (1 - COM)
    active_positions.remove(pos)
portfolio_value.append(capital)

# PASSIVE STRATEGY
capital_benchmark = 1_000_000
shares_to_buy = capital_benchmark // (technical_data.Close.values[0] * (1 + COM))
capital_benchmark -= shares_to_buy * technical_data.Close.values[0] * (1 + COM)
portfolio_value_benchmark = (shares_to_buy * technical_data.Close) + capital_benchmark
portfolio_value_benchmark_p = pd.DataFrame(portfolio_value_benchmark)
portfolio_value_benchmark_p.index = range(0, len(portfolio_value_benchmark))

# ACTIVE FINAL VALUE
print(f"Valor final del portafolio activo: {portfolio_value[-1]}")

# PLOT
plt.figure(figsize=(14, 7))
plt.title('Trading vs training')
plt.plot(portfolio_value, label="Active")
plt.plot(portfolio_value_benchmark_p, label="Passive")
plt.title(f"Active={(portfolio_value[-1] / 1_000_000 - 1)*100}%\n" +
          f"Passive={(portfolio_value_benchmark.iloc[-1] / 1_000_000 - 1)*100}%")
plt.legend()
plt.show()