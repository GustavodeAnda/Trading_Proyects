import ta
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./data/btc_project_1m_test.csv").dropna()

n_shares = 6.0156138225684685
stop_loss = 0.25350468439149887
take_profit = 0.29385546248739247
macd_slow_window = 30
macd_fast_window = 11
macd_sign_window = 15
capital = 1_000_000
COM = 0.125 / 100

macd = ta.trend.MACD(close=data["Close"], window_slow=macd_slow_window, window_fast=macd_fast_window, window_sign=macd_sign_window)
technical_data = pd.DataFrame()
technical_data["Close"] = data["Close"]
technical_data["MACD"] = macd.macd()
technical_data["MACD Signal"] = macd.macd_signal()

technical_data["BUY_SIGNAL"] = (technical_data["MACD"] > technical_data["MACD Signal"])
technical_data["SELL_SIGNAL"] = (technical_data["MACD"] < technical_data["MACD Signal"])

active_positions = []
portfolio_value = [capital]

for i, row in technical_data.iterrows():
    # Cerrar posiciones que han alcanzado tp o sl
    active_pos_copy = active_positions.copy()
    for pos in active_pos_copy:
        if row.Close < pos["stop_loss"]:
            capital += row.Close * pos["n_shares"] * (1 - COM)
            active_positions.remove(pos)
        elif row.Close > pos["take_profit"]:
            capital += row.Close * pos["n_shares"] * (1 - COM)
            active_positions.remove(pos)

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

    positions_value = sum(
        [pos["n_shares"] * row.Close for pos in active_positions if pos["type"] == "LONG"]
    )
    portfolio_value.append(capital + positions_value)

for pos in active_positions.copy():
    capital += pos["n_shares"] * technical_data.iloc[-1].Close * (1 - COM)
    active_positions.remove(pos)
portfolio_value.append(capital)

capital_benchmark = 1_000_000
shares_to_buy = capital_benchmark // (technical_data.Close.values[0] * (1 + COM))
capital_benchmark -= shares_to_buy * technical_data.Close.values[0] * (1 + COM)
portfolio_value_benchmark = (shares_to_buy * technical_data.Close) + capital_benchmark
portfolio_value_benchmark_p = pd.DataFrame(portfolio_value_benchmark)
portfolio_value_benchmark_p.index = range(0, len(portfolio_value_benchmark))

print(f"Valor final del portafolio activo: {portfolio_value[-1]}")

plt.figure(figsize=(14, 7))
plt.title('Trading vs training')
plt.plot(portfolio_value, label="Active")
plt.plot(portfolio_value_benchmark_p, label="Passive")
plt.title(f"Active={(portfolio_value[-1] / 1_000_000 - 1)*100}%\n" +
          f"Passive={(portfolio_value_benchmark.iloc[-1] / 1_000_000 - 1)*100}%")
plt.legend()
plt.show()
