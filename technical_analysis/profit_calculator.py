import pandas as pd
import optuna
from technical_analysis import create_signals


def profit(trial, data, indicators):
    capital = 1_000_000

    # Check if we are dealing with BTC or AAPL
    if 'btc' in data.columns[0].lower():
        n_shares = trial.suggest_float("n_shares", 0.01, 20)  # Allow fractional shares for BTC
        COM = 0.05 / 100
    else:
        n_shares = trial.suggest_int("n_shares", 1, 100)  # Allow only whole shares for AAPL
        COM = 0.125 / 100

    stop_loss = trial.suggest_float("stop_loss", 0.05, 0.4)
    take_profit = trial.suggest_float("take_profit", 0.05, 0.4)
    max_active_operations = 1000

    active_positions = []
    portfolio_value = [capital]

    kwargs = {}

    if "RSI" in indicators:
        kwargs["rsi_window"] = trial.suggest_int("rsi_window", 5, 50)
        kwargs["rsi_lower_threshold"] = trial.suggest_int("rsi_lower_threshold", 10, 30)
        kwargs["rsi_upper_threshold"] = trial.suggest_int("rsi_upper_threshold", 70, 90)

    if "Bollinger Bands" in indicators:
        kwargs["bollinger_window"] = trial.suggest_int("bollinger_window", 5, 50)

    if "MACD" in indicators:
        kwargs["macd_slow_window"] = trial.suggest_int("macd_slow_window", 20, 40)
        kwargs["macd_fast_window"] = trial.suggest_int("macd_fast_window", 5, 20)
        kwargs["macd_sign_window"] = trial.suggest_int("macd_sign_window", 5, 20)

    if "ATR" in indicators:
        kwargs["atr_window"] = trial.suggest_int("atr_window", 5, 20)

    technical_data = create_signals(data, indicators=indicators, **kwargs)

    for i, row in technical_data.iterrows():
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

        if row.BUY_SIGNAL and len(active_positions) < max_active_operations:
            cost = row.Close * (1 + COM) * n_shares
            if capital > cost:  # Ensure sufficient capital for the purchase
                capital -= cost
                active_positions.append({
                    "type": "LONG",
                    "bought_at": row.Close,
                    "n_shares": n_shares,
                    "stop_loss": row.Close * (1 - stop_loss),
                    "take_profit": row.Close * (1 + take_profit)
                })

        if row.SELL_SIGNAL and len(active_positions) < max_active_operations:
            cost = row.Close * (1 + COM) * n_shares
            if capital > cost:  # Ensure sufficient capital for the sale
                capital -= cost * 0.5  # Assuming selling requires half the capital for margin trading
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