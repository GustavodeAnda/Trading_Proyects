import pandas as pd
import ta

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    rsi_indicator = ta.momentum.RSIIndicator(close=data.Close, window=48)
    bollinger = ta.volatility.BollingerBands(data.Close, window=10)
    macd = ta.trend.MACD(data.Close, window_slow=26, window_fast=12, window_sign=9)
    atr = ta.volatility.AverageTrueRange(high=data.High, low=data.Low, close=data.Close, window=14)

    technical_data = pd.DataFrame()
    technical_data["Close"] = data.Close
    technical_data["RSI"] = rsi_indicator.rsi()
    technical_data["MACD"] = macd.macd()
    technical_data['BOLL'] = bollinger.bollinger_hband() - bollinger.bollinger_lband()
    technical_data["Bollinger Low"] = bollinger.bollinger_lband()
    technical_data["Bollinger High"] = bollinger.bollinger_hband()
    technical_data["ATR"] = atr.average_true_range()
    technical_data = technical_data.dropna()

    return technical_data

def create_signals(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
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

    data["BUY_SIGNAL"] = (data.RSI < kwargs["rsi_lower_threshold"])
    data["BUY_SIGNAL"] |= (data.Close < data["Bollinger Low"])
    data["BUY_SIGNAL"] |= (data.MACD > 0)
    data["BUY_SIGNAL"] |= (data.ATR > data.ATR.mean())

    data["SELL_SIGNAL"] = (data.RSI > kwargs["rsi_upper_threshold"])
    data["SELL_SIGNAL"] |= (data.Close > data["Bollinger High"])
    data["SELL_SIGNAL"] |= (data.MACD < 0)
    data["SELL_SIGNAL"] |= (data.ATR < data.ATR.mean())

    return data.dropna()

def profit(trial, data):
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
    bollinger_std = 2
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
            if capital > row.Close * (1 + COM) * n_shares:
                capital -= row.Close * (1 + COM) * n_shares
                active_positions.append({
                    "type": "LONG",
                    "bought_at": row.Close,
                    "n_shares": n_shares,
                    "stop_loss": row.Close * (1 - stop_loss),
                    "take_profit": row.Close * (1 + take_profit)
                })

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