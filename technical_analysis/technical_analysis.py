import pandas as pd
import ta
import matplotlib.pyplot as plt


def calculate_rsi(data: pd.DataFrame, window: int) -> pd.Series:
    return ta.momentum.RSIIndicator(close=data["Close"], window=window).rsi()


def calculate_macd(data: pd.DataFrame, slow: int, fast: int, sign: int) -> pd.DataFrame:
    macd = ta.trend.MACD(close=data["Close"], window_slow=slow, window_fast=fast, window_sign=sign)
    return pd.DataFrame({
        "MACD": macd.macd(),
        "MACD Signal": macd.macd_signal()
    })


def calculate_bollinger_bands(data: pd.DataFrame, window: int, window_dev: int) -> pd.DataFrame:
    bollinger = ta.volatility.BollingerBands(close=data["Close"], window=window, window_dev=window_dev)
    return pd.DataFrame({
        "Bollinger High": bollinger.bollinger_hband(),
        "Bollinger Low": bollinger.bollinger_lband(),
        "BOLL": bollinger.bollinger_hband() - bollinger.bollinger_lband()
    })


def calculate_atr(data: pd.DataFrame, window: int) -> pd.Series:
    return ta.volatility.AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"],
                                          window=window).average_true_range()


def create_signals(data: pd.DataFrame, indicators: list, **kwargs) -> pd.DataFrame:
    data = data.copy()

    if "RSI" in indicators:
        data["RSI"] = calculate_rsi(data, kwargs["rsi_window"])

    if "MACD" in indicators:
        macd_data = calculate_macd(data, slow=kwargs["macd_slow_window"], fast=kwargs["macd_fast_window"],
                                   sign=kwargs["macd_sign_window"])
        data = data.join(macd_data)

    if "Bollinger Bands" in indicators:
        bollinger_data = calculate_bollinger_bands(data, window=kwargs["bollinger_window"], window_dev=2)
        data = data.join(bollinger_data)

    if "ATR" in indicators:
        data["ATR"] = calculate_atr(data, kwargs["atr_window"])

    data["BUY_SIGNAL"] = False
    data["SELL_SIGNAL"] = False

    if "RSI" in indicators:
        data["BUY_SIGNAL"] |= (data.RSI < kwargs["rsi_lower_threshold"])
        data["SELL_SIGNAL"] |= (data.RSI > kwargs["rsi_upper_threshold"])

    if "MACD" in indicators:
        data["BUY_SIGNAL"] |= (data.MACD > 0)
        data["SELL_SIGNAL"] |= (data.MACD < 0)

    if "Bollinger Bands" in indicators:
        data["BUY_SIGNAL"] |= (data.Close < data["Bollinger Low"])
        data["SELL_SIGNAL"] |= (data.Close > data["Bollinger High"])

    if "ATR" in indicators:
        data["BUY_SIGNAL"] |= (data.ATR > data.ATR.mean())
        data["SELL_SIGNAL"] |= (data.ATR < data.ATR.mean())

    print(f"BUY_SIGNAL count: {data['BUY_SIGNAL'].sum()}")
    print(f"SELL_SIGNAL count: {data['SELL_SIGNAL'].sum()}")

    return data.dropna()

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    rsi = calculate_rsi(data, window=48)
    macd = calculate_macd(data, slow=26, fast=12, sign=9)
    bollinger = calculate_bollinger_bands(data, window=10, window_dev=2)
    atr = calculate_atr(data, window=14)

    technical_data = pd.DataFrame()
    technical_data["Close"] = data["Close"]
    technical_data["RSI"] = rsi
    technical_data["MACD"] = macd["MACD"]
    technical_data["MACD Signal"] = macd["MACD Signal"]
    technical_data["BOLL"] = bollinger["BOLL"]
    technical_data["Bollinger Low"] = bollinger["Bollinger Low"]
    technical_data["Bollinger High"] = bollinger["Bollinger High"]
    technical_data["ATR"] = atr

    return technical_data.dropna()