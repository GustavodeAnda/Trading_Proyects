import matplotlib.pyplot as plt
import pandas as pd

def plot_technical_indicators(technical_data: pd.DataFrame, title: str):
    fig, axs = plt.subplots(6, 1, figsize=(12, 24))

    axs[0].plot(technical_data["Close"], label="Close")
    axs[0].set_title('Close')

    axs[1].plot(technical_data["RSI"], label="RSI")
    axs[1].set_title('RSI')

    axs[2].plot(technical_data["MACD"], label="MACD")
    axs[2].plot(technical_data["MACD Signal"], label="MACD Signal")
    axs[2].set_title('MACD')

    axs[3].plot(technical_data["Close"], label="Close")
    axs[3].plot(technical_data["Bollinger High"], label="Bollinger High")
    axs[3].plot(technical_data["Bollinger Low"], label="Bollinger Low")
    axs[3].set_title('Bollinger Bands')

    axs[4].plot(technical_data["BOLL"], label="BOLL")
    axs[4].set_title('BOLL')

    axs[5].plot(technical_data["ATR"], label="ATR")
    axs[5].set_title('ATR')

    for ax in axs:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()