import pandas as pd
from strategy import generate_signal

def run_backtest(data):
    """
    Runs a simple backtest on the provided lower timeframe data.
    """
    signal = generate_signal(data)
    if signal:
        print("Trade signal generated:", signal)
    else:
        print("No trade signal generated.")

if __name__ == "__main__":
    # Example lower timeframe candles (requires at least lookback+2 candles, default lookback is 30)
    data = pd.DataFrame({
        "open": [100, 102, 101, 100, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127],
        "high": [103, 104, 102, 101, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128],
        "low": [99, 100, 98, 97, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124],
        "close": [102, 103, 101, 100, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
    })

    # Higher timeframe data is no longer used in this backtest call
    run_backtest(data) 