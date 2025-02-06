import pandas as pd
from strategy import generate_signal

def run_backtest(data, htf_data):
    """
    Runs a simple backtest on the provided lower timeframe and HTF data.
    """
    signal = generate_signal(data, htf_data)
    if signal:
        print("Trade signal generated:", signal)
    else:
        print("No trade signal generated.")

if __name__ == "__main__":
    # Example lower timeframe candles (3 candles required)
    data = pd.DataFrame({
        "open": [100, 102, 101],
        "high": [103, 104, 102],
        "low": [99, 100, 98],
        "close": [102, 103, 101]
    })

    # Example higher timeframe data for trend detection
    htf_data = pd.DataFrame({
        "open": [100, 101, 102, 103, 104, 105, 107, 106, 108, 109],
        "high": [101, 102, 103, 104, 105, 107, 108, 107, 109, 110],
        "low": [99, 100, 101, 102, 103, 105, 106, 105, 107, 108],
        "close": [100, 102, 102, 103, 104, 106, 107, 106, 108, 109]
    })

    run_backtest(data, htf_data) 