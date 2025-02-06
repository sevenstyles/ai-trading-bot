import pandas as pd
import numpy as np


def generate_signal(data, lookback=30):
    """
    Generate trading signal based solely on a swing breakout strategy.
    
    For a short trade:
      1. Examine the first 'lookback' candles to identify the swing high (the highest high).
      2. The breakout candle (immediately following the lookback window) must close above the swing high.
      3. The confirmation candle (the candle after the breakout candle) must close below the swing high.
      4. Entry is taken at the confirmation candle close, the stop loss is set just above the breakout candle's high,
         and the take profit is set at a 1:3 risk–reward ratio.
    
    For a long trade, the reverse logic applies:
      1. Identify the swing low in the lookback window.
      2. The breakout candle must close below the swing low.
      3. The confirmation candle must close above the swing low.
      4. Entry is taken at the confirmation candle close, the stop loss is set just below the breakout candle's low,
         and the take profit is set at a 1:3 risk–reward ratio.
    
    Parameters:
      data (DataFrame): Lower timeframe price data containing at least (lookback + 2) candles.
      lookback (int): Number of candles to use as the lookback for the swing point (min 10, max 300).
    
    Returns:
      dict or None: A dictionary containing 'side', 'entry', 'stop_loss', and 'take_profit' if a setup is detected; otherwise, None.
    """
    lookback = max(10, min(lookback, 300))
    if len(data) < lookback + 2:
        return None

    swing_window = data.iloc[:lookback]
    breakout_candle = data.iloc[lookback]
    confirmation_candle = data.iloc[lookback + 1]

    swing_high = swing_window["high"].max()
    swing_low = swing_window["low"].min()

    pos_high = swing_window["high"].argmax()
    pos_low = swing_window["low"].argmin()

    if pos_low >= pos_high:
        return None

    # Short trade setup
    if breakout_candle.close > swing_high and confirmation_candle.close < swing_high:
        entry_price = confirmation_candle.close
        stop_loss = breakout_candle.high  # Stop loss is placed just above the breakout candle's high
        risk = stop_loss - entry_price
        take_profit = entry_price - 3 * risk
        return {
            "side": "short",
            "entry": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

    # Long trade setup
    if breakout_candle.close < swing_low and confirmation_candle.close > swing_low:
        entry_price = confirmation_candle.close
        stop_loss = breakout_candle.low  # Stop loss is placed just below the breakout candle's low
        risk = entry_price - stop_loss
        take_profit = entry_price + 3 * risk
        return {
            "side": "long",
            "entry": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

    return None 