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
      dict or None: A dictionary containing 'side', 'entry', 'stop_loss', 'take_profit' and 'debug_info' if a setup is detected; otherwise, None.
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

    # Short trade setup with minimum 3 candle gap enforcement
    if breakout_candle.close > swing_high and confirmation_candle.close < swing_high:
        if (lookback - pos_high) < 4:  # require at least 3 candles between swing high candle and breakout candle
            return None
        entry_price = confirmation_candle.close
        higher_of_two = max(breakout_candle.high, confirmation_candle.high)
        stop_loss = higher_of_two * 1.001  # set SL just above the higher of the two candles
        risk = stop_loss - entry_price
        take_profit = entry_price - 3 * risk
        debug_info = {
            "lookback": lookback,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "pos_high": int(pos_high),
            "pos_low": int(pos_low),
            "breakout_candle": breakout_candle.to_dict(),
            "confirmation_candle": confirmation_candle.to_dict()
        }
        return {
            "side": "short",
            "entry": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "debug_info": debug_info
        }

    # Long trade setup with minimum 3 candle gap enforcement
    if breakout_candle.close < swing_low and confirmation_candle.close > swing_low:
        if (lookback - pos_low) < 4:  # require at least 3 candles between swing low candle and breakout candle
            return None
        entry_price = confirmation_candle.close
        lower_of_two = min(breakout_candle.low, confirmation_candle.low)
        stop_loss = lower_of_two * 0.999  # set SL just below the lower of the two candles
        risk = entry_price - stop_loss
        take_profit = entry_price + 3 * risk
        debug_info = {
            "lookback": lookback,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "pos_high": int(pos_high),
            "pos_low": int(pos_low),
            "breakout_candle": breakout_candle.to_dict(),
            "confirmation_candle": confirmation_candle.to_dict()
        }
        return {
            "side": "long",
            "entry": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "debug_info": debug_info
        }

    return None 