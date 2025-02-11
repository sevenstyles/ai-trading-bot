import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("backtester")

def generate_signal(data, lookback=40, risk_per_trade=None):
    """
    Generate trading signals based on failed breakout patterns.
    """
    if len(data) < lookback + 2:  # Need enough bars for lookback + breakout + confirmation
        return None

    swing_window = data.iloc[:lookback]
    breakout_candle = data.iloc[lookback]
    confirmation_candle = data.iloc[lookback + 1]

    swing_high = swing_window["high"].max()
    swing_low = swing_window["low"].min()
    swing_high_index = swing_window["high"].idxmax()
    swing_low_index = swing_window["low"].idxmin()

    # Get the integer location within the swing_window
    swing_high_loc = swing_window.index.get_loc(swing_high_index)
    swing_low_loc = swing_window.index.get_loc(swing_low_index)

    # --- LONG TRADE ---
    if (lookback - swing_low_loc) >= 5 and breakout_candle.close < swing_low:
        if confirmation_candle.close > swing_low:
            entry_price = confirmation_candle.close
            stop_loss = confirmation_candle.low
            risk = stop_loss - entry_price
            take_profit = entry_price - (3 * risk)  # 3:1 reward:risk ratio

            debug_info = {
                "setup_type": "failed_breakout_to_long",
                "lookback": lookback,
                "swing_high": swing_high,
                "swing_low": swing_low,
                "breakout_candle": breakout_candle.to_dict(),
                "confirmation_candle": confirmation_candle.to_dict(),
                "risk_distance": risk,
                "reward_distance": 3 * risk
            }
            return {
                "side": "long",
                "entry": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "take_profit_levels": [take_profit],
                "debug_info": debug_info,
                "risk_per_trade": risk
            }

    # --- SHORT TRADE ---
    if (lookback - swing_high_loc) >= 5 and breakout_candle.close > swing_high:
        if confirmation_candle.close < swing_high:
            entry_price = confirmation_candle.close
            stop_loss = confirmation_candle.high
            risk = stop_loss - entry_price
            take_profit = entry_price - (3 * risk)  # 3:1 reward:risk ratio

            debug_info = {
                "setup_type": "failed_breakout_to_short",
                "lookback": lookback,
                "swing_high": swing_high,
                "swing_low": swing_low,
                "breakout_candle": breakout_candle.to_dict(),
                "confirmation_candle": confirmation_candle.to_dict(),
                "risk_distance": risk,
                "reward_distance": 3 * risk
            }
            return {
                "side": "short",
                "entry": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "take_profit_levels": [take_profit],
                "debug_info": debug_info,
                "risk_per_trade": risk
            }

    return None