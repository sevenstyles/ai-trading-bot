import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("backtester")

def generate_signal(data, lookback=40):
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

    logger.debug(f"Analyzing window: Swing High: {swing_high} (Loc: {swing_high_loc}), Swing Low: {swing_low} (Loc: {swing_low_loc})")
    logger.debug(f"Breakout candle - High: {breakout_candle.high}, Low: {breakout_candle.low}, Close: {breakout_candle.close}")
    logger.debug(f"Confirmation candle - High: {confirmation_candle.high}, Low: {confirmation_candle.low}, Close: {confirmation_candle.close}")

    # --- LONG TRADE ---
    if (lookback - swing_low_loc) >= 5 and breakout_candle.close < swing_low:
        logger.debug(f"Found potential long setup (breakout below swing low), Index diff: {lookback - swing_low_loc}")
        if confirmation_candle.close > swing_low:
            logger.debug("Confirmed failed breakout (close above swing low)")
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
                "debug_info": debug_info
            }

    # --- SHORT TRADE ---
    if (lookback - swing_high_loc) >= 5 and breakout_candle.close > swing_high:
        logger.debug(f"Found potential short setup (breakout above swing high), Index diff: {lookback - swing_high_loc}")
        if confirmation_candle.close < swing_high:
            logger.debug("Confirmed failed breakout (close below swing high)")
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
                "debug_info": debug_info
            }

    return None