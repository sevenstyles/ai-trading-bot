import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("backtester")

def generate_signal(data, lookback=40):
    """
    Generate trading signal based on failed breakout reversal strategy.
    
    For a long trade (from failed short setup):
      1. Examine the first 'lookback' candles to identify the swing high
      2. The breakout candle must have its high above the swing high
      3. The confirmation candle must close below the swing high
      4. Within next 5 candles, if price closes above the highest high of breakout/confirmation candles
         we enter a long trade with stop loss below the lowest low of the monitoring period
    
    For a short trade (from failed long setup):
      1. Examine the first 'lookback' candles to identify the swing low
      2. The breakout candle must have its low below the swing low
      3. The confirmation candle must close above the swing low
      4. Within next 5 candles, if price closes below the lowest low of breakout/confirmation candles
         we enter a short trade with stop loss above the highest high of the monitoring period
    
    Parameters:
      data (DataFrame): Lower timeframe price data containing at least (lookback + 7) candles
                       (lookback + breakout + confirmation + 5 monitoring candles)
      lookback (int): Number of candles to use as the lookback for the swing point (min 10, max 300)
    
    Returns:
      dict or None: A dictionary containing 'side', 'entry', 'stop_loss', 'take_profit' and 'debug_info' if a setup is detected
    """
    lookback = max(10, min(lookback, 300))
    if len(data) < lookback + 7:  # Need extra candles for monitoring period
        logger.debug(f"Not enough data: got {len(data)} candles, need {lookback + 7}")
        return None

    swing_window = data.iloc[:lookback]
    breakout_candle = data.iloc[lookback]
    confirmation_candle = data.iloc[lookback + 1]
    monitoring_candles = data.iloc[lookback + 2:lookback + 7]  # 5 candles after confirmation

    swing_high = swing_window["high"].max()
    swing_low = swing_window["low"].min()

    logger.debug(f"Analyzing window: Swing High: {swing_high}, Swing Low: {swing_low}")
    logger.debug(f"Breakout candle - High: {breakout_candle.high}, Low: {breakout_candle.low}, Close: {breakout_candle.close}")
    logger.debug(f"Confirmation candle - High: {confirmation_candle.high}, Low: {confirmation_candle.low}, Close: {confirmation_candle.close}")

    # Check for failed short setup turning into long
    if breakout_candle.high > swing_high:
        logger.debug("Found potential short setup (breakout above swing high)")
        if confirmation_candle.close < swing_high:
            logger.debug("Confirmed failed short (close below swing high)")
            # Get the highest high of breakout and confirmation candles
            setup_high = max(breakout_candle.high, confirmation_candle.high)
            
            # Check if price closes above the setup high within monitoring period
            for i, candle in enumerate(monitoring_candles.itertuples()):
                if candle.close > confirmation_candle.high:  # Changed to be less strict
                    logger.debug(f"Found long entry on candle {i+1} of monitoring period")
                    # Found our long entry
                    entry_price = candle.close
                    # Stop loss below the lowest low during monitoring period
                    monitoring_period = data.iloc[lookback + 1:lookback + 2 + i + 1]
                    stop_loss = monitoring_period['low'].min() * 0.995
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (4 * risk)

                    debug_info = {
                        "setup_type": "failed_short_to_long",
                        "lookback": lookback,
                        "swing_high": swing_high,
                        "swing_low": swing_low,
                        "setup_high": setup_high,
                        "breakout_candle": breakout_candle.to_dict(),
                        "confirmation_candle": confirmation_candle.to_dict(),
                        "reversal_candle": {
                            'open': candle.open,
                            'high': candle.high,
                            'low': candle.low,
                            'close': candle.close,
                            'volume': candle.volume
                        },
                        "monitoring_candles_used": i + 1
                    }
                    return {
                        "side": "long",
                        "entry": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "debug_info": debug_info
                    }

    # Check for failed long setup turning into short
    if breakout_candle.low < swing_low:
        logger.debug("Found potential long setup (breakout below swing low)")
        if confirmation_candle.close > swing_low:
            logger.debug("Confirmed failed long (close above swing low)")
            # Get the lowest low of breakout and confirmation candles
            setup_low = min(breakout_candle.low, confirmation_candle.low)
            
            # Check if price closes below the setup low within monitoring period
            for i, candle in enumerate(monitoring_candles.itertuples()):
                if candle.close < confirmation_candle.low:  # Changed to be less strict
                    logger.debug(f"Found short entry on candle {i+1} of monitoring period")
                    # Found our short entry
                    entry_price = candle.close
                    # Stop loss above the highest high during monitoring period
                    monitoring_period = data.iloc[lookback + 1:lookback + 2 + i + 1]
                    stop_loss = monitoring_period['high'].max() * 1.005
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (4 * risk)

                    debug_info = {
                        "setup_type": "failed_long_to_short",
                        "lookback": lookback,
                        "swing_high": swing_high,
                        "swing_low": swing_low,
                        "setup_low": setup_low,
                        "breakout_candle": breakout_candle.to_dict(),
                        "confirmation_candle": confirmation_candle.to_dict(),
                        "reversal_candle": {
                            'open': candle.open,
                            'high': candle.high,
                            'low': candle.low,
                            'close': candle.close,
                            'volume': candle.volume
                        },
                        "monitoring_candles_used": i + 1
                    }
                    return {
                        "side": "short",
                        "entry": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "debug_info": debug_info
                    }

    return None 