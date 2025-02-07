import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("backtester")

def generate_signal(data, lookback=40):
    """
    Generate trading signals based on failed breakout patterns.
    Uses the exact lowest/highest point of the 5 bars before breakout for stop loss.
    Take profit is set at 4x the distance from entry to stop loss.
    """
    if len(data) < lookback + 7:  # Need enough bars for lookback + breakout + confirmation + monitoring
        return None
        
    swing_window = data.iloc[:lookback]
    breakout_candle = data.iloc[lookback]
    confirmation_candle = data.iloc[lookback + 1]
    monitoring_candles = data.iloc[lookback + 2:lookback + 7]  # 5 candles after confirmation

    swing_high = swing_window["high"].max()
    swing_low = swing_window["low"].min()

    # Get the 5 bars before breakout for stop loss calculation
    pre_breakout_window = data.iloc[lookback-1:lookback]
    pre_breakout_low = pre_breakout_window['low'].min()
    pre_breakout_high = pre_breakout_window['high'].max()

    logger.debug(f"Analyzing window: Swing High: {swing_high}, Swing Low: {swing_low}")
    logger.debug(f"Pre-breakout window - High: {pre_breakout_high}, Low: {pre_breakout_low}")
    logger.debug(f"Breakout candle - High: {breakout_candle.high}, Low: {breakout_candle.low}, Close: {breakout_candle.close}")
    logger.debug(f"Confirmation candle - High: {confirmation_candle.high}, Low: {confirmation_candle.low}, Close: {confirmation_candle.close}")

    # Failed long setup becomes a long trade
    if breakout_candle.high > swing_high:
        logger.debug("Found potential long setup (breakout above swing high)")
        if confirmation_candle.close < swing_high:
            logger.debug("Confirmed failed breakout (close below swing high)")
            setup_high = max(breakout_candle.high, confirmation_candle.high)
            
            for i, candle in enumerate(monitoring_candles.itertuples()):
                if candle.close > swing_high:
                    logger.debug(f"Found long entry on candle {i+1} of monitoring period")
                    entry_price = candle.close
                    
                    # Set stop loss at the pre-breakout low
                    stop_loss = pre_breakout_low
                    
                    # Calculate take profit at 4x the risk
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (4 * risk)  # 4:1 reward:risk ratio
                    
                    debug_info = {
                        "setup_type": "failed_breakout_to_long",
                        "lookback": lookback,
                        "swing_high": swing_high,
                        "swing_low": swing_low,
                        "setup_high": setup_high,
                        "pre_breakout_low": pre_breakout_low,
                        "pre_breakout_high": pre_breakout_high,
                        "breakout_candle": breakout_candle.to_dict(),
                        "confirmation_candle": confirmation_candle.to_dict(),
                        "reversal_candle": {
                            'open': candle.open,
                            'high': candle.high,
                            'low': candle.low,
                            'close': candle.close,
                            'volume': candle.volume
                        },
                        "monitoring_candles_used": i + 1,
                        "risk_distance": risk,
                        "reward_distance": 4 * risk
                    }
                    return {
                        "side": "long",
                        "entry": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "take_profit_levels": [take_profit],  # Keep this for compatibility
                        "debug_info": debug_info
                    }


    # Failed short setup becomes a short trade
    if breakout_candle.low < swing_low:
        logger.debug("Found potential short setup (breakout below swing low)")
        if confirmation_candle.close > swing_low:
            logger.debug("Confirmed failed breakout (close above swing low)")
            setup_low = min(breakout_candle.low, confirmation_candle.low)
            
            for i, candle in enumerate(monitoring_candles.itertuples()):
                if candle.close < swing_low:
                    logger.debug(f"Found short entry on candle {i+1} of monitoring period")
                    entry_price = candle.close
                    
                    # Set stop loss at the pre-breakout high
                    stop_loss = pre_breakout_high
                    
                    # Calculate take profit at 4x the risk
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (4 * risk)  # 4:1 reward:risk ratio
                    
                    debug_info = {
                        "setup_type": "failed_breakout_to_short",
                        "lookback": lookback,
                        "swing_high": swing_high,
                        "swing_low": swing_low,
                        "setup_low": setup_low,
                        "pre_breakout_low": pre_breakout_low,
                        "pre_breakout_high": pre_breakout_high,
                        "breakout_candle": breakout_candle.to_dict(),
                        "confirmation_candle": confirmation_candle.to_dict(),
                        "reversal_candle": {
                            'open': candle.open,
                            'high': candle.high,
                            'low': candle.low,
                            'close': candle.close,
                            'volume': candle.volume
                        },
                        "monitoring_candles_used": i + 1,
                        "risk_distance": risk,
                        "reward_distance": 4 * risk
                    }
                    return {
                        "side": "short",
                        "entry": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "take_profit_levels": [take_profit],  # Keep this for compatibility
                        "debug_info": debug_info
                    }

    return None 