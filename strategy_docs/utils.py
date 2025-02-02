import json
import pandas as pd
import numpy as np

def load_json_data(filepath):
    """Load OHLC JSON file into a pandas DataFrame."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

def calculate_stop_loss(entry_price, atr, volatility_ratio):
    """Dynamic stop loss calculation based on volatility regime."""
    if volatility_ratio > 1.5:
        return entry_price - (atr * 2.0)
    elif volatility_ratio < 0.8:
        return entry_price - (atr * 1.0)
    return entry_price - (atr * 1.5)

def calculate_risk_params(atr_pct):
    """Dynamic risk management parameters based on daily volatility."""
    risk_reward = min(3.0, max(1.5, 2.5 - (atr_pct / 0.5)))
    risk_percent = min(0.02, max(0.005, 0.01 * (2.0 - (atr_pct / 0.5))))
    return risk_reward, risk_percent

def get_tier_parameters(tier):
    """Return parameter thresholds based on volatility tier."""
    params = {
        'high': {'buffer_pct': 0.0045, 'sl': 0.01, 'tp': 0.022, 'vol_filter': 1.8},
        'medium': {'buffer_pct': 0.0085, 'sl': 0.015, 'tp': 0.032, 'vol_filter': 1.4},
        'low': {'buffer_pct': 0.013, 'sl': 0.022, 'tp': 0.042, 'vol_filter': 1.2},
    }
    return params[tier]

def get_volatility_tier(df):
    """Determine volatility tier based on ATR percentage."""
    # This function assumes calculate_atr imported from indicators.
    from indicators import calculate_atr
    atr = calculate_atr(df)
    atr_percentile = atr.iloc[-1] / df['close'].iloc[-1] * 100
    if atr_percentile >= 1.1:
        return 'high'
    elif atr_percentile >= 0.7:
        return 'medium'
    else:
        return 'low' 