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

def get_tier_parameters(tier):
    """Return parameter thresholds based on volatility tier."""
    params = {
        'high': {'buffer_pct': 0.0045, 'sl': 0.01, 'tp': 0.022, 'vol_filter': 1.8},
        'medium': {'buffer_pct': 0.0085, 'sl': 0.015, 'tp': 0.032, 'vol_filter': 1.4},
        'low': {'buffer_pct': 0.013, 'sl': 0.022, 'tp': 0.042, 'vol_filter': 1.2},
    }
    return params[tier]

# Removed get_volatility_tier because it depended on the deleted indicators.py 