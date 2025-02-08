import requests
import pandas as pd
import numpy as np
import ta

def calculate_atr(df, period=200):
    """Calculates Average True Range (ATR)."""
    hl = df['high'] - df['low']
    hpc = abs(df['high'] - df['close'].shift(1))
    lpc = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def calculate_parsed_high_low(df, volatility_type='atr', atr_period=200, atr_multiplier=2):
    """Calculates parsed high and low based on volatility."""
    if volatility_type == 'atr':
        atr = calculate_atr(df, period=atr_period)
        df['atr'] = atr  # Store ATR in DataFrame for later use
        df['volatility_measure'] = df['atr'] #For now, just use ATR
        df['high_low_range'] = df['high'] - df['low']
        df['high_volatility_bar'] = df['high_low_range'] >= (atr_multiplier * df['volatility_measure'])
        df['parsed_high'] = np.where(df['high_volatility_bar'], df['low'], df['high'])
        df['parsed_low'] = np.where(df['high_volatility_bar'], df['high'], df['low'])
    elif volatility_type == 'range':
        df['range'] = df['high'] - df['low']
        df['volatility_measure'] = df['range'].rolling(window=len(df), min_periods=1).mean()
        df['high_volatility_bar'] = df['range'] >= (atr_multiplier * df['volatility_measure'])
        df['parsed_high'] = np.where(df['high_volatility_bar'], df['low'], df['high'])
        df['parsed_low'] = np.where(df['high_volatility_bar'], df['high'], df['low'])
    else:
        raise ValueError("Invalid volatility_type. Choose 'atr' or 'range'.")
    return df

def load_binance_data(symbol, interval, limit=500, volatility_type='atr', atr_period=200, atr_multiplier=2):
    """
    Fetches candlestick data from Binance API and calculates parsed high/low.

    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
        interval (str): Candlestick interval (e.g., '4h').
        limit (int): Number of candlesticks to retrieve.

    Returns:
        pandas.DataFrame: DataFrame containing the candlestick data, or None if an error occurred.
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                         'close_time', 'quote_asset_volume', 'number_of_trades',
                                         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        print(f"DataFrame loaded with shape: {df.shape}")
        df = calculate_parsed_high_low(df, volatility_type, atr_period, atr_multiplier)  # Calculate parsed high/low
        print(df.head())
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Binance: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    btc_data = load_binance_data('BTCUSDT', '4h', limit=500)
    if btc_data is not None:
        print(btc_data.head())
    else:
        print("Failed to load data.") 