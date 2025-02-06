import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    """Calculate Average True Range."""
    df = df.copy()
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(period).mean()
    return df['atr']

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index."""
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window=period, min_periods=period).mean()
    avg_loss = down.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def calculate_adx(df, period=14):
    """Calculate ADX (Average Directional Index)."""
    try:
        df['prev_close'] = df['close'].shift()
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['prev_close']).abs()
        df['tr3'] = (df['low'] - df['prev_close']).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()
        df['dm_plus'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            df['high'] - df['high'].shift(1), 0
        )
        df['dm_minus'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            df['low'].shift(1) - df['low'], 0
        )
        df['tr_sma'] = df['tr'].ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        df['dm_plus_sma'] = df['dm_plus'].ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        df['dm_minus_sma'] = df['dm_minus'].ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        with np.errstate(divide='ignore', invalid='ignore'):
            df['di_plus'] = (df['dm_plus_sma'] / df['tr_sma']) * 100
            df['di_minus'] = (df['dm_minus_sma'] / df['tr_sma']) * 100
        dx = (abs(df['di_plus'] - df['di_minus']) / 
              (df['di_plus'] + df['di_minus']).replace(0, 1e-5)) * 100
        df['adx'] = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'tr',
                 'dm_plus', 'dm_minus', 'tr_sma', 'dm_plus_sma', 'dm_minus_sma',
                 'di_plus', 'di_minus'], axis=1, inplace=True)
        return df
    except Exception as e:
        print(f"ADX calculation error: {str(e)}")
        return None

def calculate_market_structure(df):
    """Identify key market structure elements."""
    df['higher_high'] = df['high'] > df['high'].shift()
    df['lower_low'] = df['low'] < df['low'].shift()
    df['range'] = df['high'] - df['low']
    return df

def calculate_trend_strength(df):
    """Quantify trend strength based on price change and range."""
    df['momentum'] = df['close'].pct_change(3).rolling(14).mean()
    df['trend_power'] = np.log(df['high'] / df['low']).rolling(14).sum()
    return df

def generate_signal(df, i):
    """Generate a trade signal based on market structure breaks and indicators."""
    if i < 20:
        return False
    structure_break = (df['higher_high'].iloc[i] and 
                       (df['volume'].iloc[i] > df['volume'].iloc[i-1] * 1.5))
    volatility_expansion = (df['range'].iloc[i] > df['range'].rolling(14).mean().iloc[i] * 1.2)
    smart_money_confirm = (
        (df['low'].iloc[i] > df['low'].iloc[i-1]) and
        (df['close'].iloc[i] > df['open'].iloc[i]) and
        (((df['close'].iloc[i] - df['open'].iloc[i]) / ((df['high'].iloc[i] - df['low'].iloc[i]) + 1e-5)) > 0.45) and
        (df['volume'].iloc[i] > df['volume'].iloc[i-1])
    )
    trend_aligned = ((df['momentum'].iloc[i] > 0) and
                     (df['trend_power'].iloc[i] > df['trend_power'].quantile(0.4)))
    adx_condition = (df['adx'].iloc[i] > 25)
    rsi_condition = (df['rsi'].iloc[i] > 45) and (df['rsi'].iloc[i] < 80)
    return all([structure_break, volatility_expansion, smart_money_confirm, trend_aligned,
                adx_condition, rsi_condition])

def generate_short_signal(df, i):
    """Generate a short trade signal based on inverted market structure and technical indicators."""
    if i < 20:
        return False
    # Inverted structure: check for lower low with a volume spike
    structure_break = (df['lower_low'].iloc[i] and 
                       (df['volume'].iloc[i] > df['volume'].iloc[i-1] * 1.5))

    # Volatility expansion remains the same
    volatility_expansion = (df['range'].iloc[i] > df['range'].rolling(14).mean().iloc[i] * 1.2)

    # Inverted smart money confirmation: bearish candle with a proper body-to-range ratio
    smart_money_confirm = (
        (df['high'].iloc[i] < df['high'].iloc[i-1]) and
        (df['close'].iloc[i] < df['open'].iloc[i]) and
        (((df['open'].iloc[i] - df['close'].iloc[i]) / ((df['high'].iloc[i] - df['low'].iloc[i]) + 1e-5)) > 0.45) and
        (df['volume'].iloc[i] > df['volume'].iloc[i-1])
    )

    # For shorts: require negative momentum and mirror the trend power check
    trend_aligned = ((df['momentum'].iloc[i] < 0) and
                     (df['trend_power'].iloc[i] < df['trend_power'].quantile(0.6)))

    adx_condition = (df['adx'].iloc[i] > 25)

    # Mirror RSI condition for shorts (avoid extreme oversold)
    rsi_condition = (df['rsi'].iloc[i] > 20) and (df['rsi'].iloc[i] < 55)

    return all([structure_break, volatility_expansion, smart_money_confirm, trend_aligned,
                adx_condition, rsi_condition])

def get_trend_direction(htf_data, period=20):
    """
    Determines the trend direction on higher timeframe data.
    
    Uses a simple moving average (SMA) to decide:
      - If the latest close is greater than the SMA, it's an 'uptrend'.
      - Otherwise, it's a 'downtrend'.
      
    Parameters:
      htf_data (DataFrame): Higher timeframe price data (must include a 'close' column).
      period (int): Lookback period for the SMA.
      
    Returns:
      str: "uptrend" or "downtrend"
    """
    sma = htf_data['close'].rolling(window=period).mean()
    last_close = htf_data['close'].iloc[-1]
    last_sma = sma.iloc[-1]
    
    if last_close > last_sma:
        return "uptrend"
    else:
        return "downtrend" 