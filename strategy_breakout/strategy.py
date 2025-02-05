import pandas as pd
import numpy as np
from datetime import datetime

def get_trend_direction(df_4h):
    """Determine trend using 4h data."""
    sma_50 = df_4h.close.rolling(50).mean().iloc[-1]
    current_price = df_4h.close.iloc[-1]
    return 'bullish' if current_price > sma_50 else 'bearish'

def detect_supply_demand_zones(data):
    """Identify supply/demand zones using 30m data."""
    df = data['30m']
    zones = []
    vol_ma = df.volume.rolling(20).mean()
    df['vol_ma'] = vol_ma
    df['vol_pct'] = df.volume / vol_ma
    i = 1
    while i < len(df) - 2:
        current = df.iloc[i]
        if current.vol_pct < 1.0 or not (current.volume > df.volume.rolling(50).mean().iloc[i] * 1.5):
            i += 1
            continue
        is_liquidity_area = (current.high == df.high.rolling(200).max().iloc[i] or
                             current.low == df.low.rolling(200).min().iloc[i])
        if current.close < current.open and is_liquidity_area:
            push_start = i
            while i < len(df) - 1 and df.iloc[i].close < df.iloc[i].open:
                i += 1
            push_end = i - 1
            if (push_end - push_start) < 2:
                continue
            extreme_candle = df.iloc[push_start:push_end+1].low.idxmin()
            extreme_candle = df.loc[extreme_candle]
            if extreme_candle.vol_pct < 1.2:
                continue
            next_candle = df.iloc[push_end+1]
            if next_candle.high > extreme_candle.high:
                zone_high = extreme_candle.high
                zone_low = extreme_candle.low
            else:
                zone_high = max(extreme_candle.open, extreme_candle.close)
                zone_low = min(extreme_candle.open, extreme_candle.close)
            zones.append({
                'type': 'supply',
                'high': zone_high,
                'low': zone_low,
                'timeframe': '30m',
                'valid': True,
                'timestamp': extreme_candle.name
            })
        elif current.close > current.open and is_liquidity_area:
            push_start = i
            while i < len(df) - 1 and df.iloc[i].close > df.iloc[i].open:
                i += 1
            push_end = i - 1
            if (push_end - push_start) < 2:
                continue
            extreme_candle = df.iloc[push_start:push_end+1].high.idxmax()
            extreme_candle = df.loc[extreme_candle]
            if extreme_candle.vol_pct < 1.2:
                continue
            next_candle = df.iloc[push_end+1]
            if next_candle.low < extreme_candle.low:
                zone_high = extreme_candle.high
                zone_low = extreme_candle.low
            else:
                zone_high = max(extreme_candle.open, extreme_candle.close)
                zone_low = min(extreme_candle.open, extreme_candle.close)
            zones.append({
                'type': 'demand',
                'high': zone_high,
                'low': zone_low,
                'timeframe': '30m',
                'valid': True,
                'timestamp': extreme_candle.name
            })
        i += 1

    for zone in zones:
        zone_df = df[df.index >= zone['timestamp']]
        consecutive_inside = (zone_df['close'] > zone['low']).astype(int) & (zone_df['close'] < zone['high']).astype(int)
        zone['valid'] = consecutive_inside.rolling(2).sum().max() < 2

    for zone in zones:
        if (datetime.now(zone['timestamp'].tzinfo) - zone['timestamp']).days > 7:
            zone['valid'] = False

    return zones

def merge_zones(zones):
    """Merge overlapping zones of the same type."""
    merged = []
    for zone in sorted(zones, key=lambda x: x['high'], reverse=True):
        if not merged:
            merged.append(zone)
            continue
        last = merged[-1]
        if (zone['type'] == last['type'] and zone['timeframe'] == last['timeframe'] and
            abs(zone['high'] - last['high']) < last['high'] * 0.005):
            merged[-1]['high'] = max(last['high'], zone['high'])
            merged[-1]['low'] = min(last['low'], zone['low'])
        else:
            merged.append(zone)
    return merged

def check_entry_conditions(data, zones, trend):
    """Determine whether entry conditions are met using 5m data."""
    ltf = data['5m']
    trade_signal = {
        'Status': 'NO TRADE',
        'Direction': 'NO TRADE',
        'Entry': None,
        'SL': None,
        'TP1': None,
        'Analysis': []
    }
    valid_zones = [z for z in zones if ((z['type'] == 'demand' if trend == 'bullish' else z['type'] == 'supply') and z['valid'])]
    volume_ok = ltf.volume.iloc[-1] > ltf.volume.rolling(20).mean().iloc[-1] * 1.5
    last_close = ltf.close.iloc[-1]
    zone_active = any((zone['low'] * 0.98 < last_close < zone['high'] * 1.02) for zone in valid_zones)
    if volume_ok and zone_active:
        direction = 'LONG' if last_close > ltf.open.iloc[-1] else 'SHORT'
        trade_signal.update({
            'Status': 'TRADE',
            'Direction': direction,
            'Entry': last_close,
            'SL': last_close * 0.99,
            'TP1': last_close * 1.04,
            'Analysis': ['Volume confirmed zone activation']
        })
    return trade_signal

def breakout_signal(df, i):
    """Simplified breakout signal logic for increased trade frequency."""
    if i < 3:
        return False
    price_buffer = 1.003
    cond_price = df['close'].iloc[i] > df['consol_high'].iloc[i] * price_buffer
    cond_volume = (df['volume'].iloc[i] / df['volume'].rolling(20).mean().iloc[i]) >= 1.5
    cond_consolidation = (df['consol_high'].iloc[i] - df['consol_low'].iloc[i]) < df['consol_high'].iloc[i] * 1.03
    cond_trend = df['close'].iloc[i] > df['ema50'].iloc[i]
    cond_macd = df['macd'].iloc[i] > df['signal'].iloc[i]
    cond_rsi = ((df['rsi'].iloc[i] > 30) and (df['rsi'].iloc[i] < 70)) if 'rsi' in df.columns else True
    return all([cond_price, cond_volume, cond_trend, cond_consolidation, cond_macd, cond_rsi])

def identify_consolidation(df, period=7):
    """Identify consolidation zones adaptively."""
    df['consol_high'] = df['high'].rolling(period, min_periods=3).max().ffill()
    df['consol_low'] = df['low'].rolling(period, min_periods=3).min().ffill()
    return df 