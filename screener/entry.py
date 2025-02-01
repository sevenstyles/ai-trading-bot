def check_entry_conditions(data, zones, trend):
    """Use 5m data with trend alignment."""
    ltf = data['5m']
    trade_signal = {
        'Status': 'NO TRADE',
        'Direction': 'NO TRADE',
        'Entry': None,
        'SL': None,
        'TP1': None,
        'Analysis': []
    }
    
    if trend == 'bullish':
        valid_zones = [z for z in zones if z['type'] == 'demand']
    else:
        valid_zones = [z for z in zones if z['type'] == 'supply']
    
    volume_ok = ltf.volume.iloc[-1] > ltf.volume.rolling(20).mean().iloc[-1] * 1.5
    
    last_close = ltf.close.iloc[-1]
    zone_active = any(
        (zone['low'] * 0.99 < last_close < zone['high'] * 1.01)
        for zone in valid_zones if zone['valid']
    )
    
    if volume_ok and zone_active:
        direction = 'LONG' if last_close > ltf.open.iloc[-1] else 'SHORT'
        trade_signal.update({
            'Status': 'TRADE',
            'Direction': direction,
            'Entry': last_close,
            'SL': last_close * 0.99,
            'TP1': last_close * 1.03,
            'Analysis': ['Volume confirmed zone activation']
        })
        
    if trend == 'bullish' and not any(z['type'] == 'demand' for z in valid_zones):
        return trade_signal
    
    return trade_signal 

def bat_strategy(df):
    signals = []
    asian = df.between_time('00:00','08:00')
    
    for i, candle in asian.iterrows():
        prev_range = calculate_range(df, i)
        if (candle.close > asian.high.rolling(5).max() * 1.012 or 
            candle.close < asian.low.rolling(5).min() * 0.988):
            
            entry = {
                'time': i,
                'direction': 'short' if candle.close < candle.open else 'long',
                'entry': candle.close,
                'sl': candle.close * (0.975 if 'short' else 1.025),
                'tp': candle.close * (0.96 if 'short' else 1.04)
            }
            
            if bat_volume_check(candle.volume, df.volume.rolling(20).mean()):
                signals.append(entry)
    
    return signals 