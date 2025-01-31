import pandas as pd
import matplotlib.pyplot as plt
from binance.client import Client
import mplfinance as mpf
from datetime import datetime
import os
import re

# Load strategy parameters
with open('strategy_3_prompt.txt', 'r') as f:
    strategy_rules = f.read()

# Load trade pair    
with open('trade_pair.txt', 'r') as f:
    # Read with BOM handling and comment stripping
    raw_content = f.read().lstrip('\ufeff').split('#')[0].strip().upper()
    SYMBOL = re.sub(r'[^A-Z0-9]', '', raw_content)  # Remove any non-alphanum chars

# Binance API setup
client = Client()

def fetch_data(symbol):
    """Fetch multi-timeframe data"""
    intervals = {
        '1h': 200,   # New HTF (Higher Time Frame)
        '5m': 500    # New LTF (Lower Time Frame)
    }
    
    data = {}
    for interval, limit in intervals.items():
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                   'close_time', 'quote_asset_volume', 'number_of_trades',
                   'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        # Include volume in the selected columns
        data[interval] = df[['open', 'high', 'low', 'close', 'volume']]
    return data

def detect_supply_demand_zones(data):
    """Institutional-grade zone detection with volume validation"""
    zones = []
    for tf, df in data.items():
        # Calculate volume benchmarks
        vol_ma = df.volume.rolling(20).mean()
        df['vol_ma'] = vol_ma
        df['vol_pct'] = df.volume / vol_ma
        
        i = 1
        while i < len(df)-2:
            current = df.iloc[i]
            
            # Institutional volume threshold
            if current.vol_pct < 1.2:  # Skip low volume formations
                i += 1
                continue
                
            # Liquidity check (price at swing high/low)
            is_liquidity_area = (
                current.high == df.high.rolling(50).max().iloc[i] or 
                current.low == df.low.rolling(50).min().iloc[i]
            )
            
            if current.close < current.open and is_liquidity_area:  # Supply
                push_start = i
                while i < len(df)-1 and df.iloc[i].close < df.iloc[i].open:
                    i += 1
                push_end = i-1
                
                # Institutional filter: Minimum 3 candles in push
                if (push_end - push_start) < 2:
                    continue
                    
                extreme_candle = df.iloc[push_start:push_end+1].low.idxmin()
                extreme_candle = df.loc[extreme_candle]
                
                # Volume confirmation
                if extreme_candle.vol_pct < 1.5:
                    continue
                    
                # Zone formation
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
                    'timeframe': tf,
                    'valid': True
                })

            elif current.close > current.open and is_liquidity_area:  # Demand
                push_start = i
                while i < len(df)-1 and df.iloc[i].close > df.iloc[i].open:
                    i += 1
                push_end = i-1
                
                if (push_end - push_start) < 2:
                    continue
                    
                extreme_candle = df.iloc[push_start:push_end+1].high.idxmax()
                extreme_candle = df.loc[extreme_candle]
                
                if extreme_candle.vol_pct < 1.5:
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
                    'timeframe': tf,
                    'valid': True
                })
            
            i += 1
    
    # Add zone invalidation check
    current_price = data['5m'].close.iloc[-1]
    
    for zone in zones:
        # Mark zone as invalid if price has entered
        if zone['low'] < current_price < zone['high']:
            zone['valid'] = False
            
    return zones

def merge_zones(zones):
    """Combine overlapping zones of same type"""
    merged = []
    for zone in sorted(zones, key=lambda x: x['high'], reverse=True):
        if not merged:
            merged.append(zone)
            continue
            
        last = merged[-1]
        if (zone['type'] == last['type'] and 
            zone['timeframe'] == last['timeframe'] and
            abs(zone['high'] - last['high']) < last['high'] * 0.005):
            
            # Merge zones
            merged[-1]['high'] = max(last['high'], zone['high'])
            merged[-1]['low'] = min(last['low'], zone['low'])
        else:
            merged.append(zone)
    
    for zone in merged:
        # Preserve validity status when merging
        if any(z['valid'] for z in merged if z['type'] == zone['type']):
            zone['valid'] = True
            
    return [z for z in merged if z['valid']]  # Remove invalid zones

def check_entry_conditions(data, zones):
    """Institutional entry validation"""
    trade_signal = {}
    ltf = data['5m']
    
    # Fix deprecated indexing
    recent_volume = ltf.volume.iloc[-3:].mean()
    vol_ma = ltf.volume.rolling(20).mean().iloc[-1]
    volume_ok = recent_volume > vol_ma * 2
    
    # Price action validation
    last_close = ltf.close.iloc[-1]
    zone_active = any(
        zone['low'] < last_close < zone['high'] 
        for zone in zones if zone['valid']
    )
    
    if volume_ok and zone_active:
        # Basic entry logic (needs expansion)
        trade_signal = {
            'Direction': 'LONG' if last_close > ltf.open.iloc[-1] else 'SHORT',
            'Entry': last_close,
            'SL': last_close * 0.99,  # 1% stop loss
            'TP1': last_close * 1.03,  # 3% take profit
            'Analysis': ['Volume confirmed zone activation']
        }
        
    return trade_signal

def plot_trade_setup(df, signal, zones):
    """Plot with properly shaded zones using matplotlib"""
    # Create main plot
    fig, axes = mpf.plot(
        df,
        type='candle',
        style='charles',
        title=f'{SYMBOL} Trade Setup',
        ylabel='Price',
        figsize=(12,6),
        returnfig=True
    )
    
    # Get main axis
    ax = axes[0]
    
    # Plot zones as shaded regions
    for zone in zones:
        if zone['valid']:
            color = 'red' if zone['type'] == 'supply' else 'green'
            alpha = 0.3
            
            # Create horizontal lines for zone boundaries
            ax.axhspan(zone['low'], zone['high'], 
                       facecolor=color, alpha=alpha, edgecolor='none')
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    filename = f"plots/{SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(filename)
    plt.close(fig)

def run_screener():
    data = fetch_data(SYMBOL)
    zones = detect_supply_demand_zones(data)
    
    # Pass full data dict to entry check
    signal = check_entry_conditions(data, zones)
    
    # Use 1h data for plotting
    plot_df = data['1h']
    
    # Generate output
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output = {
        'Timestamp': timestamp,
        'Status': 'NO TRADE',
        'Direction': 'NO TRADE',
        'Analysis': []
    }
    
    if signal:
        output.update(signal)
        output['Status'] = 'TRADE'
    
    # Print formatted output
    print(f"\n{'='*40}")
    print("Strategy 3 Screener Results:")
    for k, v in output.items():
        print(f"{k}: {v}")
    print(f"{'='*40}\n")
    
    # Plot the chart with actual DataFrame
    plot_trade_setup(plot_df, output, zones)
    
if __name__ == '__main__':
    run_screener()
