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

# Load trade pairs    
with open('trade_pair.txt', 'r') as f:
    # Read all valid pairs
    raw_content = f.read().lstrip('\ufeff')
    SYMBOLS = [
        re.sub(r'[^A-Z0-9]', '', line.split('#')[0].strip())
        for line in raw_content.split('\n')
        if line.strip() and not line.strip().startswith('#')
    ]

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
    
    # After zone creation, check for closes within zones
    for zone in zones:
        zone_df = data[zone['timeframe']]
        # Check if any candle CLOSED within the zone
        closes_inside = (zone_df['close'] > zone['low']) & (zone_df['close'] < zone['high'])
        if closes_inside.any():
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
    # Modify zone plotting to only show valid zones
    valid_zones = [z for z in zones if z['valid']]
    
    # Create main plot
    fig, axes = mpf.plot(
        df,
        type='candle',
        style='charles',
        title=f"{signal['Symbol']} Trade Setup",
        ylabel='Price',
        figsize=(12,6),
        returnfig=True
    )
    
    # Get main axis
    ax = axes[0]
    
    # Plot zones as shaded regions
    for zone in valid_zones:
        color = 'red' if zone['type'] == 'supply' else 'green'
        alpha = 0.3
        
        # Create horizontal lines for zone boundaries
        ax.axhspan(zone['low'], zone['high'], 
                   facecolor=color, alpha=alpha, edgecolor='none')
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    filename = f"plots/{signal['Symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(filename)
    plt.close(fig)

def run_screener():
    for symbol in SYMBOLS:
        print(f"\n{'='*40}")
        print(f"Analyzing {symbol}")
        try:
            data = fetch_data(symbol)
            zones = detect_supply_demand_zones(data)
            signal = check_entry_conditions(data, zones)

            # Generate output
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output = {
                'Symbol': symbol,
                'Timestamp': timestamp,
                'Status': 'NO TRADE',
                'Direction': 'NO TRADE',
                'Analysis': []
            }

            if signal:
                output.update(signal)
                output['Status'] = 'TRADE'

            # Print results
            print(f"\nStrategy 3 Results for {symbol}:")
            for k, v in output.items():
                if k != 'Symbol':
                    print(f"{k}: {v}")

            # Plot using 1h data
            plot_trade_setup(data['1h'], output, zones)

        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
        print(f"{'='*40}\n")

if __name__ == '__main__':
    run_screener()
