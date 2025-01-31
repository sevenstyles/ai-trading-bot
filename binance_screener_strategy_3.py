import pandas as pd
import matplotlib.pyplot as plt
from binance.client import Client
import mplfinance as mpf
from datetime import datetime
import os
import re
import numpy as np

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
    """Extended historical data ranges"""
    intervals = {
        '30m': 4032,  # 3 months of data (4032 candles)
        '4h': 1512,    # 6 months of data
        '5m': 576     
    }
    
    data = {}
    for interval, limit in intervals.items():
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                   'close_time', 'quote_asset_volume', 'number_of_trades',
                   'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        # Include volume in the selected columns
        data[interval] = df[['open', 'high', 'low', 'close', 'volume']]
    return data

def get_trend_direction(df_4h):
    """Determine trend using 4h data"""
    sma_50 = df_4h.close.rolling(50).mean().iloc[-1]
    current_price = df_4h.close.iloc[-1]
    return 'bullish' if current_price > sma_50 else 'bearish'

def detect_supply_demand_zones(data):
    """Simplified validation - only check current close"""
    df = data['30m']
    current_close = df.close.iloc[-1]
    
    zones = []
    
    # Calculate volume benchmarks
    vol_ma = df.volume.rolling(20).mean()
    df['vol_ma'] = vol_ma
    df['vol_pct'] = df.volume / vol_ma
    
    i = 1
    while i < len(df)-2:
        current = df.iloc[i]
        
        # Institutional volume threshold
        if current.vol_pct < 1.0 or \
           not (current.volume > df.volume.rolling(50).mean().iloc[i] * 1.5):
            i += 1
            continue
            
        # Liquidity check (price at swing high/low)
        is_liquidity_area = (
            current.high == df.high.rolling(200).max().iloc[i] or  # 200-period lookback
            current.low == df.low.rolling(200).min().iloc[i]
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
            if extreme_candle.vol_pct < 1.2:  # Changed from 1.5
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
                'timeframe': '30m',
                'valid': True,
                'timestamp': extreme_candle.name
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
            
            if extreme_candle.vol_pct < 1.2:  # Changed from 1.5
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
    
    # Final validation fix - only check closes after zone formation
    for zone in zones:
        # Get full historical closes AFTER zone creation
        zone_df = df[df.index >= zone['timestamp']]
        # Require 2+ consecutive closes inside zone
        consecutive_inside = (zone_df['close'] > zone['low']).astype(int) & \
                             (zone_df['close'] < zone['high']).astype(int)
        zone['valid'] = consecutive_inside.rolling(2).sum().max() < 2
    
    # Zones expire after 7 days
    for zone in zones:
        if (datetime.now(zone['timestamp'].tzinfo) - zone['timestamp']).days > 7:
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
    
    return merged

def check_entry_conditions(data, zones, trend):
    """Use 5m data with trend alignment"""
    ltf = data['5m']
    trade_signal = {
        'Status': 'NO TRADE',
        'Direction': 'NO TRADE',
        'Entry': None,
        'SL': None,
        'TP1': None,
        'Analysis': []
    }
    
    # Require trend alignment
    if trend == 'bullish':
        valid_zones = [z for z in zones if z['type'] == 'demand']
    else:
        valid_zones = [z for z in zones if z['type'] == 'supply']
    
    # Relax volume requirement
    volume_ok = ltf.volume.iloc[-1] > ltf.volume.rolling(20).mean().iloc[-1] * 1.5
    
    # Price must be in zone with 1% buffer
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
        
    # Trend alignment check
    if trend == 'bullish' and not any(z['type'] == 'demand' for z in valid_zones):
        return trade_signal
    
    return trade_signal

def plot_trade_setup(df, signal, zones):
    # Increase visible candles to 500 (≈10 days of 30m data)
    plot_df = df.iloc[-500:]  # Changed from -200
    
    # Create main plot with adjusted scale
    fig, axes = mpf.plot(
        plot_df,
        type='candle',
        style='charles',
        title=f"{signal['Symbol']} Trade Setup",
        ylabel='Price',
        figsize=(16,8),  # Larger figure size
        returnfig=True,
        tight_layout=True,
        ylim=(
            plot_df.low.min() * 0.997,  # Reduced padding
            plot_df.high.max() * 1.003
        )
    )
    ax = axes[0]
    
    # Filter only 30m timeframe zones
    valid_zones = [z for z in zones if z['timeframe'] == '30m' and z['valid']]
    
    # Debug output
    print(f"Valid 30m zones: {len(valid_zones)}")
    for z in valid_zones:
        print(f"{z['type']} zone {z['low']}-{z['high']} formed at "
              f"{z['timestamp'].tz_convert('America/New_York')}")
    
    # Plot zones as shaded boxes
    for zone in valid_zones:
        color = '#FF0000' if zone['type'] == 'supply' else '#00FF00'
        alpha = 0.2
        
        # Extend zone plotting to current candle
        mask = (plot_df.index >= zone['timestamp']) & (plot_df.index <= plot_df.index[-1])
        
        # Plot shaded zone
        ax.fill_between(
            plot_df.index[mask],
            zone['low'],
            zone['high'],
            facecolor=color,
            alpha=alpha,
            edgecolor='none',
            zorder=0  # Behind candles
        )
        
        # Add zone borders
        ax.plot(
            plot_df.index[mask], 
            [zone['high']]*sum(mask),
            color=color,
            alpha=0.5,
            linewidth=1
        )
        ax.plot(
            plot_df.index[mask],
            [zone['low']]*sum(mask),
            color=color,
            alpha=0.5,
            linewidth=1
        )
    
    # Plot zones even if partially out of view
    for zone in valid_zones:
        # Extend plot boundaries to include zones
        y_min = min(df.low.min(), zone['low'])
        y_max = max(df.high.max(), zone['high'])
        ax.set_ylim(y_min * 0.995, y_max * 1.005)
    
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
            trend = get_trend_direction(data['4h'])
            signal = check_entry_conditions(data, zones, trend)

            # Generate output
            output = {
                'Symbol': symbol,
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Status': 'NO TRADE',
                'Direction': 'NO TRADE',
                'Analysis': []
            }

            # Merge signal data
            output.update(signal)

            # Print results
            print(f"\nStrategy 3 Results for {symbol}:")
            for k, v in output.items():
                if k != 'Symbol':
                    print(f"{k}: {v}")

            # Plot using 30m data
            plot_trade_setup(data['30m'], output, zones)

            # Add debug info
            print(f"Found {len(zones)} zones")
            for z in zones:
                print(f"{z['type']} zone: {z['low']}-{z['high']} ({z['timeframe']})")

            # Clean up output formatting
            if output['Status'] == 'TRADE':
                print(f"Direction: {output['Direction']}")
                print(f"Entry: {output['Entry']:.4f}")
                print(f"SL: {output['SL']:.4f}") 
                print(f"TP1: {output['TP1']:.4f}")
            else:
                print("Status: NO TRADE")
                print("Direction: NO TRADE")

        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
        print(f"{'='*40}\n")

if __name__ == '__main__':
    run_screener()
