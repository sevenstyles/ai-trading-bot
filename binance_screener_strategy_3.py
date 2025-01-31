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
        '4h': 200,  # HTF (Higher Time Frame)
        '1h': 100   # LTF (Lower Time Frame)
    }
    
    data = {}
    for interval, limit in intervals.items():
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        # Include all 12 columns from Binance API response
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                   'close_time', 'quote_asset_volume', 'number_of_trades',
                   'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        # Select only the columns we actually need
        data[interval] = df[['open', 'high', 'low', 'close']]
    return data

def detect_supply_demand_zones(data):
    """Identify valid supply/demand zones"""
    zones = []
    for tf, df in data.items():
        for i in range(1, len(df)-1):
            # Identify swing points
            prev_candle = df.iloc[i-1]
            current_candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Supply Zone (Bearish)
            if current_candle.high > prev_candle.high and current_candle.high > next_candle.high:
                if current_candle.close < current_candle.open:  # Bearish candle
                    zone_high = max(current_candle.high, prev_candle.high)
                    zone_low = min(current_candle.low, prev_candle.low)
                    zones.append({
                        'type': 'supply',
                        'high': zone_high,
                        'low': zone_low,
                        'timeframe': tf
                    })
                    
            # Demand Zone (Bullish)
            elif current_candle.low < prev_candle.low and current_candle.low < next_candle.low:
                if current_candle.close > current_candle.open:  # Bullish candle
                    zone_high = max(current_candle.high, prev_candle.high)
                    zone_low = min(current_candle.low, prev_candle.low)
                    zones.append({
                        'type': 'demand',
                        'high': zone_high,
                        'low': zone_low,
                        'timeframe': tf
                    })
    return zones

def check_entry_conditions(df, zones):
    """Check strategy entry conditions"""
    trade_signal = {}
    # Implement strategy entry rules here
    return trade_signal

def plot_trade_setup(df, signal, zones):
    """Plot with zones and trade setup"""
    apds = []
    colors = {'supply': 'red', 'demand': 'green'}
    
    # Plot zones
    for zone in zones:
        if zone['timeframe'] == '4h':  # Only plot HTF zones
            style = mpf.make_addplot(
                pd.Series([zone['high']]*len(df), index=df.index),
                type='line', color=colors[zone['type']], alpha=0.3, width=2
            )
            apds.append(style)
            
            style = mpf.make_addplot(
                pd.Series([zone['low']]*len(df), index=df.index),
                type='line', color=colors[zone['type']], alpha=0.3, width=2
            )
            apds.append(style)
    
    # Plot trade signals
    if signal.get('Status') == 'TRADE':
        entry_line = mpf.make_addplot(
            pd.Series(signal['Entry'], index=df.index),
            type='scatter', color='blue', markersize=50
        )
        apds.append(entry_line)
    
    # Configure plot
    mpf.plot(df, type='candle', style='charles',
             title=f'{SYMBOL} Trade Setup',
             ylabel='Price',
             addplot=apds,
             figsize=(12,6),
             savefig=f"plots/{SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
def run_screener():
    # Get multi-timeframe data
    data = fetch_data(SYMBOL)  # Returns dict of DataFrames
    zones = detect_supply_demand_zones(data)
    
    # Use 1h data for plotting
    plot_df = data['1h']  # Get LTF dataframe for charting
    signal = check_entry_conditions(plot_df, zones)
    
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
