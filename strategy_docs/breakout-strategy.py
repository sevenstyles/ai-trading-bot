import pandas as pd
import matplotlib.pyplot as plt
from binance.client import Client
import mplfinance as mpf
from datetime import datetime, timedelta, time, timezone
import os
import re
import numpy as np
import random
import json
import logging
from pathlib import Path

# Determine the base directory (i.e. the directory where this script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add this after BASE_DIR definition
DEBUG_LOG_DIR = Path(BASE_DIR) / 'debug_logs'
DEBUG_LOG_DIR.mkdir(exist_ok=True)

# Update path for the strategy prompt file if it remains in the same folder as before or moved.
# For example, if you want to keep it in binance_data, update accordingly:
# STRATEGY_FILE = os.path.join(BASE_DIR, '../strategy_3_prompt.txt')
# If you have moved it into the screener folder too, you can simply do:
STRATEGY_FILE = os.path.join(BASE_DIR, '..', 'strategy_3_prompt.txt')

# Load strategy parameters
with open(STRATEGY_FILE, 'r') as f:
    strategy_rules = f.read()

# Binance API setup
client = Client()

CONFIRMATION_WINDOW = 24  # 6 hours (24*15min)
MIN_SESSION_RANGE_PCT = 1.5
MIN_PRICE = 0.000001  # Minimum valid price to prevent micro moves
POSITION_SIZE_CAP = 5000  # Lower cap for low-priced tokens
MIN_QUOTE_VOLUME = 25000  # Further reduce to accommodate lower liquidity pairs
MIN_TRADE_VALUE = 250  # Minimum $250 trade value

# Further relaxed parameters
MIN_CONSOL_RANGE_PCT = 3.0    # 3% consolidation range
BUFFER_PCT = 0.0025          # 0.25% buffer
MIN_CONSOL_BARS = 8          # 32 hours (8*4h)
VOL_SPIKE_MULTIPLIER = 1.0   # Match average volume
MIN_CONSOL_DAYS = 1          # 1 day minimum consolidation
VOL_MA_PERIOD = 20           # 20-bar volume average

# Add time-based exit to prevent holding through reversions
MAX_HOLD_PERIOD = 72  # 3 days (72 hours)

# Improved risk management parameters
RISK_REWARD_RATIO = 2.0      # More achievable 2:1 ratio
ATR_PERIOD = 14        # For volatility-based stops

# Improve stop loss calculation
ATR_MULTIPLIER = 1.5        # Wider stop for 4h volatility

def fetch_data(symbol, start_time=None, end_time=None):
    """
    Fetch extended historical data for the given symbol.
    Optionally filters the data by passing startTime and endTime (in milliseconds).
    Returns a dictionary of DataFrames for different intervals.
    """
    intervals = {
        '30m': 4032,   # 3 months of data (4032 candles)
        '4h': 1512,    # 6+ months of data (252 days)
        '5m': 576      # 2 days of data
    }
    data = {}
    for interval, limit in intervals.items():
        # Pass startTime and endTime if provided.
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit, startTime=start_time, endTime=end_time)
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                   'close_time', 'quote_asset_volume', 'number_of_trades',
                   'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=columns)
        # Convert all numeric columns properly
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC')
        df.set_index('timestamp', inplace=True)
        df['quote_asset_volume'] = df['quote_asset_volume'].astype(float)
        df['volume'] = df['volume'].astype(float)
        data[interval] = df[['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']]
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
    volume_ok = ltf.volume.iloc[-1] > ltf.volume.rolling(20).mean().iloc[-1] * 1.2
    
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
    if False:  # Disable trend filter temporarily for testing
        print(f"Skipping {direction} setup - trend mismatch")
        return False
    
    return trade_signal

def plot_trade_setup(df, signal, zones):
    """Guarantee zone visibility on charts"""
    # Get full historical data range
    plot_df = df.iloc[-1000:]
    
    # Create base plot
    fig, axes = mpf.plot(
        plot_df,
        type='candle',
        style='charles',
        title=f"{signal['Symbol']} Trade Setup",
        ylabel='Price',
        figsize=(18, 9),
        returnfig=True,
        tight_layout=True
    )
    ax = axes[0]

    # Get all valid zones
    valid_zones = [z for z in zones if z['valid']]
    print(f"Plotting {len(valid_zones)} valid zones")
    
    # Calculate unified y-axis limits
    if valid_zones:
        min_low = min([z['low'] for z in valid_zones] + [plot_df.low.min()])
        max_high = max([z['high'] for z in valid_zones] + [plot_df.high.max()])
        ax.set_ylim(min_low * 0.98, max_high * 1.02)

    # Plot each zone across full chart width
    for zone in valid_zones:
        color = '#FF0000' if zone['type'] == 'supply' else '#00FF00'
        
        # Full width zone rectangle
        ax.axhspan(
            zone['low'],
            zone['high'],
            facecolor=color,
            alpha=0.2,
            zorder=0
        )
        
        # Zone borders
        ax.axhline(zone['high'], color=color, alpha=0.5, linestyle='--')
        ax.axhline(zone['low'], color=color, alpha=0.5, linestyle='--')

    # Save plot: if you wish to write the plots to binance_data/screener/plots you can do:
    plots_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    filename = os.path.join(plots_dir, f"{signal['Symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(filename)
    plt.close(fig)

def run_screener():
    # Get random 10 USDT pairs
    all_tickers = client.get_all_tickers()
    usdt_pairs = [t['symbol'] for t in all_tickers if t['symbol'].endswith('USDT')]
    SYMBOLS = random.sample(usdt_pairs, 20)

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

def load_json_data(filepath):
    """Load OHLC JSON file into a pandas DataFrame."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Convert timestamp to datetime and set it as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    df = df.copy()
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(period).mean()
    return df['atr']

def get_session_data(df, start=time(0, 0), end=time(8, 0)):
    """Return df subset for each day between specified times."""
    # Ensure index is UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df.between_time('00:00', '08:00')

def get_position_cap(entry_price):
    if entry_price < 0.001:  # Micro-cap tokens
        return POSITION_SIZE_CAP * 5  # 25,000
    elif entry_price < 0.1:   
        return 250000   # Higher capacity for low-price
    else:                     
        return 10000     # Double standard cap

def simulate_trade(data, entry_time, direction, tp_multiplier, sl_multiplier):
    entry_price = round(data.iloc[0]['close'], 8)
    
    # Dynamic position cap
    position_cap = get_position_cap(entry_price)
    min_units = max(MIN_TRADE_VALUE / entry_price, 100)  # Absolute minimum 100 units
    position_size = min(1000000 / entry_price, position_cap)
    
    # Final validation
    position_size = max(position_size, min_units)
    if position_size > position_cap:
        print(f"Position cap exceeded: {position_size:.2f} > {position_cap}")
        return None

    # Calculate targets
    if direction == 'long':
        tp_price = entry_price * (1 + tp_multiplier)
        sl_price = entry_price * (1 - sl_multiplier)
    else:
        tp_price = entry_price * (1 - tp_multiplier)
        sl_price = entry_price * (1 + sl_multiplier)
    
    # Find exit conditions
    for idx, row in data.iterrows():
        if direction == 'long':
            if row['high'] >= tp_price:
                return {
                    'exit_time': idx,
                    'exit_price': tp_price,
                    'profit_pct': tp_multiplier * 100,
                    'position_size': position_size,
                    'duration': (idx - entry_time).total_seconds()/60
                }
            elif row['low'] <= sl_price:
                return {
                    'exit_time': idx,
                    'exit_price': sl_price,
                    'profit_pct': -sl_multiplier * 100,
                    'position_size': position_size,
                    'duration': (idx - entry_time).total_seconds()/60
                }
        else:
            if row['low'] <= tp_price:
                return {
                    'exit_time': idx,
                    'exit_price': tp_price,
                    'profit_pct': tp_multiplier * 100,
                    'position_size': position_size,
                    'duration': (idx - entry_time).total_seconds()/60
                }
            elif row['high'] >= sl_price:
                return {
                    'exit_time': idx,
                    'exit_price': sl_price,
                    'profit_pct': -sl_multiplier * 100,
                    'position_size': position_size,
                    'duration': (idx - entry_time).total_seconds()/60
                }
    
    print(f"No exit found within {len(data)} candles")
    return None

def fetch_data_api(symbol, interval="15m", limit=1000):
    """
    Fetch historical OHLC data from Binance API for the given symbol.
    Using 15m timeframe as per playbook.
    """
    try:
        klines = client.get_klines(
            symbol=symbol, 
            interval=interval, 
            limit=limit
        )
        
        # Create DataFrame with proper timestamp handling
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # Convert string values to float for OHLCV data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Convert timestamp to datetime index
        df.index = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df.index = df.index.tz_localize('UTC')
        
        # Select only the OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        print(f"Successfully fetched {len(df)} candles of {interval} data for {symbol}")
        return df
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        print(f"Full error details: ", e)
        return None

def get_volatility_tier(df):
    """Determine volatility tier based on ATR"""
    atr = calculate_atr(df)
    atr_percentile = atr.iloc[-1] / df['close'].iloc[-1] * 100
    
    # Relaxed volatility tier conditions as per playbook
    if atr_percentile >= 1.1:  # Changed from 1.7
        return 'high'
    elif atr_percentile >= 0.7:  # Changed from 1.3
        return 'medium'
    else:
        return 'low'

def get_tier_parameters(tier):
    """Adjusted parameters to match playbook v3.1 requirements"""
    params = {
        'high': {
            'buffer_pct': 0.0045,  # 0.45% buffer (0.3-0.6% range)
            'sl': 0.01,           # 1.0% stop loss
            'tp': 0.022,          # 2.2% take profit
            'vol_filter': 1.8      # 1.8x volume filter
        },
        'medium': {
            'buffer_pct': 0.0085, # 0.85% buffer (0.7-1.0% range)
            'sl': 0.015,          # 1.5% stop loss
            'tp': 0.032,          # 3.2% take profit
            'vol_filter': 1.4      # 1.4x volume filter
        },
        'low': {
            'buffer_pct': 0.013,  # 1.3% buffer (1.1-1.5% range)
            'sl': 0.022,          # 2.2% stop loss
            'tp': 0.042,          # 4.2% take profit
            'vol_filter': 1.2      # 1.2x volume filter
        }
    }
    return params[tier]

def backtest_asian_fakeout(df):
    """Modified backtest with simplified entry conditions"""
    trades = []
    df['ATR'] = calculate_atr(df)
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['date'] = df.index.date
    
    total_sessions = 0
    sessions_with_range = 0
    sessions_with_volume = 0
    potential_setups = 0
    
    for day, day_df in df.groupby('date'):
        session_df = get_session_data(day_df)
        if len(session_df) < 3:
            continue
            
        total_sessions += 1
        
        # Calculate session range
        session_high = session_df['high'].max()
        session_low = session_df['low'].min()
        session_range = session_high - session_low
        session_range_pct = (session_range / session_df['close'].mean()) * 100
        
        print(f"\nAnalyzing session {day}:")
        print(f"Session range: {session_range_pct:.2f}%")
        
        # Daily range filter - make it more lenient
        daily_range = day_df['high'].max() - day_df['low'].min()
        avg_daily_range = df['high'].rolling(20).max() - df['low'].rolling(20).min()
        if daily_range < (0.3 * avg_daily_range.mean()):  # More lenient filter
            print("Failed daily range filter")
            continue
            
        sessions_with_range += 1
        
        # Get volatility tier and parameters
        try:
            tier = get_volatility_tier(session_df)
        except ZeroDivisionError:
            continue  # Skip sessions with invalid ATR calculations
        params = get_tier_parameters(tier)
        
        # Calculate buffer
        base_buffer_pct = params['buffer_pct']
        atr_factor = min(session_df['ATR'].iloc[-1] / session_df['ATR'].mean(), 1.5)
        buffer_pct = base_buffer_pct * atr_factor
        buffer = session_df['close'].mean() * buffer_pct
        
        print(f"Volatility Tier: {tier.upper()} | Session Range: {session_range_pct:.2f}% | ATR Factor: {atr_factor:.2f}")
        print(f"Buffer size: {buffer_pct*100:.2f}%")
        print(f"Buffer absolute: {buffer:.8f}")
        print(f"Session price: {session_df['close'].mean():.8f}")
        
        sessions_with_volume += 1
        
        # Check each candle in session
        for i in range(2, len(session_df)-5):  # Ensure minimum 5 candles remaining
            current = session_df.iloc[i]
            prev = session_df.iloc[i-1]
            
            # Simplified volume check
            current_vol_ratio = (current['volume'] / current['vol_ma20']) 
            if not (0 < current_vol_ratio < 10):  # Sanity check for volume ratio
                continue
                
            potential_setups += 1
            
            # Increase confirmation window to 5 for high volatility and 4 for lower volatility
            confirmation_window = 5 if tier == 'high' else 4

            # SHORT setup
            if current['high'] > session_high:
                print(f"\nPotential SHORT setup at {session_df.index[i]}")
                print(f"Current high: {current['high']:.8f}")
                print(f"Session high: {session_high:.8f}")
                
                # Look for confirmation in next few candles within the confirmation window
                for j in range(1, min(confirmation_window, len(session_df)-i)):
                    future = session_df.iloc[i+j]
                    
                    # Round prices to 8 decimals for valid move calculation
                    future_close = round(future['close'], 8)
                    current_high = round(current['high'], 8)
                    current_low = round(current['low'], 8)
                    
                    # Compute price move for SHORT setup
                    price_move = (current_high - future_close) / current_high
                    required_move = max(
                        (buffer_pct/100) * (0.015 if tier == 'high' else 0.005),  # Reduce multiplier
                        0.0005  # More lenient minimum
                    )
                    print(f"(DEBUG) Candle {j}: Close={future_close:.8f}, Required Move={required_move:.6f}")
                    if price_move >= required_move:
                        print(f"Setup confirmed! (SHORT) Price move: {price_move*100:.2f}% (Required: {required_move*100:.2f}%)")
                        
                        trade = simulate_trade(
                            session_df.iloc[i+j:],
                            session_df.index[i+j],
                            'short',
                            params['tp'],
                            params['sl']
                        )
                        if trade:
                            trade.update({
                                'entry_time': session_df.index[i+j],
                                'setup_time': session_df.index[i],
                                'direction': 'short',
                                'tier': tier,
                                'buffer_pct': buffer_pct*100,
                                'volume_ratio': current_vol_ratio,
                                'session_range_pct': session_range_pct,
                                'price_move_pct': price_move*100,
                                'entry_price': future['close']
                            })
                            trades.append(trade)
                            # Debug trade parameters
                            print(f"Trade params - TP: {params['tp']}x, SL: {params['sl']}x, Size: {trade['position_size']}")
                            print(f"Trade executed at {future['close']:.8f}")
                            print(f"Price: {future['close']:.8f}, Size: {trade['position_size']:.2f}, Valid: {trade['position_size'] > 0}")

                            # Add price-tiered debug logging
                            print(f"Price Tier: {'Micro' if future['close'] < 0.01 else 'Low' if future['close'] < 0.1 else 'Standard'} | "
                                  f"Cap: {get_position_cap(future['close']):,} | "
                                  f"Size: {trade['position_size']:.2f} units")
                        break
            
            # LONG setup
            if current['low'] < session_low:
                print(f"\nPotential LONG setup at {session_df.index[i]}")
                print(f"Current low: {current['low']:.8f}")
                print(f"Session low: {session_low:.8f}")
                
                for j in range(1, min(confirmation_window, len(session_df)-i)):
                    future = session_df.iloc[i+j]
                    
                    # Round prices to 8 decimals for valid move calculation
                    future_close = round(future['close'], 8)
                    current_low = round(current['low'], 8)
                    
                    # Compute price move for LONG setup
                    price_move = (future_close - current_low) / current_low
                    required_move = max(
                        (buffer_pct/100) * (0.015 if tier == 'high' else 0.005),  # Reduce multiplier
                        0.0005  # More lenient minimum
                    )
                    print(f"\n[DEBUG] Checking {direction.upper()} setup at {session_df.index[i]}")
                    print(f"Buffer: {buffer_pct:.2f}% ({buffer:.8f})")
                    print(f"Required move: {required_move*100:.4f}%")
                    print(f"Trend direction: {trend} | Vol tier: {tier}")
                    print(f"Session range: {session_range_pct:.2f}% | Volume ratio: {current_vol_ratio:.2f}x")

                    # Inside the confirmation window loop
                    print(f"  Candle {j}: Close={future_close:.8f} (Move: {price_move*100:.4f}%)")
                    if price_move >= required_move:
                        print(f"  !! VALID MOVE !! Needed {required_move*100:.4f}% got {price_move*100:.4f}%")
                    else:
                        print(f"  Insufficient move: {price_move*100:.4f}% < {required_move*100:.4f}%")
                        
                        trade = simulate_trade(
                            session_df.iloc[i+j:],
                            session_df.index[i+j],
                            'long',
                            params['tp'],
                            params['sl']
                        )
                        if trade:
                            trade.update({
                                'entry_time': session_df.index[i+j],
                                'setup_time': session_df.index[i],
                                'direction': 'long',
                                'tier': tier,
                                'buffer_pct': buffer_pct*100,
                                'volume_ratio': current_vol_ratio,
                                'session_range_pct': session_range_pct,
                                'price_move_pct': price_move*100,
                                'entry_price': future['close']
                            })
                            trades.append(trade)
                            # Debug trade parameters
                            print(f"Trade params - TP: {params['tp']}x, SL: {params['sl']}x, Size: {trade['position_size']}")
                            print(f"Trade executed at {future['close']:.8f}")
                            print(f"Price: {future['close']:.8f}, Size: {trade['position_size']:.2f}, Valid: {trade['position_size'] > 0}")

                            # Add price-tiered debug logging
                            print(f"Price Tier: {'Micro' if future['close'] < 0.01 else 'Low' if future['close'] < 0.1 else 'Standard'} | "
                                  f"Cap: {get_position_cap(future['close']):,} | "
                                  f"Size: {trade['position_size']:.2f} units")
                        break
            
            session_volume = session_df['volume'].sum()
            session_price = session_df['close'].mean()
            
            # Convert to dollar volume check
            quote_volume = session_volume * session_price
            if quote_volume < MIN_QUOTE_VOLUME:
                print(f"Insufficient liquidity (${quote_volume:.2f} < ${MIN_QUOTE_VOLUME})")
                continue

    print("\nBacktest Statistics:")
    print(f"Total sessions analyzed: {total_sessions}")
    print(f"Sessions with sufficient range: {sessions_with_range}")
    print(f"Sessions with sufficient volume: {sessions_with_volume}")
    print(f"Potential setup opportunities: {potential_setups}")
    print(f"Actual trades taken: {len(trades)}")
    
    # Add debug logging after position sizing
    if trades:
        print(f"\nFinal Position Sizes:")
        [print(f"Trade {i+1}: {t['position_size']:.2f} units (${t['position_size']*t['entry_price']:.2f})") for i,t in enumerate(trades)]
    return trades

def summarize_trades(trades):
    """Enhanced trade analysis with detailed statistics"""
    if not trades:
        print("No trades to summarize")
        return
    
    df = pd.DataFrame(trades)
    df['profit_pct'] = df['profit_pct'].round(2)
    
    print("\n=== TRADE SUMMARY ===")
    print(f"Total Trades: {len(df)}")
    print(f"Time Period: {df['entry_time'].min()} to {df['entry_time'].max()}")
    
    # Win/Loss Analysis
    wins = df[df['outcome'] == 'WIN']
    losses = df[df['outcome'] == 'LOSS']
    print(f"\nWins: {len(wins)} ({len(wins)/len(df)*100:.1f}%)")
    print(f"Losses: {len(losses)} ({len(losses)/len(df)*100:.1f}%)")
    print(f"No Exits: {len(df) - len(wins) - len(losses)}")
    
    # Profit Metrics
    print("\nProfit Analysis:")
    print(f"Total Return: {df['profit_pct'].sum():.2f}%")
    print(f"Average Win: {wins['profit_pct'].mean():.2f}%")
    print(f"Average Loss: {losses['profit_pct'].mean():.2f}%")
    print(f"Risk/Reward Ratio: {abs(wins['profit_pct'].mean()/losses['profit_pct'].mean()):.2f}:1")
    
    # Duration Analysis
    print("\nDuration Analysis:")
    print(f"Average Trade Duration: {df['duration_hours'].mean():.1f} hours")
    print(f"Longest Trade: {df['duration_hours'].max():.1f} hours")
    print(f"Shortest Trade: {df['duration_hours'].min():.1f} hours")
    
    # Show sample trades
    print("\nSample Trades:")
    # Add risk/reward ratio calculation
    df['risk_reward'] = abs((df['TP'] - df['entry_price']) / (df['entry_price'] - df['SL']))
    print(df[['entry_time', 'entry_price', 'SL', 'TP', 'exit_price', 
            'profit_pct', 'risk_reward', 'outcome', 'duration_hours']].head(5))

def backtest_breakout_strategy(df_4h, df_15m=None, symbol="UNKNOWN"):
    trades = []
    df = df_4h.copy()
    
    # Calculate adaptive volume threshold
    df['vol_ma'] = df['volume'].rolling(VOL_MA_PERIOD).mean().bfill()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    
    # Dynamic consolidation range calculation
    df['consol_high'] = df['high'].rolling(MIN_CONSOL_BARS).max().shift(1)
    df['consol_low'] = df['low'].rolling(MIN_CONSOL_BARS).min().shift(1)
    df['consol_range_pct'] = ((df['consol_high'] - df['consol_low']) / df['consol_low'] * 100)
    
    # Trend filter using EMA
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    for i in range(MIN_CONSOL_BARS+1, len(df)):
        try:
            # Current market conditions
            current_high = df['high'].iloc[i]
            current_close = df['close'].iloc[i]
            vol_ratio = df['vol_ratio'].iloc[i]
            consol_high = df['consol_high'].iloc[i]
            ema = df['ema_50'].iloc[i]
            
            # Entry conditions
            price_condition = current_high > consol_high * (1 + BUFFER_PCT)
            volume_condition = vol_ratio >= VOL_SPIKE_MULTIPLIER
            trend_condition = current_close > ema  # Bullish filter
            
            if price_condition and volume_condition and trend_condition:
                # Calculate risk parameters
                atr = df['high'].iloc[i] - df['low'].iloc[i]
                entry_price = consol_high * (1 + BUFFER_PCT)
                stop_loss = entry_price - (atr * 1.5)
                take_profit = entry_price + (entry_price - stop_loss) * 2
                
                # Track trade outcome
                trade = {
                    'entry_time': df.index[i],
                    'entry_price': entry_price,
                    'SL': stop_loss,
                    'TP': take_profit,
                    'exit_time': None,
                    'exit_price': None,
                    'duration_hours': 0,
                    'outcome': 'OPEN',
                    'profit_pct': 0,
                    'consol_range_pct': ((consol_high - df['consol_low'].iloc[i])/df['consol_low'].iloc[i] * 100),
                    'vol_ratio': vol_ratio,
                    'atr': atr
                }
                
                # Simulate trade outcome
                for j in range(i+1, len(df)):
                    current_low = df['low'].iloc[j]
                    current_high = df['high'].iloc[j]
                    
                    # Check for stop loss hit
                    if current_low <= stop_loss:
                        trade.update({
                            'exit_time': df.index[j],
                            'exit_price': stop_loss,
                            'outcome': 'LOSS',
                            'duration_hours': (df.index[j] - df.index[i]).total_seconds()/3600
                        })
                        break
                        
                    # Check for take profit hit
                    if current_high >= take_profit:
                        trade.update({
                            'exit_time': df.index[j],
                            'exit_price': take_profit,
                            'outcome': 'WIN',
                            'duration_hours': (df.index[j] - df.index[i]).total_seconds()/3600
                        })
                        break
                else:
                    # No exit triggered - close at final price
                    trade.update({
                        'exit_time': df.index[-1],
                        'exit_price': df['close'].iloc[-1],
                        'outcome': 'NO EXIT',
                        'duration_hours': (df.index[-1] - df.index[i]).total_seconds()/3600
                    })
                
                # Calculate final PNL
                trade['profit_pct'] = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100
                trades.append(trade)

        except Exception as e:
            print(f"Error processing candle {i}: {str(e)}")
    
    return trades

def main():
    """Enhanced backtester with complete reporting"""
    SYMBOLS = ['BTCUSDT']
    
    for symbol in SYMBOLS:
        print(f"\n=== BACKTESTING {symbol} ===")
        try:
            # Get fresh market data directly from Binance
            data = fetch_data(symbol)
            df_4h = data['4h'].copy()
            
            # Run backtest
            trades = backtest_breakout_strategy(df_4h, symbol=symbol)
            
            # Generate report
            if trades:
                print(f"\nFirst Trade:")
                print(f"Entry: {trades[0]['entry_price']:.2f} at {trades[0]['entry_time']}")
                print(f"Exit: {trades[0]['exit_price']:.2f} ({trades[0]['profit_pct']:.2f}%) after {trades[0]['duration_hours']:.1f}h")
                
                print(f"\nLast Trade:")
                print(f"Entry: {trades[-1]['entry_price']:.2f} at {trades[-1]['entry_time']}")
                print(f"Exit: {trades[-1]['exit_price']:.2f} ({trades[-1]['profit_pct']:.2f}%) after {trades[-1]['duration_hours']:.1f}h")
                
                summarize_trades(trades)
            else:
                print("No trades generated")
                
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
