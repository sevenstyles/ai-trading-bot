import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from dotenv import load_dotenv
import os
import json
import argparse
import sys

# Load environment variables
load_dotenv()

class SupplyDemandScanner:
    def __init__(self):
        self.base_url = os.getenv('BINANCE_BASE_URL', 'https://api.binance.com')
        self.min_volume_usdt = 1000000  # Minimum 24h volume in USDT
        self.min_price_usdt = 0.1  # Minimum price in USDT
        self.major_pairs = ['BTCUSDT', 'ETHUSDT']  # Major pairs have stricter rules
        self.major_pair_volume_multiplier = 0.8  # 80% of normal volume requirement
        
    def _send_request(self, endpoint, params=None):
        """Send public API request"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    def check_trend_alignment(self, df):
        """Check for higher highs/lower lows trend alignment"""
        try:
            # Get recent swing points (last 30 candles)
            recent_df = df.tail(30)
            
            swing_highs = []
            swing_lows = []
            
            for i in range(1, len(recent_df)-1):
                if recent_df['high'].iloc[i] > recent_df['high'].iloc[i-1] and \
                   recent_df['high'].iloc[i] > recent_df['high'].iloc[i+1]:
                    swing_highs.append(recent_df['high'].iloc[i])
                if recent_df['low'].iloc[i] < recent_df['low'].iloc[i-1] and \
                   recent_df['low'].iloc[i] < recent_df['low'].iloc[i+1]:
                    swing_lows.append(recent_df['low'].iloc[i])
            
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                higher_highs = swing_highs[-1] > swing_highs[0]
                higher_lows = swing_lows[-1] > swing_lows[0]
                lower_highs = swing_highs[-1] < swing_highs[0]
                lower_lows = swing_lows[-1] < swing_lows[0]
                
                if higher_highs and higher_lows:
                    return 'up'
                elif lower_highs and lower_lows:
                    return 'down'
            
            return None
        except Exception as e:
            print(f"Error in check_trend_alignment: {str(e)}")
            return None

    def is_strong_move(self, df):
        """Check if there's been a strong recent move"""
        try:
            # Should also consider single strong candle that creates the zone
            recent_candles = df.tail(3)
            
            # Check for 3 consecutive candles OR one strong candle (>2x avg size)
            avg_body = abs(recent_candles['close'] - recent_candles['open']).mean()
            strong_single = any(abs(recent_candles['close'] - recent_candles['open']) > 2*avg_body)
            
            consecutive = all(recent_candles['close'] > recent_candles['open']) or \
                         all(recent_candles['close'] < recent_candles['open'])
            
            return consecutive or strong_single
        except Exception as e:
            print(f"Error in is_strong_move: {str(e)}")
            return False

    def is_retracing(self, df):
        """Check if price is retracing after a strong move"""
        try:
            # Modified to allow shallower retracements
            recent_data = df.tail(10)
            current_price = df['close'].iloc[-1]
            
            # Get nearest swing high/low
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            
            # Allow retracements up to 15% from swing point
            valid_retracement = (
                current_price <= swing_high * 0.85 or  # 15% retracement
                current_price >= swing_low * 1.15      # 15% retracement
            )
            
            return valid_retracement
        except Exception as e:
            print(f"Error in is_retracing: {str(e)}")
            return False

    def get_valid_pairs(self):
        """Get all USDT pairs meeting criteria"""
        print("\nGetting valid pairs...")
        exchange_info = self._send_request("/api/v3/exchangeInfo")
        valid_pairs = []
        
        if not exchange_info:
            print("Failed to get exchange info")
            return valid_pairs
            
        total_symbols = len([s for s in exchange_info['symbols'] if s['quoteAsset'] == 'USDT'])
        print(f"Found {total_symbols} total USDT pairs")
        
        for symbol_info in exchange_info['symbols']:
            symbol = symbol_info['symbol']
            
            if symbol_info['quoteAsset'] == 'USDT' and symbol_info['status'] == 'TRADING':
                ticker = self._send_request("/api/v3/ticker/24hr", params={'symbol': symbol})
                if ticker:
                    volume_usdt = float(ticker['quoteVolume'])
                    price = float(ticker['lastPrice'])
                    
                    print(f"\nChecking {symbol}:")
                    print(f"Volume: {volume_usdt:.2f} USDT")
                    print(f"Price: {price:.8f} USDT")
                    
                    if volume_usdt >= self.min_volume_usdt and price >= self.min_price_usdt:
                        print("Passed volume and price criteria")
                        valid_pairs.append(symbol)
                    else:
                        print("Failed volume or price criteria")
        
        print(f"\nFound {len(valid_pairs)} pairs meeting volume/price criteria")
        return valid_pairs

    def process_klines_data(self, data):
        """Process raw klines data into DataFrame"""
        if not data:
            return None
            
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                                       'ignore'])
        
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def get_klines(self, symbol, interval, limit=200):
        """Get kline/candlestick data"""
        endpoint = "/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        data = self._send_request(endpoint, params)
        return self.process_klines_data(data)

    def calculate_atr(self, df, period=14):
        """Calculate Average True Range with pandas compatibility"""
        try:
            high_low = df['high'].values - df['low'].values
            high_close = np.abs(df['high'].values - df['close'].shift().values)
            low_close = np.abs(df['low'].values - df['close'].shift().values)
            
            true_range = pd.Series(np.maximum.reduce([high_low, high_close, low_close]))
            return true_range.rolling(period).mean()
        except Exception as e:
            print(f"ATR calculation error: {str(e)}")
            return pd.Series(np.zeros(len(df)))

    def detect_zones(self, df):
        """Final sensitivity adjustments"""
        try:
            # Store original timestamps as a separate series
            original_index = df.index
            df = df.reset_index(drop=True)
            df['original_timestamp'] = original_index
            
            # Existing calculations and filters...
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            
            # Reset index after each filter operation
            print(f"Initial candles: {len(df)}")
            df = df[df['volume'] > df['volume'].rolling(20).mean() * 0.5].reset_index(drop=True)
            print(f"After volume filter: {len(df)}")
            atr = self.calculate_atr(df, 14)
            df = df[df['body_size'] > atr * 0.3].reset_index(drop=True)
            print(f"After ATR filter: {len(df)}")

            zones = []
            # Add bounds check for loop range
            for i in range(2, min(len(df)-3, 1000)):  # Dynamic range calculation
                current = df.iloc[i]
                prev1 = df.iloc[i-1]
                next1 = df.iloc[i+1]
                
                # Loosen rejection candle criteria
                is_bearish_rejection = (current['close'] < current['open'] and 
                                       current['upper_wick'] > current['body_size'] * 1.2)  # Changed from 1.5
                is_bullish_rejection = (current['close'] > current['open'] and 
                                       current['lower_wick'] > current['body_size'] * 1.2)  # Changed from 1.5
                
                if not (is_bearish_rejection or is_bullish_rejection):
                    continue
                    
                # Loosen volume confirmation
                if symbol in self.major_pairs:
                    volume_threshold = prev1['volume'] * self.major_pair_volume_multiplier
                else:
                    volume_threshold = prev1['volume'] * 1.05
                
                if current['volume'] < volume_threshold:
                    print(f"Rejected zone at {i} - insufficient volume")
                    continue
                    
                # Add price buffer for subsequent validation
                if is_bearish_rejection:
                    if next1['high'] > current['high'] * 1.002 or next1['close'] > current['close'] * 1.002:
                        print(f"Rejected supply zone at {i} - failed bearish confirmation")
                        continue
                    zone = {
                        'high': float(current['high']),
                        'low': float(current['low']),
                        'type': 'supply',
                        'timestamp': df['original_timestamp'].iloc[i],
                        'volume': float(current['volume']),
                        'original_index': i  # Track original position in filtered DF
                    }
                else:
                    if next1['low'] < current['low'] * 0.998 or next1['close'] < current['close'] * 0.998:
                        print(f"Rejected demand zone at {i} - failed bullish confirmation")
                        continue
                    zone = {
                        'high': float(current['high']),
                        'low': float(current['low']),
                        'type': 'demand',
                        'timestamp': df['original_timestamp'].iloc[i],
                        'volume': float(current['volume']),
                        'original_index': i  # Track original position in filtered DF
                    }
                
                zones.append(zone)
            
            # Loosen zone validation criteria
            valid_zones = []
            for zone in zones:
                post_zone = df.iloc[zone['original_index']+1:]
                
                # Allow up to 5% of candles to revisit zone
                if zone['type'] == 'supply':
                    revisit_count = (post_zone['high'] > zone['low']).sum()
                    if revisit_count > len(post_zone)*0.05:
                        continue
                else:
                    revisit_count = (post_zone['low'] < zone['high']).sum()
                    if revisit_count > len(post_zone)*0.05:
                        continue
                
                valid_zones.append(zone)
            
            print(f"Found {len(valid_zones)} valid zones after sensitivity adjustments")
            return valid_zones
            
        except Exception as e:
            print(f"Zone detection error: {str(e)}")
            return []

    def calculate_zone_strength(self, df, zone):
        """Calculate zone strength score based on strategy rules"""
        strength = 0
        
        # Base points for zone type
        strength += 2 if zone['type'] == 'accuracy' else 1
        
        # Add points for proximity to swing point
        swing_high = df['high'].max()
        swing_low = df['low'].min()
        
        if zone['type'] == 'supply':
            strength += 1 if zone['high'] >= swing_high * 0.95 else 0
        else:
            strength += 1 if zone['low'] <= swing_low * 1.05 else 0
        
        # New confluence checks
        if zone['type'] == 'supply':
            # Check for overhead resistance
            resistance = df['high'].rolling(50).max()
            strength += 2 if zone['high'] >= resistance.iloc[-1] * 0.98 else 0
        else:
            # Check for underlying support
            support = df['low'].rolling(50).min()
            strength += 2 if zone['low'] <= support.iloc[-1] * 1.02 else 0

        # Add volume profile confluence
        vp = df['volume'].rolling(20).mean()
        strength += 1 if zone['volume'] > vp.iloc[-1] else 0
        
        return strength

    def validate_zone(self, df, zone):
        """Updated validation with proper tap checking"""
        try:
            # Changed to allow wick taps but reject closes inside
            post_zone_data = df.iloc[zone['original_index']+1:]
            
            # Check for candle CLOSURES inside zone
            closes_inside = post_zone_data[
                (post_zone_data['close'] > zone['low']) & 
                (post_zone_data['close'] < zone['high'])
            ]
            if not closes_inside.empty:
                print(f"Rejecting zone - {len(closes_inside)} closes inside")
                return False
            
            # Trend alignment
            df_1h = self.get_klines(zone['symbol'], '1h', 50)  # 1h trend must agree
            trend = self.check_trend_alignment(df_1h)
            
            # Risk-Reward
            if self.calculate_risk_reward(zone['entry_price'], zone['stop_price'], zone['tp1_price']) >= 3:  # Minimum 1:3 required
                return True
            
            return False
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False

    def calculate_risk_reward(self, entry_price, stop_price, target_price):
        """Calculate risk-reward ratio"""
        risk = abs(entry_price - stop_price)
        reward = abs(target_price - entry_price)
        return reward / risk if risk > 0 else 0

    def find_targets(self, df, entry_price, stop_price, direction):
        """Find TP levels using swing highs/lows and fibonacci extensions"""
        risk = abs(entry_price - stop_price)
        
        # Target calculation
        tp1 = entry_price + (risk * 3)  # Minimum 1:3
        tp2 = entry_price + (risk * 10)  # Max 1:10 or 1.618 Fib extension

        # Partial profit taking
        if direction == 'LONG':
            # Take 50% at TP1, rest at TP2
            pass  # Actual position sizing logic would go here
        else:
            # Similar logic for shorts
            min_tp1 = entry_price - (risk * 3)
            future_lows = df[df['low'] < min_tp1]['low']
            tp1 = future_lows.iloc[0] if not future_lows.empty else min_tp1
            
            tp2 = max(entry_price - (risk * 10),
                     entry_price - (risk * 1.618))
        
        return tp1, tp2

    def validate_entry_sequence(self, df, zone, zone_type, symbol):
        """Updated entry validation for long setups"""
        try:
            post_zone_data = df.iloc[zone['original_index']+1:]
            
            # After zone touch (wick in zone)
            for i in range(len(post_zone_data)-1):
                current = post_zone_data.iloc[i]     # Candle that touched zone
                next_candle = post_zone_data.iloc[i+1]  # Confirmation candle
                
                # For LONG (DEMAND) setups
                if zone_type == 'DEMAND':
                    # 1. Price must touch zone with wick (2% buffer)
                    tapped = current['low'] <= zone['high'] * 1.02
                    
                    # 2. Next candle must close above midpoint of its range
                    confirmed = next_candle['close'] > (next_candle['open'] + next_candle['low'])/2
                    
                # For SHORT (SUPPLY) setups    
                else:
                    # 1. Price must touch zone with wick (2% buffer)
                    tapped = current['high'] >= zone['low'] * 0.98
                    
                    # 2. Next candle must close below midpoint of its range
                    confirmed = next_candle['close'] < (next_candle['open'] + next_candle['high'])/2
                
                # If both conditions met → Trade executes
                if tapped and confirmed:
                    return True, {
                        'entry_price': next_candle['close'],  # Entry at confirmation candle close
                        'stop_price': zone['low'] if zone_type == 'SUPPLY' else zone['high']
                    }
            
            return False, None
        except Exception as e:
            print(f"Entry validation error: {str(e)}")
            return False, None

    def calculate_confidence(self, setup):
        """Simple confidence scoring for visualization"""
        score = 0
        # Add your confidence metrics here
        score += 3 if setup['rr_ratio'] >= 3 else 0
        score += 2 if setup['zone_accuracy'] == 'accuracy' else 1
        return min(10, score)

    def find_setup_sequence(self, df, symbol):
        """Find valid supply and demand setups with updated rules"""
        print(f"\n=== 5m Zone Detection for {symbol} ===")
        try:
            df_1h = self.get_klines(symbol, '1h', 100)  # HTF 1h
            trend = self.check_trend_alignment(df_1h)
            
            zones = self.detect_zones(df)
            print(f"Found {len(zones)} zones")
            
            valid_setups = []
            
            # Process each zone individually
            for zone in zones:
                try:
                    # Add index bounds check
                    if zone['original_index'] >= len(df) - 2:
                        continue
                        
                    # Validate entry sequence for this zone
                    valid_entry, entry_setup = self.validate_entry_sequence(
                        df, 
                        zone, 
                        'SUPPLY' if zone['type'] == 'supply' else 'DEMAND', 
                        symbol
                    )
                    
                    if valid_entry and entry_setup:
                        # Calculate targets
                        risk = abs(entry_setup['entry_price'] - entry_setup['stop_price'])
                        tp1 = entry_setup['entry_price'] + (risk * 3) if zone['type'] == 'demand' else entry_setup['entry_price'] - (risk * 3)
                        tp2 = entry_setup['entry_price'] + (risk * 10) if zone['type'] == 'demand' else entry_setup['entry_price'] - (risk * 10)
                        
                        setup = {
                            'symbol': symbol,
                            'direction': 'LONG' if zone['type'] == 'demand' else 'SHORT',
                            'zone_type': zone['type'],
                            'entry_price': float(entry_setup['entry_price']),
                            'stop_price': float(entry_setup['stop_price']),
                            'tp1': float(tp1),
                            'tp2': float(tp2),
                            'rr_ratio': abs(tp1 - entry_setup['entry_price']) / risk,
                            'entry_triggered': False,  # Boolean flag
                            'trigger_time': None,      # Datetime of confirmation candle
                            'actual_entry': None      # Price of confirmation candle close
                        }
                        
                        # Add this check before appending to potential_setups:
                        df_check = self.get_klines(symbol, '5m', 2)  # Get last 2 candles
                        if setup['direction'] == 'LONG':
                            triggered = (df_check.iloc[-2]['low'] <= setup['entry_price']) and \
                                        (df_check.iloc[-1]['close'] > setup['entry_price'])
                        else:
                            triggered = (df_check.iloc[-2]['high'] >= setup['entry_price']) and \
                                        (df_check.iloc[-1]['close'] < setup['entry_price'])

                        if triggered:
                            setup.update({
                                'entry_triggered': True,
                                'trigger_time': df_check.index[-1].strftime('%Y-%m-%d %H:%M'),
                                'actual_entry': df_check.iloc[-1]['close']
                            })
                        
                        valid_setups.append(setup)
                        print(f"Valid {setup['direction']} setup found for {symbol}")
                        
                except Exception as e:
                    print(f"Error processing zone: {str(e)}")
                    continue
            
            return valid_setups
            
        except Exception as e:
            print(f"Error in find_setup_sequence: {str(e)}")
            return []

    def scan_for_setups(self):
        """Scan for setups matching the strategy"""
        print("\nStarting scan for setups...")
        valid_pairs = self.get_valid_pairs()
        potential_setups = []
        
        for symbol in valid_pairs:
            try:
                # Get data
                df_1h = self.get_klines(symbol, '1h', 200)
                df_5m = self.get_klines(symbol, '5m', 720)
                
                if df_1h is None or df_5m is None:
                    continue
                
                # Find and validate setups
                setups = self.find_setup_sequence(df_5m, symbol)
                for setup in setups:
                    if self.validate_setup_with_lower_timeframe(df_1h, setup):
                        potential_setups.append(setup)
            
            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        print(f"\nAnalysis complete. Found {len(potential_setups)} total setups")
        return pd.DataFrame(potential_setups) if potential_setups else pd.DataFrame()

    def validate_setup_with_lower_timeframe(self, df_1h, setup):
        """Validate setup using 1h timeframe instead of 4h"""
        try:
            # Using 24 candles (24 hours) for daily trend validation
            recent_candles = df_1h.tail(24)
            if setup['direction'] == 'LONG':
                return recent_candles['close'].iloc[-1] > recent_candles['close'].iloc[0]
            else:
                return recent_candles['close'].iloc[-1] < recent_candles['close'].iloc[0]
        except Exception as e:
            print(f"Error in validate_setup_with_lower_timeframe: {str(e)}")
            return False

    def is_zone_tapped(self, df, zone):
        """Check if price has entered zone since creation - FIXED"""
        post_zone_data = df.iloc[zone['original_index']+1:]
        
        # Changed to check wicks rather than closes for taps
        if zone['type'] == 'supply':
            # Considered tapped if price WICK enters the zone
            taps = (post_zone_data['high'] > zone['low'])
        else:
            taps = (post_zone_data['low'] < zone['high'])
        
        return taps.any()

    def test_visualization(self, symbol="BTCUSDT", interval="4h"):
        """Generate test visualization with sample data"""
        print(f"\nGenerating test visualization for {symbol}...")
        
        # Get historical data
        df = self.get_klines(symbol, interval, 100)
        if df is None:
            print("Failed to get test data")
            return

        # Create manual test setup
        test_setup = {
            'symbol': symbol,
            'direction': 'LONG' if np.random.rand() > 0.5 else 'SHORT',
            'zone_type': 'DEMAND',
            'zone_accuracy': 'accuracy',
            'entry_price': df['close'].iloc[-1],
            'stop_price': df['low'].iloc[-5],
            'tp1_price': df['close'].iloc[-1] * 1.03,
            'tp2_price': df['close'].iloc[-1] * 1.10,
            'rr_ratio': 3.5
        }
        
        # Plot with actual price data
        self.plot_setup(df, test_setup, "test_visualization.png")
        print("Test visualization saved to test_visualization.png")

    def plot_zones(self, df, zones, symbol):
        """Plot zones on price chart"""
        plt.figure(figsize=(12,6))
        plt.plot(df['close'], label='Price')
        
        for zone in zones:
            color = 'red' if zone['type'] == 'SUPPLY' else 'green'
            plt.axhspan(zone['low'], zone['high'], alpha=0.2, color=color)
        
        plt.title(f"{symbol} Zones")
        plt.legend()
        plt.show()

    def merge_zones(self, zones):
        """Merge overlapping zones"""
        merged = []
        for zone in sorted(zones, key=lambda x: x['high'], reverse=True):
            if not merged:
                merged.append(zone)
            else:
                last = merged[-1]
                if zone['high'] > last['low']:  # Overlapping
                    merged[-1] = {
                        'high': max(zone['high'], last['high']),
                        'low': min(zone['low'], last['low']),
                        'type': 'merged'
                    }
                else:
                    merged.append(zone)
        return merged

    class Backtester:
        def __init__(self, scanner):
            self.scanner = scanner
            self.results = []
            
        def backtest_pair(self, symbol, interval, days_back=3):
            """Backtest a single pair"""
            print(f"\nBacktesting {symbol} for last {days_back} days...")
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Get historical data
            df = self.scanner.get_historical_klines(symbol, interval, start_time, end_time)
            if df is None or len(df) < 100:
                return
                
            # Find all setups in historical data
            all_setups = []
            window_step = 10  # Process every 10 candles instead of 1
            for i in range(100, len(df), window_step):
                window = df.iloc[:i]
                setups = self.scanner.find_setup_sequence(window, symbol)
                all_setups.extend(setups)
                
            # Evaluate setups
            for setup in all_setups:
                if setup['entry_triggered']:
                    self.evaluate_trade(setup, df)
            
        def evaluate_trade(self, setup, full_df):
            """Evaluate trade outcome based on historical data"""
            try:
                # Add timestamp validation
                if pd.Timestamp(setup['trigger_time']) > full_df.index.max():
                    print(f"Invalid trigger time {setup['trigger_time']} - beyond data range")
                    return
                    
                entry_index = full_df.index.get_loc(setup['trigger_time'])
                trade_data = full_df.iloc[entry_index:]
                
                if setup['direction'] == 'LONG':
                    stop_hit = (trade_data['low'] < setup['stop_price']).any()
                    tp1_hit = (trade_data['high'] >= setup['tp1']).any()
                    tp2_hit = (trade_data['high'] >= setup['tp2']).any() if not stop_hit else False
                else:
                    stop_hit = (trade_data['high'] > setup['stop_price']).any()
                    tp1_hit = (trade_data['low'] <= setup['tp1']).any()
                    tp2_hit = (trade_data['low'] <= setup['tp2']).any() if not stop_hit else False
                
                # Calculate result
                result = {
                    'symbol': setup['symbol'],
                    'direction': setup['direction'],
                    'entry_time': setup['trigger_time'],
                    'entry_price': setup['entry_price'],
                    'stop_price': setup['stop_price'],
                    'tp1': setup['tp1'],
                    'tp2': setup['tp2'],
                    'outcome': 'STOP' if stop_hit else 'TP1' if tp1_hit else 'TP2' if tp2_hit else 'OPEN',
                    'rr_achieved': self.calculate_actual_rr(trade_data, setup)
                }
                self.results.append(result)
                
            except KeyError as e:
                print(f"Skipping invalid setup: {str(e)}")
                return
            except Exception as e:
                print(f"Error evaluating trade: {str(e)}")
                return
            
        def calculate_actual_rr(self, trade_data, setup):
            """Calculate actual risk/reward achieved"""
            entry = setup['entry_price']
            risk = abs(entry - setup['stop_price'])
            
            if setup['direction'] == 'LONG':
                max_price = trade_data['high'].max()
                reward = max_price - entry
            else:
                min_price = trade_data['low'].min()
                reward = entry - min_price
                
            return reward / risk if risk > 0 else 0
            
        def generate_report(self):
            """Generate backtest performance report"""
            if not self.results:
                return None
                
            df = pd.DataFrame(self.results)
            df['win'] = df['outcome'].isin(['TP1', 'TP2'])
            
            report = {
                'total_trades': len(df),
                'win_rate': df['win'].mean(),
                'avg_rr': df['rr_achieved'].mean(),
                'profit_factor': self.calculate_profit_factor(df),
                'max_drawdown': self.calculate_drawdown(df),
                'best_trade': df['rr_achieved'].max(),
                'worst_trade': df['rr_achieved'].min()
            }
            return report, df
            
        def calculate_profit_factor(self, df):
            winners = df[df['win']]
            losers = df[~df['win']]
            return winners['rr_achieved'].sum() / abs(losers['rr_achieved'].sum()) if not losers.empty else np.inf
            
        def calculate_drawdown(self, df):
            cumulative = df['rr_achieved'].cumsum()
            peak = cumulative.expanding(min_periods=1).max()
            drawdown = (peak - cumulative).max()
            return drawdown

    def get_historical_klines(self, symbol, interval, start_time, end_time):
        """Get historical klines between dates"""
        endpoint = "/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000),
            'limit': 1000
        }
        
        all_data = []
        while True:
            data = self._send_request(endpoint, params)
            if not data:
                break
            all_data.extend(data)
            
            # Update params for next request
            params['startTime'] = data[-1][0] + 1
            if params['startTime'] > params['endTime']:
                break
            
        return self.process_klines_data(all_data)

def main():
    try:
        scanner = SupplyDemandScanner()
        
        if '--backtest' in sys.argv:
            backtester = scanner.Backtester(scanner)
            
            # Check for pair file in CURRENT DIRECTORY
            pair_file_path = 'trade_pair.txt'  # Changed from 'binance_data/trade_pair.txt'
            if not os.path.exists(pair_file_path):
                print(f"Error: {pair_file_path} not found in current directory")
                return
                
            with open(pair_file_path) as f:
                pairs = [line.strip() for line in f.readlines() if line.strip()]
                
            if not pairs:
                print("No pairs found in trade_pair.txt")
                return
                
            print(f"Backtesting {len(pairs)} pairs from file...")
            
            for pair in pairs:
                backtester.backtest_pair(pair, '5m', days_back=3)
            
            report_result = backtester.generate_report()
            if report_result is None:
                print("\nNo trades executed during backtest period")
                return
                
            report, detailed_df = report_result
            print("\nBacktest Report:")
            print(json.dumps(report, indent=2))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'screener-results/backtest_{pairs[0]}_{timestamp}.csv' if len(pairs) == 1 else \
                      f'screener-results/backtest_{len(pairs)}_pairs_{timestamp}.csv'
            detailed_df.to_csv(filename, index=False)
            print(f"\nSaved backtest results to {filename}")
            return
        else:
            # Existing main logic
            setups = scanner.scan_for_setups()
            # ... rest of existing main code ...

        if setups is None or (isinstance(setups, pd.DataFrame) and setups.empty):
            print("No setups found matching criteria")
            return
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs('screener-results', exist_ok=True)
            
            # Generate timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed results to CSV
            csv_filename = f'screener-results/setups_strategy3_{timestamp}.csv'
            setups.to_csv(csv_filename, index=False)
            
            # Create a simplified list of symbols
            symbols_list = setups['symbol'].unique().tolist()
            symbols_filename = 'trade_pair_strategy3.txt'
            with open(symbols_filename, 'w') as f:
                f.write('\n'.join(symbols_list))
            
            # Print summary to console
            print("\nTop potential setups found:")
            print(setups.to_string())
            print(f"\nDetailed results saved to: {csv_filename}")
            print(f"Symbols list saved to: {symbols_filename}")
            print(f"\nFound {len(setups)} unique setups across {len(symbols_list)} symbols")
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
        
    except Exception as e:
        print(f"Error running scanner: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())

def test():
    scanner = SupplyDemandScanner()
    scanner.test_visualization()  # Test BTC
    # scanner.test_visualization("ETHUSDT")  # Uncomment to test ETH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run visualization test')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    args = parser.parse_args()
    
    if args.test:
        test()
    elif args.backtest:
        main()
    else:
        main() 