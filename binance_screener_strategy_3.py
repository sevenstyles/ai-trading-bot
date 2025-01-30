import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from dotenv import load_dotenv
import os
import json
import mplfinance as mpf
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

class SupplyDemandScanner:
    def __init__(self):
        self.base_url = os.getenv('BINANCE_BASE_URL', 'https://api.binance.com')
        self.min_volume_usdt = 1000000  # Minimum 24h volume in USDT
        self.min_price_usdt = 0.1  # Minimum price in USDT
        self.major_pairs = ['BTCUSDT', 'ETHUSDT']  # Major pairs have stricter rules
        
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
            # Calculate EMAs for trend context
            df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            
            # Get recent swing points
            recent_df = df.tail(30)
            
            # Calculate swings
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
                # Check EMA alignment
                price = df['close'].iloc[-1]
                above_emas = price > df['ema20'].iloc[-1] and price > df['ema50'].iloc[-1]
                below_emas = price < df['ema20'].iloc[-1] and price < df['ema50'].iloc[-1]
                
                # Check swing patterns
                higher_highs = swing_highs[-1] > swing_highs[0]
                higher_lows = swing_lows[-1] > swing_lows[0]
                lower_highs = swing_highs[-1] < swing_highs[0]
                lower_lows = swing_lows[-1] < swing_lows[0]
                
                if (higher_highs and higher_lows) or above_emas:
                    return 'up'
                elif (lower_highs and lower_lows) or below_emas:
                    return 'down'
            
            return None
            
        except Exception as e:
            print(f"Error in check_trend_alignment: {str(e)}")
            return None

    def is_strong_move(self, df):
        """Check if there's been a strong recent move"""
        try:
            # Calculate average candle size and percentage moves
            df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
            df['candle_size'] = abs(df['close'] - df['open'])
            df['body_size_pct'] = df['candle_size'] / df['open'] * 100  # Size as percentage
            
            # Get recent data
            recent_candles = df.tail(5)
            avg_size = df['candle_size'].rolling(20).mean()
            recent_avgs = avg_size.tail(5)
            
            # Consider a move strong if:
            # 1. Single candle > 1.2x average
            strong_single = (recent_candles['candle_size'] > (recent_avgs * 1.2)).any()
            
            # 2. Three consecutive candles showing momentum
            consecutive_momentum = False
            for i in range(len(recent_candles)-2):
                three_candles = recent_candles.iloc[i:i+3]
                if all(three_candles['close'] > three_candles['open']) or \
                   all(three_candles['close'] < three_candles['open']):
                    consecutive_momentum = True
                    break
            
            # 3. Check for strong volume
            df['vol_sma'] = df['volume'].rolling(20).mean()
            recent_vol = df['volume'].tail(5)
            vol_sma = df['vol_sma'].tail(5)
            strong_volume = (recent_vol > vol_sma * 1.2).any()
            
            result = strong_single or consecutive_momentum or strong_volume
            
            print(f"Strong move check:")
            print(f"Recent candle sizes: {recent_candles['candle_size'].values}")
            print(f"Average sizes: {recent_avgs.values}")
            print(f"Strong single candle: {strong_single}")
            print(f"Consecutive momentum: {consecutive_momentum}")
            print(f"Strong volume: {strong_volume}")
            
            return result
            
        except Exception as e:
            print(f"Error in is_strong_move: {str(e)}")
            print("Full error:")
            import traceback
            print(traceback.format_exc())
            return False

    def is_retracing(self, df):
        """Check if price is retracing after a strong move"""
        try:
            df = df.copy()
            recent_data = df.tail(10)
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()
            current_price = df['close'].iloc[-1]
            
            # Calculate retracement levels
            up_retracement = (recent_high - current_price) / (recent_high - recent_low)
            down_retracement = (current_price - recent_low) / (recent_high - recent_low)
            
            # Calculate momentum
            price_change = abs(df['close'].iloc[-1] - df['close'].iloc[-2])
            price_change_pct = price_change / df['close'].iloc[-2] * 100
            
            # Tighten retracement criteria
            valid_retracement = (
                (0.3 <= up_retracement <= 0.7) or    # Changed from 10-90% to 30-70%
                (0.3 <= down_retracement <= 0.7) or  
                (price_change_pct >= 1.0)            # Increased from 0.5% 
            )
            
            print(f"Retracement analysis:")
            print(f"Recent high: {recent_high:.8f}")
            print(f"Recent low: {recent_low:.8f}")
            print(f"Current price: {current_price:.8f}")
            print(f"Up retracement: {up_retracement:.2%}")
            print(f"Down retracement: {down_retracement:.2%}")
            print(f"Price change %: {price_change_pct:.2f}%")
            print(f"Valid retracement: {valid_retracement}")
            
            return valid_retracement
            
        except Exception as e:
            print(f"Error in is_retracing: {str(e)}")
            print("Full error:")
            import traceback
            print(traceback.format_exc())
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

    def get_klines(self, symbol, interval, limit=200):
        """Get kline/candlestick data"""
        endpoint = "/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        data = self._send_request(endpoint, params)
        if not data:
            return None
            
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                                       'ignore'])
        
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        print(f"First candle time: {df.index[0]} | Last candle time: {df.index[-1]}")
        return df

    def detect_zones(self, df):
        """Detect supply and demand zones with accuracy rules"""
        original_index = df.index
        df = df.reset_index()
        
        # Calculate candle characteristics
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        supply_zones = []
        demand_zones = []
        
        for i in range(1, len(df)-1):
            # Modified strong move detection
            lookback = max(5, int(len(df)*0.1))  # Dynamic lookback
            avg_body_size = df['body_size'].iloc[i-lookback:i].mean()
            is_strong_push = df['body_size'].iloc[i] > avg_body_size * 0.9
            
            # Add volume confirmation
            volume_above_avg = df['volume'].iloc[i] > df['volume'].iloc[i-lookback:i].mean()
            
            if is_strong_push and volume_above_avg:  # Added volume filter
                # Supply zone (bearish push down)
                if df['close'].iloc[i] < df['open'].iloc[i]:
                    # Check for push down
                    if df['close'].iloc[i] < df['low'].iloc[i-1]:
                        # Accuracy zone check
                        if df['low'].iloc[i] < df['low'].iloc[i+1]:
                            zone = {
                                'high': df['high'].iloc[i],
                                'low': df['close'].iloc[i],
                                'type': 'accuracy',
                                'index': i
                            }
                        else:
                            zone = {
                                'high': df['high'].iloc[i],
                                'low': df['low'].iloc[i],
                                'type': 'normal',
                                'index': i
                            }
                        # Moved wick validation inside zone creation block
                        candle_body = abs(df['close'].iloc[i] - df['open'].iloc[i])
                        wick_size = df['high'].iloc[i] - df['low'].iloc[i] - candle_body
                        if zone['type'] == 'accuracy' and wick_size < candle_body * 0.5:
                            continue  # Skip this zone if wick criteria not met
                        supply_zones.append(zone)
                
                # Demand zone (bullish push up)
                elif df['close'].iloc[i] > df['open'].iloc[i]:
                    # Check for push up
                    if df['close'].iloc[i] > df['high'].iloc[i-1]:
                        # Accuracy zone check
                        if df['high'].iloc[i] > df['high'].iloc[i+1]:
                            zone = {
                                'high': df['open'].iloc[i],
                                'low': df['low'].iloc[i],
                                'type': 'accuracy',
                                'index': i
                            }
                        else:
                            zone = {
                                'high': df['high'].iloc[i],
                                'low': df['low'].iloc[i],
                                'type': 'normal',
                                'index': i
                            }
                        # Moved wick validation inside zone creation block
                        candle_body = abs(df['close'].iloc[i] - df['open'].iloc[i])
                        wick_size = df['high'].iloc[i] - df['low'].iloc[i] - candle_body
                        if zone['type'] == 'accuracy' and wick_size < candle_body * 0.5:
                            continue  # Skip this zone if wick criteria not met
                        demand_zones.append(zone)
            
            # FIXED: Use original datetime index for age calculation
            zone_age = (original_index[-1] - original_index[i]).total_seconds()/3600
            print(f"Potential zone at index {i} ({zone_age:.1f}h old) - Type: {'Supply' if df['close'].iloc[i] < df['open'].iloc[i] else 'Demand'}")
        
        return supply_zones, demand_zones

    def detect_liquidity_formation(self, df, symbol, zone):
        """Detect liquidity formation near zones"""
        # Look for multiple liquidity sweeps
        recent_lows = df['low'].tail(20)
        recent_highs = df['high'].tail(20)
        
        if 'high' in zone:  # Supply zone
            swing_lows = recent_lows[recent_lows < zone['low']]
            valid = len(swing_lows) >= 2
            # Check volume on last tap
            return valid and df['volume'].iloc[-1] > df['volume'].mean() * 1.2
        else:  # Demand zone
            swing_highs = recent_highs[recent_highs > zone['high']]
            valid = len(swing_highs) >= 2
            return valid and df['volume'].iloc[-1] > df['volume'].mean() * 1.2

    def validate_zone(self, df, zone):
        """Validate zone based on criteria"""
        # Fix zone age calculation
        zone_time = df.iloc[zone['index']].name
        current_time = df.index[-1]
        zone_age = (current_time - zone_time).total_seconds() / 3600
        if zone_age > 48:
            print(f"Rejecting zone: Age {zone_age:.1f}h > 48h")
            return False
        
        # Keep existing price validation
        current_price = df['close'].iloc[-1]
        if 'high' in zone:
            return current_price < zone['high'] * 1.05
        else:
            return current_price > zone['low'] * 0.95

    def calculate_risk_reward(self, entry_price, stop_price, target_price):
        """Calculate risk-reward ratio"""
        risk = abs(entry_price - stop_price)
        reward = abs(target_price - entry_price)
        return reward / risk if risk > 0 else 0

    def find_targets(self, df, entry_price, stop_price, direction):
        """Find TP levels using swing highs/lows and fibonacci extensions"""
        risk = abs(entry_price - stop_price)
        
        if direction == 'LONG':
            # TP1: Nearest swing high (minimum 1:3)
            min_tp1 = entry_price + (risk * 3)
            future_highs = df[df['high'] > min_tp1]['high']
            tp1 = future_highs.iloc[0] if not future_highs.empty else min_tp1
            
            # TP2: 1.618 extension or max 1:10
            tp2 = min(entry_price + (risk * 10),  # Max 1:10
                     entry_price + (risk * 1.618))  # Fib extension
        else:
            # Similar logic for shorts
            min_tp1 = entry_price - (risk * 3)
            future_lows = df[df['low'] < min_tp1]['low']
            tp1 = future_lows.iloc[0] if not future_lows.empty else min_tp1
            
            tp2 = max(entry_price - (risk * 10),
                     entry_price - (risk * 1.618))
        
        return tp1, tp2

    def validate_entry_sequence(self, df, zone, zone_type):
        """Validate exact entry sequence"""
        # Require confirmation candle
        if zone_type == 'SUPPLY':
            if df['close'].iloc[-1] > df['open'].iloc[-1]:  # Reject if bullish close
                return False, None
        else:
            if df['close'].iloc[-1] < df['open'].iloc[-1]:  # Reject if bearish close
                return False, None
        
        last_candles = df.tail(5)
        
        if zone_type == 'SUPPLY':
            # Expanded entry conditions
            bearish_close = last_candles['close'].iloc[-1] < last_candles['open'].iloc[-1]
            break_entry = last_candles['low'].iloc[-1] < min(last_candles['low'].iloc[-2], zone['low'])  # More lenient
            flip_entry = (last_candles['high'].iloc[-1] > last_candles['high'].iloc[-2] and 
                         last_candles['close'].iloc[-1] < last_candles['close'].iloc[-2])  # More flexible
            
            if any([bearish_close, break_entry, flip_entry]):
                entry_price = last_candles['close'].iloc[-1]
                return True, entry_price
        else:  # DEMAND
            # Expanded entry conditions
            normal_entry = last_candles['close'].iloc[-1] > last_candles['open'].iloc[-1]
            break_entry = (last_candles['high'].iloc[-1] > last_candles['high'].iloc[-2] and 
                          last_candles['close'].iloc[-1] > last_candles['high'].iloc[-2])
            flip_entry = (last_candles['low'].iloc[-1] < last_candles['low'].iloc[-2] and 
                          last_candles['close'].iloc[-1] > last_candles['low'].iloc[-2])
            
            if any([normal_entry, break_entry, flip_entry]):
                entry_price = last_candles['close'].iloc[-1]
                return True, entry_price
        
        return False, None

    def plot_setup(self, df, setup, filename=None):
        """Plot the setup with zones and entry points"""
        # Create a copy of the dataframe for plotting
        plot_df = df.copy()
        
        # Configure style
        mc = mpf.make_marketcolors(up='green', down='red',
                                  edge='inherit',
                                  wick='inherit',
                                  volume='in')
        s = mpf.make_mpf_style(marketcolors=mc)
        
        # Create the plot
        fig, axes = mpf.plot(plot_df, type='candle', style=s,
                            volume=True, returnfig=True,
                            figsize=(15, 10))
        
        # Add zone
        if setup['zone_type'] == 'SUPPLY':
            plt.axhline(y=setup['stop_price'], color='r', linestyle='--', label='Supply Zone High')
            plt.axhline(y=setup['entry_price'], color='r', linestyle='--', label='Supply Zone Low')
        else:
            plt.axhline(y=setup['stop_price'], color='g', linestyle='--', label='Demand Zone Low')
            plt.axhline(y=setup['entry_price'], color='g', linestyle='--', label='Demand Zone High')
        
        # Add targets
        plt.axhline(y=setup['tp1_price'], color='b', linestyle=':', label='TP1')
        plt.axhline(y=setup['tp2_price'], color='b', linestyle='--', label='TP2')
        
        # Add title and legend
        plt.title(f"{setup['symbol']} {setup['zone_type']} Zone Setup (RR: {setup['rr_ratio']:.1f})")
        plt.legend()
        
        # Save or show
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def find_setup_sequence(self, df, symbol):
        """Find valid supply and demand setups with updated rules"""
        print(f"\n=== Zone Detection Debug for {symbol} ===")
        supply_zones, demand_zones = self.detect_zones(df)
        print(f"Found {len(supply_zones)} supply zones, {len(demand_zones)} demand zones")
        
        for zone in supply_zones:
            print(f"Supply Zone: {zone} | Age: {(df.index[-1] - df.index[zone['index']]).total_seconds()/3600:.1f}h")
            valid = self.validate_zone(df, zone)
            print(f"Zone validation: {valid}")
        
        # Similar debug for demand zones...
        valid_setups = []
        
        # Get trend
        trend = self.check_trend_alignment(df)
        print(f"Current trend: {trend}")
        
        # Check supply zones
        for zone in supply_zones:
            # Only consider supply zones in downtrend or no clear trend
            if trend == 'up':
                continue
            
            # Check zone age (max 48 hours as per strategy)
            zone_age = (df.index[-1] - df.index[zone['index']]).total_seconds() / 3600
            if zone_age > 48:
                continue
            
            # Validate sequence
            valid_entry, entry_price = self.validate_entry_sequence(df, zone, 'SUPPLY')
            if not valid_entry:
                continue
            
            # Check for liquidity formation
            if self.detect_liquidity_formation(df, symbol, zone):
                stop_price = zone['high']
                
                # Find targets
                tp1, tp2 = self.find_targets(df, entry_price, stop_price, 'SHORT')
                
                # Calculate RR
                risk = abs(entry_price - stop_price)
                reward_tp1 = abs(tp1 - entry_price)
                rr_ratio = reward_tp1 / risk if risk > 0 else 0
                
                if rr_ratio >= 3:  # Minimum 1:3 RR as per strategy
                    setup = {
                        'symbol': symbol,
                        'direction': 'SHORT',
                        'zone_type': 'SUPPLY',
                        'zone_accuracy': zone['type'],
                        'entry_price': float(entry_price),
                        'stop_price': float(stop_price),
                        'tp1_price': float(tp1),
                        'tp2_price': float(tp2),
                        'rr_ratio': float(rr_ratio),
                        'zone_age': zone_age
                    }
                    valid_setups.append(setup)
                    print(f"Found valid SUPPLY setup for {symbol}")
        
        # Similar logic for demand zones
        for zone in demand_zones:
            # Only consider demand zones in uptrend or no clear trend
            if trend == 'down':
                continue
            
            # Check zone age
            zone_age = (df.index[-1] - df.index[zone['index']]).total_seconds() / 3600
            if zone_age > 48:
                continue
            
            # Validate sequence
            valid_entry, entry_price = self.validate_entry_sequence(df, zone, 'DEMAND')
            if not valid_entry:
                continue
            
            # Check for liquidity formation
            if self.detect_liquidity_formation(df, symbol, zone):
                stop_price = zone['low']
                
                # Find targets
                tp1, tp2 = self.find_targets(df, entry_price, stop_price, 'LONG')
                
                # Calculate RR
                risk = abs(entry_price - stop_price)
                reward_tp1 = abs(tp1 - entry_price)
                rr_ratio = reward_tp1 / risk if risk > 0 else 0
                
                if rr_ratio >= 3:  # Minimum 1:3 RR as per strategy
                    setup = {
                        'symbol': symbol,
                        'direction': 'LONG',
                        'zone_type': 'DEMAND',
                        'zone_accuracy': zone['type'],
                        'entry_price': float(entry_price),
                        'stop_price': float(stop_price),
                        'tp1_price': float(tp1),
                        'tp2_price': float(tp2),
                        'rr_ratio': float(rr_ratio),
                        'zone_age': zone_age
                    }
                    valid_setups.append(setup)
                    print(f"Found valid DEMAND setup for {symbol}")
        
        return valid_setups

    def scan_for_setups(self):
        """Scan for setups matching the strategy"""
        print("\nStarting scan for setups...")
        valid_pairs = self.get_valid_pairs()
        potential_setups = []
        
        print(f"\nFound {len(valid_pairs)} pairs meeting initial criteria")
        
        for symbol in valid_pairs:
            try:
                print(f"\n=== Analyzing {symbol} ===")
                
                # Get data for both timeframes
                df_4h = self.get_klines(symbol, '4h', 300)
                df_1h = self.get_klines(symbol, '1h', 300)
                
                if df_4h is None or df_1h is None:
                    print("Failed to get kline data")
                    continue
                
                # Check trend alignment
                trend = self.check_trend_alignment(df_4h)
                print(f"Trend alignment: {trend}")
                
                # Continue analysis even if no clear trend
                # Check for strong move
                strong_move = self.is_strong_move(df_4h)
                print(f"Strong move detected: {strong_move}")
                
                if not strong_move:
                    print("No strong move detected, skipping pair")
                    continue
                    
                # Check for retracement
                retracing = self.is_retracing(df_4h)
                print(f"Price retracing: {retracing}")
                
                if not retracing:
                    print("No valid retracement, skipping pair")
                    continue
                
                # Find setups on 4h timeframe
                setups = self.find_setup_sequence(df_4h, symbol)
                print(f"Number of potential setups found: {len(setups)}")
                
                for setup in setups:
                    # Validate setup with 1h timeframe
                    if self.validate_setup_with_lower_timeframe(df_1h, setup):
                        potential_setups.append(setup)
                        print(f"Valid {setup['direction']} setup found!")
                        print(f"Entry: {setup['entry_price']:.8f}")
                        print(f"Stop: {setup['stop_price']:.8f}")
                        print(f"TP1: {setup['tp1_price']:.8f}")
                        print(f"RR Ratio: {setup['rr_ratio']:.2f}")
                    else:
                        print("Setup failed 1h timeframe validation")
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        print(f"\nAnalysis complete. Found {len(potential_setups)} total setups")
        return pd.DataFrame(potential_setups) if potential_setups else pd.DataFrame()

    def validate_setup_with_lower_timeframe(self, df_1h, setup):
        """Validate setup using 1h timeframe"""
        try:
            # Get recent 1h candles
            recent_candles = df_1h.tail(12).copy()  # Last 12 hours
            
            # Calculate EMAs for trend context
            recent_candles['ema20'] = ta.trend.EMAIndicator(recent_candles['close'], window=20).ema_indicator()
            
            if setup['direction'] == 'LONG':  # For demand zones
                # Check if we're in an uptrend or consolidation on 1h
                price = recent_candles['close'].iloc[-1]
                ema = recent_candles['ema20'].iloc[-1]
                
                # Modified EMA check (3% threshold)
                ema_distance = abs(price - ema) / ema
                valid_ema = ema_distance <= 0.03  # Increased from 1% to 3%
                
                # Relaxed price action check
                clean_price = True
                overlapping_count = 0
                for i in range(1, len(recent_candles)):
                    curr_candle = recent_candles.iloc[i]
                    prev_candle = recent_candles.iloc[i-1]
                    
                    # Avoid overlapping candles
                    if (curr_candle['high'] > prev_candle['high'] and 
                        curr_candle['low'] < prev_candle['low']):
                        overlapping_count += 1
                # Allow up to 20% overlapping candles        
                clean_price = overlapping_count/len(recent_candles) < 0.2
                
                print(f"1H Validation - Valid EMA: {valid_ema}, Clean price: {clean_price}")
                return valid_ema and clean_price
                
            else:  # For supply zones (SHORT)
                # Check if we're in a downtrend or consolidation on 1h
                price = recent_candles['close'].iloc[-1]
                ema = recent_candles['ema20'].iloc[-1]
                
                # Modified EMA check (3% threshold)
                ema_distance = abs(price - ema) / ema
                valid_ema = ema_distance <= 0.03  # Increased from 1% to 3%
                
                # Relaxed price action check
                clean_price = True
                overlapping_count = 0
                for i in range(1, len(recent_candles)):
                    curr_candle = recent_candles.iloc[i]
                    prev_candle = recent_candles.iloc[i-1]
                    
                    # Avoid overlapping candles
                    if (curr_candle['high'] > prev_candle['high'] and 
                        curr_candle['low'] < prev_candle['low']):
                        overlapping_count += 1
                # Allow up to 20% overlapping candles        
                clean_price = overlapping_count/len(recent_candles) < 0.2
                
                print(f"1H Validation - Valid EMA: {valid_ema}, Clean price: {clean_price}")
                return valid_ema and clean_price
                
        except Exception as e:
            print(f"Error in validate_setup_with_lower_timeframe: {str(e)}")
            print(traceback.format_exc())
            return False

    def detect_liquidity_formation(self, df, symbol, zone):
        # Add minimum volume requirement
        min_volume = df['volume'].quantile(0.7)  # Top 30% volume
        if df['volume'].iloc[-1] < min_volume:
            print(f"Liquidity tap volume too low: {df['volume'].iloc[-1]:.2f} < {min_volume:.2f}")
            return False

def main():
    try:
        # Initialize scanner
        print("Initializing scanner...")
        scanner = SupplyDemandScanner()
        
        print("Scanning for potential setups...")
        try:
            setups = scanner.scan_for_setups()
        except Exception as e:
            print(f"Error during scan_for_setups: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            return
        
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

if __name__ == "__main__":
    main() 