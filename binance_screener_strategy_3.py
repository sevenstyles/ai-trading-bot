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
            # Calculate average candle size
            df['candle_size'] = abs(df['close'] - df['open'])
            avg_size = df['candle_size'].rolling(20).mean()
            
            # Look for candles significantly larger than average
            recent_candles = df.tail(5).reset_index(drop=True)
            recent_avgs = avg_size.tail(5).reset_index(drop=True)
            
            # More lenient strong move detection (1.3x instead of 1.5x)
            strong_moves = recent_candles['candle_size'] > (recent_avgs * 1.3)
            
            result = strong_moves.any()
            print(f"Strong move check - Recent sizes: {recent_candles['candle_size'].values}, Averages: {recent_avgs.values}, Result: {result}")
            return result
            
        except Exception as e:
            print(f"Error in is_strong_move: {str(e)}")
            return False

    def is_retracing(self, df):
        """Check if price is retracing after a strong move"""
        try:
            recent_data = df.tail(10)
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()
            current_price = df['close'].iloc[-1]
            
            # Calculate retracement levels
            up_retracement = (recent_high - current_price) / (recent_high - recent_low)
            down_retracement = (current_price - recent_low) / (recent_high - recent_low)
            
            # More lenient retracement range (20-80%)
            result = 0.2 <= up_retracement <= 0.8 or 0.2 <= down_retracement <= 0.8
            
            print(f"Retracement check - High: {recent_high}, Low: {recent_low}, Current: {current_price}")
            print(f"Up retracement: {up_retracement:.2%}, Down retracement: {down_retracement:.2%}")
            
            return result
            
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
        
        return df

    def detect_zones(self, df):
        """Detect supply and demand zones with accuracy rules"""
        df = df.reset_index()
        
        # Calculate candle characteristics
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        supply_zones = []
        demand_zones = []
        
        for i in range(1, len(df)-1):
            # Strong move detection (1.3x average)
            is_strong = df['body_size'].iloc[i] > df['body_size'].iloc[i-10:i].mean() * 1.3
            
            if is_strong:
                # Supply zone (bearish candle)
                if df['close'].iloc[i] < df['open'].iloc[i]:
                    # Accuracy zone: If wick extends below next candle, use body + upper wick only
                    if df['low'].iloc[i] < df['low'].iloc[i+1]:
                        zone = {
                            'high': df['high'].iloc[i],
                            'low': df['close'].iloc[i],  # Use close for accuracy zone
                            'type': 'accuracy',
                            'index': i
                        }
                    else:
                        # Normal zone: use full candle
                        zone = {
                            'high': df['high'].iloc[i],
                            'low': df['low'].iloc[i],
                            'type': 'normal',
                            'index': i
                        }
                    supply_zones.append(zone)
                
                # Demand zone (bullish candle)
                elif df['close'].iloc[i] > df['open'].iloc[i]:
                    zone = {
                        'high': df['open'].iloc[i],  # Use open for more accurate zone
                        'low': df['low'].iloc[i],
                        'type': 'accuracy' if df['high'].iloc[i] > df['high'].iloc[i+1] else 'normal',
                        'index': i
                    }
                    demand_zones.append(zone)
        
        return supply_zones, demand_zones

    def detect_liquidity_formation(self, df, symbol, zone):
        """Detect liquidity formation near zones"""
        # Check for liquidity formation (small retracement not tapping zone)
        recent_candles = df.tail(5)  # Look at last 5 candles
        
        if 'high' in zone:  # Supply zone
            # Check if we've broken the low that created liquidity
            liquidity_low = recent_candles['low'].min()
            previous_low = df['low'].iloc[-6:-1].min()  # Low before recent candles
            liquidity_valid = liquidity_low < previous_low
            
            return liquidity_valid
        else:  # Demand zone
            # Check if we've broken the high that created liquidity
            liquidity_high = recent_candles['high'].max()
            previous_high = df['high'].iloc[-6:-1].max()  # High before recent candles
            liquidity_valid = liquidity_high > previous_high
            
            return liquidity_valid

    def validate_zone(self, df, zone):
        """Validate zone based on criteria"""
        # Check for candle closures inside zone
        price_in_zone = ((df['close'] >= zone['low']) & (df['close'] <= zone['high']))
        if price_in_zone.any():
            return False
            
        # Check trend alignment using higher timeframe
        df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        current_trend = 'up' if df['close'].iloc[-1] > df['ema50'].iloc[-1] else 'down'
        
        # Zone should align with trend
        if 'high' in zone and current_trend == 'up':  # Supply zone should be in downtrend
            return False
        if 'low' in zone and current_trend == 'down':  # Demand zone should be in uptrend
            return False
            
        return True

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
        last_candles = df.tail(5)
        
        if zone_type == 'SUPPLY':
            # Find candle that tapped the zone
            tap_mask = last_candles['high'] >= zone['high']
            if not tap_mask.any():
                return False, None
            
            tap_idx = tap_mask.idxmax()
            tap_candle = last_candles.loc[tap_idx]
            
            # Check for first bearish close after tap
            post_tap_candles = last_candles.loc[tap_idx:]
            
            # No closes inside zone
            if ((post_tap_candles['close'] >= zone['low']) & 
                (post_tap_candles['close'] <= zone['high'])).any():
                return False, None
            
            # Look for entry confirmation
            last_candle = last_candles.iloc[-1]
            prev_candle = last_candles.iloc[-2]
            
            # Primary entry - bearish close
            bearish_close = last_candle['close'] < last_candle['open']
            
            # Secondary entry - break of candle
            break_entry = (last_candle['low'] < prev_candle['low'] and 
                          last_candle['close'] < prev_candle['low'])
            
            # Flip entry (rare) - momentum flip in final minutes
            flip_entry = (last_candle['high'] > prev_candle['high'] and
                         last_candle['close'] < prev_candle['low'])
            
            if bearish_close or break_entry or flip_entry:
                entry_price = last_candle['close']
                return True, entry_price
        else:  # DEMAND
            # Find the candle that tapped the zone
            tap_mask = last_candles['low'] <= zone['low']
            if not tap_mask.any():
                return False, None
            
            tap_idx = tap_mask.idxmax()
            tap_candle = last_candles.loc[tap_idx]
            
            # Check candles after tap
            post_tap_candles = last_candles.loc[tap_idx:]
            
            # No closes inside zone
            if ((post_tap_candles['close'] >= zone['low']) & 
                (post_tap_candles['close'] <= zone['high'])).any():
                return False, None
            
            # Look for entry confirmation
            last_candle = last_candles.iloc[-1]
            prev_candle = last_candles.iloc[-2]
            
            # Normal entry - bullish close
            normal_entry = last_candle['close'] > last_candle['open']
            
            # Break of candle entry
            break_entry = (last_candle['high'] > prev_candle['high'] and 
                          last_candle['close'] > prev_candle['high'])
            
            if normal_entry or break_entry:
                entry_price = last_candle['close']
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
        valid_setups = []
        
        # Detect zones
        supply_zones, demand_zones = self.detect_zones(df)
        
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
        """
        Validate setup using 1h timeframe
        - Check for clean price action
        - Verify trend alignment
        """
        # Get recent candles
        recent_candles = df_1h.tail(12)  # Last 12 1h candles
        
        # Calculate average body size
        recent_candles['body_size'] = abs(recent_candles['close'] - recent_candles['open'])
        avg_body_size = recent_candles['body_size'].mean()
        
        # Check for clean price action (no excessive wicks)
        for _, candle in recent_candles.iterrows():
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            
            # Wicks should not be more than 80% of body size
            if upper_wick > (avg_body_size * 0.8) or lower_wick > (avg_body_size * 0.8):
                return False
        
        # Check trend alignment
        if setup['direction'] == 'LONG':
            # For longs, we want to see higher lows
            lows = recent_candles['low'].values
            higher_lows = all(lows[i] >= lows[i-1] for i in range(1, len(lows)))
            if not higher_lows:
                return False
        else:  # SHORT
            # For shorts, we want to see lower highs
            highs = recent_candles['high'].values
            lower_highs = all(highs[i] <= highs[i-1] for i in range(1, len(highs)))
            if not lower_highs:
                return False
        
        return True

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