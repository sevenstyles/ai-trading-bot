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
import argparse

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
            # Modified to allow extreme retracements near swing points
            recent_data = df.tail(10)
            current_price = df['close'].iloc[-1]
            
            # Get nearest swing high/low
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            
            # Allow retracements up to 90% from swing point
            valid_retracement = (
                current_price <= swing_high * 0.9 or  # 10% below swing high
                current_price >= swing_low * 1.1      # 10% above swing low
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
        # Add recency filter to only look at last 100 candles
        df = df.iloc[-100:] if len(df) > 100 else df
        original_index = df.index
        df = df.reset_index()
        
        # Calculate candle characteristics
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        supply_zones = []
        demand_zones = []
        
        for i in range(1, len(df)-1):
            # Remove dynamic lookback and volume confirmation
            is_strong_push = df['body_size'].iloc[i] > df['body_size'].iloc[i-5:i].mean() * 0.9
            
            # Modified condition (remove volume_above_avg check)
            if is_strong_push:  # Simplified condition
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
                        zone['tapped'] = self.is_zone_tapped(df, zone)
                        zone['strength'] = self.calculate_zone_strength(df, zone)
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
                        zone['tapped'] = self.is_zone_tapped(df, zone)
                        zone['strength'] = self.calculate_zone_strength(df, zone)
                        demand_zones.append(zone)
            
            # FIXED: Use original datetime index for age calculation
            zone_age = (original_index[-1] - original_index[i]).total_seconds()/3600
            print(f"Potential zone at index {i} ({zone_age:.1f}h old) - Type: {'Supply' if df['close'].iloc[i] < df['open'].iloc[i] else 'Demand'}")
        
        return supply_zones, demand_zones

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
        
        return strength

    def detect_liquidity_formation(self, df, symbol, zone):
        """Simplified liquidity check without sweep counting"""
        try:
            # Check for break of previous high/low only
            if zone['type'] == 'supply':
                return df['high'].iloc[-1] > df['high'].iloc[-2]
            else:
                return df['low'].iloc[-1] < df['low'].iloc[-2]
        except Exception as e:
            print(f"Error in detect_liquidity_formation: {str(e)}")
            return False

    def validate_zone(self, df, zone):
        """Updated validation with proper tap checking"""
        try:
            # Changed to allow wick taps but reject closes inside
            post_zone_data = df.iloc[zone['index']+1:]
            
            # Check for candle CLOSURES inside zone
            closes_inside = post_zone_data[
                (post_zone_data['close'] > zone['low']) & 
                (post_zone_data['close'] < zone['high'])
            ]
            if not closes_inside.empty:
                print(f"Rejecting zone - {len(closes_inside)} closes inside")
                return False
            
            return True
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
        """Now using 30m candles for entry validation"""
        post_zone_data = df.iloc[zone['index']+1:]  # 30m data
        
        # Existing entry logic remains but operates on 30m timeframe
        for i in range(len(post_zone_data)):
            if zone_type == 'SUPPLY':
                if post_zone_data['close'].iloc[i] < post_zone_data['open'].iloc[i]:
                    return True, post_zone_data['close'].iloc[i]
            else:
                # LONG entry - first bullish close after zone touch
                if post_zone_data['close'].iloc[i] > post_zone_data['open'].iloc[i]:
                    return True, post_zone_data['close'].iloc[i]
        
        return False, None

    def plot_setup(self, df, setup, filename=None):
        """Enhanced visualization of strategy setup"""
        # Create plot with annotations
        apds = [
            mpf.make_addplot(df['close'], panel=0, color='dodgerblue', ylabel='Price'),
        ]
        
        # Create figure
        fig, axes = mpf.plot(df, 
                           type='candle',
                           style='yahoo',
                           addplot=apds,
                           volume=True,
                           figsize=(20, 12),
                           returnfig=True)
        
        ax = axes[0]
        
        # Draw supply/demand zone
        if setup['zone_type'] == 'SUPPLY':
            zone_color = 'red'
            ax.axhspan(setup['stop_price'], setup['entry_price'], 
                     facecolor='red', alpha=0.1, label='Supply Zone')
        else:
            zone_color = 'green'
            ax.axhspan(setup['entry_price'], setup['stop_price'],
                     facecolor='green', alpha=0.1, label='Demand Zone')
        
        # Plot entry point
        entry_time = df.index[-1]  # Simplified for example
        ax.plot(entry_time, setup['entry_price'], 
              marker='^' if setup['direction'] == 'LONG' else 'v', 
              markersize=15, color='lime', label='Entry')
        
        # Plot liquidity levels
        ax.axhline(y=setup['stop_price'], color=zone_color, 
                 linestyle='--', linewidth=2, label='Liquidity Level')
        
        # Plot targets and stops
        ax.axhline(setup['tp1_price'], color='blue', linestyle=':', 
                 label='TP1 (1:3 RR)')
        ax.axhline(setup['tp2_price'], color='navy', linestyle='--', 
                 label='TP2 (1:10 RR)')
        ax.axhline(setup['stop_price'], color='red', 
                 linestyle='-', label='Stop Loss')
        
        # Add strategy info box
        info_text = (
            f"Strategy 3 Setup\n"
            f"Direction: {setup['direction']}\n"
            f"RR Ratio: 1:{setup['rr_ratio']:.1f}\n"
            f"Zone Type: {setup['zone_accuracy'].title()}\n"
            f"Confidence: {self.calculate_confidence(setup)}/10"
        )
        ax.text(0.05, 0.95, info_text, 
              transform=ax.transAxes, 
              bbox=dict(facecolor='white', alpha=0.8))
        
        # Add legend and labels
        ax.set_title(f"{setup['symbol']} - Supply & Demand Strategy Visualization", fontsize=16)
        ax.legend(loc='upper left')
        
        # Save or show
        if filename:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def calculate_confidence(self, setup):
        """Simple confidence scoring for visualization"""
        score = 0
        # Add your confidence metrics here
        score += 3 if setup['rr_ratio'] >= 3 else 0
        score += 2 if setup['zone_accuracy'] == 'accuracy' else 1
        return min(10, score)

    def find_setup_sequence(self, df, symbol):
        """Find valid supply and demand setups with updated rules"""
        print(f"\n=== Zone Detection Debug for {symbol} ===")
        # Get 4H trend separately
        df_4h = self.get_klines(symbol, '4h', 50)  # Shorter lookback
        trend = self.check_trend_alignment(df_4h)
        
        # Existing zone analysis continues but on 30m df
        supply_zones, demand_zones = self.detect_zones(df)
        print(f"Found {len(supply_zones)} supply zones, {len(demand_zones)} demand zones")
        
        for zone in supply_zones:
            print(f"Supply Zone: {zone} | Age: {(df.index[-1] - df.index[zone['index']]).total_seconds()/3600:.1f}h")
            valid = self.validate_zone(df, zone)
            print(f"Zone validation: {valid}")
        
        # Similar debug for demand zones...
        valid_setups = []
        
        # Check supply zones
        for zone in supply_zones:
            # Only consider supply zones in downtrend or no clear trend
            if trend == 'up':
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
                        'rr_ratio': float(rr_ratio)
                    }
                    valid_setups.append(setup)
                    print(f"Found valid SUPPLY setup for {symbol}")
        
        # Similar logic for demand zones
        for zone in demand_zones:
            # Only consider demand zones in uptrend or no clear trend
            if trend == 'down':
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
                        'rr_ratio': float(rr_ratio)
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
                df_4h = self.get_klines(symbol, '4h', 300)  # For trend only
                df_30m = self.get_klines(symbol, '30m', 300)  # For zone analysis
                
                if df_4h is None or df_30m is None:
                    print("Failed to get kline data")
                    continue
                
                # Check trend alignment
                trend = self.check_trend_alignment(df_4h)
                print(f"Trend alignment: {trend}")
                
                # Continue analysis even if no clear trend
                # Check for strong move
                strong_move = self.is_strong_move(df_30m)
                print(f"Strong move detected: {strong_move}")
                
                if not strong_move:
                    print("No strong move detected, skipping pair")
                    continue
                    
                # Check for retracement
                retracing = self.is_retracing(df_30m)
                print(f"Price retracing: {retracing}")
                
                if not retracing:
                    print("No valid retracement, skipping pair")
                    continue
                
                # Find setups on 30m data with 4h trend
                setups = self.find_setup_sequence(df_30m, symbol)
                print(f"Number of potential setups found: {len(setups)}")
                
                for setup in setups:
                    # Validate setup with 4h timeframe
                    if self.validate_setup_with_lower_timeframe(df_4h, setup):
                        potential_setups.append(setup)
                        print(f"Valid {setup['direction']} setup found!")
                        print(f"Entry: {setup['entry_price']:.8f}")
                        print(f"Stop: {setup['stop_price']:.8f}")
                        print(f"TP1: {setup['tp1_price']:.8f}")
                        print(f"RR Ratio: {setup['rr_ratio']:.2f}")
                    else:
                        print("Setup failed 4h timeframe validation")
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        print(f"\nAnalysis complete. Found {len(potential_setups)} total setups")
        return pd.DataFrame(potential_setups) if potential_setups else pd.DataFrame()

    def validate_setup_with_lower_timeframe(self, df_4h, setup):
        """Validate setup using 4h timeframe"""
        try:
            # Simply confirm direction matches
            recent_candles = df_4h.tail(12)
            if setup['direction'] == 'LONG':
                return recent_candles['close'].iloc[-1] > recent_candles['close'].iloc[0]
            else:
                return recent_candles['close'].iloc[-1] < recent_candles['close'].iloc[0]
        except Exception as e:
            print(f"Error in validate_setup_with_lower_timeframe: {str(e)}")
            return False

    def is_zone_tapped(self, df, zone):
        """Check if price has entered zone since creation - FIXED"""
        post_zone_data = df.iloc[zone['index']+1:]
        
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

def test():
    scanner = SupplyDemandScanner()
    scanner.test_visualization()  # Test BTC
    # scanner.test_visualization("ETHUSDT")  # Uncomment to test ETH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run visualization test')
    args = parser.parse_args()
    
    if args.test:
        test()
    else:
        main() 