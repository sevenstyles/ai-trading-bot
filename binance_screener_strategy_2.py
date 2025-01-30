import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

class ForeverModelScanner:
    def __init__(self):
        self.base_url = os.getenv('BINANCE_BASE_URL', 'https://api.binance.com')
        self.min_volume_usdt = 1000000  # Minimum 24h volume in USDT
        self.min_price_usdt = 0.1  # Minimum price in USDT
        
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
        
    def get_valid_pairs(self):
        """Get all USDT pairs meeting minimum criteria"""
        exchange_info = self._send_request("/api/v3/exchangeInfo")
        valid_pairs = []
        
        if not exchange_info:
            return valid_pairs
            
        for symbol in exchange_info['symbols']:
            if symbol['quoteAsset'] == 'USDT' and symbol['status'] == 'TRADING':
                # Get 24h stats
                ticker = self._send_request("/api/v3/ticker/24hr", params={'symbol': symbol['symbol']})
                if ticker:
                    volume_usdt = float(ticker['quoteVolume'])
                    price = float(ticker['lastPrice'])
                    
                    if volume_usdt >= self.min_volume_usdt and price >= self.min_price_usdt:
                        valid_pairs.append(symbol['symbol'])
                        print(f"Found valid pair: {symbol['symbol']} (Volume: {volume_usdt:.2f} USDT, Price: {price:.2f} USDT)")
                        
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
        
        # Convert numeric columns
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df

    def calculate_average_volume(self, df, periods=20):
        """Calculate average volume for specified periods"""
        return df['volume'].rolling(window=periods).mean()

    def detect_liquidity_sweep(self, df, avg_volume):
        """Detect institutional liquidity sweeps with balanced criteria"""
        # Calculate key levels with improved swing detection
        df['swing_high'] = (df['high'] > df['high'].shift(1)) & \
                          (df['high'] > df['high'].shift(2)) & \
                          (df['high'] > df['high'].shift(-1)) & \
                          (df['high'] > df['high'].shift(-2))
                          
        df['swing_low'] = (df['low'] < df['low'].shift(1)) & \
                         (df['low'] < df['low'].shift(2)) & \
                         (df['low'] < df['low'].shift(-1)) & \
                         (df['low'] < df['low'].shift(-2))
        
        # Volume requirements - more flexible
        volume_spike = df['volume'] > (avg_volume * 2.0)
        not_excessive = df['volume'] < (avg_volume * 5)
        volume_valid = volume_spike & not_excessive
        
        # Calculate candle characteristics
        body_size = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        
        # Detect sweeps with relaxed criteria
        bullish_sweep = (
            (df['low'] < df['low'].shift(1)) &     # Takes out previous low
            (lower_wick > body_size * 1.2) &       # Reduced from 1.5
            (body_size < total_range * 0.5) &      # Increased from 0.4
            (df['close'] >= df['low'] + total_range * 0.3)  # Closes in upper half
        )
        
        bearish_sweep = (
            (df['high'] > df['high'].shift(1)) &   # Takes out previous high
            (upper_wick > body_size * 1.2) &       # Reduced from 1.5
            (body_size < total_range * 0.5) &      # Increased from 0.4
            (df['close'] <= df['high'] - total_range * 0.3)  # Closes in lower half
        )
        
        # Combine with volume confirmation
        sweeps = volume_valid & (bullish_sweep | bearish_sweep)
        
        # Label sweep direction
        sweep_direction = pd.Series(index=df.index, dtype=str)
        sweep_direction[bullish_sweep & volume_valid] = 'BULLISH'
        sweep_direction[bearish_sweep & volume_valid] = 'BEARISH'
        
        return sweeps, sweep_direction

    def detect_consolidation(self, df, min_candles=8, max_candles=12):
        """Check for 8-12 candle consolidation with strict requirements"""
        # Calculate price range as percentage
        price_range = (df['high'] - df['low']) / df['low'] * 100
        
        # Strict consolidation check (1.2% range)
        is_consolidating = price_range < 1.2
        
        # Volume must show consistent decrease
        volume_ma = df['volume'].rolling(window=3).mean()
        volume_decrease = volume_ma < volume_ma.shift(1)
        
        # Both conditions must be met for minimum candles
        consolidation = is_consolidating & volume_decrease
        
        # Look for sustained consolidation periods
        sustained_consolidation = consolidation.rolling(window=min_candles).sum() >= min_candles
        
        return sustained_consolidation

    def detect_fvg(self, df, min_gap_percent=1.2):
        """Detect Fair Value Gaps with improved validation"""
        # Calculate ATR for validation
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Calculate trend direction using EMA
        df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        bullish_trend = df['close'] > df['ema20']
        bearish_trend = df['close'] < df['ema20']
        
        # FVG detection with ATR and trend validation
        fvg_bullish = (
            (df['low'].shift(1) > df['high'].shift(-1)) & \
            ((df['low'].shift(1) - df['high'].shift(-1)) / df['close'] * 100 >= min_gap_percent) & \
            ((df['low'].shift(1) - df['high'].shift(-1)) > atr) & \
            bullish_trend  # Must align with trend
        )
        
        fvg_bearish = (
            (df['high'].shift(-1) > df['low'].shift(1)) & \
            ((df['high'].shift(-1) - df['low'].shift(1)) / df['close'] * 100 >= min_gap_percent) & \
            ((df['high'].shift(-1) - df['low'].shift(1)) > atr) & \
            bearish_trend  # Must align with trend
        )
        
        # Combine conditions
        fvg = fvg_bullish | fvg_bearish
        
        # Label FVG direction
        fvg_direction = pd.Series(index=df.index, dtype=str)
        fvg_direction[fvg_bullish] = 'BULLISH'
        fvg_direction[fvg_bearish] = 'BEARISH'
        
        return fvg, fvg_direction

    def check_timeframe_alignment(self, symbol):
        """Check alignment between 4h and 15m timeframes with less strict validation"""
        # Get data for both timeframes
        df_4h = self.get_klines(symbol, '4h', 200)
        df_15m = self.get_klines(symbol, '15m', 800)
        
        if df_4h is None or df_15m is None:
            return False, None
            
        # Get recent candles (last 12 15m candles = 3 hours)
        recent_15m = df_15m.tail(12)
        last_4h_candle = df_4h.iloc[-1]
        
        # Check for relatively clean price action (relaxed criteria)
        clean_wicks = all(
            abs(c['high'] - c['close']) < abs(c['close'] - c['open']) * 0.8 and  # Increased from 0.5
            abs(c['low'] - c['open']) < abs(c['close'] - c['open']) * 0.8        # Increased from 0.5
            for _, c in recent_15m.iterrows()
        )
        
        # Simplified momentum alignment check - removed EMA and RSI requirements
        if last_4h_candle['close'] > last_4h_candle['open']:  # Bullish 4h
            momentum_aligned = recent_15m['close'].iloc[-1] > recent_15m['close'].iloc[-3]  # Overall upward movement
            direction = 'BULLISH' if momentum_aligned and clean_wicks else None
        else:  # Bearish 4h
            momentum_aligned = recent_15m['close'].iloc[-1] < recent_15m['close'].iloc[-3]  # Overall downward movement
            direction = 'BEARISH' if momentum_aligned and clean_wicks else None
            
        return momentum_aligned and clean_wicks, direction

    def detect_structural_change(self, df):
        """Detect structural changes with more relaxed criteria"""
        # Calculate basic trend changes
        df['higher_high'] = df['high'] > df['high'].shift(1)
        df['lower_low'] = df['low'] < df['low'].shift(1)
        
        # Volume confirmation - more relaxed
        avg_volume = self.calculate_average_volume(df)
        volume_confirmed = df['volume'] > avg_volume * 1.1  # Reduced from 1.2
        
        # Detect BOS with relaxed criteria
        bullish_bos = (df['close'] > df['high'].shift(1)) & volume_confirmed
        bearish_bos = (df['close'] < df['low'].shift(1)) & volume_confirmed
        
        # Simplified CHoCH with relaxed criteria
        bullish_choch = bullish_bos & (df['low'].shift(1) > df['low'].shift(3))  # Changed from shift(2)
        bearish_choch = bearish_bos & (df['high'].shift(1) < df['high'].shift(3))  # Changed from shift(2)
        
        # Combine conditions
        structural_change = bullish_choch | bearish_choch
        
        # Label direction
        direction = pd.Series(index=df.index, dtype=str)
        direction[bullish_choch] = 'BULLISH'
        direction[bearish_choch] = 'BEARISH'
        
        return structural_change, direction

    def detect_smt(self, df):
        """Detect Smart Money Trap (SMT) patterns with relaxed criteria"""
        # Calculate price rejection
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Volume-weighted rejection - relaxed
        avg_volume = self.calculate_average_volume(df)
        volume_confirmed = df['volume'] > avg_volume * 1.5  # Reduced from 2.0
        
        # Detect rejection patterns - relaxed
        bullish_rejection = (
            (df['lower_wick'] > df['body_size'] * 1.5) &  # Reduced from 2
            (df['close'] > df['open']) &
            volume_confirmed &
            (df['close'] > df['close'].shift(1))  # Closes higher
        )
        
        bearish_rejection = (
            (df['upper_wick'] > df['body_size'] * 1.5) &  # Reduced from 2
            (df['close'] < df['open']) &
            volume_confirmed &
            (df['close'] < df['close'].shift(1))  # Closes lower
        )
        
        # Combine conditions
        smt = bullish_rejection | bearish_rejection
        
        # Label direction
        direction = pd.Series(index=df.index, dtype=str)
        direction[bullish_rejection] = 'BULLISH'
        direction[bearish_rejection] = 'BEARISH'
        
        return smt, direction

    def detect_opposing_liquidity(self, df, min_candles=12, max_target_distance=0.2):  # Increased from 0.15
        """Detect opposing liquidity targets with balanced criteria"""
        # Calculate swing points with relaxed window
        df['swing_high'] = df['high'].rolling(3).max() == df['high']
        df['swing_low'] = df['low'].rolling(3).min() == df['low']
        
        # Group similar levels with wider tolerance
        df['level_high'] = df['high'].round(decimals=3)  # Reduced precision
        df['level_low'] = df['low'].round(decimals=3)    # Reduced precision
        
        # Count touches at each level
        df['high_touches'] = df.groupby('level_high')['high'].transform('count')
        df['low_touches'] = df.groupby('level_low')['low'].transform('count')
        
        # Valid liquidity levels - relaxed criteria
        valid_highs = (df['high_touches'] >= 2) & df['swing_high']  # Reduced from 3 touches
        valid_lows = (df['low_touches'] >= 2) & df['swing_low']     # Reduced from 3 touches
        
        # Get target levels
        target_levels = pd.Series(index=df.index, dtype=float)
        target_levels[valid_highs] = df.loc[valid_highs, 'high']
        target_levels[valid_lows] = df.loc[valid_lows, 'low']
        
        # Filter targets by maximum distance (increased to 20%)
        current_price = df['close'].iloc[-1]
        max_distance = current_price * max_target_distance
        
        target_levels = target_levels[abs(target_levels - current_price) <= max_distance]
        
        return target_levels

    def find_setup_sequence(self, df):
        """Find setup sequences with more relaxed criteria"""
        avg_volume = self.calculate_average_volume(df)
        valid_setups = []
        
        # Detect all components
        sweeps, sweep_direction = self.detect_liquidity_sweep(df, avg_volume)
        structural_change, struct_direction = self.detect_structural_change(df)
        smt, smt_direction = self.detect_smt(df)
        target_levels = self.detect_opposing_liquidity(df, max_target_distance=0.2)  # Increased from 0.15
        
        # Look for valid sweeps
        sweep_indices = sweeps[sweeps].index
        for sweep_idx in sweep_indices:
            sweep_dir = sweep_direction.loc[sweep_idx]
            
            # Check 15m alignment with relaxed criteria
            aligned, align_direction = self.check_timeframe_alignment(df.name)
            if not aligned or align_direction != sweep_dir:
                continue
                
            # Look for structural change after sweep (increased window)
            struct_window = df.loc[sweep_idx:].head(25)  # Increased from 20
            if not (structural_change.loc[struct_window.index] & (struct_direction.loc[struct_window.index] == sweep_dir)).any():
                continue
            
            # Look for FVG after structural change (increased window)
            fvg_window = df.loc[sweep_idx:].head(30)  # Increased from 25
            fvg, fvg_direction = self.detect_fvg(fvg_window)
            
            if not (fvg & (fvg_direction == sweep_dir)).any():
                continue
            
            # Find first valid FVG
            fvg_idx = fvg[fvg & (fvg_direction == sweep_dir)].index[0]
            
            # Check for SMT (optional)
            has_smt = False
            if smt.loc[sweep_idx:fvg_idx].any():
                smt_matches = smt_direction.loc[sweep_idx:fvg_idx] == sweep_dir
                has_smt = smt_matches.any()
            
            # Find closest valid opposing liquidity target
            target_level = None
            if sweep_dir == 'BULLISH':
                future_targets = target_levels[target_levels > df.loc[fvg_idx, 'high']]
                if not future_targets.empty:
                    target_level = future_targets.iloc[0]
            else:
                future_targets = target_levels[target_levels < df.loc[fvg_idx, 'low']]
                if not future_targets.empty:
                    target_level = future_targets.iloc[-1]
            
            if target_level is None:
                continue
            
            # Calculate metrics
            entry_price = df.loc[fvg_idx, 'close']
            stop_price = df.loc[sweep_idx, 'high' if sweep_dir == 'BULLISH' else 'low']
            current_price = df['close'].iloc[-1]
            
            # Validate stop distance with minimum pip size (reduced)
            min_stop_distance = current_price * 0.002  # Reduced from 0.003
            if abs(entry_price - stop_price) < min_stop_distance:
                continue
            
            # Validate stop distance (increased)
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            current_atr = atr.iloc[-1]
            if abs(entry_price - stop_price) > (current_atr * 3.0):  # Increased from 2.5
                continue
                
            risk = abs(entry_price - stop_price)
            reward = abs(target_level - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # More flexible RR requirements
            if 2.0 <= rr_ratio <= 20:  # Widened range from 2.5-15
                valid_setups.append({
                    'direction': sweep_dir,
                    'sweep_index': sweep_idx,
                    'fvg_index': fvg_idx,
                    'sweep_price': float(df.loc[sweep_idx, 'close']),
                    'entry_price': float(entry_price),
                    'stop_price': float(stop_price),
                    'target_price': float(target_level),
                    'rr_ratio': float(rr_ratio),
                    'has_smt': has_smt,
                    'setup_quality': 'A++' if has_smt else 'A'
                })
        
        return valid_setups

    def scan_for_setups(self):
        """Scan for setups matching the updated strategy"""
        valid_pairs = self.get_valid_pairs()
        potential_setups = []
        
        for symbol in valid_pairs:
            try:
                print(f"\nAnalyzing {symbol}...")
                
                # Get 4h data (primary timeframe)
                df_4h = self.get_klines(symbol, '4h', 200)
                if df_4h is None:
                    continue
                
                # Check timeframe alignment
                aligned, align_direction = self.check_timeframe_alignment(symbol)
                if not aligned:
                    continue
                
                # Find valid setup sequences
                setups = self.find_setup_sequence(df_4h)
                if not setups:
                    continue
                
                for setup in setups:
                    if setup['direction'] == align_direction:  # Must align with 15m timeframe
                        setup_info = {
                            'symbol': symbol,
                            'direction': setup['direction'],
                            'entry_price': setup['entry_price'],
                            'stop_price': setup['stop_price'],
                            'target_price': setup['target_price'],
                            'rr_ratio': setup['rr_ratio'],
                            'setup_quality': setup['setup_quality'],
                            'has_smt': setup['has_smt']
                        }
                        potential_setups.append(setup_info)
                        print(f"Found {setup['direction']} setup for {symbol} (RR: {setup['rr_ratio']:.1f}, Quality: {setup['setup_quality']})")
                    
            except Exception as e:
                print(f"Error scanning {symbol}: {str(e)}")
                continue
        
        # Create DataFrame and sort by quality
        df_setups = pd.DataFrame(potential_setups)
        if not df_setups.empty:
            df_setups = df_setups.sort_values(['setup_quality', 'rr_ratio'], ascending=[False, False])
            df_setups = df_setups.drop_duplicates(subset=['symbol', 'direction'], keep='first')
        
        return df_setups

    def get_setup_details(self, symbol):
        """Get detailed analysis for a specific setup with proper JSON serialization"""
        df_4h = self.get_klines(symbol, '4h', 200)
        if df_4h is None:
            return None
            
        # Calculate ATR
        df_4h['atr'] = ta.volatility.AverageTrueRange(df_4h['high'], df_4h['low'], df_4h['close']).average_true_range()
        
        # Add setup components
        setups = self.find_setup_sequence(df_4h)
        if setups:
            latest_setup = setups[-1]
            # Create JSON-serializable dictionary
            details_dict = {
                'timestamp': df_4h['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'setup_direction': latest_setup['direction'],
                'sweep_price': float(latest_setup['sweep_price']),
                'entry_price': float(latest_setup['entry_price']),
                'stop_price': float(latest_setup['stop_price']),
                'target_price': float(latest_setup['target_price']),
                'rr_ratio': float(latest_setup['rr_ratio']),
                'has_smt': latest_setup['has_smt'],
                'setup_quality': latest_setup['setup_quality'],
                'current_price': float(df_4h['close'].iloc[-1]),
                'current_atr': float(df_4h['atr'].iloc[-1])
            }
            return details_dict
        
        return None

def main():
    try:
        # Initialize scanner
        scanner = ForeverModelScanner()
        
        print("Scanning for potential setups...")
        setups = scanner.scan_for_setups()
        
        if setups.empty:
            print("No setups found matching criteria")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs('screener-results', exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Filter out setups with zero gap
        setups = setups[setups['rr_ratio'] > 0]
        
        # Save detailed results to CSV
        csv_filename = f'screener-results/setups_{timestamp}.csv'
        setups.to_csv(csv_filename, index=False)
        
        # Create a simplified list of symbols for trade_pair.txt
        # Only include symbols with significant gaps
        significant_setups = setups[setups['rr_ratio'] > 1.0]
        symbols_list = significant_setups['symbol'].unique().tolist()
        symbols_filename = 'trade_pair.txt'
        with open(symbols_filename, 'w') as f:
            f.write('\n'.join(symbols_list))
        
        # Print summary to console
        print("\nTop potential setups found:")
        print(setups.to_string())
        print(f"\nDetailed results saved to: {csv_filename}")
        print(f"Symbols list saved to: {symbols_filename}")
        print(f"\nFound {len(setups)} unique setups across {len(symbols_list)} symbols")
        
        # Get detailed analysis for top setup
        if not setups.empty:
            top_symbol = setups.iloc[0]['symbol']
            print(f"\nDetailed analysis for top symbol {top_symbol}:")
            details = scanner.get_setup_details(top_symbol)
            
            if details is not None:
                # Save detailed analysis to JSON
                json_filename = f'screener-results/detailed_analysis_{top_symbol}_{timestamp}.json'
                with open(json_filename, 'w') as f:
                    json.dump(details, f, indent=2)
                print(f"Detailed analysis saved to: {json_filename}")
            
    except Exception as e:
        print(f"Error running scanner: {str(e)}")

if __name__ == "__main__":
    main()