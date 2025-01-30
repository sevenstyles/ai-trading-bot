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
        """Detect potential liquidity sweeps with volume requirements"""
        # Volume must be between 2.5x and 5x average (relaxed upper limit)
        volume_valid = (df['volume'] > (avg_volume * 2.5)) & (df['volume'] < (avg_volume * 7))  # Relaxed upper limit
        
        # Price action requirements - relaxed wick requirements
        high_wicks = (df['high'] - df['close']) > (df['high'] - df['low']) * 0.4  # Relaxed from 0.6
        low_wicks = (df['close'] - df['low']) > (df['high'] - df['low']) * 0.4   # Relaxed from 0.6
        
        # Must show clean sweep with weak closing price - relaxed conditions
        bearish_sweep = high_wicks & (df['close'] <= df['open'])
        bullish_sweep = low_wicks & (df['close'] >= df['open'])
        
        return (volume_valid & (bearish_sweep | bullish_sweep))

    def detect_consolidation(self, df, min_candles=6, max_candles=12):  # Reduced min candles
        """Check for 6-12 candle consolidation pattern"""
        # Calculate price range as percentage
        price_range = (df['high'] - df['low']) / df['low'] * 100
        
        # Check for tight consolidation - relaxed threshold
        is_consolidating = price_range < 2.0  # Increased from 1.2
        
        # Check volume decrease during consolidation - simplified
        volume_decrease = df['volume'] < df['volume'].rolling(window=3).mean()
        
        # Combine conditions and look for sustained periods - relaxed requirement
        consolidation = is_consolidating & volume_decrease
        return consolidation.rolling(window=min_candles).sum() >= (min_candles * 0.7)  # Allow some flexibility

    def detect_fvg(self, df, min_gap_percent=1.0):  # Reduced gap requirement
        """Detect Fair Value Gaps with validation"""
        # Basic FVG detection
        fvg_bullish = (df['low'].shift(1) > df['high'].shift(-1)) & \
                     ((df['low'].shift(1) - df['high'].shift(-1)) / df['close'] * 100 >= min_gap_percent)
        
        fvg_bearish = (df['high'].shift(-1) > df['low'].shift(1)) & \
                     ((df['high'].shift(-1) - df['low'].shift(1)) / df['close'] * 100 >= min_gap_percent)
        
        # Calculate ATR for validation
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Validate FVG with ATR - relaxed condition
        fvg = (fvg_bullish | fvg_bearish) & (df['high'] - df['low'] > atr * 0.5)  # Reduced ATR requirement
        return fvg

    def check_setup_sequence(self, df):
        """Verify core setup components exist"""
        # Get component signals
        avg_volume = self.calculate_average_volume(df)
        sweeps = self.detect_liquidity_sweep(df, avg_volume)
        fvg = self.detect_fvg(df)
        
        # Just check if we have both sweep and FVG within the last 20 candles
        recent_sweeps = sweeps.iloc[-20:].any()
        recent_fvg = fvg.iloc[-20:].any()
        
        return recent_sweeps and recent_fvg

    def check_timeframe_alignment(self, symbol):
        """Check alignment between 4h and 15m timeframes"""
        # Get data for both timeframes
        df_4h = self.get_klines(symbol, '4h', 200)
        df_15m = self.get_klines(symbol, '15m', 100)
        
        if df_4h is None or df_15m is None:
            return False
            
        # Check if recent 15m candles confirm 4h trend
        last_4h_candle = df_4h.iloc[-1]
        recent_15m = df_15m.tail(3)  # Last 3 15m candles
        
        # Relaxed trend alignment - only need 2 out of 3 candles to align
        if last_4h_candle['close'] > last_4h_candle['open']:  # Bullish 4h
            aligned_candles = sum(1 for _, c in recent_15m.iterrows() if c['close'] > c['open'])
            return aligned_candles >= 2
        else:  # Bearish 4h
            aligned_candles = sum(1 for _, c in recent_15m.iterrows() if c['close'] < c['open'])
            return aligned_candles >= 2

    def scan_for_setups(self):
        """Main scanning function focusing on core components"""
        valid_pairs = self.get_valid_pairs()
        potential_setups = []
        
        for symbol in valid_pairs:
            try:
                print(f"\nAnalyzing {symbol}...")
                
                # Get 4h data (primary timeframe)
                df_4h = self.get_klines(symbol, '4h', 200)
                if df_4h is None:
                    continue
                
                # Calculate indicators
                df_4h['rsi'] = ta.momentum.RSIIndicator(df_4h['close']).rsi()
                
                # Check for core setup components
                has_setup = self.check_setup_sequence(df_4h)
                
                if not has_setup:
                    continue
                
                # RSI criteria - slightly relaxed
                rsi = df_4h['rsi'].iloc[-1]
                rsi_extreme = (rsi < 37) or (rsi > 63)  # Relaxed from 35/65
                
                # Volume criteria
                avg_volume = self.calculate_average_volume(df_4h)
                recent_volume = df_4h['volume'].iloc[-1]
                volume_ratio = recent_volume / avg_volume.iloc[-1]
                
                # Check timeframe alignment
                has_tf_alignment = self.check_timeframe_alignment(symbol)
                
                # Accept setups with core components and either RSI extreme or timeframe alignment
                if has_setup and (rsi_extreme or has_tf_alignment):
                    setup = {
                        'symbol': symbol,
                        'current_price': df_4h['close'].iloc[-1],
                        'volume_ratio': volume_ratio,
                        'rsi': rsi,
                        'timeframe_aligned': has_tf_alignment,
                        'setup_quality': 'A+' if (rsi_extreme and has_tf_alignment) else 'A'
                    }
                    potential_setups.append(setup)
                    print(f"Found potential setup for {symbol} (Quality: {setup['setup_quality']})")
                    
            except Exception as e:
                print(f"Error scanning {symbol}: {str(e)}")
                continue
        
        return pd.DataFrame(potential_setups)

    def get_setup_details(self, symbol):
        """Get detailed analysis for a specific setup"""
        # Get data for both timeframes
        df_4h = self.get_klines(symbol, '4h', 200)
        df_15m = self.get_klines(symbol, '15m', 100)
        
        if df_4h is None:
            return None
            
        # Calculate additional metrics
        df_4h['atr'] = ta.volatility.AverageTrueRange(df_4h['high'], df_4h['low'], df_4h['close']).average_true_range()
        df_4h['rsi'] = ta.momentum.RSIIndicator(df_4h['close']).rsi()
        
        # Add 15m confirmation if available
        if df_15m is not None:
            df_4h['15m_aligned'] = self.check_timeframe_alignment(symbol)
        
        return df_4h

def main():
    try:
        # Initialize scanner
        scanner = ForeverModelScanner()
        
        print("Scanning for potential setups...")
        setups = scanner.scan_for_setups()
        
        if setups.empty:
            print("No setups found matching criteria")
            return
        
        # Sort by setup quality and volume ratio
        setups_sorted = setups.sort_values(['setup_quality', 'volume_ratio'], ascending=[False, False])
        
        # Create output directory if it doesn't exist
        os.makedirs('screener-results', exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to CSV
        csv_filename = f'screener-results/setups_{timestamp}.csv'
        setups_sorted.to_csv(csv_filename, index=False)
        
        # Create a simplified list of symbols for trade_pair.txt
        symbols_list = setups_sorted['symbol'].tolist()
        symbols_filename = 'trade_pair.txt'
        with open(symbols_filename, 'w') as f:
            f.write('\n'.join(symbols_list))
        
        # Print summary to console
        print("\nTop potential setups found:")
        print(setups_sorted.to_string())
        print(f"\nDetailed results saved to: {csv_filename}")
        print(f"Symbols list saved to: {symbols_filename}")
        
        # Get detailed analysis for top setup
        if not setups_sorted.empty:
            top_symbol = setups_sorted.iloc[0]['symbol']
            print(f"\nDetailed analysis for top symbol {top_symbol}:")
            details = scanner.get_setup_details(top_symbol)
            
            if details is not None:
                # Save detailed analysis to JSON
                details_dict = details.tail().to_dict('records')
                json_filename = f'screener-results/detailed_analysis_{top_symbol}_{timestamp}.json'
                with open(json_filename, 'w') as f:
                    json.dump(details_dict, f, indent=2)
                print(f"Detailed analysis saved to: {json_filename}")
            
    except Exception as e:
        print(f"Error running scanner: {str(e)}")

if __name__ == "__main__":
    main()