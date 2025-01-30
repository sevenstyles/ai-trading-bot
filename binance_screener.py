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

    def get_klines(self, symbol, interval, limit=100):
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
        """Detect potential liquidity sweeps"""
        volume_spikes = df['volume'] > (avg_volume * 2.5)
        high_wicks = (df['high'] - df['close']) > (df['high'] - df['low']) * 0.6
        low_wicks = (df['close'] - df['low']) > (df['high'] - df['low']) * 0.6
        
        return volume_spikes & (high_wicks | low_wicks)

    def detect_fvg(self, df, min_gap_percent=1.2):
        """Detect Fair Value Gaps"""
        fvg_bullish = (df['low'].shift(1) > df['high'].shift(-1)) & \
                     ((df['low'].shift(1) - df['high'].shift(-1)) / df['close'] * 100 >= min_gap_percent)
        
        fvg_bearish = (df['high'].shift(-1) > df['low'].shift(1)) & \
                     ((df['high'].shift(-1) - df['low'].shift(1)) / df['close'] * 100 >= min_gap_percent)
        
        return fvg_bullish | fvg_bearish

    def scan_for_setups(self, timeframe='4h', limit=100):
        """Main scanning function"""
        valid_pairs = self.get_valid_pairs()
        potential_setups = []
        
        for symbol in valid_pairs:
            try:
                print(f"\nAnalyzing {symbol}...")
                
                # Get klines data
                df = self.get_klines(symbol, timeframe, limit)
                if df is None:
                    continue
                
                # Calculate indicators
                avg_volume = self.calculate_average_volume(df)
                df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
                
                # Detect potential setups
                has_liquidity_sweep = self.detect_liquidity_sweep(df, avg_volume).any()
                has_fvg = self.detect_fvg(df).any()
                
                # Volume criteria
                recent_volume = df['volume'].iloc[-1]
                avg_vol_last = avg_volume.iloc[-1]
                recent_volume_increase = recent_volume > avg_vol_last * 1.8
                
                # RSI criteria (oversold or overbought)
                rsi = df['rsi'].iloc[-1]
                rsi_extreme = (rsi < 35) or (rsi > 65)
                
                # If meeting initial criteria, add to potential setups
                if has_liquidity_sweep and has_fvg and recent_volume_increase and rsi_extreme:
                    setup = {
                        'symbol': symbol,
                        'current_price': df['close'].iloc[-1],
                        'volume_ratio': recent_volume / avg_vol_last,
                        'rsi': rsi,
                        'has_liquidity_sweep': has_liquidity_sweep,
                        'has_fvg': has_fvg
                    }
                    potential_setups.append(setup)
                    print(f"Found potential setup for {symbol}")
                    
            except Exception as e:
                print(f"Error scanning {symbol}: {str(e)}")
                continue
        
        return pd.DataFrame(potential_setups)

    def get_setup_details(self, symbol, timeframe='4h'):
        """Get detailed analysis for a specific setup"""
        df = self.get_klines(symbol, timeframe, limit=100)
        if df is None:
            return None
            
        # Calculate additional metrics for detailed analysis
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        return df

def main():
    try:
        # Initialize scanner
        scanner = ForeverModelScanner()
        
        print("Scanning for potential setups...")
        setups = scanner.scan_for_setups()
        
        if setups.empty:
            print("No setups found matching criteria")
            return
        
        # Sort by volume ratio to prioritize most active setups
        setups_sorted = setups.sort_values('volume_ratio', ascending=False)
        
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