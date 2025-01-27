from datetime import datetime
import os
import requests
from dotenv import load_dotenv
import json
import csv
import re

class BinanceClient:
    def __init__(self):
        load_dotenv()
        self.base_url = os.getenv('BINANCE_BASE_URL')
        
    def get_ohlc_data(self, symbol, interval, limit=100):
        """Retrieve OHLC data from Binance"""
        endpoint = "/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        return self._send_request(endpoint, params)
    
    def _send_request(self, endpoint, params=None):
        """Send public API request"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return self._parse_ohlc(response.json())
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    def _parse_ohlc(self, data):
        """Parse raw OHLC data into structured format"""
        return [{
            'timestamp': datetime.fromtimestamp(d[0]/1000).strftime('%Y-%m-%d %H:%M:%S'),
            'open': float(d[1]),
            'high': float(d[2]),
            'low': float(d[3]),
            'close': float(d[4]),
            'volume': float(d[5])
        } for d in data]

    def get_multi_timeframe_data(self, symbol, intervals):
        """Retrieve OHLC data for multiple timeframes"""
        data = {}
        for interval, limit in intervals.items():
            endpoint = "/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            data[interval] = self._send_request(endpoint, params)
        return data

    def save_trade_to_csv(self, symbol, analysis_text):
        """Save trade analysis with clean text formatting"""
        try:
            # Remove emojis and special characters
            clean_text = re.sub(r'[^\x00-\x7F]+', '', analysis_text)
            
            # Determine trade status
            trade_status = "TRADE" if "SETUP" in clean_text else "NO TRADE"
            
            # Parse details only for valid trades
            entry_match = sl_match = tp1_match = tp2_match = confidence_match = None
            if trade_status == "TRADE":
                entry_match = re.search(r"Entry Zone: ([\d\.]+)-([\d\.]+)", clean_text)
                sl_match = re.search(r"Stop Loss: ([\d\.]+)", clean_text)
                tp1_match = re.search(r"TP1: ([\d\.]+)", clean_text)
                tp2_match = re.search(r"TP2: ([\d\.]+)", clean_text)
                confidence_match = re.search(r"Confidence: (\d/5)", clean_text)

            # Create trades directory if needed
            os.makedirs('trades', exist_ok=True)
            
            # CSV file path
            csv_path = f"trades/{symbol}.csv"
            
            # Trade data structure
            trade_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'symbol': symbol,
                'status': trade_status,
                'entry_low': entry_match.group(1) if entry_match else '',
                'entry_high': entry_match.group(2) if entry_match else '',
                'sl': sl_match.group(1) if sl_match else '',
                'tp1': tp1_match.group(1) if tp1_match else '',
                'tp2': tp2_match.group(1) if tp2_match else '',
                'confidence': confidence_match.group(1) if confidence_match else '',
                'analysis': clean_text[:500]  # Store first 500 chars of clean text
            }
            
            # Write/append to CSV
            file_exists = os.path.exists(csv_path)
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=trade_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(trade_data)
                
            print(f"Logged {trade_status} to {csv_path}")
            
        except Exception as e:
            print(f"Error saving trade: {str(e)}")

def send_to_deepseek(data, symbol):
    """Send formatted OHLC data to DeepSeek API using the reasoning model"""
    load_dotenv()
    api_key = os.getenv('DEEPSEEK_API_KEY')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, 'deepseek_prompt.txt')
        
        # Read prompt from file
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
            
        # Formatting multi-timeframe data
        formatted = []
        for tf, candles in data.items():
            tf_data = "\n".join([f"{c['timestamp']} | O:{c['open']} H:{c['high']} L:{c['low']} C:{c['close']} V:{c['volume']}"
                                for c in candles])
            formatted.append(f"{tf} Timeframe:\n{tf_data}")
        
        data_str = "\n\n".join(formatted)
        
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {
                    "role": "system", 
                    "content": system_prompt  # Use file content here
                },
                {
                    "role": "user",
                    "content": f"Here's the latest OHLC data for trade analysis:\n{data_str}\nPlease provide your analysis:"
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        # Update to correct API endpoint
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        
        # Save response
        response_data = response.json()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        response_filename = f"deepseek_response_{timestamp}.json"
        
        with open(response_filename, 'w') as f:
            json.dump(response_data, f, indent=2)
            
        # Also save text version
        analysis_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', 'No analysis content')
        
        BinanceClient().save_trade_to_csv(symbol, analysis_text)
            
        return response_data
        
    except Exception as e:
        print(f"DeepSeek API error: {str(e)[:200]}")  # Truncate long errors
        return None

def get_trading_pair():
    """Read trading pair from text file"""
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'trade_pair.txt')
        with open(file_path, 'r') as f:
            for line in f:
                clean_line = line.split('#')[0].strip()  # Handle inline comments
                if clean_line:
                    if not clean_line.isalpha():
                        print(f"Error: Invalid characters in pair '{clean_line}'")
                        exit(1)
                    return clean_line.upper()
            print("Error: Empty trade_pair.txt")
            exit(1)
    except FileNotFoundError:
        print("Error: Missing binance_data/trade_pair.txt")
        exit(1)

if __name__ == "__main__":
    client = BinanceClient()
    symbol = get_trading_pair()  # Get pair from file
    intervals = {
        '1d': 90,   # 3 months daily (covers quarterly institutional cycles)
        '4h': 180,  # 30 days of 4h (720 hours - captures full market rotations)
        '1h': 168   # 7 days of 1h (full week of intraday liquidity patterns)
    }
    multi_data = client.get_multi_timeframe_data(symbol, intervals)  # Use dynamic symbol
    if multi_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n=== DEBUG ===")
        print(f"1. Using trading pair: '{symbol}'")
        print(f"2. Type of symbol: {type(symbol)}")
        print(f"3. Time intervals: {intervals}\n")
        for timeframe, data in multi_data.items():
            filename = f"ohlc_{symbol}_{timeframe}_{timestamp}.json"
            print(f"4. Generating filename: {filename}")
            with open(filename, 'w') as f:
                json.dump(data, f, default=str, indent=2)
            print(f"Saved {len(data)} {timeframe} candles to {filename}")
        
        response = send_to_deepseek(multi_data, symbol)
        if response:
            print("DeepSeek response:", response)
            # Get symbol from data (first timeframe's first candle)
            symbol = get_trading_pair()  # Get symbol from trade_pair.txt
            BinanceClient().save_trade_to_csv(symbol, response.get('choices', [{}])[0].get('message', {}).get('content', 'No analysis content')) 