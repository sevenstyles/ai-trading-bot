from datetime import datetime
import os
import requests
from dotenv import load_dotenv
import json

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

def send_to_deepseek(data):
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
        with open(prompt_path, 'r') as f:
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
                    "content": f"Here's the latest BTC/USDT hourly data:\n{data_str}\nPlease provide your analysis:"
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
        text_filename = f"deepseek_analysis_{timestamp}.txt"
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(f"DeepSeek Analysis ({timestamp})\n")
            f.write("="*40 + "\n\n")
            f.write(analysis_text)
            
        print(f"Saved analysis to {text_filename}")
            
        return response_data
        
    except Exception as e:
        print(f"DeepSeek API error: {str(e)[:200]}")  # Truncate long errors
        return None

if __name__ == "__main__":
    client = BinanceClient()
    intervals = {
        '1d': 30,   # 30 daily candles (~1 month)
        '1h': 100,  # 100 hourly candles (~4 days)
        '15m': 50   # 50 15-min candles (~12 hours)
    }
    multi_data = client.get_multi_timeframe_data("BTCUSDT", intervals)
    if multi_data:
        response = send_to_deepseek(multi_data)
        if response:
            print("DeepSeek response:", response) 