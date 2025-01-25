from datetime import datetime
import os
import requests
from dotenv import load_dotenv
import json

class BinanceClient:
    def __init__(self):
        load_dotenv()
        self.base_url = os.getenv('BINANCE_BASE_URL')
        
    def get_ohlc_data(self, symbol, interval, limit=20):
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

def send_to_deepseek(data):
    """Send formatted OHLC data to DeepSeek API using the reasoning model"""
    load_dotenv()
    api_key = os.getenv('DEEPSEEK_API_KEY')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        # Format data for DeepSeek's chat API
        data_str = "\n".join([f"{candle['timestamp']} O:{candle['open']} H:{candle['high']} L:{candle['low']} C:{candle['close']}" 
                             for candle in data])
        
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {
                    "role": "system", 
                    "content": "Analyze this OHLC data. Provide technical analysis focusing on key support/resistance levels, trends, and potential trading opportunities."
                },
                {
                    "role": "user",
                    "content": f"Here's the recent OHLC data:\n{data_str}\nPlease provide your analysis:"
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
            
        return response_data
        
    except Exception as e:
        print(f"DeepSeek API error: {str(e)[:200]}")  # Truncate long errors
        return None

if __name__ == "__main__":
    client = BinanceClient()
    ohlc_data = client.get_ohlc_data("BTCUSDT", "1h")
    if ohlc_data:
        response = send_to_deepseek(ohlc_data)
        if response:
            print("DeepSeek response:", response) 