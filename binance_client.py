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
        
    def _read_end_date(self):
        """Read and parse human-readable END date from text file"""
        try:
            file_path = os.path.join(os.path.dirname(__file__), 'ohlc_data_end_date.txt')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.split('#')[0].strip()
                        if line:
                            for fmt in ('%Y-%m-%d %H:%M:%S', 
                                      '%Y-%m-%d %H:%M',
                                      '%Y-%m-%d'):
                                try:
                                    dt = datetime.strptime(line, fmt)
                                    return int(dt.timestamp() * 1000)
                                except ValueError:
                                    continue
                            print(f"Warning: Couldn't parse date: {line}")
            return None
        except Exception as e:
            print(f"Error reading end date: {e}")
            return None

    def get_ohlc_data(self, symbol, interval, limit=100):
        """Retrieve historical OHLC data from Binance"""
        endpoint = "/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        # Add END time if specified
        end_time = self._read_end_date()
        if end_time:
            params['endTime'] = end_time
            human_time = datetime.fromtimestamp(end_time/1000).strftime('%Y-%m-%d %H:%M:%S')
            print(f"\nDEBUG: Fetching {limit} candles ending at {human_time} (UTC)")

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
        """Retrieve historical OHLC data for multiple timeframes"""
        data = {}
        end_time = self._read_end_date()
        end_suffix = ""
        
        if end_time:
            end_suffix = f"_until_{datetime.fromtimestamp(end_time/1000).strftime('%Y%m%d')}"
            
        for interval, limit in intervals.items():
            endpoint = "/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            if end_time:
                params['endTime'] = end_time
                
            data[interval] = self._send_request(endpoint, params)
            
        return data, end_suffix

    def save_trade_to_csv(self, symbol, analysis_text):
        """Save trade analysis without symbol column"""
        try:
            # Clean and parse analysis text
            clean_text = re.sub(r'[^\x00-\x7F]+', '', analysis_text)
            
            # Parse fields with END RESPONSE delimiter
            analysis_match = re.search(
                r"Analysis: (.*?)\n?END RESPONSE",
                clean_text, 
                re.DOTALL | re.IGNORECASE
            )
            
            # Extract other fields
            timestamp_match = re.search(r"Timestamp: (.+)", clean_text)
            status_match = re.search(r"Status: (TRADE|NO TRADE)", clean_text)
            direction_match = re.search(r"Direction: (LONG|SHORT|NO TRADE)", clean_text)
            entry_match = re.search(r"Entry: ([\d\.]+)", clean_text)
            sl_match = re.search(r"SL: ([\d\.]+)", clean_text)
            tp_match = re.search(r"TP: ([\d\.]+)", clean_text)
            rr_match = re.search(r"RR Ratio: (1:[\d\.]+)", clean_text)
            confidence_match = re.search(r"Confidence: ([\d/5]+)", clean_text)

            # Update CSV fieldnames and data
            fieldnames = [
                'timestamp',
                'status', 
                'direction',
                'entry',
                'sl',
                'tp',
                'rr_ratio',
                'confidence',
                'analysis'
            ]

            # Remove symbol from trade_data
            trade_data = {
                'timestamp': timestamp_match.group(1) if timestamp_match else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': status_match.group(1) if status_match else 'NO TRADE',
                'direction': direction_match.group(1) if direction_match else '',
                'entry': entry_match.group(1) if entry_match else '',
                'sl': sl_match.group(1) if sl_match else '',
                'tp': tp_match.group(1) if tp_match else '',
                'rr_ratio': rr_match.group(1) if rr_match else '',
                'confidence': confidence_match.group(1) if confidence_match else '',
                'analysis': analysis_match.group(1).strip()[:500] if analysis_match else clean_text[:500]
            }

            # Remove END RESPONSE from analysis if present
            if analysis_match:
                trade_data['analysis'] = trade_data['analysis'].replace('END RESPONSE', '').strip()

            # Create trades directory if needed
            os.makedirs('trades', exist_ok=True)
            
            # CSV file path
            csv_path = f"trades/{symbol}.csv"
            
            # Write to CSV with strict column order
            file_exists = os.path.exists(csv_path)
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                    
                if all([trade_data['status'], trade_data['direction']]):
                    writer.writerow(trade_data)
                    print(f"Logged {trade_data['status']} to {csv_path}")
                else:
                    print("Invalid format - skipped CSV logging")

        except Exception as e:
            print(f"CSV save error: {str(e)}")

def send_to_deepseek(data, symbol):
    """Send formatted OHLC data to DeepSeek API with full response logging"""
    load_dotenv()
    api_key = os.getenv('DEEPSEEK_API_KEY')
    
    if not api_key:
        print("Error: Missing DEEPSEEK_API_KEY in environment variables")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        # Validate data before sending
        if not data or not any(data.values()):
            print("Error: Empty or invalid OHLC data provided to DeepSeek")
            return None

        # Read prompt template
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, 'deepseek_trade_prompt.txt')
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
            
        # Format data payload with abbreviated fields
        formatted = []
        for tf, candles in data.items():
            if not candles:
                print(f"Warning: Empty candles for {tf} timeframe")
                continue
                
            # Use abbreviated field names: t=timestamp, o=open, h=high, l=low, c=close, v=volume
            tf_data = "\n".join([f"t:{c['timestamp']} o:{c['open']} h:{c['high']} l:{c['low']} c:{c['close']} v:{c['volume']}"
                                for c in candles])
            formatted.append(f"{tf}:\n{tf_data}")

        if not formatted:
            print("Error: No valid data to send to DeepSeek")
            return None
            
        data_str = "\n\n".join(formatted)
        
        # Update payload to enable streaming
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this market data:\n{data_str}"}
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
            "stream": True  # Add this line to enable streaming
        }

        # Create debug directory and file
        os.makedirs('debug', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_filename = f"debug/deepseek_response_full_{timestamp}.txt"
        
        with open(debug_filename, 'w', encoding='utf-8') as debug_file:
            print(f"\nStarting DeepSeek analysis stream (debug log: {debug_filename}):")
            
            # Make streaming request
            with requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=True
            ) as response:
                try:
                    response.raise_for_status()
                    full_response = ""
                    buffer = ""
                    in_analysis = False
                    
                    debug_file.write(f"Request URL: {response.url}\n")
                    debug_file.write(f"Status Code: {response.status_code}\n")
                    debug_file.write(f"Headers: {dict(response.headers)}\n\n")
                    
                    # Process streaming chunks
                    for chunk in response.iter_lines():
                        debug_file.write(f"RAW CHUNK: {chunk}\n")
                        
                        if chunk:
                            decoded_chunk = chunk.decode('utf-8')
                            debug_file.write(f"DECODED: {decoded_chunk}\n")
                            
                            # Handle keep-alive and empty lines
                            if decoded_chunk.startswith(':') or not decoded_chunk.strip():
                                continue
                                
                            # Handle [DONE] marker
                            if decoded_chunk.strip() == 'data: [DONE]':
                                # Process remaining buffer
                                if buffer:
                                    full_response += buffer
                                    print(buffer, end='', flush=True)
                                    buffer = ""
                                print("\n\nStream completed successfully")
                                debug_file.write("\n[STREAM COMPLETED]\n")
                                continue
                                
                            if decoded_chunk.startswith("data: "):
                                try:
                                    json_chunk = json.loads(decoded_chunk[6:])
                                    debug_file.write(f"PARSED JSON: {json_chunk}\n")
                                    
                                    # Check for API errors
                                    if 'error' in json_chunk:
                                        error_msg = f"\nAPI Error: {json_chunk['error'].get('message', 'Unknown error')}"
                                        print(error_msg)
                                        debug_file.write(f"ERROR: {error_msg}\n")
                                        return None
                                        
                                    content = json_chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                    if content:
                                        buffer += content
                                        debug_file.write(f"CONTENT BUFFER: {buffer}\n")
                                        
                                        if "Analysis:" in buffer and not in_analysis:
                                            in_analysis = True
                                            print("Analysis: ", end='', flush=True)
                                            debug_file.write("[ANALYSIS START DETECTED]\n")
                                            
                                        if '\n' in buffer:
                                            parts = buffer.split('\n')
                                            for part in parts[:-1]:
                                                print(part, end='\n', flush=True)
                                                full_response += part + '\n'
                                            buffer = parts[-1]
                                            debug_file.write(f"FLUSHED BUFFER: {parts[:-1]}\n")
                                            
                                except json.JSONDecodeError:
                                    error_msg = f"\nFailed to parse chunk: {decoded_chunk[:100]}"
                                    print(error_msg)
                                    debug_file.write(f"PARSE ERROR: {error_msg}\n")
                            else:
                                error_msg = f"\nUnexpected chunk format: {decoded_chunk[:100]}"
                                print(error_msg)
                                debug_file.write(f"UNEXPECTED CHUNK: {error_msg}\n")
                                
                except Exception as e:
                    error_msg = f"\nStream error: {str(e)}"
                    print(error_msg)
                    debug_file.write(f"STREAM ERROR: {error_msg}\n")
                    return None

            # Save final analysis after stream
            debug_file.write(f"\nFULL RESPONSE: {full_response}\n")
            
            # Save full response after stream completes
            analysis_text = full_response.strip()
            if analysis_text:
                BinanceClient().save_trade_to_csv(symbol, analysis_text)
            else:
                print("\nWarning: Empty analysis content in stream")
            
            return analysis_text
        
    except Exception as e:
        print(f"DeepSeek API Error: {str(e)[:200]}")
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
    symbol = get_trading_pair()
    intervals = {
        '1d': 10,
        '4h': 10,
        '1h': 10
    }
    
    multi_data, end_suffix = client.get_multi_timeframe_data(symbol, intervals)
    
    if multi_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create ohlc-data directory
        os.makedirs('ohlc-data', exist_ok=True)
        
        for timeframe, data in multi_data.items():
            filename = os.path.join('ohlc-data', f"ohlc_{symbol}_{timeframe}{end_suffix}_{timestamp}.json")
            print(f"\nSaving {len(data)} {timeframe} candles to {filename}")
            with open(filename, 'w') as f:
                json.dump(data, f, default=str, indent=2)
        
        response = send_to_deepseek(multi_data, symbol)
        if response:
            print("DeepSeek response:", response)
            # Get symbol from data (first timeframe's first candle)
            symbol = get_trading_pair()  # Get symbol from trade_pair.txt
            BinanceClient().save_trade_to_csv(symbol, response) 