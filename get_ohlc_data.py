import requests
import datetime
import json
import logging
import re
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv
import os
from openai import OpenAI
from tabulate import tabulate
from binance.um_futures import UMFutures
from binance.error import ClientError
from binance_client import BinanceClient, send_to_deepseek
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

def main():
    client = BinanceClient()
    print("Fetching OHLC data...")
    
    # Read trading pair from file
    with open('trade_pair.txt', 'r') as f:
        symbol = f.read().strip().upper()  # Read and sanitize the symbol
    
    # Update the interval values here
    intervals = {
        '1d': 10,   
        '4h': 10,   
        '1h': 10    
    }
    
    # Get multi-timeframe data using the symbol from file
    ohlc_data, start_suffix = client.get_multi_timeframe_data(symbol, intervals)  # Unpack the tuple
    
    if ohlc_data:
        # Save each timeframe's data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for timeframe, data in ohlc_data.items():
            filename = f"ohlc_{symbol}_{timeframe}{start_suffix}_{timestamp}.json"
            print(f"Saving {len(data)} {timeframe} candles to {filename}")
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        
        # Send to DeepSeek
        deepseek_response = send_to_deepseek(ohlc_data, symbol)
        if deepseek_response:
            print("Received DeepSeek analysis")
    else:
        print("Failed to retrieve data")

if __name__ == "__main__":
    main()
