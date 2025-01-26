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

# Remove the file logging configuration
# logging.basicConfig(
#     level=logging.INFO,
#     filename='trading_bot.log',
#     filemode='a',
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

def main():
    client = BinanceClient()
    print("Fetching OHLC data...")
    
    # Read trading pair from file
    with open('trade_pair.txt', 'r') as f:
        symbol = f.read().strip().upper()  # Read and sanitize the symbol
    
    # Update the interval values here
    intervals = {
        '1d': 90,    # 3 months daily (covers quarterly institutional cycles)
        '4h': 180,   # 30 days of 4h (720 hours - captures full market rotations)
        '1h': 168    # 7 days of 1h (full week of intraday liquidity patterns)
    }
    
    # Get multi-timeframe data using the symbol from file
    ohlc_data = client.get_multi_timeframe_data(symbol, intervals)  # Changed from "BTCUSDT"
    
    if ohlc_data:
        # Save each timeframe's data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for timeframe, data in ohlc_data.items():
            filename = f"ohlc_{symbol}_{timeframe}_{timestamp}.json"  # Use actual symbol variable
            with open(filename, 'w') as f:
                json.dump(data, f, default=str, indent=2)
            print(f"Saved {len(data)} {timeframe} candles to {filename}")
        
        # Send to DeepSeek
        deepseek_response = send_to_deepseek(ohlc_data)
        if deepseek_response:
            print("Received DeepSeek analysis")
    else:
        print("Failed to retrieve data")

if __name__ == "__main__":
    main()
