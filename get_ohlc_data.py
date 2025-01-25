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
from precision import PrecisionHandler
from strategy_handler import get_trading_strategy
from utils import validate_price_levels, calculate_position_size
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    filename='trading_bot.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Trading parameters
ACCOUNT_BALANCE = 15000  
RISK_PERCENTAGE = 0.05  
LEVERAGE = 20  
MIN_NOTIONAL = 200  
PRICE_BUFFER = 0.002  
SYMBOL = 'BTCUSDT'  

def main():
    client = BinanceClient()
    print("Fetching OHLC data...")
    ohlc_data = client.get_ohlc_data("BTCUSDT", "1h", 100)
    
    if ohlc_data:
        # Save OHLC data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ohlc_filename = f"ohlc_data_{timestamp}.json"
        with open(ohlc_filename, 'w') as f:
            json.dump(ohlc_data, f, default=str, indent=2)  # Handle datetime serialization
        print(f"Saved {len(ohlc_data)} candles to {ohlc_filename}")
        
        # Send to DeepSeek and save response
        deepseek_response = send_to_deepseek(ohlc_data)
        if deepseek_response:
            print("Received DeepSeek analysis:", deepseek_response.get('summary', 'No summary available'))
    else:
        print("Failed to retrieve data")

if __name__ == "__main__":
    main()
