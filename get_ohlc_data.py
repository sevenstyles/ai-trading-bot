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
from binance_client import initialize_binance_client, get_symbol_info, place_orders
from precision import PrecisionHandler
from strategy_handler import get_trading_strategy
from utils import validate_price_levels, calculate_position_size

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
    try:
        # Initialize Binance Futures client
        client = initialize_binance_client()

        # Get symbol information and create precision handler
        symbol_info = get_symbol_info(client, SYMBOL)
        precision_handler = PrecisionHandler(symbol_info)
        logging.info(f"Symbol info - Price precision: {precision_handler.price_precision}, Tick size: {precision_handler.tick_size}")
        logging.info(f"Symbol info - Quantity precision: {precision_handler.quantity_precision}, Step size: {precision_handler.step_size}")

        # Fetch OHLC data
        url = 'https://fapi.binance.com/fapi/v1/klines'
        params = {
            'symbol': SYMBOL,
            'interval': '15m',
            'limit': 100
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Parse OHLC data
        ohlc_data = [{
            'Open Time': datetime.datetime.fromtimestamp(candle[0]/1000).strftime('%Y-%m-%d %H:%M:%S'),
            'Open': float(candle[1]),
            'High': float(candle[2]),
            'Low': float(candle[3]),
            'Close': float(candle[4]),
            'Volume': float(candle[5])
        } for candle in data]

        # Export OHLC data to JSON file
        export_filename = f"ohlc_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(export_filename, 'w') as f:
            json.dump(ohlc_data, f, indent=2)
        logging.info(f"Exported OHLC data to {export_filename}")

        current_price = float(ohlc_data[-1]['Close'])
        logging.info(f"Current price: {current_price}")

        # Get trading strategy
        trade_strategy = get_trading_strategy(ohlc_data)
        logging.info(f"Trade strategy received: {trade_strategy}")
        print(f"Current price: {current_price}")
        print(f"Strategy details:")
        print(f"  Direction: {trade_strategy['direction']}")
        print(f"  Entry: {trade_strategy['entry']}")
        print(f"  Stop-loss: {trade_strategy['stop_loss']}")
        print(f"  Take-profit: {trade_strategy['exit']}")
        print(f"  Reasoning: {trade_strategy['reasoning']}")

        # Check if trade is recommended
        if isinstance(trade_strategy, dict) and not trade_strategy.get('trade', True):
            logging.info(f"No trade recommended: {trade_strategy.get('reasoning')}")
            print(f"No trade recommended: {trade_strategy.get('reasoning')}")
            return

        # Calculate and validate position size
        quantity = calculate_position_size(
            current_price,
            float(trade_strategy['entry']),
            float(trade_strategy['stop_loss']),
            precision_handler,
            ACCOUNT_BALANCE,
            RISK_PERCENTAGE,
            LEVERAGE,
            MIN_NOTIONAL
        )

        # Validate and normalize price levels
        exit_level, stop_loss = validate_price_levels(
            current_price,
            float(trade_strategy['entry']),
            float(trade_strategy['exit']),
            float(trade_strategy['stop_loss']),
            trade_strategy['direction'],
            precision_handler
        )

        # Double-check notional value before placing orders
        if not precision_handler.check_notional(quantity, current_price):
            raise ValueError(f"Order notional value {quantity * current_price:.2f} USDT is below minimum {MIN_NOTIONAL} USDT")

        # Place orders
        orders = place_orders(
            client,
            SYMBOL,
            trade_strategy['direction'],
            quantity,
            current_price,
            stop_loss,
            exit_level,
            LEVERAGE
        )
        
        logging.info("All orders placed successfully")
        print("All orders placed successfully")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
