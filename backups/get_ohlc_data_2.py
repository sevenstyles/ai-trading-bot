import requests
import datetime
import json
import logging
from dotenv import load_dotenv
import os
from openai import OpenAI
from tabulate import tabulate
from binance.um_futures import UMFutures
from binance.error import ClientError

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, filename='trading_bot.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

# Fetch data from Binance API
url = 'https://api.binance.com/api/v3/klines'
params = {
    'symbol': 'BTCUSDT',
    'interval': '15m',
    'limit': 100
}
response = requests.get(url, params=params)
data = response.json()

# Parse and organize data
ohlc_data = []
for candle in data:
    ohlc = {
        'Open Time': datetime.datetime.fromtimestamp(candle[0]/1000).strftime('%Y-%m-%d %H:%M:%S'),
        'Open': float(candle[1]),
        'High': float(candle[2]),
        'Low': float(candle[3]),
        'Close': float(candle[4]),
        'Volume': float(candle[5])
    }
    ohlc_data.append(ohlc)

# Convert data to a table string
table = tabulate(ohlc_data, headers='keys', tablefmt='grid')

# Set up OpenAI client for DeepSeek API
client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),  # Use environment variable
    base_url='https://api.deepseek.com'
)

# Prepare messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"Analyse the following data and provide a trade strategy with entry, exit, and stop-loss levels in JSON format.\n\nData:\n{table}"}
]

# Send chat completion request
try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    # Extract DeepSeek's response
    ds_response = response.choices[0].message.content.strip()
    
    # Parse the JSON response
    try:
        trade_strategy = json.loads(ds_response)
        entry = trade_strategy['entry']
        exit_level = trade_strategy['exit']
        stop_loss = trade_strategy['stop_loss']
        
        # Set up Binance Futures testnet API connection
        client_binance = UMFutures(
            key=os.getenv('BINANCE_API_KEY'),  # Use environment variable
            secret=os.getenv('BINANCE_API_SECRET')  # Use environment variable
        )
        
        # Define order parameters
        symbol = 'BTCUSDT'
        side = 'BUY' if entry > ohlc_data[-1]['Close'] else 'SELL'
        type_order = 'MARKET'
        quantity = 0.001  # Adjust based on your strategy
        
        # Place market order
        try:
            order_response = client_binance.new_order(symbol=symbol, side=side, type=type_order, quantity=quantity)
            logging.info(f"Order placed: {order_response}")
            print(f"Order placed: {order_response}")
        except ClientError as e:
            logging.error(f"Binance API error: {e}")
            print(f"Binance API error: {e}")
        
        # Place stop-loss order
        stop_side = 'SELL' if side == 'BUY' else 'BUY'
        stop_type = 'STOP_MARKET'
        stop_price = stop_loss
        try:
            stop_order_response = client_binance.new_order(symbol=symbol, side=stop_side, type=stop_type, stopPrice=stop_price, quantity=quantity)
            logging.info(f"Stop-loss order placed: {stop_order_response}")
            print(f"Stop-loss order placed: {stop_order_response}")
        except ClientError as e:
            logging.error(f"Binance API error: {e}")
            print(f"Binance API error: {e}")
        
    except json.JSONDecodeError:
        logging.error("Error: DeepSeek response is not in valid JSON format.")
        print("Error: DeepSeek response is not in valid JSON format.")
    
except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"An error occurred: {e}")