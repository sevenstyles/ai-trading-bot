import requests
import datetime
import json
import logging
import re
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

# Trading parameters
ACCOUNT_BALANCE = 15000  # Total account balance in USDT
RISK_PERCENTAGE = 0.025  # Risk 2.5% per trade
LEVERAGE = 20  # 20x leverage
MIN_QUANTITY = 0.001  # Minimum trade quantity for BTC

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
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url='https://api.deepseek.com'
)

# Prepare messages with clear JSON formatting instructions
messages = [
    {"role": "system", "content": """You are a trading assistant that analyzes OHLC data and provides trading strategies.
    Respond ONLY with a raw JSON object (no markdown, no code blocks) in this format:
    {
        "entry": number,
        "exit": number,
        "stop_loss": number,
        "direction": "LONG" or "SHORT",
        "reasoning": "string explaining the strategy"
    }"""},
    {"role": "user", "content": f"""Analyze this BTCUSDT OHLC data and provide a trade strategy.
    Requirements:
    1. Entry price should be based on the most recent closing price
    2. Stop loss should be 1% away from entry for LONG trades, -1% for SHORT trades
    3. Take profit should be 2% away from entry for LONG trades, -2% for SHORT trades
    4. Based on the recent price action, determine if this should be a LONG or SHORT trade
    
    Data:
    {table}
    
    Respond with ONLY the raw JSON object - no markdown, no code blocks."""}
]

# Send chat completion request
try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.1,
        stream=False
    )
    # Extract DeepSeek's response and clean it
    ds_response = response.choices[0].message.content.strip()
    
    # Clean the response by removing markdown code blocks
    ds_response = re.sub(r'^```json\s*', '', ds_response)  # Remove opening ```json
    ds_response = re.sub(r'\s*```$', '', ds_response)      # Remove closing ```
    
    # Parse the JSON response
    try:
        trade_strategy = json.loads(ds_response)
        entry = float(trade_strategy['entry'])
        exit_level = float(trade_strategy['exit'])
        stop_loss = float(trade_strategy['stop_loss'])
        direction = trade_strategy['direction']
        
        # Log the strategy
        logging.info(f"Trade strategy received: {trade_strategy}")
        print(f"Trade strategy received: {trade_strategy}")
        
        # Calculate position size based on risk parameters
        risk_amount = ACCOUNT_BALANCE * RISK_PERCENTAGE
        stop_loss_distance = abs(entry - stop_loss)
        position_size = (risk_amount / stop_loss_distance) * LEVERAGE
        
        # Calculate and round the quantity (BTC amount)
        quantity = position_size / entry
        
        # Ensure quantity meets minimum requirements and round to 3 decimal places
        quantity = max(round(quantity, 3), MIN_QUANTITY)
        
        # Log the calculated quantity
        logging.info(f"Calculated quantity: {quantity} BTC")
        print(f"Calculated quantity: {quantity} BTC")
        
        # Set up Binance Futures testnet API connection
        client_binance = UMFutures(
            key=os.getenv('BINANCE_API_KEY'),
            secret=os.getenv('BINANCE_API_SECRET'),
            base_url='https://testnet.binancefuture.com'
        )
        
        # Set leverage first
        try:
            client_binance.change_leverage(symbol='BTCUSDT', leverage=LEVERAGE)
            print(f"Leverage set to {LEVERAGE}x")
        except ClientError as e:
            logging.error(f"Error setting leverage: {e}")
            print(f"Error setting leverage: {e}")
            raise
        
        # Define order parameters
        symbol = 'BTCUSDT'
        side = 'BUY' if direction == 'LONG' else 'SELL'
        type_order = 'MARKET'
        
        # Place market order
        try:
            order_response = client_binance.new_order(
                symbol=symbol,
                side=side,
                type=type_order,
                quantity=quantity
            )
            logging.info(f"Order placed: {order_response}")
            print(f"Order placed: {order_response}")
            
            # Place stop-loss order
            stop_side = 'SELL' if side == 'BUY' else 'BUY'
            try:
                stop_order_response = client_binance.new_order(
                    symbol=symbol,
                    side=stop_side,
                    type='STOP_MARKET',
                    stopPrice=stop_loss,
                    quantity=quantity,
                    timeInForce='GTC'  # Good Till Cancel
                )
                logging.info(f"Stop-loss order placed: {stop_order_response}")
                print(f"Stop-loss order placed: {stop_order_response}")
                
                # Place take-profit order
                tp_order_response = client_binance.new_order(
                    symbol=symbol,
                    side=stop_side,
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=exit_level,
                    quantity=quantity,
                    timeInForce='GTC'  # Good Till Cancel
                )
                logging.info(f"Take-profit order placed: {tp_order_response}")
                print(f"Take-profit order placed: {tp_order_response}")
                
            except ClientError as e:
                logging.error(f"Binance API error placing stop orders: {e}")
                print(f"Binance API error placing stop orders: {e}")
                
        except ClientError as e:
            logging.error(f"Binance API error placing market order: {e}")
            print(f"Binance API error placing market order: {e}")
        
    except json.JSONDecodeError as e:
        logging.error(f"Error: DeepSeek response is not in valid JSON format. Response: {ds_response}")
        print(f"Error: DeepSeek response is not in valid JSON format. Response: {ds_response}")
    
except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"An error occurred: {e}")