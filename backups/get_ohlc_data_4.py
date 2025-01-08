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
    'limit': 500
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
    1. Based on the recent price action, determine if this should be a LONG or SHORT trade
    2. If there is no suitable trade, do not place a trade.
    
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
        
        # Get the latest closing price for calculations
        latest_close_price = ohlc_data[-1]['Close']
        
        # Calculate minimum quantity needed for 100 USDT notional value
        if latest_close_price > 0:  # Prevent division by zero
            min_quantity_needed = 100 / latest_close_price
            min_quantity_needed = round(min_quantity_needed, 3)
        else:
            min_quantity_needed = MIN_QUANTITY  # Fallback to MIN_QUANTITY if price is 0

        # Calculate position size based on risk parameters
        risk_amount = ACCOUNT_BALANCE * RISK_PERCENTAGE
        stop_loss_distance = abs(entry - stop_loss)
        position_size = (risk_amount / stop_loss_distance) * LEVERAGE
        
        # Calculate initial quantity (BTC amount)
        quantity = round(position_size / entry, 3)
        
        # If initial quantity is below minimum, adjust risk parameters
        if quantity < min_quantity_needed:
            warning_msg = f"Initial quantity {quantity} BTC is below minimum {min_quantity_needed} BTC - adjusting to meet Binance requirements"
            logging.warning(warning_msg)
            print(warning_msg)
            
            # Calculate required risk amount to meet minimum quantity with 1% buffer
            buffer = 1.01  # 1% buffer to ensure we exceed minimum
            required_risk = (min_quantity_needed * entry * stop_loss_distance * buffer) / LEVERAGE
            risk_percentage = required_risk / ACCOUNT_BALANCE
            
            # Update quantity to meet minimum with buffer
            quantity = round(min_quantity_needed * buffer, 3)
            
            # Ensure quantity meets minimum even after rounding
            if quantity * latest_close_price < 100:
                quantity = round(100 / latest_close_price, 3) + 0.001
            
            # Log the adjusted risk parameters
            logging.info(f"Adjusted risk percentage: {risk_percentage:.4f}")
            print(f"Adjusted risk percentage: {risk_percentage:.4f}")
        
        # Calculate actual notional value
        notional_value = quantity * latest_close_price
        
        # Validate notional value meets Binance requirements
        if notional_value < 100:
            error_msg = f"Notional value {notional_value} USDT is below minimum 100 USDT requirement"
            logging.error(error_msg)
            print(error_msg)
            raise ValueError(error_msg)

        # Log the quantity and notional value
        logging.info(f"Minimum quantity needed: {min_quantity_needed} BTC")
        logging.info(f"Calculated quantity: {quantity} BTC")
        logging.info(f"Notional value: {notional_value} USDT")
        print(f"Minimum quantity needed: {min_quantity_needed} BTC")
        print(f"Calculated quantity: {quantity} BTC")
        print(f"Notional value: {notional_value} USDT")
        
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
                try:
                    tp_order_response = client_binance.new_order(
                        symbol=symbol,
                        side='SELL' if side == 'BUY' else 'BUY',
                        type='TAKE_PROFIT_MARKET',
                        stopPrice=exit_level,
                        quantity=quantity,
                        timeInForce='GTC'  # Good Till Cancel
                    )
                    logging.info(f"Take-profit order placed: {tp_order_response}")
                    print(f"Take-profit order placed: {tp_order_response}")
                
                except ClientError as e:
                    logging.error(f"Binance API error placing take-profit order: {e}")
                    print(f"Binance API error placing take-profit order: {e}")
                    raise
                
                except Exception as e:
                    logging.error(f"Unexpected error placing take-profit order: {e}")
                    print(f"Unexpected error placing take-profit order: {e}")
                    raise
                    
            except Exception as e:
                logging.error(f"Error placing stop-loss order: {e}")
                print(f"Error placing stop-loss order: {e}")
                raise
                
        except Exception as e:
            logging.error(f"Error placing market order: {e}")
            print(f"Error placing market order: {e}")
            raise
            
    except Exception as e:
        logging.error(f"Error parsing trade strategy: {e}")
        print(f"Error parsing trade strategy: {e}")
        raise
        
except Exception as e:
    logging.error(f"Error in main trading logic: {e}")
    print(f"Error in main trading logic: {e}")
    raise
