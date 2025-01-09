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

def close_existing_position(client, symbol):
    """Close any existing position for the given symbol"""
    try:
        # Get current position information
        position_info = client.get_position_risk(symbol=symbol)
        if not position_info:
            return False
            
        position = position_info[0]
        position_amt = float(position['positionAmt'])
        
        if position_amt == 0:
            return False
            
        # Determine side based on position amount
        side = 'SELL' if position_amt > 0 else 'BUY'
        quantity = abs(position_amt)
        
        # Place market order to close position
        close_order = client.new_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        logging.info(f"Closed existing position: {close_order}")
        print(f"Closed existing position: {close_order}")
        return True
        
    except Exception as e:
        logging.error(f"Error closing position: {e}")
        print(f"Error closing position: {e}")
        raise

def main():
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

    # Calculate date range
    start_time = datetime.datetime.fromtimestamp(data[0][0]/1000)
    end_time = datetime.datetime.fromtimestamp(data[-1][0]/1000)
    date_range = f"{start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}"
    logging.info(f"Fetched data range: {date_range}")
    print(f"Fetched data range: {date_range}")

    # Convert data to a table string
    table = tabulate(ohlc_data, headers='keys', tablefmt='grid')

    # Set up OpenAI client for DeepSeek API
    client = OpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url='https://api.deepseek.com'
    )

    try:
        # Read system message from file
        with open('deepseek_prompt.txt', 'r') as f:
            system_message = f.read().strip()
    except Exception as e:
        logging.error(f"Error reading deepseek_prompt.txt: {e}")
        raise

    # Prepare messages with clear JSON formatting instructions
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"""Analyze this BTCUSDT OHLC data and provide a trade strategy.
        Requirements:
        1. Based on the price data, determine if this should be a LONG or SHORT trade
        2. If there is no suitable trade, do not place a trade.
        
        Data:
        {table}
        
        Respond with ONLY the raw JSON object - no markdown, no code blocks."""}
    ]

    try:
        # Send chat completion request
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
        
        # Save DeepSeek response to file
        with open('deepseek_response.txt', 'w') as f:
            f.write(ds_response)
        logging.info("Saved DeepSeek response to deepseek_response.txt")
        print("Saved DeepSeek response to deepseek_response.txt")
        
        # Parse the JSON response
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

        # Set up Binance Futures testnet API connection
        client_binance = UMFutures(
            key=os.getenv('BINANCE_API_KEY'),
            secret=os.getenv('BINANCE_API_SECRET'),
            base_url='https://testnet.binancefuture.com'
        )
        
        # Get symbol precision from Binance
        exchange_info = client_binance.exchange_info()
        symbol_info = next(
            (s for s in exchange_info['symbols'] if s['symbol'] == 'BTCUSDT'),
            None
        )
        
        if not symbol_info:
            raise ValueError("Could not get BTCUSDT symbol info")
            
        # Get quantity precision
        quantity_precision = next(
            f['stepSize'] for f in symbol_info['filters'] 
            if f['filterType'] == 'LOT_SIZE'
        ).split('.')[1].find('1') + 1
        
        # Get price precision
        price_precision = next(
            f['tickSize'] for f in symbol_info['filters']
            if f['filterType'] == 'PRICE_FILTER'
        ).split('.')[1].find('1') + 1
        
        # Calculate initial quantity with proper precision
        quantity = round(position_size / entry, quantity_precision)
        
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
            quantity = round(min_quantity_needed * buffer, quantity_precision)
            
            # Ensure quantity meets minimum even after rounding
            if quantity * latest_close_price < 100:
                quantity = round(100 / latest_close_price, quantity_precision) + (10 ** -quantity_precision)

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

        # Validate and adjust stop-loss with proper precision
        min_stop_distance = entry * 0.005  # Minimum 0.5% distance from entry
        if abs(entry - stop_loss) < min_stop_distance:
            stop_loss = entry - min_stop_distance if direction == 'LONG' else entry + min_stop_distance
            stop_loss = round(stop_loss, price_precision)
            warning_msg = f"Stop-loss adjusted to maintain minimum distance from entry price"
            logging.warning(warning_msg)
            print(warning_msg)
            
        # Round all prices to required precision
        entry = round(entry, price_precision)
        stop_loss = round(stop_loss, price_precision)
        exit_level = round(exit_level, price_precision)

        # Log the quantity and notional value
        logging.info(f"Minimum quantity needed: {min_quantity_needed} BTC")
        logging.info(f"Calculated quantity: {quantity} BTC")
        logging.info(f"Notional value: {notional_value} USDT")
        logging.info(f"Adjusted stop-loss: {stop_loss}")
        print(f"Minimum quantity needed: {min_quantity_needed} BTC")
        print(f"Calculated quantity: {quantity} BTC")
        print(f"Notional value: {notional_value} USDT")
        print(f"Adjusted stop-loss: {stop_loss}")

        # Set leverage first
        try:
            client_binance.change_leverage(symbol='BTCUSDT', leverage=LEVERAGE)
            print(f"Leverage set to {LEVERAGE}x")
        except ClientError as e:
            logging.error(f"Error setting leverage: {e}")
            print(f"Error setting leverage: {e}")
            raise

        # Close any existing position before placing new trade
        if close_existing_position(client_binance, 'BTCUSDT'):
            logging.info("Successfully closed existing position")
            print("Successfully closed existing position")
        else:
            logging.info("No existing position to close")
            print("No existing position to close")

        # Define order parameters
        symbol = 'BTCUSDT'
        side = 'BUY' if direction == 'LONG' else 'SELL'
        type_order = 'MARKET'

        # Place market order
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
            side='SELL' if side == 'BUY' else 'BUY',
            type='TAKE_PROFIT_MARKET',
            stopPrice=exit_level,
            quantity=quantity,
            timeInForce='GTC'  # Good Till Cancel
        )
        logging.info(f"Take-profit order placed: {tp_order_response}")
        print(f"Take-profit order placed: {tp_order_response}")

    except Exception as e:
        logging.error(f"Error in trading logic or calculations: {e}")
        print(f"Error in trading logic or calculations: {e}")
        raise

if __name__ == "__main__":
    main()