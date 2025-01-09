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
ACCOUNT_BALANCE = 15000  # Total account balance in USDT
RISK_PERCENTAGE = 0.05  # Risk 2.5% per trade
LEVERAGE = 20  # 20x leverage
MIN_NOTIONAL = 200  # Minimum notional value in USDT
PRICE_BUFFER = 0.002  # 0.2% buffer for take-profit and stop-loss
SYMBOL = 'BTCUSDT'  # Trading symbol

class PrecisionHandler:
    def __init__(self, symbol_info):
        self.symbol_info = symbol_info
        self.quantity_precision = self._get_quantity_precision()
        self.price_precision = self._get_price_precision()
        self.tick_size = self._get_tick_size()
        self.step_size = self._get_step_size()
        self.min_notional = self._get_min_notional()

    def _get_filter(self, filter_type):
        return next(
            (f for f in self.symbol_info['filters'] if f['filterType'] == filter_type),
            None
        )

    def _get_quantity_precision(self):
        lot_size = self._get_filter('LOT_SIZE')
        return str(lot_size['stepSize']).rstrip('0').find('.')

    def _get_price_precision(self):
        price_filter = self._get_filter('PRICE_FILTER')
        return str(price_filter['tickSize']).rstrip('0').find('.')

    def _get_tick_size(self):
        price_filter = self._get_filter('PRICE_FILTER')
        return float(price_filter['tickSize'])

    def _get_step_size(self):
        lot_size = self._get_filter('LOT_SIZE')
        return float(lot_size['stepSize'])

    def _get_min_notional(self):
        notional = self._get_filter('MIN_NOTIONAL')
        return float(notional['notional']) if notional else 100.0

    def normalize_quantity(self, quantity):
        """Normalize quantity according to symbol's quantity precision"""
        step_size = Decimal(str(self.step_size))
        quantity = Decimal(str(quantity))
        normalized = float(
            (quantity / step_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * step_size
        )
        return normalized

    def normalize_price(self, price):
        """Normalize price according to symbol's price precision"""
        tick_size = Decimal(str(self.tick_size))
        price = Decimal(str(price))
        normalized = float(
            (price / tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * tick_size
        )
        return normalized

    def check_notional(self, quantity, price):
        """Check if order meets minimum notional value requirement"""
        notional = quantity * price
        return notional >= self.min_notional

def get_symbol_info(client, symbol):
    """Get symbol information from exchange"""
    try:
        exchange_info = client.exchange_info()
        symbol_info = next(
            (s for s in exchange_info['symbols'] if s['symbol'] == symbol),
            None
        )
        if not symbol_info:
            raise ValueError(f"Could not find info for symbol {symbol}")
        return symbol_info
    except Exception as e:
        logging.error(f"Error getting symbol info: {e}")
        raise

def validate_price_levels(current_price, entry, exit_level, stop_loss, direction, precision_handler):
    """Validate and adjust price levels based on current market price and direction"""
    if direction == 'LONG':
        if exit_level <= current_price:
            raise ValueError(f"Take-profit {exit_level} is below current price {current_price} for LONG position")
        if stop_loss >= current_price:
            raise ValueError(f"Stop-loss {stop_loss} is above current price {current_price} for LONG position")
    else:  # SHORT
        if exit_level >= current_price:
            raise ValueError(f"Take-profit {exit_level} is above current price {current_price} for SHORT position")
        if stop_loss <= current_price:
            raise ValueError(f"Stop-loss {stop_loss} is below current price {current_price} for SHORT position")
    
    # Normalize prices
    exit_level = precision_handler.normalize_price(exit_level)
    stop_loss = precision_handler.normalize_price(stop_loss)
    
    return exit_level, stop_loss

def calculate_position_size(current_price, entry, stop_loss, precision_handler):
    """Calculate and validate position size"""
    risk_amount = ACCOUNT_BALANCE * RISK_PERCENTAGE
    stop_loss_distance = abs(entry - stop_loss)
    position_size = (risk_amount / stop_loss_distance) * LEVERAGE
    
    # Calculate initial quantity
    quantity = position_size / current_price
    
    # Ensure minimum notional value
    min_quantity = MIN_NOTIONAL / current_price
    if quantity < min_quantity:
        logging.warning(f"Adjusting quantity to meet minimum notional value requirement")
        quantity = min_quantity * 1.01  # Add 1% buffer
    
    # Normalize quantity
    quantity = precision_handler.normalize_quantity(quantity)
    
    # Verify notional value
    if not precision_handler.check_notional(quantity, current_price):
        # If still below minimum notional, increase quantity
        quantity = precision_handler.normalize_quantity((MIN_NOTIONAL / current_price) * 1.01)
        
    logging.info(f"Calculated quantity: {quantity} BTC")
    logging.info(f"Notional value: {quantity * current_price:.2f} USDT")
    
    return quantity

def place_orders(client, symbol, direction, quantity, current_price, stop_loss, exit_level):
    """Place the main order, stop-loss, and take-profit orders with proper error handling"""
    orders = []
    try:
        # Set leverage
        client.change_leverage(symbol=symbol, leverage=LEVERAGE)
        logging.info(f"Leverage set to {LEVERAGE}x")

        # Close any existing positions first
        try:
            position_info = client.get_position_risk(symbol=symbol)
            if position_info and float(position_info[0]['positionAmt']) != 0:
                close_side = 'SELL' if float(position_info[0]['positionAmt']) > 0 else 'BUY'
                client.new_order(
                    symbol=symbol,
                    side=close_side,
                    type='MARKET',
                    quantity=abs(float(position_info[0]['positionAmt'])),
                    reduceOnly=True
                )
                logging.info("Closed existing position")
        except Exception as e:
            logging.error(f"Error closing existing position: {e}")
            raise

        # Place main market order
        side = 'BUY' if direction == 'LONG' else 'SELL'
        main_order = client.new_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        orders.append(('Main', main_order))
        logging.info(f"Main order placed: {main_order}")

        # Place stop-loss order
        stop_side = 'SELL' if direction == 'LONG' else 'BUY'
        sl_order = client.new_order(
            symbol=symbol,
            side=stop_side,
            type='STOP_MARKET',
            stopPrice=stop_loss,
            quantity=quantity,
            reduceOnly=True
        )
        orders.append(('Stop-loss', sl_order))
        logging.info(f"Stop-loss order placed: {sl_order}")

        # Place take-profit order
        tp_order = client.new_order(
            symbol=symbol,
            side=stop_side,
            type='TAKE_PROFIT_MARKET',
            stopPrice=exit_level,
            quantity=quantity,
            reduceOnly=True
        )
        orders.append(('Take-profit', tp_order))
        logging.info(f"Take-profit order placed: {tp_order}")

        return orders

    except ClientError as e:
        error_msg = f"Binance API error: {e.error_code} - {e.error_message}"
        logging.error(error_msg)
        print(error_msg)  # Added print statement for immediate feedback
        
        # Cancel any successful orders if we hit an error
        if orders:
            logging.warning("Cancelling successful orders due to error...")
            for order_type, order in orders:
                try:
                    client.cancel_order(symbol=symbol, orderId=order['orderId'])
                    logging.info(f"Cancelled {order_type} order: {order['orderId']}")
                except Exception as cancel_error:
                    logging.error(f"Error cancelling {order_type} order: {cancel_error}")
        
        raise

def get_trading_strategy(ohlc_data):
    """Get trading strategy from DeepSeek API"""
    try:
        client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url='https://api.deepseek.com'
        )

        with open('deepseek_prompt.txt', 'r') as f:
            system_message = f.read().strip()

        table = tabulate(ohlc_data, headers='keys', tablefmt='grid')
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

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.1,
            stream=False
        )
        
        ds_response = response.choices[0].message.content.strip()
        ds_response = re.sub(r'^```json\s*', '', ds_response)
        ds_response = re.sub(r'\s*```$', '', ds_response)
        
        strategy = json.loads(ds_response)
        logging.info(f"Received strategy: {strategy}")
        return strategy
    
    except Exception as e:
        logging.error(f"Error getting trading strategy: {e}")
        raise

def main():
    try:
        # Initialize Binance Futures client
        client = UMFutures(
            key=os.getenv('BINANCE_API_KEY'),
            secret=os.getenv('BINANCE_API_SECRET'),
            base_url='https://testnet.binancefuture.com'
        )

        # Get symbol information and create precision handler
        symbol_info = get_symbol_info(client, SYMBOL)
        precision_handler = PrecisionHandler(symbol_info)

        # Fetch OHLC data
        url = 'https://api.binance.com/api/v3/klines'
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

        current_price = float(ohlc_data[-1]['Close'])
        logging.info(f"Current price: {current_price}")

        # Get trading strategy
        trade_strategy = get_trading_strategy(ohlc_data)
        logging.info(f"Trade strategy received: {trade_strategy}")

        # Calculate and validate position size
        quantity = calculate_position_size(
            current_price,
            float(trade_strategy['entry']),
            float(trade_strategy['stop_loss']),
            precision_handler
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
            exit_level
        )
        
        logging.info("All orders placed successfully")
        print("All orders placed successfully")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()