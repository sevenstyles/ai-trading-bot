import os
import logging
from binance.um_futures import UMFutures
from binance.error import ClientError
from precision import PrecisionHandler
import time

# Initialize Binance Futures client
def initialize_binance_client():
    return UMFutures(
        key=os.getenv('BINANCE_API_KEY'),
        secret=os.getenv('BINANCE_API_SECRET'),
        base_url='https://testnet.binancefuture.com'
    )

# Get symbol information from exchange
def get_symbol_info(client, symbol):
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

# Place the main order, stop-loss, and take-profit orders with proper error handling
def place_orders(client, symbol, direction, quantity, current_price, stop_loss, exit_level, leverage):
    orders = []
    try:
        # Get symbol info for price normalization
        symbol_info = client.exchange_info()['symbols']
        symbol_info = next((s for s in symbol_info if s['symbol'] == symbol), None)
        if not symbol_info:
            raise ValueError(f"Could not find info for symbol {symbol}")
        
        precision_handler = PrecisionHandler(symbol_info)
        
        # Set leverage
        client.change_leverage(symbol=symbol, leverage=leverage)
        logging.info(f"Leverage set to {leverage}x")

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

        # Wait a moment for the order to be filled and get the fill price
        time.sleep(1)  # Wait for 1 second
        orders_info = client.get_orders(symbol=symbol, orderId=main_order['orderId'])
        if not orders_info or orders_info[0]['status'] != 'FILLED':
            raise ValueError("Main order not filled yet")
        
        fill_price = float(orders_info[0]['avgPrice'])
        logging.info(f"Main order filled at price: {fill_price}")

        # Add buffer for stop orders (1% from fill price)
        buffer = fill_price * 0.01
        logging.info(f"Using price buffer of {buffer} USDT")

        # Place stop-loss order with buffer
        stop_side = 'SELL' if direction == 'LONG' else 'BUY'
        sl_trigger = stop_loss - buffer if direction == 'LONG' else stop_loss + buffer
        sl_trigger = precision_handler.normalize_price(sl_trigger)
        logging.info(f"Stop-loss trigger price: {sl_trigger} (original: {stop_loss})")
        sl_order = client.new_order(
            symbol=symbol,
            side=stop_side,
            type='STOP_MARKET',
            stopPrice=sl_trigger,
            quantity=quantity,
            reduceOnly=True
        )
        orders.append(('Stop-loss', sl_order))
        logging.info(f"Stop-loss order placed: {sl_order}")

        # Place take-profit order with buffer
        tp_trigger = exit_level + buffer if direction == 'LONG' else exit_level - buffer
        tp_trigger = precision_handler.normalize_price(tp_trigger)
        logging.info(f"Take-profit trigger price: {tp_trigger} (original: {exit_level})")
        tp_order = client.new_order(
            symbol=symbol,
            side=stop_side,
            type='TAKE_PROFIT_MARKET',
            stopPrice=tp_trigger,
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