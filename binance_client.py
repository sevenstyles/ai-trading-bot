import os
import logging
from binance.um_futures import UMFutures
from binance.error import ClientError

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