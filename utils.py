from decimal import Decimal, ROUND_DOWN
import logging

# Validate and adjust price levels based on current market price and direction
def validate_price_levels(current_price, entry, exit_level, stop_loss, direction, precision_handler):
    # Minimum price distance (0.5% from current price)
    min_distance = current_price * 0.005
    
    if direction == 'LONG':
        if exit_level <= current_price:
            raise ValueError(f"Take-profit {exit_level} is below current price {current_price} for LONG position")
        if stop_loss >= current_price:
            raise ValueError(f"Stop-loss {stop_loss} is above current price {current_price} for LONG position")
        if exit_level - current_price < min_distance:
            exit_level = current_price + min_distance
            logging.info(f"Adjusted take-profit to maintain minimum distance: {exit_level}")
        if current_price - stop_loss < min_distance:
            stop_loss = current_price - min_distance
            logging.info(f"Adjusted stop-loss to maintain minimum distance: {stop_loss}")
    else:  # SHORT
        if exit_level >= current_price:
            raise ValueError(f"Take-profit {exit_level} is above current price {current_price} for SHORT position")
        if stop_loss <= current_price:
            raise ValueError(f"Stop-loss {stop_loss} is below current price {current_price} for SHORT position")
        if current_price - exit_level < min_distance:
            exit_level = current_price - min_distance
            logging.info(f"Adjusted take-profit to maintain minimum distance: {exit_level}")
        if stop_loss - current_price < min_distance:
            stop_loss = current_price + min_distance
            logging.info(f"Adjusted stop-loss to maintain minimum distance: {stop_loss}")
    
    # Normalize prices
    exit_level = precision_handler.normalize_price(exit_level)
    stop_loss = precision_handler.normalize_price(stop_loss)
    
    return exit_level, stop_loss

# Calculate and validate position size
def calculate_position_size(current_price, entry, stop_loss, precision_handler, account_balance, risk_percentage, leverage, min_notional):
    risk_amount = account_balance * risk_percentage
    stop_loss_distance = abs(entry - stop_loss)
    position_size = (risk_amount / stop_loss_distance) * leverage
    
    # Calculate initial quantity
    quantity = position_size / current_price
    
    # Ensure minimum notional value
    min_quantity = min_notional / current_price
    if quantity < min_quantity:
        logging.warning(f"Adjusting quantity to meet minimum notional value requirement")
        quantity = min_quantity * 1.01  # Add 1% buffer
    
    # Normalize quantity
    quantity = precision_handler.normalize_quantity(quantity)
    
    # Verify notional value
    if not precision_handler.check_notional(quantity, current_price):
        # If still below minimum notional, increase quantity
        quantity = precision_handler.normalize_quantity((min_notional / current_price) * 1.01)
        
    logging.info(f"Calculated quantity: {quantity} BTC")
    logging.info(f"Notional value: {quantity * current_price:.2f} USDT")
    
    return quantity