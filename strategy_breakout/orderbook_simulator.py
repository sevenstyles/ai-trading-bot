def simulate_market_order(order_book, quantity, side):
    """
    Simulate a market order execution given an order book snapshot.
    
    Parameters:
        order_book (dict): A dictionary with 'bids' and 'asks', each a list of [price, quantity].
        quantity (float): The quantity of asset to trade.
        side (str): 'buy' or 'sell'.
    
    Returns:
        tuple: (average_price, executed_quantity). If not enough liquidity, may return a partial fill.
    """
    filled_qty = 0.0
    total_cost = 0.0
    if side == 'buy':
        # For buys, use lowest asks first
        levels = sorted(order_book.get('asks', []), key=lambda x: x[0])
    elif side == 'sell':
        # For sells, use highest bids first
        levels = sorted(order_book.get('bids', []), key=lambda x: x[0], reverse=True)
    else:
        raise ValueError("Side must be 'buy' or 'sell'")

    for level in levels:
        price, available_qty = level
        trade_qty = min(quantity - filled_qty, available_qty)
        total_cost += trade_qty * price
        filled_qty += trade_qty
        if filled_qty >= quantity:
            break

    if filled_qty == 0:
        # No liquidity available
        return 0, 0
    average_price = total_cost / filled_qty
    return average_price, filled_qty 