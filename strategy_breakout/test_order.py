from config import BINANCE_API_KEY, BINANCE_API_SECRET, RISK_PER_TRADE, LEVERAGE
from binance.client import Client
import math

# Initialize the Binance Futures Testnet client
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)


def dynamic_round_quantity(symbol, quantity):
    try:
        info = client.get_symbol_info(symbol)
        for f in info.get('filters', []):
            if f.get('filterType') == 'LOT_SIZE':
                step_size = float(f.get('stepSize'))
                if step_size > 0:
                    precision = int(round(-math.log10(step_size)))
                    return round(quantity, precision)
        return round(quantity, 3)
    except Exception as e:
        return round(quantity, 3)


def test_order():
    # Set the trading pair and order parameters
    symbol = "BTCUSDT"  # Change if needed
    side = "BUY"       # Use "BUY" for a long order; "SELL" for a short order

    try:
        # Set leverage for the symbol
        client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
    except Exception as e:
        print(f"Error setting leverage for {symbol}: {e}")
        return

    try:
        # Fetch current price for the symbol
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker["price"])
    except Exception as e:
        print(f"Error fetching ticker for {symbol}: {e}")
        return

    try:
        # Retrieve USDT balance and calculate order quantity based on RISK_PER_TRADE
        balance_info = client.futures_account_balance()
        usdt_balance = None
        for b in balance_info:
            if b.get('asset') == 'USDT':
                usdt_balance = float(b.get('balance', 0))
                break
        if usdt_balance is None or usdt_balance <= 0:
            quantity = 0.001
        else:
            order_value = usdt_balance * RISK_PER_TRADE
            quantity = order_value / current_price
            quantity = dynamic_round_quantity(symbol, quantity)

        # Place a test market order
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )
        print("Test order placed successfully:")
        print(order)
    except Exception as e:
        print("Error placing test order:", e)


if __name__ == "__main__":
    test_order() 