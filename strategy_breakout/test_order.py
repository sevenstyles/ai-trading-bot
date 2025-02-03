from config import BINANCE_API_KEY, BINANCE_API_SECRET
from binance.client import Client

# Initialize the Binance Futures Testnet client
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)


def test_order():
    # Set the trading pair and order parameters
    symbol = "BTCUSDT"  # You can change this to any desired symbol
    side = "SELL"        # "BUY" for a long order; change to "SELL" to test a short order
    quantity = 0.005      # Example fixed quantity; adjust as needed
    try:
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