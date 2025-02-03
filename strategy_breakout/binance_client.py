from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)

# Monkey-patch for Binance WebSocket issue
try:
    from binance.streams import ClientConnection
except ImportError:
    pass
else:
    ClientConnection.fail_connection = lambda self: None 