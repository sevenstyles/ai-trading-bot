import time
from binance import Client, ThreadedWebsocketManager
from config import API_KEY, API_SECRET, SYMBOL, TICK_INTERVAL

print("binance_data.py script starting...")

class BinanceData:
    def __init__(self, order_flow_analyzer):
        print("BinanceData.__init__ starting...")
        try:
            print("Creating Binance Client...")
            # Use testnet=True for the testnet
            self.client = Client(API_KEY, API_SECRET, testnet=True)
            print("Binance Client created.")

            print("Creating ThreadedWebsocketManager...")
            self.twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
            print("ThreadedWebsocketManager created.")

            self.order_flow_analyzer = order_flow_analyzer
            self.order_book = {"bids": {}, "asks": {}}
            self.last_trade = None
            self.running = False
            print("BinanceData.__init__ finished.")

        except Exception as e:
            print(f"Error in BinanceData.__init__: {e}")
            raise

    def start(self):
        self.twm.start()

        # Start combined stream for order book and trades
        self.twm.start_multiplex_socket(
            callback=self.handle_message,
            streams=[
                f"{SYMBOL.lower()}@depth@100ms",  # Order book updates
                f"{SYMBOL.lower()}@trade",  # Trade updates
            ],
        )
        self.running = True
        print(f"Started data streams for {SYMBOL}")

    def stop(self):
        if self.running:
            self.twm.stop()
            self.running = False
            print("Stopped data streams.")

    def handle_message(self, msg):
        if "stream" not in msg:
            print(f"Unexpected message format: {msg}")
            return

        stream_type = msg["stream"]
        data = msg["data"]

        if stream_type == f"{SYMBOL.lower()}@depth":
            self.update_order_book(data)
        elif stream_type == f"{SYMBOL.lower()}@trade":
            self.process_trade(data)

    def update_order_book(self, data):
        # Update bids
        for bid in data["b"]:
            price_level = float(bid[0])
            quantity = float(bid[1])
            if quantity == 0.0:
                if price_level in self.order_book["bids"]:
                    del self.order_book["bids"][price_level]
            else:
                self.order_book["bids"][price_level] = quantity

        # Update asks
        for ask in data["a"]:
            price_level = float(ask[0])
            quantity = float(bid[1])
            if quantity == 0.0:
                if price_level in self.order_book["asks"]:
                    del self.order_book["asks"][price_level]
            else:
                self.order_book["asks"][price_level] = quantity

        # Trigger order book processing
        self.order_flow_analyzer.process_order_book(self.order_book.copy())

    def process_trade(self, data):
        trade = {
            "price": float(data["p"]),
            "quantity": float(data["q"]),
            "is_buyer_maker": data["m"],
            "timestamp": data["T"],
        }
        self.last_trade = trade

        # Trigger trade processing
        self.order_flow_analyzer.process_trade(trade)

    def get_current_price(self):
        # Use the last trade price as an approximation.  For more accuracy,
        # you could use the midpoint of the best bid and ask.
        if self.last_trade:
            return self.last_trade['price']
        else:
            return None

def test_binance_connection():
    try:
        client = Client(API_KEY, API_SECRET, testnet=True)
        server_time = client.get_server_time()
        print(f"Binance Testnet Server Time: {server_time}")
        print("Binance API connection successful!")
        return True
    except Exception as e:
        print(f"Binance API connection failed: {e}")
        return False

if __name__ == "__main__":
    test_binance_connection() 