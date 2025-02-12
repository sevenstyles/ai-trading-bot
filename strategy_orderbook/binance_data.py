import os
import time
import threading
from binance import Client
from config import API_KEY, API_SECRET, SYMBOL
import logging
from binance_data_handler import BinanceDataHandler  # Import the new class

# Configure logging
logging.basicConfig(filename='error_log.txt', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

print("binance_data.py script starting...")

class BinanceData:
    def __init__(self, order_flow_analyzer):
        print("BinanceData.__init__ starting...")
        try:
            self.order_flow_analyzer = order_flow_analyzer
            self.symbol = SYMBOL
            self.data_handler = BinanceDataHandler(order_flow_analyzer)  # Use the new handler
            self.bid_depth = 0.0
            self.ask_depth = 0.0
            self.current_bid = 0.0
            self.current_ask = 0.0
            self.trade_volume = 0.0 # Keep this here, as it's used by update_data_periodically
            self.last_update_time = None
            self.update_interval = 1  # Update interval in seconds
            self.current_price = None
            self.client = Client(API_KEY, API_SECRET) # Need a client instance for get_current_price
            print("BinanceData.__init__ finished.")

        except Exception as e:
            print(f"Error in BinanceData.__init__: {e}")
            raise

    def start(self):
        print("Starting data streams for", self.symbol)
        self.data_handler.start()  # Start through the handler
        self.last_update_time = time.time()
        self.data_update_thread = threading.Thread(target=self.update_data_periodically)
        self.data_update_thread.daemon = True
        self.data_update_thread.start()
        print("Data streams started.")

    def stop(self):
        print("Stopping Binance WebSocket...")
        self.data_handler.stop()  # Stop through the handler
        print("Binance WebSocket stopped.")

    def update_data_periodically(self):
        while True:
            try:
                time.sleep(2)  # Initial delay to allow depth cache to populate
                if self.last_update_time is not None:
                    elapsed_time = time.time() - self.last_update_time
                    if elapsed_time > self.update_interval:
                        current_price = self.get_current_price()
                        if current_price:
                            self.current_price = current_price
                            logging.info(f"Current price updated: {self.current_price}")
                        else:
                            log_message = "Could not retrieve current price."
                            logging.error(log_message)

                        # Get best bid and ask from depth cache
                        if self.data_handler.depth_cache and self.data_handler.depth_cache['bids'] and self.data_handler.depth_cache['asks']:
                            best_bid_price = float(list(self.data_handler.depth_cache['bids'].keys())[0])
                            best_ask_price = float(list(self.data_handler.depth_cache['asks'].keys())[0])
                            self.current_bid = best_bid_price
                            self.current_ask = best_ask_price
                        else:
                            log_message = "Could not retrieve depth information or bids/asks are empty."
                            logging.error(log_message)

                        self.last_update_time = time.time()

                        # Calculate liquidity
                        raw_liquidity = self.current_bid + self.current_ask
                        print(f"Current liquidity: {raw_liquidity}")

                        # Calculate bid and ask depth from the depth cache - get from handler
                        self.bid_depth, self.ask_depth = self.data_handler.calculate_depth()

                        # --- MOVE CHECK_FOR_SIGNALS HERE ---
                        print(f"About to call check_for_signals from binance_data.py. order_flow_analyzer: {self.order_flow_analyzer}")
                        signal = self.order_flow_analyzer.check_for_signals(raw_liquidity, self.bid_depth, self.ask_depth)  # Pass bid and ask depth
                        if signal:
                            print(f"Received signal: {signal}")
                            self.order_flow_analyzer.trading_bot.trading_signal = signal  # Set the shared variable
                        # ------------------------------------

            except Exception as e:
                log_message = f"Error updating data periodically: {e}"
                logging.error(log_message)

            time.sleep(self.update_interval)

    def get_current_price(self):
        """
        Fetches the current price from Binance API.
        """
        try:
            ticker = self.client.get_ticker(symbol=self.symbol)
            if ticker and 'lastPrice' in ticker:
                return ticker['lastPrice']
            else:
                log_message = "Could not retrieve ticker information or lastPrice not found."
                logging.error(log_message)
                return None
        except Exception as e:
            log_message = f"Error getting current price: {e}"
            logging.error(log_message)
            return None

    def log_error(self, message):
        """Logs an error message to a file with a timestamp."""
        log_file_path = "error_log.txt"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        try:
            with open(log_file_path, "a") as log_file:
                log_file.write(log_message)
            print(f"Error logged to {log_file_path}")
        except Exception as e:
            print(f"Error writing to log file: {e}")