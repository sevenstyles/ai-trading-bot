import os
print("os module imported")
import time
print("time module imported")
import threading
print("threading module imported")
from binance import Client, ThreadedWebsocketManager
print("Binance Client and ThreadedWebsocketManager imported")
from config import API_KEY, API_SECRET, SYMBOL
print("Config imported")
import logging
print("logging imported")

# Configure logging
logging.basicConfig(filename='error_log.txt', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

print("binance_data.py script starting...")

class BinanceData:
    def __init__(self, order_flow_analyzer):
        print("BinanceData.__init__ starting...")
        try:
            print("Creating Binance Client...")
            self.client = Client(API_KEY, API_SECRET)
            print("Binance Client created.")

            print("Creating ThreadedWebsocketManager...")
            self.twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            print("ThreadedWebsocketManager created.")

            self.order_flow_analyzer = order_flow_analyzer
            self.symbol = SYMBOL
            self.depth_cache = {'bids': {}, 'asks': {}}
            self.bid_depth = 0.0
            self.ask_depth = 0.0
            self.current_bid = 0.0
            self.current_ask = 0.0
            self.trade_volume = 0.0
            self.last_update_time = None
            self.update_interval = 1  # Update interval in seconds
            self.failed_attempts = 0  # Counter for consecutive failed attempts
            self.max_failed_attempts = 5  # Maximum number of consecutive failed attempts before restarting
            self.current_price = None  # Added for the new get_current_price method
            print("BinanceData.__init__ finished.")

        except Exception as e:
            print(f"Error in BinanceData.__init__: {e}")
            raise

    def start(self):
        print("Starting data streams for", self.symbol)
        self.twm.start()
        self.twm.start_depth_socket(callback=self.handle_depth_data, symbol=self.symbol)
        self.twm.start_trade_socket(callback=self.handle_trade_data, symbol=self.symbol)
        self.last_update_time = time.time()
        self.data_update_thread = threading.Thread(target=self.update_data_periodically)
        self.data_update_thread.daemon = True
        self.data_update_thread.start()
        print("Data streams started.")

    def stop(self):
        print("Stopping Binance WebSocket...")
        self.twm.stop()
        print("Binance WebSocket stopped.")

    def handle_depth_data(self, msg):
        try:
            if not isinstance(msg, dict):
                log_message = f"Invalid depth data received: Expected a dict, got {type(msg)}"
                logging.error(log_message)
                self.failed_attempts += 1
                return

            if 'b' not in msg:
                log_message = f"Invalid depth data received: Missing 'bids' key in {msg}"
                logging.error(log_message)
                self.failed_attempts += 1
                return

            if 'a' not in msg:
                log_message = f"Invalid depth data received: Missing 'asks' key in {msg}"
                logging.error(log_message)
                self.failed_attempts += 1
                return

            bids = msg['b']
            asks = msg['a']

            if not isinstance(bids, list) or not isinstance(asks, list):
                log_message = "Invalid depth data: Bids or asks are not lists."
                logging.error(log_message)
                self.failed_attempts += 1
                return

            self.update_depth_cache(bids, asks)
            self.calculate_depth()
            self.failed_attempts = 0  # Reset failed attempts counter

        except ValueError as e:
            log_message = f"Invalid depth data format: Could not convert to float: {msg}. Error: {e}"
            logging.error(log_message)
            self.failed_attempts += 1
        except exceptions.BinanceAPIException as e:  # Catch Binance API exceptions
            log_message = f"Binance API Exception in handle_depth_data: {e}"
            logging.error(log_message)
            self.failed_attempts += 1
        except Exception as e:
            log_message = f"Error processing depth event: {e}"
            logging.error(log_message)
            self.failed_attempts += 1

        # Check if the maximum number of failed attempts has been reached
        if self.failed_attempts > self.max_failed_attempts:
            print("Maximum number of failed attempts reached. Restarting BinanceData...")
            self.restart()

    def handle_trade_data(self, msg):
        try:
            # Check if msg is a dictionary
            if not isinstance(msg, dict):
                log_message = f"Invalid trade data received: Expected a dict, got {type(msg)}"
                logging.error(log_message)
                self.failed_attempts += 1
                return

            # Check if the message is an error message
            if 'e' in msg and msg['e'] == 'error':
                log_message = f"Received error message from WebSocket: {msg['m']}"
                logging.error(log_message)
                self.failed_attempts += 1
                return

            if 'p' not in msg:
                log_message = f"Invalid trade data received: Missing 'p' key in {msg}"
                logging.error(log_message)
                self.failed_attempts += 1
                return

            if 'q' not in msg:
                log_message = f"Invalid trade data received: Missing 'q' key in {msg}"
                logging.error(log_message)
                self.failed_attempts += 1
                return

            price = float(msg['p'])
            qty = float(msg['q'])
            is_buy = not msg['m']

            self.trade_volume += qty
            self.order_flow_analyzer.process_trade(price, {"price": price, "volume": qty, "is_buy": is_buy})
            self.failed_attempts = 0  # Reset failed attempts counter

        except ValueError as e:
            log_message = f"Invalid trade data format: Could not convert to float: {msg}. Error: {e}"
            logging.error(log_message)
            self.failed_attempts += 1

        except Exception as e:
            log_message = f"Error handling trade data: {e}"
            logging.error(log_message)
            self.failed_attempts += 1

        # Check if the maximum number of failed attempts has been reached
        if self.failed_attempts > self.max_failed_attempts:
            print("Maximum number of failed attempts reached. Restarting BinanceData...")
            self.restart()

    def update_data_periodically(self):
        while True:
            try:
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
                            self.failed_attempts += 1

                        # Get best bid and ask from depth cache
                        if self.depth_cache and self.depth_cache['bids'] and self.depth_cache['asks']:
                            best_bid_price = float(list(self.depth_cache['bids'].keys())[0])
                            best_ask_price = float(list(self.depth_cache['asks'].keys())[0])
                            self.current_bid = best_bid_price
                            self.current_ask = best_ask_price
                        else:
                            log_message = "Could not retrieve depth information or bids/asks are empty."
                            logging.error(log_message)
                            self.failed_attempts += 1

                        self.last_update_time = time.time()
                        self.failed_attempts = 0  # Reset failed attempts counter

                        # Calculate liquidity
                        raw_liquidity = self.current_bid + self.current_ask
                        print(f"Current liquidity: {raw_liquidity}")

                        # Calculate bid and ask depth from the depth cache
                        self.bid_depth = sum(self.depth_cache['bids'].values())
                        self.ask_depth = sum(self.depth_cache['asks'].values())

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
                self.failed_attempts += 1

            # Check if the maximum number of failed attempts has been reached
            if self.failed_attempts > self.max_failed_attempts:
                print("Maximum number of failed attempts reached. Restarting BinanceData...")
                self.restart()

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

    def restart(self):
        print("Restarting BinanceData...")
        self.stop()
        self.failed_attempts = 0
        self.start()

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

    def update_depth_cache(self, bids, asks):
        """Updates the depth cache with new bid and ask data."""
        try:
            self.depth_cache = {'bids': {}, 'asks': {}}

            # Process each bid
            for bid in bids:
                if not isinstance(bid, list):
                    log_message = f"Invalid bid format: Expected a list, got {type(bid)}"
                    logging.error(log_message)
                    continue

                if len(bid) != 2:
                    log_message = f"Invalid bid format: Expected length 2, got {len(bid)}"
                    logging.error(log_message)
                    continue

                try:
                    price, size = map(float, bid)
                    self.depth_cache['bids'][price] = size
                except ValueError:
                    log_message = f"Invalid bid format: Could not convert to float: {bid}"
                    logging.error(log_message)

            # Process each ask
            for ask in asks:
                if not isinstance(ask, list):
                    log_message = f"Invalid ask format: Expected a list, got {type(ask)}"
                    logging.error(log_message)
                    continue

                if len(ask) != 2:
                    log_message = f"Invalid ask format: Expected length 2, got {len(ask)}"
                    logging.error(log_message)
                    continue

                try:
                    price, size = map(float, ask)
                    self.depth_cache['asks'][price] = size
                except ValueError:
                    log_message = f"Invalid ask format: Could not convert to float: {ask}"
                    logging.error(log_message)

            if not self.depth_cache['bids'] or not self.depth_cache['asks']:
                log_message = "No bid or ask data available in depth cache."
                logging.error(log_message)
                self.failed_attempts += 1
                return
            else:
                print(f"Bids in depth cache: {len(self.depth_cache['bids'])}, Asks in depth cache: {len(self.depth_cache['asks'])}")

            # Process the order book data
            self.order_flow_analyzer.process_order_book(self.depth_cache)

        except Exception as e:
            log_message = f"Error updating depth cache: {e}"
            logging.error(log_message)
            self.failed_attempts += 1

    def calculate_depth(self):
        """Calculates the total bid and ask depth from the depth cache."""
        try:
            self.bid_depth = sum(self.depth_cache['bids'].values())
            self.ask_depth = sum(self.depth_cache['asks'].values())
            print(f"Bid Depth: {self.bid_depth}, Ask Depth: {self.ask_depth}")
        except Exception as e:
            log_message = f"Error calculating depth: {e}"
            logging.error(log_message)

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