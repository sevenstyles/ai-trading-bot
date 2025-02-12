import os
import time
import threading
from binance import Client, ThreadedWebsocketManager
from config import API_KEY, API_SECRET, SYMBOL
import logging

# Configure logging
logging.basicConfig(filename='error_log.txt', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class BinanceDataHandler:
    def __init__(self, order_flow_analyzer):
        print("BinanceDataHandler.__init__ starting...")
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
            self.failed_attempts = 0
            self.max_failed_attempts = 5
            print("BinanceDataHandler.__init__ finished.")

        except Exception as e:
            print(f"Error in BinanceDataHandler.__init__: {e}")
            raise

    def start(self):
        print("Starting data streams for", self.symbol)
        self.twm.start()
        self.twm.start_depth_socket(callback=self.handle_depth_data, symbol=self.symbol)
        self.twm.start_trade_socket(callback=self.handle_trade_data, symbol=self.symbol)
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
            print("Maximum number of failed attempts reached. Restarting BinanceDataHandler...")
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
            print("Maximum number of failed attempts reached. Restarting BinanceDataHandler...")
            self.restart()
    
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
            bid_depth = sum(self.depth_cache['bids'].values())
            ask_depth = sum(self.depth_cache['asks'].values())
            print(f"Bid Depth: {bid_depth}, Ask Depth: {ask_depth}")
            return bid_depth, ask_depth
        except Exception as e:
            log_message = f"Error calculating depth: {e}"
            logging.error(log_message)
            return 0, 0

    def restart(self):
        print("Restarting BinanceDataHandler...")
        self.stop()
        self.failed_attempts = 0
        self.start()