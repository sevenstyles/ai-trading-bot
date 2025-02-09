import time
print("time module imported")
import csv
print("csv module imported")
import os  # Import the 'os' module
print("os module imported")
from config import SYMBOL, ORDER_TYPE, LIMIT_PRICE_OFFSET, STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE, MAX_POSITION_SIZE_PERCENTAGE, API_KEY, API_SECRET, ORDER_FLOW_DELTA_Z_SCORE_THRESHOLD, TICK_INTERVAL
print("Config imported")
from binance_data import BinanceData
print("BinanceData imported")
import order_flow_analyzer
from order_flow_analyzer import OrderFlowAnalyzer
print("OrderFlowAnalyzer imported")
from binance import Client, exceptions
print("Binance Client and exceptions imported")
import logging

print("Trading Bot script starting...")
print(f"Executing trading_bot.py from: {os.path.abspath(__file__)}")
print(f"API_KEY from config: {API_KEY}")  # Print API_KEY
print(f"API_SECRET from config: {API_SECRET}")  # Print API_SECRET

class TradingBot:
    def __init__(self):
        print("TradingBot.__init__ starting...")
        try:
            print("Creating OrderFlowAnalyzer...")
            self.order_flow_analyzer = OrderFlowAnalyzer(self)  # Pass self to the constructor
            print("OrderFlowAnalyzer created.")

            print("Creating BinanceData...")
            self.binance_data = BinanceData(self.order_flow_analyzer)
            print("BinanceData created.")

            self.symbol = SYMBOL
            self.order_type = ORDER_TYPE
            self.limit_price_offset = LIMIT_PRICE_OFFSET
            self.stop_loss_percentage = STOP_LOSS_PERCENTAGE
            self.take_profit_percentage = TAKE_PROFIT_PERCENTAGE
            self.max_position_size_percentage = MAX_POSITION_SIZE_PERCENTAGE
            self.open_trades = []  # List to store open trades
            self.client = Client(API_KEY, API_SECRET)
            self.csv_directory = 'trade_history'  # Directory to store CSV files
            self.csv_file_path = os.path.join(self.csv_directory, f"trades_{self.symbol}_{int(time.time())}.csv")
            self.big_move_directory = 'big_move_history'  # Directory for big move CSV
            self.big_move_csv_file_path = os.path.join(self.big_move_directory, f"big_move_potential_{self.symbol}_{int(time.time())}.csv")
            self.initialize_csv()
            print("TradingBot.__init__ finished.")

            # Configure logging
            logging.basicConfig(filename='error_log.txt', level=logging.ERROR,
                                format='%(asctime)s - %(levelname)s - %(message)s')

            # Initialize big move CSV
            self.big_move_csv_file = None
            self.big_move_csv_writer = None
            self.initialize_big_move_csv()

            self.trading_signal = None  # Shared variable for trading signal

        except Exception as e:
            print(f"Error in TradingBot.__init__: {e}")
            logging.error(f"Error in TradingBot.__init__: {e}")
            raise  # Re-raise the exception to see the full traceback

    def run(self):
        print("Creating BinanceData in run()...")
        self.binance_data.start()
        print("BinanceData started in run().")

        try:
            while True:
                # Check for big move potential
                self.check_big_move_potential()

                # Check for trading signals
                #print(f"About to call check_for_signals. order_flow_analyzer: {self.order_flow_analyzer}")
                #signal = self.order_flow_analyzer.check_for_signals()

                #if signal:
                #    print(f"Received signal: {signal}")
                #    self.execute_trade(signal)
                if self.trading_signal:
                    signal = self.trading_signal
                    self.trading_signal = None  # Reset the signal
                    print(f"Received signal from shared variable: {signal}")
                    self.execute_trade(signal)

                # Monitor open trades
                self.monitor_open_trades()

                # Wait for the specified tick interval
                time.sleep(TICK_INTERVAL)

        except Exception as e:
            print(f"Error in run: {e}")
            logging.error(f"Error in run: {e}")
        finally:
            self.binance_data.stop()

    def ensure_directory_exists(self, directory):
        """Ensures that the specified directory exists."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def initialize_csv(self):
        self.ensure_directory_exists(self.csv_directory)  # Ensure directory exists
        timestamp = int(time.time())
        filename = f"trades_{self.symbol}_{timestamp}.csv"
        self.csv_file_path = os.path.join(self.csv_directory, filename)  # Set the full file path
        self.csv_file = open(self.csv_file_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Write the header to the CSV file
        header = ['Signal', 'Entry Price', 'Quantity', 'Stop Loss', 'Take Profit', 'Exit Price', 'Exit Timestamp', 'PNL', 'Status']
        self.csv_writer.writerow(header)
        self.csv_file.flush()

    def write_trade_to_csv(self, signal, entry_price, quantity, stop_loss, take_profit, exit_price, exit_timestamp, pnl, status):
        if self.csv_writer:
            row = [signal, entry_price, quantity, stop_loss, take_profit, exit_price, exit_timestamp, pnl, status]
            self.csv_writer.writerow(row)
            self.csv_file.flush()  # Ensure data is written immediately

    def get_available_balance(self):
        try:
            # Get account balances
            balances = self.client.get_account()['balances']
            # Find the balance for the quote currency (e.g., USDT)
            for balance in balances:
                if balance['asset'] == SYMBOL[-3:]:  # Assumes USDT as quote currency
                    return float(balance['free'])
            print(f"Quote currency {SYMBOL[-3:]} not found in account.")
            return 0.0
        except exceptions.BinanceAPIException as e:
            print(f"Binance API Exception: {e}")
            logging.error(f"Binance API Exception in get_available_balance: {e}")
            return 0.0
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            logging.error(f"Unexpected error in get_available_balance: {e}")
            return 0.0

    def calculate_quantity(self, entry_price):
        try:
            available_balance = self.get_available_balance()
            # Calculate the maximum position size based on the available balance
            max_position_size = available_balance * self.max_position_size_percentage
            # Calculate the quantity to trade based on the entry price
            quantity = max_position_size / entry_price
            # Get the symbol information to determine the quantity precision
            symbol_info = self.client.get_symbol_info(symbol=self.symbol)
            if not symbol_info:
                print(f"Could not retrieve symbol information for {self.symbol}")
                return None
            # Extract the filters from the symbol information
            filters = symbol_info['filters']
            # Find the LOT_SIZE filter to determine the step size
            lot_size_filter = next((f for f in filters if f['filterType'] == 'LOT_SIZE'), None)
            if not lot_size_filter:
                print("Could not retrieve LOT_SIZE filter.")
                return None
            # Determine the step size from the LOT_SIZE filter
            step_size = float(lot_size_filter['stepSize'])
            # Round the quantity down to the nearest step size
            quantity = round(quantity / step_size) * step_size
            return quantity
        except exceptions.BinanceAPIException as e:
            print(f"Binance API Exception: {e}")
            logging.error(f"Binance API Exception in calculate_quantity: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            logging.error(f"Unexpected error in calculate_quantity: {e}")
            return None

    def place_order(self, side, quantity, price=None):
        try:
            if self.order_type == 'MARKET':
                order = self.client.order_market(
                    symbol=self.symbol,
                    side=side,
                    quantity=quantity)
            elif self.order_type == 'LIMIT' and price:
                order = self.client.order_limit(
                    symbol=self.symbol,
                    side=side,
                    quantity=quantity,
                    price=price)
            else:
                print("Invalid order type or missing price for LIMIT order.")
                return None
            return order
        except exceptions.BinanceAPIException as e:
            print(f"Binance API Exception: {e}")
            logging.error(f"Binance API Exception in place_order: {e}")
            return None
        except exceptions.BinanceOrderException as e:
            print(f"Binance Order Exception: {e}")
            logging.error(f"Binance Order Exception in place_order: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            logging.error(f"Unexpected error in place_order: {e}")
            return None

    def execute_trade(self, signal):
        """
        Executes a trade based on the signal.
        """
        try:
            # Get current price
            current_price = float(self.binance_data.current_price)

            # Calculate quantity based on max position size percentage
            account_balance = float(self.client.get_account()['balances'][0]['free'])  # USDT balance
            max_position_size = account_balance * self.max_position_size_percentage
            quantity = max_position_size / current_price

            # Round quantity to appropriate precision
            quantity = self.round_quantity(quantity)

            if signal == "buy":
                # Execute buy order
                if self.order_type == 'MARKET':
                    order = self.client.order_market_buy(symbol=self.symbol, quantity=quantity)
                elif self.order_type == 'LIMIT':
                    limit_price = current_price * (1 + self.limit_price_offset)  # Example offset
                    order = self.client.order_limit_buy(symbol=self.symbol, quantity=quantity, price=limit_price)
                else:
                    print(f"Invalid order type: {self.order_type}")
                    return

                order_id = order['orderId']

                # Calculate stop loss and take profit prices
                stop_loss = current_price * (1 - self.stop_loss_percentage)
                take_profit = current_price * (1 + self.take_profit_percentage)

                # Store trade information
                timestamp = int(time.time())
                self.open_trades.append({
                    'timestamp': timestamp,
                    'signal': signal,
                    'price': current_price,
                    'quantity': quantity,
                    'order_id': order_id,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'pnl': 0,  # Initial PnL
                    'is_open': True
                })

                self.write_trade_to_csv(signal, current_price, quantity, stop_loss, take_profit, 0, timestamp, 0, "OPEN")
                print(f"Executed {signal} order. Order ID: {order_id}, Price: {current_price}, Quantity: {quantity}")

            elif signal == "sell":
                # Execute sell order (similar to buy order)
                if self.order_type == 'MARKET':
                    order = self.client.order_market_sell(symbol=self.symbol, quantity=quantity)
                elif self.order_type == 'LIMIT':
                    limit_price = current_price * (1 - self.limit_price_offset)  # Example offset
                    order = self.client.order_limit_sell(symbol=self.symbol, quantity=quantity, price=limit_price)
                else:
                    print(f"Invalid order type: {self.order_type}")
                    return

                order_id = order['orderId']

                # Calculate stop loss and take profit prices
                stop_loss = current_price * (1 + self.stop_loss_percentage)
                take_profit = current_price * (1 - self.take_profit_percentage)

                # Store trade information
                timestamp = int(time.time())
                self.open_trades.append({
                    'timestamp': timestamp,
                    'signal': signal,
                    'price': current_price,
                    'quantity': quantity,
                    'order_id': order_id,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'pnl': 0,  # Initial PnL
                    'is_open': True
                })

                self.write_trade_to_csv(signal, current_price, quantity, stop_loss, take_profit, 0, timestamp, 0, "OPEN")
                print(f"Executed {signal} order. Order ID: {order_id}, Price: {current_price}, Quantity: {quantity}")

            else:
                print("No signal to execute.")

        except exceptions.BinanceAPIException as e:
            print(f"Binance API exception: {e}")
            logging.error(f"Binance API exception: {e}")
        except Exception as e:
            print(f"Error executing trade: {e}")
            logging.error(f"Trade Execution Error: {e}")

    def check_stop_loss_take_profit(self):
        if not self.in_position:
            return

        try:
            current_price = self.binance_data.get_current_price()
            if not current_price:
                print("Could not retrieve current price.")
                return

            current_price = float(current_price)

            if self.signal == "buy":
                # Trailing Stop Loss Logic
                new_stop_loss = current_price * (1 - self.stop_loss_percentage)
                if new_stop_loss > self.stop_loss_price:
                    print(f"Moving stop loss from {self.stop_loss_price} to {new_stop_loss}")
                    self.stop_loss_price = new_stop_loss
                    # Update CSV with new stop loss price
                    self.update_csv_stop_loss(self.stop_loss_price)

                if current_price <= self.stop_loss_price:
                    self.exit_position("stop_loss", current_price)
                elif current_price >= self.take_profit_price:
                    self.exit_position("take_profit", current_price)
            elif self.signal == "sell":
                # Trailing Stop Loss Logic
                new_stop_loss = current_price * (1 + self.stop_loss_percentage)
                if new_stop_loss < self.stop_loss_price:
                    print(f"Moving stop loss from {self.stop_loss_price} to {new_stop_loss}")
                    self.stop_loss_price = new_stop_loss
                    # Update CSV with new stop loss price
                    self.update_csv_stop_loss(self.stop_loss_price)

                if current_price >= self.stop_loss_price:
                    self.exit_position("stop_loss", current_price)
                elif current_price <= self.take_profit_price:
                    self.exit_position("take_profit", current_price)

        except exceptions.BinanceAPIException as e:
            print(f"Binance API Exception: {e}")
        except exceptions.BinanceOrderException as e:
            print(f"Binance Order Exception: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def exit_position(self, reason, exit_price):
        if not self.in_position:
            print("Not in a position. Skipping exit.")
            return

        try:
            quantity = self.calculate_quantity(exit_price)  # Use exit_price to calculate quantity
            if not quantity:
                print("Could not calculate quantity for exit.")
                return

            if self.signal == "buy":
                side = "SELL"
            elif self.signal == "sell":
                side = "BUY"
            else:
                print("Invalid signal.")
                return

            order = self.place_order(side, quantity)

            if order:
                self.in_position = False
                exit_timestamp = int(time.time())
                pnl = (exit_price - self.entry_price) * quantity if self.signal == "buy" else (self.entry_price - exit_price) * quantity
                print(f"Position exited. Reason: {reason}, Exit Price: {exit_price}, PNL: {pnl}")

                # Write trade exit to CSV
                self.write_trade_to_csv(self.signal, self.entry_price, quantity, self.stop_loss_price, self.take_profit_price, exit_price, exit_timestamp, pnl, "CLOSED")

                self.entry_price = 0.0
                self.stop_loss_price = 0.0
                self.take_profit_price = 0.0

            else:
                print("Order placement failed.")

        except exceptions.BinanceAPIException as e:
            print(f"Binance API Exception while exiting position: {e}")
        except exceptions.BinanceOrderException as e:
            print(f"Binance Order Exception while exiting position: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while exiting position: {e}")

    def update_csv_stop_loss(self, new_stop_loss):
        # Read all rows from the CSV file
        with open(self.csv_file.name, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            rows = list(reader)

        # Find the last row with status "OPEN"
        for i in range(len(rows) - 1, 0, -1):
            if rows[i][8] == "OPEN":
                # Update the stop loss price in that row
                rows[i][3] = new_stop_loss
                break

        # Write the updated rows back to the CSV file
        with open(self.csv_file.name, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(rows)
        self.csv_file.flush()

    def initialize_big_move_csv(self):
        self.ensure_directory_exists(self.big_move_directory)  # Ensure directory exists
        timestamp = int(time.time())
        filename = f"big_move_potential_{self.symbol}_{timestamp}.csv"
        self.big_move_csv_file_path = os.path.join(self.big_move_directory, filename)  # Set the full file path
        self.big_move_csv_file = open(self.big_move_csv_file_path, mode='w', newline='')
        self.big_move_csv_writer = csv.writer(self.big_move_csv_file)

        # Write the header to the CSV file
        header = ['Timestamp', 'Order Flow Delta Z-Score', 'Bid Depth', 'Ask Depth', 'Slope', 'Best Bid', 'Best Ask']
        self.big_move_csv_writer.writerow(header)
        self.big_move_csv_file.flush()

    def write_big_move_to_csv(self, timestamp, z_score, bid_depth, ask_depth, slope, best_bid, best_ask):
        if self.big_move_csv_writer:
            row = [timestamp, z_score, bid_depth, ask_depth, slope, best_bid, best_ask]
            self.big_move_csv_writer.writerow(row)
            self.big_move_csv_file.flush()

    def check_big_move_potential(self):
        """
        Checks for strong signs of a potential big price movement and saves the details to a separate CSV file.
        """
        try:
            # Define your criteria for a "big move" here
            z_score = self.order_flow_analyzer.order_flow_delta_z_score
            bid_depth = self.order_flow_analyzer.bid_depth
            ask_depth = self.order_flow_analyzer.ask_depth
            slope = self.order_flow_analyzer.order_book_slope
            best_bid = self.binance_data.current_bid
            best_ask = self.binance_data.current_ask

            # Example criteria:
            if (abs(z_score) > (ORDER_FLOW_DELTA_Z_SCORE_THRESHOLD * 1.5) and  # Increased Z-score threshold
                abs(slope) > 0.05):  # Increased order book slope
                print("Potential big move detected! Saving details to big_move_potential.csv")
                timestamp = int(time.time())
                self.write_big_move_to_csv(timestamp, z_score, bid_depth, ask_depth, slope, best_bid, best_ask)

        except Exception as e:
            print(f"Error checking big move potential: {e}")
            logging.error(f"Error checking big move potential: {e}")

    def write_trade_to_csv(self, timestamp, signal, price, quantity, order_id, stop_loss, take_profit):
        """
        Writes trade details to a CSV file.
        """
        try:
            file_exists = os.path.isfile(self.csv_file_path)

            with open(self.csv_file_path, mode='a', newline='') as csvfile:
                fieldnames = ['timestamp', 'signal', 'price', 'quantity', 'order_id', 'stop_loss', 'take_profit', 'pnl']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'timestamp': timestamp,
                    'signal': signal,
                    'price': price,
                    'quantity': quantity,
                    'order_id': order_id,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'pnl': 0  # Initial PnL is 0
                })
            print(f"Trade details written to {self.csv_file_path}")

        except Exception as e:
            print(f"Error writing trade details to CSV: {e}")
            logging.error(f"CSV Write Error: {e}")

    def monitor_open_trades(self):
        """
        Monitors open trades and closes them based on stop-loss and take-profit levels.
        """
        try:
            current_price = self.binance_data.get_current_price()

            if current_price is None:
                print("Current price is None. Skipping trade monitoring.")
                return

            current_price = float(current_price)

            for trade in self.open_trades:
                if trade['is_open']:
                    if trade['signal'] == 'buy':
                        if current_price <= trade['stop_loss'] or current_price >= trade['take_profit']:
                            # Close buy order
                            close_signal = 'sell'
                            pnl = (current_price - trade['price']) * trade['quantity']
                            self.close_trade(trade, current_price, pnl)
                    elif trade['signal'] == 'sell':
                        if current_price >= trade['stop_loss'] or current_price <= trade['take_profit']:
                            # Close sell order
                            close_signal = 'buy'
                            pnl = (trade['price'] - current_price) * trade['quantity']
                            self.close_trade(trade, current_price, pnl)
        except Exception as e:
            print(f"Error monitoring open trades: {e}")
            logging.error(f"Monitoring Error: {e}")

    def close_trade(self, trade, close_price, pnl):
        """
        Closes an open trade.
        """
        try:
            if trade['signal'] == 'buy':
                order = self.client.order_market_sell(symbol=self.symbol, quantity=trade['quantity'])
            elif trade['signal'] == 'sell':
                order = self.client.order_market_buy(symbol=self.symbol, quantity=trade['quantity'])
            else:
                print(f"Invalid signal {trade['signal']} in trade data.")
                return

            order_id = order['orderId']
            trade['is_open'] = False
            trade['pnl'] = pnl

            self.update_csv_with_pnl(trade['timestamp'], pnl)
            print(f"Closed trade with order ID: {order_id}, PnL: {pnl}")

        except exceptions.BinanceAPIException as e:
            print(f"Binance API exception: {e}")
            logging.error(f"Binance API exception: {e}")
        except Exception as e:
            print(f"Error closing trade: {e}")
            logging.error(f"Trade Closing Error: {e}")

    def update_csv_with_pnl(self, timestamp, pnl):
        """
        Updates the PnL for a specific trade in the CSV file.
        """
        try:
            # Read all rows from the CSV file
            with open(self.csv_file_path, mode='r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)

            # Find the row with the matching timestamp and update the PnL
            for row in rows:
                if row['timestamp'] == str(timestamp):
                    row['pnl'] = str(pnl)
                    break

            # Write the updated rows back to the CSV file
            with open(self.csv_file_path, mode='w', newline='') as csvfile:
                fieldnames = ['timestamp', 'signal', 'price', 'quantity', 'order_id', 'stop_loss', 'take_profit', 'pnl']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"PnL updated in {self.csv_file_path} for timestamp: {timestamp}")

        except Exception as e:
            print(f"Error updating PnL in CSV: {e}")
            logging.error(f"CSV Update Error: {e}")

    def round_quantity(self, quantity):
        """
        Rounds the quantity to the correct precision based on Binance's LOT_SIZE filter.
        """
        try:
            symbol_info = self.client.get_symbol_info(symbol=self.symbol)
            if not symbol_info:
                print(f"Could not retrieve symbol information for {self.symbol}")
                return None

            filters = symbol_info['filters']
            lot_size_filter = next((f for f in filters if f['filterType'] == 'LOT_SIZE'), None)

            if not lot_size_filter:
                print("Could not retrieve LOT_SIZE filter.")
                return None

            step_size = float(lot_size_filter['stepSize'])
            # Round down to the nearest step size
            rounded_quantity = round(quantity / step_size) * step_size
            return rounded_quantity

        except exceptions.BinanceAPIException as e:
            print(f"Binance API Exception: {e}")
            logging.error(f"Binance API Exception in round_quantity: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            logging.error(f"Unexpected error in round_quantity: {e}")
            return None

# Instantiate the TradingBot and run it
bot = TradingBot()
bot.run() 