import time
print("time module imported")
import csv
print("csv module imported")
import os  # Import the 'os' module
print("os module imported")
from config import SYMBOL, ORDER_TYPE, LIMIT_PRICE_OFFSET, STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE, MAX_POSITION_SIZE_PERCENTAGE, API_KEY, API_SECRET
print("Config imported")
from binance_data import BinanceData
print("BinanceData imported")
from order_flow_analyzer import OrderFlowAnalyzer
print("OrderFlowAnalyzer imported")
from binance import Client, exceptions
print("Binance Client and exceptions imported")

print("Trading Bot script starting...")
print(f"Executing trading_bot.py from: {os.path.abspath(__file__)}")
print(f"API_KEY from config: {API_KEY}")  # Print API_KEY
print(f"API_SECRET from config: {API_SECRET}")  # Print API_SECRET

class TradingBot:
    def __init__(self):
        print("TradingBot.__init__ starting...")
        try:
            print("Creating OrderFlowAnalyzer...")
            self.order_flow_analyzer = OrderFlowAnalyzer()
            print("OrderFlowAnalyzer created.")

            # Delay BinanceData creation
            #print("Creating BinanceData...")
            self.binance_data = None #BinanceData(self.order_flow_analyzer)
            #print("BinanceData created.")

            # Explicitly set testnet=True here
            #print("Creating Binance Client in TradingBot...")
            #self.client = self.binance_data.client #Client(API_KEY, API_SECRET, testnet=True) # Use the same client
            #print("Binance Client in TradingBot created.")
            self.client = None

            self.in_position = False
            self.entry_price = 0.0
            self.stop_loss_price = 0.0
            self.take_profit_price = 0.0
            self.csv_file = f"trades_{SYMBOL}_{int(time.time())}.csv"  # Unique filename
            self.csv_writer = None  # Initialize csv_writer
            print("Calling initialize_csv...")
            self.initialize_csv()
            print("initialize_csv completed.")
            print("TradingBot.__init__ finished.")

        except Exception as e:
            print(f"Error in TradingBot.__init__: {e}")
            raise  # Re-raise the exception to see the full traceback

    def initialize_csv(self):
        """Creates the CSV file and writes the header row."""
        file_exists = os.path.isfile(self.csv_file)
        with open(self.csv_file, 'a', newline='') as file:
            self.csv_writer = csv.writer(file)
            if not file_exists:
                self.csv_writer.writerow([
                    "Timestamp", "Signal", "Entry Price", "Quantity",
                    "Stop Loss", "Take Profit", "Exit Price", "Exit Timestamp", "PNL"
                ])

    def write_trade_to_csv(self, signal, entry_price, quantity, stop_loss, take_profit, exit_price=None, exit_timestamp=None, pnl=None):
        """Writes trade data to the CSV file."""
        timestamp = int(time.time())  # Use consistent timestamp format
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp, signal, entry_price, quantity, stop_loss,
                take_profit, exit_price, exit_timestamp, pnl
            ])

    def run(self):
        try:
            # Initialize BinanceData here, after OrderFlowAnalyzer
            print("Creating BinanceData in run()...")
            self.binance_data = BinanceData(self.order_flow_analyzer)
            print("BinanceData created in run().")
            self.client = self.binance_data.client

            self.binance_data.start()
            while True:
                signal = self.order_flow_analyzer.check_for_signals()
                if signal:
                    print(f"Signal detected: {signal}")
                    self.execute_trade(signal)
                self.check_stop_loss_take_profit()
                time.sleep(1)  # Check every second. Adjust as needed.
        except Exception as e:
            print(f"Error in TradingBot.run(): {e}")
            raise
        finally:
            if self.binance_data:
                self.binance_data.stop()

    def execute_trade(self, signal):
        if self.in_position:
            print("Already in a position. Skipping trade execution.")
            return  # Already in a position, don't enter another

        current_price = self.binance_data.get_current_price()
        if current_price is None:
            print("Could not retrieve current price.  Skipping trade execution.")
            return

        # Get account balance and calculate quantity
        balance = self.get_available_balance()
        if balance is None:
            return
        quantity = self.calculate_quantity(balance, current_price)

        try:
            if signal == "buy":
                if ORDER_TYPE == 'MARKET':
                    order = self.client.order_market_buy(symbol=SYMBOL, quantity=quantity)
                elif ORDER_TYPE == 'LIMIT':
                    bid_price = max(self.order_flow_analyzer.order_book['bids'].keys()) #best bid
                    limit_price = round(bid_price * (1 + LIMIT_PRICE_OFFSET), 8) #ensure correct decimal places
                    order = self.client.order_limit_buy(symbol=SYMBOL, quantity=quantity, price=str(limit_price))
                else:
                    raise ValueError(f"Invalid ORDER_TYPE: {ORDER_TYPE}")

                print(f"Buy order placed: {order}")
                # Extract actual entry price from order fills (for MARKET orders)
                self.entry_price = float(order['fills'][0]['price']) if order['fills'] else current_price
                self.set_stop_loss_take_profit(signal)
                self.in_position = True
                # Write trade entry to CSV
                self.write_trade_to_csv(signal, self.entry_price, quantity, self.stop_loss_price, self.take_profit_price)

            elif signal == "sell":
                if ORDER_TYPE == 'MARKET':
                    order = self.client.order_market_sell(symbol=SYMBOL, quantity=quantity)
                elif ORDER_TYPE == 'LIMIT':
                    ask_price = min(self.order_flow_analyzer.order_book['asks'].keys()) #best ask
                    limit_price = round(ask_price * (1- LIMIT_PRICE_OFFSET), 8) #ensure correct decimal places
                    order = self.client.order_limit_sell(symbol=SYMBOL, quantity=quantity, price=str(limit_price))
                else:
                    raise ValueError(f"Invalid ORDER_TYPE: {ORDER_TYPE}")

                print(f"Sell order placed: {order}")
                self.entry_price = float(order['fills'][0]['price']) if order['fills'] else current_price
                self.set_stop_loss_take_profit(signal)
                self.in_position = True
                # Write trade entry to CSV
                self.write_trade_to_csv(signal, self.entry_price, quantity, self.stop_loss_price, self.take_profit_price)

        except exceptions.BinanceAPIException as e:
            print(f"Binance API Exception: {e}")
        except exceptions.BinanceOrderException as e:
            print(f"Binance Order Exception: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get_available_balance(self):
        """Gets the available balance of the quote currency (e.g., USDT)."""
        try:
            account_info = self.client.get_account()
            for balance in account_info['balances']:
                if balance['asset'] == SYMBOL[-4:]:  # Assumes quote currency is last 4 chars (USDT, BUSD, etc.)
                    return float(balance['free'])
            print(f"Could not find balance for {SYMBOL[-4:]} in account.")
            return None # Quote currency not found
        except exceptions.BinanceAPIException as e:
            print(f"Binance API Exception while getting balance: {e}")
            return None

    def calculate_quantity(self, balance, price):
        """Calculates the quantity to trade based on risk parameters."""
        position_size = balance * MAX_POSITION_SIZE_PERCENTAGE
        quantity = position_size / price
        # Adjust quantity based on Binance's minimum order size and step size (lot size)
        info = self.client.get_symbol_info(SYMBOL)
        step_size = None
        for f in info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                step_size = float(f['stepSize'])
                break
        if step_size is None:
            print("Could not determine step size for quantity.")
            return 0
        quantity = round(quantity / step_size) * step_size
        return quantity

    def set_stop_loss_take_profit(self, signal):
        """Sets stop-loss and take-profit prices based on entry price and percentages."""
        if signal == "buy":
            self.stop_loss_price = self.entry_price * (1 - STOP_LOSS_PERCENTAGE)
            self.take_profit_price = self.entry_price * (1 + TAKE_PROFIT_PERCENTAGE)
        elif signal == "sell":
            self.stop_loss_price = self.entry_price * (1 + STOP_LOSS_PERCENTAGE)
            self.take_profit_price = self.entry_price * (1 - TAKE_PROFIT_PERCENTAGE)

        print(f"Stop Loss: {self.stop_loss_price}, Take Profit: {self.take_profit_price}")

    def check_stop_loss_take_profit(self):
        """Checks if stop loss or take profit has been hit and exits the position."""
        if not self.in_position:
            return

        current_price = self.binance_data.get_current_price()
        if current_price is None:
            print("Could not retrieve current price for stop loss/take profit check.")
            return

        if self.signal == "buy":  # Long position
            if current_price <= self.stop_loss_price or current_price >= self.take_profit_price:
                self.exit_position(current_price, "stop_loss" if current_price <= self.stop_loss_price else "take_profit")
        elif self.signal == "sell":  # Short position (Not implemented in your code)
            if current_price >= self.stop_loss_price or current_price <= self.take_profit_price:
                self.exit_position(current_price, "stop_loss" if current_price >= self.stop_loss_price else "take_profit")

    def exit_position(self, exit_price, reason):
        """Exits the current position."""
        if not self.in_position:
            return

        try:
            quantity = self.get_current_position_size()
            if quantity is None:
                return

            if self.signal == "buy":
                if ORDER_TYPE == 'MARKET':
                    order = self.client.order_market_sell(symbol=SYMBOL, quantity=quantity)
                elif ORDER_TYPE == 'LIMIT':
                    ask_price = min(self.order_flow_analyzer.order_book['asks'].keys()) #best ask
                    limit_price = round(ask_price * (1 - LIMIT_PRICE_OFFSET), 8) #ensure correct decimal places
                    order = self.client.order_limit_sell(symbol=SYMBOL, quantity=quantity, price=str(limit_price))
                else:
                    raise ValueError(f"Invalid ORDER_TYPE: {ORDER_TYPE}")
                print(f"Sell order placed to exit position: {order}")
            elif self.signal == "sell": #not implemented
                if ORDER_TYPE == 'MARKET':
                    order = self.client.order_market_buy(symbol=SYMBOL, quantity=quantity)
                elif ORDER_TYPE == 'LIMIT':
                    bid_price = max(self.order_flow_analyzer.order_book['bids'].keys()) #best bid
                    limit_price = round(bid_price * (1 + LIMIT_PRICE_OFFSET), 8) #ensure correct decimal places
                    order = self.client.order_limit_buy(symbol=SYMBOL, quantity=quantity, price=str(limit_price))
                else:
                    raise ValueError(f"Invalid ORDER_TYPE: {ORDER_TYPE}")
                print(f"Buy order placed to exit position: {order}")

            self.in_position = False
            exit_timestamp = int(time.time())
            pnl = (exit_price - self.entry_price) * quantity if self.signal == "buy" else (self.entry_price - exit_price) * quantity
            print(f"Position exited. Reason: {reason}, Exit Price: {exit_price}, PNL: {pnl}")

            # Write trade exit to CSV
            self.write_trade_to_csv(self.signal, self.entry_price, quantity, self.stop_loss_price, self.take_profit_price, exit_price, exit_timestamp, pnl)

            self.entry_price = 0.0
            self.stop_loss_price = 0.0
            self.take_profit_price = 0.0

        except exceptions.BinanceAPIException as e:
            print(f"Binance API Exception while exiting position: {e}")
        except exceptions.BinanceOrderException as e:
            print(f"Binance Order Exception while exiting position: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while exiting position: {e}")

# Instantiate the TradingBot and run it
bot = TradingBot()
bot.run() 