import time
print("time module imported")
import numpy as np
print("numpy imported")
from collections import deque
print("deque imported")
from config import (
    MOVING_AVERAGE_WINDOW,
    VOLUME_SPIKE_MULTIPLIER,
    ORDER_FLOW_DELTA_Z_SCORE_THRESHOLD,
    CONFIRMATION_TICKS,
    MINIMUM_LIQUIDITY_THRESHOLD,
    TICK_INTERVAL,
    ATR_PERIOD,
    CONFIRMATION_THRESHOLD
)
from utils import calculate_z_score
from indicators import ATR  # Import the ATR class

print("order_flow_analyzer.py script starting...")

class OrderFlowAnalyzer:
    def __init__(self):
        print("OrderFlowAnalyzer.__init__ starting...")
        try:
            self.trade_volumes = deque(maxlen=MOVING_AVERAGE_WINDOW)
            self.volume_deltas = deque(maxlen=MOVING_AVERAGE_WINDOW)
            self.order_book_slopes = deque(maxlen=MOVING_AVERAGE_WINDOW)
            self.order_flow_deltas = deque(maxlen=MOVING_AVERAGE_WINDOW)  # Store raw deltas
            self.order_flow_delta_z_scores = deque(maxlen=MOVING_AVERAGE_WINDOW)  # Store Z-scores
            self.bid_depth = 0.0
            self.ask_depth = 0.0
            self.order_book_slope = 0.0
            self.trade_volume = 0.0
            self.order_flow_delta = 0.0 # Current order flow delta
            self.order_flow_delta_z_score = 0.0 # Current Z-score
            self.order_flow_delta_history = deque(maxlen=MOVING_AVERAGE_WINDOW)
            self.volume_history = deque(maxlen=MOVING_AVERAGE_WINDOW)
            self.bid_volume_history = deque(maxlen=MOVING_AVERAGE_WINDOW) # added
            self.ask_volume_history = deque(maxlen=MOVING_AVERAGE_WINDOW) # added
            self.last_timestamp = None
            self.confirmation_counter = 0
            self.current_signal = None  # 'buy', 'sell', or None
            self.recent_trades = deque(maxlen=MOVING_AVERAGE_WINDOW) # added to store for tick interval
            self.signal_history = deque(maxlen=CONFIRMATION_TICKS)  # Store recent signals
            self.atr = ATR(ATR_PERIOD)  # Initialize ATR indicator
            self.prices = deque(maxlen=MOVING_AVERAGE_WINDOW)  # Store recent prices for ATR
            self.signal_sum = 0  # Sum of recent signals for moving average
            print("OrderFlowAnalyzer.__init__ finished.")
        except Exception as e:
            print(f"Error in OrderFlowAnalyzer.__init__: {e}")
            raise

    def calculate_moving_average(self, data_series):
        if len(data_series) == 0:
            return 0
        return sum(data_series) / len(data_series)

    def calculate_volume_delta(self):
        if len(self.trade_volumes) < 2:
            return 0
        return self.trade_volumes[-1] - self.trade_volumes[-2]

    def calculate_order_flow_delta(self, bid_depth, ask_depth):
        # Simple order flow delta calculation
        return bid_depth - ask_depth

    def calculate_order_book_slope(self, bid_depth, ask_depth):
        # Calculate a simple slope using bid and ask depth
        if (bid_depth + ask_depth) == 0:
            return 0  # Avoid division by zero
        return (bid_depth - ask_depth) / (bid_depth + ask_depth)

    def calculate_z_score(self, data_point, data_series):
        if not data_series:
            return 0  # Avoid calculations on empty series
        mean = np.mean(data_series)
        std_dev = np.std(data_series)
        if std_dev == 0:
            return 0  # Avoid division by zero
        return (data_point - mean) / std_dev

    def update_order_flow_data(self, bid_depth, ask_depth, trade_volume):
        # Update depths
        self.bid_depth = bid_depth
        self.ask_depth = ask_depth
        self.trade_volume = trade_volume

        # Calculate and append the new order flow delta
        order_flow_delta = self.calculate_order_flow_delta(bid_depth, ask_depth)
        self.order_flow_deltas.append(order_flow_delta)
        self.order_flow_delta = order_flow_delta # Update current value

        # Calculate and append the new order book slope
        order_book_slope = self.calculate_order_book_slope(bid_depth, ask_depth)
        self.order_book_slopes.append(order_book_slope)
        self.order_book_slope = order_book_slope # Update current value

        # Calculate and append the new trade volume
        self.trade_volumes.append(trade_volume)

        # Calculate the Z-score for the order flow delta
        order_flow_delta_z_score = self.calculate_z_score(order_flow_delta, list(self.order_flow_deltas))
        self.order_flow_delta_z_scores.append(order_flow_delta_z_score)
        self.order_flow_delta_z_score = order_flow_delta_z_score # Update current value

    def process_order_book(self, order_book):
        """Analyzes order book data for imbalances."""
        try:
            # Check if the order book is empty
            if not order_book['bids'] or not order_book['asks']:
                print("Order book is empty. Skipping processing.")
                self.order_book_imbalance = 0
                return

            # Get the best bid and ask prices and sizes
            best_bid_price = float(list(order_book['bids'].keys())[0])
            best_bid_size = order_book['bids'][best_bid_price]
            best_ask_price = float(list(order_book['asks'].keys())[0])
            best_ask_size = order_book['asks'][best_ask_price]

            # Calculate the bid-ask spread
            bid_ask_spread = best_ask_price - best_bid_price

            # Calculate the order book imbalance
            order_book_imbalance = best_bid_size - best_ask_size

            # Calculate order book depth (sum of sizes for the top N levels)
            bid_depth = sum(float(order_book['bids'][price]) for price in list(order_book['bids'].keys())[:min(5, len(order_book['bids']))])  # Top 5 bid levels
            ask_depth = sum(float(order_book['asks'][price]) for price in list(order_book['asks'].keys())[:min(5, len(order_book['asks']))])  # Top 5 ask levels

            # Calculate order book slope (difference in depth between bid and ask)
            order_book_slope = bid_depth - ask_depth

            # Print the order book data for monitoring
            print(f"Best Bid: {best_bid_price} @ {best_bid_size}, Best Ask: {best_ask_price} @ {best_ask_size}, Imbalance: {order_book_imbalance}, Bid Depth: {bid_depth}, Ask Depth: {ask_depth}, Slope: {order_book_slope}")

            # You can add more sophisticated analysis here, such as:
            # - Tracking changes in the order book over time
            # - Identifying large orders (icebergs)
            # - Detecting order book spoofing

            # Store the order book data for use in check_for_signals
            self.best_bid_price = best_bid_price
            self.best_bid_size = best_bid_size
            self.best_ask_price = best_ask_price
            self.best_ask_size = best_ask_size
            self.order_book_imbalance = order_book_imbalance
            self.bid_depth = bid_depth
            self.ask_depth = ask_depth
            self.order_book_slope = order_book_slope

        except Exception as e:
            print(f"Error processing order book: {e}")

    def process_trade(self, price, trade):
        """Processes individual trade data."""
        #print(f"Processing trade: Price: {price}, Volume: {volume}") #debug
        #print(f"Processing trade: {trade}") #debug
        if not isinstance(trade, dict):
            print(f"Invalid trade format: {trade}")
            return

        if 'price' not in trade or 'volume' not in trade:
            print(f"Invalid trade format, missing 'price' or 'volume': {trade}")
            return

        volume = trade["volume"]
        #print(f"Processing trade: Volume: {volume}") #debug
        if volume is None:
            print(f"Trade volume is None, skipping update.")
            return

        if len(self.volume_history) >= MOVING_AVERAGE_WINDOW:
            self.trade_volumes.popleft()  # Remove the oldest trade volume

        self.trade_volumes.append(volume)
        self.volume_history.append(volume)
        print(f"Trade Volume: {volume}, Recent Trades Length: {len(self.recent_trades)}")
        self.recent_trades.append(volume) # was volume # CORRECT THIS LINE
        self.prices.append(trade["price"])  # Append the trade price

        # Update ATR
        self.atr.update(trade["price"], trade["price"], trade["price"])  # high, low, close are the same for trade price

    def check_for_signals(self):
        """Checks for trading signals based on order flow analysis."""
        if len(self.volume_history) < MOVING_AVERAGE_WINDOW:
            print("Not enough volume data to generate signals.")
            return None

        # Calculate order flow delta (difference between bid and ask volume)
        order_flow_delta = np.sum(self.volume_history)
        self.order_flow_delta_history.append(order_flow_delta)

        # Calculate Z-score of order flow delta
        order_flow_delta_z_score = calculate_z_score(self.order_flow_delta_history)
        print(f"Order Flow Delta Z-Score: {order_flow_delta_z_score}")

        # --- Incorporate Order Book Data into Signal Logic ---
        # Example:
        if self.order_book_imbalance > MINIMUM_LIQUIDITY_THRESHOLD and order_flow_delta_z_score > ORDER_FLOW_DELTA_Z_SCORE_THRESHOLD:
            signal = "buy"
            print("Buy signal generated based on order book imbalance and order flow!")
        elif self.order_book_imbalance < -MINIMUM_LIQUIDITY_THRESHOLD and order_flow_delta_z_score < -ORDER_FLOW_DELTA_Z_SCORE_THRESHOLD:
            signal = "sell"
            print("Sell signal generated based on order book imbalance and order flow!")
        else:
            signal = None

        if abs(order_flow_delta_z_score) > ORDER_FLOW_DELTA_Z_SCORE_THRESHOLD:
            if order_flow_delta_z_score > 0:
                signal = "buy"
                print("Buy signal generated!")
            else:
                signal = "sell"
                print("Sell signal generated!")

            # Confirmation logic (simplified)
            self.confirmation_counter += 1
            if self.confirmation_counter >= CONFIRMATION_TICKS:
                self.confirmation_counter = 0
                return signal
            else:
                print(f"Signal needs {CONFIRMATION_TICKS - self.confirmation_counter} more ticks to confirm.")
                return None
        else:
            self.confirmation_counter = 0
            return None 