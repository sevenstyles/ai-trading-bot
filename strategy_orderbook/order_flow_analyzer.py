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

# Trivial change to force re-import
class OrderFlowAnalyzer:
    def __init__(self, trading_bot):
        print("OrderFlowAnalyzer.__init__ starting...")
        try:
            self.trading_bot = trading_bot
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
            self.buy_volume_history = deque(maxlen=MOVING_AVERAGE_WINDOW)  # Track buy volume
            self.sell_volume_history = deque(maxlen=MOVING_AVERAGE_WINDOW) # Track sell volume
            self.signal_strength_history = deque(maxlen=MOVING_AVERAGE_WINDOW) # For confirmation
        except Exception as e:
            print(f"Error in OrderFlowAnalyzer.__init__: {e}")
            raise
    
    def calculate_moving_average(self, data_series):
        if len(data_series) == 0:
            return 0
        return sum(data_series) / len(data_series)
    
    def calculate_volume_delta(self):
        # Not used anymore, replaced by order flow delta
        return 0
    
    def calculate_order_flow_delta(self, bid_depth, ask_depth):
        # Not used directly, calculated in check_for_signals now
        return 0
    
    
    def calculate_order_book_slope(self, bid_depth, ask_depth):
        # Calculate a simple slope using bid and ask depth.  Not used directly anymore.
        if (bid_depth + ask_depth) == 0:
            return 0  # Avoid division by zero
        return (bid_depth - ask_depth) / (bid_depth + ask_depth)
    
    def update_order_flow_data(self, bid_depth, ask_depth, trade_volume):
        # Update depths - these are now updated in process_order_book
        self.bid_depth = bid_depth
        self.ask_depth = ask_depth
        self.trade_volume = trade_volume # Still used for liquidity check
    
        # order_flow_delta and z-score are now calculated in check_for_signals
    
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
    
            # --- Enhanced Order Book Imbalance Calculation ---
            # bid_depth_weighted = 0
            # ask_depth_weighted = 0
            # weights = [0.8, 0.4, 0.2, 0.1, 0.05]  # Less aggressive weights

            # # Log raw depths here
            raw_bid_depth = sum(order_book['bids'].values())
            raw_ask_depth = sum(order_book['asks'].values())
            print(f"Raw Bid Depth: {raw_bid_depth}, Raw Ask Depth: {raw_ask_depth}")

            # for i in range(min(5, len(order_book['bids']))):
            #     price = list(order_book['bids'].keys())[i]
            #     bid_depth_weighted += order_book['bids'][price] * weights[i]
            # for i in range(min(5, len(order_book['asks']))):
            #     price = list(order_book['asks'].keys())[i]
            #     ask_depth_weighted += order_book['asks'][price] * weights[i]
    
            # print(f"Bid Depth Before Weighting: {bid_depth_weighted}")
            # print(f"Ask Depth Before Weighting: {ask_depth_weighted}")
    
            # self.bid_depth = bid_depth_weighted # Update for liquidity check
            # self.ask_depth = ask_depth_weighted  # Update for liquidity check
            self.bid_depth = raw_bid_depth
            self.ask_depth = raw_ask_depth
            self.order_book_imbalance = self.bid_depth - self.ask_depth
            # Normalize the imbalance
            self.order_book_imbalance_normalized = self.order_book_imbalance / (self.bid_depth + self.ask_depth) if (self.bid_depth + self.ask_depth) >0 else 0
            print(f"Order Book: bid_depth={self.bid_depth}, ask_depth={self.ask_depth}, imbalance={self.order_book_imbalance}, normalized={self.order_book_imbalance_normalized}")
    
    
            # Calculate order book slope (difference in depth between bid and ask)
            # Not used in signal generation anymore, but keeping for potential future use
            order_book_slope = self.order_book_imbalance
    
            # Print the order book data for monitoring
            print(f"Best Bid: {best_bid_price} @ {best_bid_size}, Best Ask: {best_ask_price} @ {best_ask_size}, Imbalance: {self.order_book_imbalance}, Bid Depth: {self.bid_depth}, Ask Depth: {self.ask_depth}, Slope: {order_book_slope}")
    
            # Store the order book data for use in check_for_signals
            self.best_bid_price = best_bid_price
            self.best_bid_size = best_bid_size
            self.best_ask_price = best_ask_price
            self.best_ask_size = best_ask_size
            # self.order_book_imbalance = order_book_imbalance # Replaced by weighted calculation
            # self.bid_depth = bid_depth  # Replaced
            # self.ask_depth = ask_depth  # Replaced
            self.order_book_slope = order_book_slope
    
        except Exception as e:
            print(f"Error processing order book: {e}")
    
    def process_trade(self, price, trade):
        """Processes individual trade data."""
        if not isinstance(trade, dict):
            print(f"Invalid trade format: {trade}")
            return
    
        if 'price' not in trade or 'volume' not in trade or 'is_buy' not in trade:
            print(f"Invalid trade format, missing 'price', 'volume', or 'is_buy': {trade}")
            return
    
        volume = trade["volume"]
        is_buy = trade['is_buy']
    
        if volume is None:
            print(f"Trade volume is None, skipping update.")
            return
    
        # Update buy/sell volume history
        if is_buy:
            self.buy_volume_history.appendleft(volume)
            self.sell_volume_history.appendleft(0) # Add 0 to sell to keep lengths consistent
        else:
            self.sell_volume_history.appendleft(volume)
            self.buy_volume_history.appendleft(0) # Add 0 to buy
    
        self.recent_trades.append(volume)  # Keep for potential future use
        self.prices.append(trade["price"])  # Append the trade price
    
        # Update ATR
        self.atr.update(trade["price"], trade["price"], trade["price"])  # high, low, close are the same for trade price
    
    def check_for_signals(self, raw_liquidity):
        """Checks for trading signals based on order flow analysis."""
    
        # --- General Liquidity Check ---
        #raw_liquidity = self.bid_depth + self.ask_depth  # Use raw depths
        print(f"Current liquidity: {raw_liquidity}")
        if raw_liquidity < MINIMUM_LIQUIDITY_THRESHOLD:
            print("Liquidity below minimum threshold. Skipping signal generation.")
            return None
    
        if len(self.buy_volume_history) < MOVING_AVERAGE_WINDOW:
            print("Not enough volume data to generate signals.")
            return None
    
        # --- Calculate Order Flow Delta and Z-score ---
        order_flow_delta = sum(self.buy_volume_history) - sum(self.sell_volume_history)
        self.order_flow_delta_history.append(order_flow_delta)

        if len(self.order_flow_delta_history) < 2:
            print("Not enough order flow delta history to calculate Z-score.")
            return None

        order_flow_delta_z_score = self.calculate_z_score(order_flow_delta, list(self.order_flow_delta_history))
        print(f"Order Flow Delta Z-Score: {order_flow_delta_z_score}, Order Flow Delta: {order_flow_delta}")
    
        # --- Unified Signal Logic ---
        z_score_weight = 1.0  # Configurable weight
        imbalance_weight = 1.0 # Configurable weight
    
        signal_strength = (
            order_flow_delta_z_score * z_score_weight +
            self.order_book_imbalance_normalized * imbalance_weight
        )
        self.signal_strength_history.append(signal_strength)
    
        # --- Improved Confirmation (Moving Average of Signal Strength) ---
        signal_strength_ma = self.calculate_moving_average(self.signal_strength_history)
    
        print(f"Signal Strength: {signal_strength}, Moving Average: {signal_strength_ma}")
    
        if signal_strength_ma > ORDER_FLOW_DELTA_Z_SCORE_THRESHOLD:
            signal = "buy"
            print("Buy signal generated!")
        elif signal_strength_ma < -ORDER_FLOW_DELTA_Z_SCORE_THRESHOLD:
            signal = "sell"
            print("Sell signal generated!")
        else:
            signal = None
    
        return signal

    def calculate_z_score(self, data_point, data_series):
        """Calculates the Z-score of a single data point against a series."""
        if len(data_series) < 2:  # Need at least 2 data points for standard deviation
            return 0  # Return 0 if not enough data points

        std_dev = np.std(data_series)
        if std_dev == 0:
            return 0  # Return 0 if standard deviation is zero
        else:
            return (data_point - np.mean(data_series)) / std_dev