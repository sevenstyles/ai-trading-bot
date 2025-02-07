import numpy as np
from collections import deque
from config import (
    MOVING_AVERAGE_WINDOW,
    VOLUME_SPIKE_MULTIPLIER,
    ORDER_FLOW_DELTA_Z_SCORE_THRESHOLD,
    CONFIRMATION_TICKS,
    MINIMUM_LIQUIDITY_THRESHOLD,
    TICK_INTERVAL
)
from utils import calculate_z_score

print("order_flow_analyzer.py script starting...")

class OrderFlowAnalyzer:
    def __init__(self):
        print("OrderFlowAnalyzer.__init__ starting...")
        try:
            # Uncomment the deque initializations
            self.order_flow_delta_history = deque(maxlen=MOVING_AVERAGE_WINDOW)
            self.volume_history = deque(maxlen=MOVING_AVERAGE_WINDOW)
            self.bid_volume_history = deque(maxlen=MOVING_AVERAGE_WINDOW) # added
            self.ask_volume_history = deque(maxlen=MOVING_AVERAGE_WINDOW) # added
            self.last_timestamp = None
            self.confirmation_counter = 0
            self.current_signal = None  # 'buy', 'sell', or None
            self.recent_trades = deque(maxlen=MOVING_AVERAGE_WINDOW) # added to store for tick interval
            print("OrderFlowAnalyzer.__init__ finished.")
        except Exception as e:
            print(f"Error in OrderFlowAnalyzer.__init__: {e}")
            raise

    def process_order_book(self, order_book):
        # Placeholder for order book absorption analysis.  This is a more advanced
        # concept and requires significant logic to implement accurately.
        # This simplified version doesn't use the order book directly for signals,
        # but you could add logic here to detect large orders being absorbed.

        # Basic idea (you'll need to expand this):
        # 1. Track changes in order book levels over time.
        # 2. Identify large orders (relative to average size).
        # 3. If a large sell order appears at a bid level, and the price doesn't drop,
        #    it *might* indicate absorption.  This is NOT foolproof.

        pass  # No order book analysis in this simplified version

    def process_trade(self, trade):
        """Processes individual trade data."""
        volume = trade["price"] * trade["quantity"]
        self.volume_history.append(volume)
        print(f"Trade Volume: {volume}, Recent Trades Length: {len(self.recent_trades)}")
        self.recent_trades.append(trade["price"]) # was volume

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