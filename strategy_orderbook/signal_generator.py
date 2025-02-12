import numpy as np
from collections import deque
from config import (
    MOVING_AVERAGE_WINDOW,
    ORDER_FLOW_DELTA_Z_SCORE_THRESHOLD,
    MINIMUM_LIQUIDITY_THRESHOLD
)

class SignalGenerator:
    def __init__(self):
        print("SignalGenerator.__init__ starting...")
        self.order_flow_delta_history = deque([0] * 5, maxlen=MOVING_AVERAGE_WINDOW)
        self.signal_strength_history = deque(maxlen=MOVING_AVERAGE_WINDOW)
        print("SignalGenerator.__init__ finished.")

    def calculate_moving_average(self, data_series):
        if len(data_series) == 0:
            return 0
        return sum(data_series) / len(data_series)

    def calculate_z_score(self, data_point, data_series):
        """Calculates the Z-score of a single data point against a series."""
        if len(data_series) < 2:  # Need at least 2 data points for standard deviation
            return 0  # Return 0 if not enough data points

        std_dev = np.std(data_series)
        if std_dev == 0:
            return 0  # Return 0 if standard deviation is zero
        else:
            return (data_point - np.mean(data_series)) / std_dev

    def check_for_signals(self, raw_liquidity, bid_depth, ask_depth):
        """Checks for trading signals based on order flow analysis."""

        # --- General Liquidity Check ---
        print(f"Current liquidity: {raw_liquidity}")
        if raw_liquidity < MINIMUM_LIQUIDITY_THRESHOLD:
            print("Liquidity below minimum threshold. Skipping signal generation.")
            return None

        # --- Calculate Order Flow Delta and Z-score ---
        order_flow_delta = bid_depth - ask_depth  # Use raw depths from order book
        self.order_flow_delta_history.append(order_flow_delta)

        if len(self.order_flow_delta_history) < 2:
            print("Not enough order flow delta history to calculate Z-score.")
            return None

        order_flow_delta_z_score = self.calculate_z_score(order_flow_delta, list(self.order_flow_delta_history))
        print(f"Order Flow Delta Z-Score: {order_flow_delta_z_score}, Order Flow Delta: {order_flow_delta}")

        # --- Unified Signal Logic ---
        z_score_weight = 1.0  # Configurable weight
        imbalance_weight = 1.0  # Configurable weight

        # Calculate normalized order book imbalance
        order_book_imbalance_normalized = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0

        signal_strength = (
            order_flow_delta_z_score * z_score_weight +
            order_book_imbalance_normalized * imbalance_weight
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