import numpy as np
from collections import deque

class ATR:
    def __init__(self, period):
        self.period = period
        self.highs = deque(maxlen=period)
        self.lows = deque(maxlen=period)
        self.closes = deque(maxlen=period)
        self.true_ranges = deque(maxlen=period)
        self.atr = None

    def update(self, high, low, close):
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)

        if len(self.closes) < 2:
            return  # Not enough data to calculate TR

        # True Range calculation
        current_high = self.highs[-1]
        current_low = self.lows[-1]
        previous_close = self.closes[-2]

        tr = max(
            current_high - current_low,
            abs(current_high - previous_close),
            abs(current_low - previous_close)
        )
        self.true_ranges.append(tr)

        if len(self.true_ranges) == self.period:
            self.atr = np.mean(self.true_ranges)
        elif len(self.true_ranges) > self.period:
            # Wilder's smoothing method
            self.atr = (self.atr * (self.period - 1) + tr) / self.period

    def get_atr(self):
        return self.atr 