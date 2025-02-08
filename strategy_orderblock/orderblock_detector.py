import pandas as pd
import numpy as np
import ta
from scipy.signal import argrelextrema
from collections import namedtuple

# Define namedtuples for data organization
OrderBlock = namedtuple('OrderBlock', ['bar_high', 'bar_low', 'bar_time', 'bias', 'index'])
SwingPoint = namedtuple('SwingPoint', ['current_level', 'last_level', 'crossed', 'bar_time', 'bar_index'])
Trend = namedtuple('Trend', ['bias'])

BULLISH = 1
BEARISH = -1

def identify_swing_points(df, swing_length=50):
    """Identifies swing points based on parsed high/low, replicating Pine Script logic."""
    swing_highs = []
    swing_lows = []

    def leg(size, i):
        """Replicates the leg() function from the Pine Script."""
        if i < size:
            return 0  # Not enough data

        new_leg_high = df['high'][i] > df['high'][i-size:i].max()
        new_leg_low = df['low'][i] < df['low'][i-size:i].min()

        if new_leg_high:
            return BEARISH  #BEARISH_LEG = 0
        elif new_leg_low:
            return BULLISH #BULLISH_LEG = 1
        else:
            return 0  # Neither

    def start_of_new_leg(leg_value, prev_leg_value):
        """Replicates the startOfNewLeg() function from the Pine Script."""
        return leg_value != prev_leg_value

    # Iterate through the data, starting from the swing_length
    for i in range(swing_length, len(df)):
        # Calculate the leg value
        current_leg = leg(swing_length, i)
        previous_leg = leg(1, i-1) # Get the previous leg value

        # Check if it's the start of a new leg
        if start_of_new_leg(current_leg, previous_leg):
            if current_leg == BULLISH:  #Bullish Leg
                swing_lows.append(i)
            elif current_leg == BEARISH:  #Bearish Leg
                swing_highs.append(i)

    return swing_highs, swing_lows

def is_high_volatility(df, i, atr_period=200, atr_multiplier=2, volatility_type='atr'):
    """Checks if a candle is a high volatility candle using ATR or Cumulative Mean Range."""
    if volatility_type == 'atr':
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=atr_period)
        atr_series = atr.average_true_range()
        if i < atr_period or pd.isna(atr_series[i]):
            return False
        is_volatile = (df['high'][i] - df['low'][i]) >= (atr_multiplier * atr_series[i])
        #print(f"ATR: {atr_series[i]}, High-Low Range: {df['high'][i] - df['low'][i]}, Is Volatile: {is_volatile}")
        return is_volatile
    elif volatility_type == 'range':
        cumulative_mean_range = (df['high'] - df['low']).rolling(window=i+1).mean()
        if i < 1 or pd.isna(cumulative_mean_range[i]):
            return False
        return (df['high'][i] - df['low'][i]) >= (atr_multiplier * cumulative_mean_range[i])
    else:
        raise ValueError("Invalid volatility_type. Choose 'atr' or 'range'.")

def identify_order_blocks(df, swing_length=50, volatility_type='atr', order_block_mitigation='highlow'):
    """Identifies order blocks based on swing points."""
    swing_highs, swing_lows = identify_swing_points(df, swing_length)
    order_blocks = []

    # Implement the order block identification logic from the Pine Script
    # For each swing high/low, determine the order block boundaries
    # using parsed_high and parsed_low.

    for high_index in swing_highs:
        # Find the highest parsed high within the swing range
        swing_start = max(0, high_index - swing_length)
        swing_end = high_index
        if swing_start >= swing_end:
            continue  # Skip if the swing range is invalid

        highest_parsed_high_index = df['parsed_high'][swing_start:swing_end+1].idxmax()
        order_blocks.append(OrderBlock(df['high'][highest_parsed_high_index], df['low'][highest_parsed_high_index], df['timestamp'][highest_parsed_high_index], BEARISH, highest_parsed_high_index))

    for low_index in swing_lows:
        # Find the lowest parsed low within the swing range
        swing_start = max(0, low_index - swing_length)
        swing_end = low_index
        if swing_start >= swing_end:
            continue  # Skip if the swing range is invalid

        lowest_parsed_low_index = df['parsed_low'][swing_start:swing_end+1].idxmin()
        order_blocks.append(OrderBlock(df['high'][lowest_parsed_low_index], df['low'][lowest_parsed_low_index], df['timestamp'][lowest_parsed_low_index], BULLISH, lowest_parsed_low_index))

    # Limit the number of order blocks
    order_blocks = order_blocks[:100]

    return order_blocks

def is_order_block_mitigated(order_block, df, current_index, mitigation_type='close'):
    """Checks if an order block has been mitigated."""
    if order_block.bias == BULLISH:
        if mitigation_type == 'close':
            mitigated = df['close'][current_index] < order_block.bar_low
        elif mitigation_type == 'highlow':
            mitigated = df['low'][current_index] < order_block.bar_low
    elif order_block.bias == BEARISH:
        if mitigation_type == 'close':
            mitigated = df['close'][current_index] > order_block.bar_high
        elif mitigation_type == 'highlow':
            mitigated = df['high'][current_index] > order_block.bar_high
    else:
        mitigated = False
    return mitigated

if __name__ == '__main__':
    # Example Usage
    data = {
        'open': [10, 11, 12, 11, 13, 14, 13, 15, 14, 16, 15, 14, 13, 12, 11, 10],
        'close': [11, 12, 11, 13, 14, 13, 15, 16, 15, 17, 16, 15, 14, 13, 12, 11],
        'high': [12, 13, 12, 14, 15, 14, 16, 17, 16, 18, 17, 16, 15, 14, 13, 12],
        'low': [9, 10, 10, 10, 12, 12, 14, 15, 14, 16, 15, 14, 13, 12, 11],
        'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12', '2024-01-13', '2024-01-14', '2024-01-15', '2024-01-16'])
    }
    df = pd.DataFrame(data)
    df = calculate_parsed_high_low(df)
    order_blocks = identify_order_blocks(df)
    print("Order Blocks:", order_blocks)

    # Example of checking mitigation
    if order_blocks:
        is_mitigated = is_order_block_mitigated(order_blocks[0], df, len(df) - 1)
        print("Is Mitigated:", is_mitigated) 