import pandas as pd
import numpy as np
import ta
from collections import namedtuple
import logging

# Configure logging
logging.basicConfig(filename='orderblock_debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define namedtuples for data organization
OrderBlock = namedtuple('OrderBlock', ['bar_high', 'bar_low', 'bar_time', 'bias', 'index'])
SwingPoint = namedtuple('SwingPoint', ['current_level', 'last_level', 'crossed', 'bar_time', 'bar_index'])
Trend = namedtuple('Trend', ['bias'])

BULLISH = 1
BEARISH = -1
BULLISH_LEG = 1
BEARISH_LEG = 0

def leg(df, size, index):
    """
    Determines the current leg of the market (bullish or bearish).

    Args:
        df (pd.DataFrame): DataFrame containing candlestick data.
        size (int): Period to consider for swing identification.
        index (int): Current index in the DataFrame.

    Returns:
        int: BULLISH (1) or BEARISH (-1) or 0 if no new leg
    """
    if index < size:
        return 0  # Not enough data to determine leg

    highest_high = df['high'][index-size+1:index+1].max()
    lowest_low = df['low'][index-size+1:index+1].min()

    logging.debug(f"leg() - index: {index}, size: {size}")
    logging.debug(f"leg() - highest_high: {highest_high}, lowest_low: {lowest_low}")
    logging.debug(f"leg() - df['high'][{index-size}]: {df['high'][index-size] if index-size >= 0 else 'N/A'}")
    logging.debug(f"leg() - df['low'][{index-size}]: {df['low'][index-size] if index-size >= 0 else 'N/A'}")

    if index - size >= 0:
        if df['high'][index] > highest_high:
            logging.debug(f"leg() - New Bearish Leg. High[{index}] ({df['high'][index]}) > highest_high ({highest_high})")
            return BEARISH_LEG  # New bearish leg
        elif df['low'][index] < lowest_low:
            logging.debug(f"leg() - New Bullish Leg. Low[{index}] ({df['low'][index]}) < lowest_low ({lowest_low})")
            return BULLISH_LEG  # New bullish leg
        else:
            logging.debug("leg() - No new leg")
            return 0
    else:
        return 0

def identify_swing_points(df, swing_length=50, swing_type='swing'):
    """Identifies swing points based on the leg function."""
    highs = []
    lows = []
    current_leg = 0
    previous_leg = 0

    for i in range(swing_length, len(df)):
        current_leg = leg(df, swing_length, i)
        logging.debug(f"Index: {i}, Current Leg: {current_leg}, Previous Leg: {previous_leg}")

        if current_leg != 0 and current_leg != previous_leg:
            # Correctly identify the swing point index
            if current_leg == BULLISH_LEG:
                swing_low_index = df['low'][i-swing_length:i+1].idxmin()
                lows.append(swing_low_index)
                logging.debug(f"New Swing Low at index: {swing_low_index}, Low: {df['low'][swing_low_index]}")
            elif current_leg == BEARISH_LEG:
                swing_high_index = df['high'][i-swing_length:i+1].idxmax()
                highs.append(swing_high_index)
                logging.debug(f"New Swing High at index: {swing_high_index}, High: {df['high'][swing_high_index]}")
            previous_leg = current_leg

    logging.debug(f"Identified Swing Highs: {highs}")
    logging.debug(f"Identified Swing Lows: {lows}")
    return highs, lows

def filter_mitigated_order_blocks(order_blocks, df, mitigation_type='close'):
    """Filters out mitigated order blocks."""
    filtered_order_blocks = []
    for ob in order_blocks:
        if not is_order_block_mitigated(ob, df, len(df) -1, mitigation_type):
            filtered_order_blocks.append(ob)

    #Return the filtered list with the most recent order blocks
    return filtered_order_blocks

def is_high_volatility(df, index, volatility_type='atr', atr_period=200, atr_multiplier=2):
    """Checks if a candle is a high volatility candle based on ATR or range."""
    if volatility_type == 'atr':
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=atr_period).average_true_range().iloc[index]
        if pd.isna(atr):
            return False  # Handle NaN values
        return (df['high'][index] - df['low'][index]) >= (atr_multiplier * atr)
    elif volatility_type == 'range':
        # Calculate cumulative mean range (approximation)
        cumulative_range = (df['high'][:index+1] - df['low'][:index+1]).sum()
        mean_range = cumulative_range / (index + 1)
        return (df['high'][index] - df['low'][index]) >= (atr_multiplier * mean_range)
    else:
        raise ValueError("Invalid volatility_type. Choose 'atr' or 'range'.")

def calculate_parsed_high_low(df, volatility_type='atr', atr_period=200, atr_multiplier=2):
    """Calculates parsed high and low based on volatility."""
    df['parsed_high'] = df['high']
    df['parsed_low'] = df['low']

    for i in range(len(df)):
        if is_high_volatility(df, i, volatility_type, atr_period, atr_multiplier):
            df.loc[i, 'parsed_high'] = df['low'][i]
            df.loc[i, 'parsed_low'] = df['high'][i]
    return df

def identify_order_blocks(df, swing_length=50, internal_length=5, volatility_type='atr', atr_period=200, atr_multiplier=2):
    """Identifies order blocks based on swing points and internal swings."""
    swing_highs, swing_lows = identify_swing_points(df, swing_length, swing_type='swing')
    internal_highs, internal_lows = identify_swing_points(df, internal_length, swing_type='internal')
    swing_order_blocks = []
    internal_order_blocks = []

    logging.debug(f"Swing Highs: {swing_highs}")
    logging.debug(f"Swing Lows: {swing_lows}")

    for high_index in swing_highs:
        # Find the highest parsed_high between the previous swing low and this swing high
        logging.debug(f"Processing swing high at index: {high_index}")
        previous_swing_low_index = -1
        for low_idx in swing_lows:
            if low_idx < high_index:
                previous_swing_low_index = max(previous_swing_low_index, low_idx)
            else:
                break  # swing_lows is sorted, so we can stop searching
        logging.debug(f"  Previous swing low index: {previous_swing_low_index}")

        if previous_swing_low_index != -1:
            highest_parsed_high_index = df['parsed_high'][previous_swing_low_index:high_index+1].idxmax()
            logging.debug(f"  Highest parsed high index between {previous_swing_low_index} and {high_index}: {highest_parsed_high_index}")
            logging.debug(f"    Parsed High Value: {df['parsed_high'][highest_parsed_high_index]}")
            logging.debug(f"    Parsed Low Value: {df['parsed_low'][highest_parsed_high_index]}")

            # Create the order block.
            ob = OrderBlock(df['parsed_high'][highest_parsed_high_index], df['parsed_low'][lowest_parsed_low_index], df['timestamp'][highest_parsed_high_index], BEARISH, highest_parsed_high_index)
            logging.debug(f"  Potential Bearish Swing Order Block: {ob}")

            # Check for mitigation immediately.
            mitigated = False
            for i in range(highest_parsed_high_index + 1, len(df)):
                if is_order_block_mitigated(ob, df, i):
                    mitigated = True
                    logging.debug(f"    Order Block mitigated at index {i}")
                    break
            if not mitigated:
                swing_order_blocks.insert(0, ob)
                logging.debug(f"    Order Block added: {ob}")
        else:
            logging.debug("  No previous swing low found.")

    for low_index in swing_lows:
        # Find the lowest parsed_low between the previous swing high and this swing low
        logging.debug(f"Processing swing low at index: {low_index}")
        previous_swing_high_index = -1
        for high_idx in swing_highs:
            if high_idx < low_index:
                previous_swing_high_index = max(previous_swing_high_index, high_idx)
            else:
                break # swing_highs is sorted
        logging.debug(f"  Previous swing high index: {previous_swing_high_index}")

        if previous_swing_high_index != -1:
            lowest_parsed_low_index = df['parsed_low'][previous_swing_high_index:low_index+1].idxmin()
            logging.debug(f"  Lowest parsed low index between {previous_swing_high_index} and {low_index}: {lowest_parsed_low_index}")
            logging.debug(f"    Parsed High Value: {df['parsed_high'][lowest_parsed_low_index]}")
            logging.debug(f"    Parsed Low Value: {df['parsed_low'][lowest_parsed_low_index]}")

            # Create the order block
            ob = OrderBlock(df['parsed_high'][lowest_parsed_low_index], df['parsed_low'][lowest_parsed_low_index], df['timestamp'][lowest_parsed_low_index], BULLISH, lowest_parsed_low_index)
            logging.debug(f"  Potential Bullish Swing Order Block: {ob}")

            # Check for mitigation immediately
            mitigated = False
            for i in range(lowest_parsed_low_index + 1, len(df)):
                if is_order_block_mitigated(ob, df, i):
                    mitigated = True
                    logging.debug(f"    Order Block mitigated at index {i}")
                    break
            if not mitigated:
                swing_order_blocks.insert(0, ob)
                logging.debug(f"    Order Block added: {ob}")
        else:
            logging.debug("  No previous swing high found.")

    for high_index in internal_highs:
        # Find the highest parsed_high between the previous internal low and this internal high
        logging.debug(f"Processing internal high at index: {high_index}")
        previous_internal_low_index = -1
        for low_idx in internal_lows:
            if low_idx < high_index:
                previous_internal_low_index = max(previous_internal_low_index, low_idx)
            else:
                break
        logging.debug(f"  Previous internal low index: {previous_internal_low_index}")

        if previous_internal_low_index != -1:
            highest_parsed_high_index = df['parsed_high'][previous_internal_low_index:high_index+1].idxmax()
            logging.debug(f"  Highest parsed high index between {previous_internal_low_index} and {high_index}: {highest_parsed_high_index}")
            logging.debug(f"    Parsed High Value: {df['parsed_high'][highest_parsed_high_index]}")
            logging.debug(f"    Parsed Low Value: {df['parsed_low'][highest_parsed_high_index]}")

            # Create the order block
            ob = OrderBlock(df['parsed_high'][highest_parsed_high_index], df['parsed_low'][highest_parsed_high_index], df['timestamp'][highest_parsed_high_index], BEARISH, highest_parsed_high_index)
            logging.debug(f"  Potential Bearish Internal Order Block: {ob}")

            # Check for mitigation immediately
            mitigated = False
            for i in range(highest_parsed_high_index + 1, len(df)):
                if is_order_block_mitigated(ob, df, i):
                    mitigated = True
                    logging.debug(f"    Order Block mitigated at index {i}")
                    break
            if not mitigated:
                internal_order_blocks.insert(0, ob)
                logging.debug(f"    Order Block added: {ob}")
        else:
            logging.debug("  No previous internal low found.")

    for low_index in internal_lows:
        # Find the lowest parsed_low between the previous internal high and this internal low
        logging.debug(f"Processing internal low at index: {low_index}")
        previous_internal_high_index = -1
        for high_idx in internal_highs:
            if high_idx < low_index:
                previous_internal_high_index = max(previous_internal_high_index, high_idx)
            else:
                break
        logging.debug(f"  Previous internal high index: {previous_internal_high_index}")

        if previous_internal_high_index != -1:
            lowest_parsed_low_index = df['parsed_low'][previous_internal_high_index:low_index+1].idxmin()
            logging.debug(f"  Lowest parsed low index between {previous_internal_high_index} and {low_index}: {lowest_parsed_low_index}")
            logging.debug(f"    Parsed High Value: {df['parsed_high'][lowest_parsed_low_index]}")
            logging.debug(f"    Parsed Low Value: {df['parsed_low'][lowest_parsed_low_index]}")

            # Create the order block
            ob = OrderBlock(df['parsed_high'][lowest_parsed_low_index], df['parsed_low'][lowest_parsed_low_index], df['timestamp'][lowest_parsed_low_index], BULLISH, lowest_parsed_low_index)
            logging.debug(f"  Potential Bullish Internal Order Block: {ob}")

            # Check for mitigation immediately
            mitigated = False
            for i in range(lowest_parsed_low_index + 1, len(df)):
                if is_order_block_mitigated(ob, df, i):
                    mitigated = True
                    logging.debug(f"    Order Block mitigated at index {i}")
                    break
            if not mitigated:
                internal_order_blocks.insert(0, ob)
                logging.debug(f"    Order Block added: {ob}")
        else:
            logging.debug("  No previous internal high found.")

    # Limit the number of order blocks (optional, but good practice)
    swing_order_blocks = swing_order_blocks[:100]
    internal_order_blocks = internal_order_blocks[:100]

    return swing_order_blocks, internal_order_blocks # Return both lists

def calculate_atr(df, period):
    """
    Calculates the Average True Range (ATR) for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing candlestick data.
        period (int): Period to calculate the ATR.

    Returns:
        pd.Series: A Series containing the ATR values.
    """
    df['tr'] = abs(df['high'] - df['low'])
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df['atr']

def is_valid_order_block(order_block, df, volatility_type):
    """
    Checks if the given order block is valid based on the volatility filter.

    Args:
        order_block (OrderBlock): The order block to check.
        df (pd.DataFrame): DataFrame containing candlestick data.
        volatility_type (str): 'atr' or 'range' to determine volatility.

    Returns:
        bool: True if the order block is valid, False otherwise.
    """
    atr_multiplier = 2
    if volatility_type == 'atr':
        atr_value = df['atr'][order_block.index]
        order_block_size = abs(order_block.bar_high - order_block.bar_low)
        return order_block_size <= atr_multiplier * atr_value
    else:
        return True  # Implement range-based filter if needed

def is_order_block_mitigated(order_block, df, current_index, mitigation_type='close'):
    """Checks if an order block has been mitigated."""
    logging.debug(f"Checking mitigation for order block: {order_block} at index {current_index}")
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
    logging.debug(f"Mitigation result: {mitigated}")
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
    df = calculate_parsed_high_low(df)  # Calculate parsed high/low
    swing_order_blocks, internal_order_blocks = identify_order_blocks(df)
    logging.info(f"Swing Order Blocks: {swing_order_blocks}")
    logging.info(f"Internal Order Blocks: {internal_order_blocks}")

    # Example of checking mitigation
    if swing_order_blocks:
        is_mitigated = is_order_block_mitigated(swing_order_blocks[0], df, len(df) - 1)
        print("Is Mitigated:", is_mitigated)