import pandas as pd
from orderblock_detector import identify_order_blocks, is_order_block_mitigated, calculate_parsed_high_low


def backtest_order_blocks(df, mitigation_type='close', swing_length=50, internal_length=5):
    """
    Backtests order block strategy on the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data and timestamps.
        mitigation_type (str): 'close' or 'highlow' to determine mitigation.
        swing_length (int): Length for swing calculation.
        internal_length (int): Length for internal swing calculation.

    Returns:
        tuple: (swing_order_blocks, internal_order_blocks)
    """
    df = calculate_parsed_high_low(df)
    swing_order_blocks, internal_order_blocks = identify_order_blocks(df, swing_length=swing_length, internal_length=internal_length)
    # No need to filter here, it is done inside identify_order_blocks
    return swing_order_blocks, internal_order_blocks


if __name__ == '__main__':
    # Example Usage
    data = {
        'open': [10, 11, 12, 11, 13, 14, 13, 15, 14, 16, 15, 14, 13, 12, 11, 10],
        'close': [11, 12, 11, 13, 14, 13, 15, 16, 15, 17, 16, 15, 14, 13, 12, 11],
        'high': [12, 13, 12, 14, 15, 14, 16, 17, 16, 18, 17, 16, 15, 14, 13, 12],
        'low': [9, 10, 10, 10, 12, 12, 14, 15, 14, 16, 15, 14, 13, 12, 11, 10],
        'timestamp': pd.to_datetime(
            ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07',
             '2024-01-08', '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12', '2024-01-13', '2024-01-14',
             '2024-01-15', '2024-01-16'])
    }
    df = pd.DataFrame(data)
    swing_order_blocks, internal_order_blocks = backtest_order_blocks(df, mitigation_type='highlow', swing_length=5, internal_length=5)
    print("Swing Order Blocks:", swing_order_blocks)
    print("Internal Order Blocks:", internal_order_blocks)