import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from orderblock_detector import is_order_block_mitigated
from orderblock_detector import BULLISH, BEARISH
import numpy as np

def visualize_order_blocks(df, swing_order_blocks, internal_order_blocks, filename="order_blocks.png", mitigation_type='close'):
    """
    Visualizes order blocks on a candlestick chart.

    Args:
        df (pd.DataFrame): DataFrame containing candlestick data with 'timestamp', 'open', 'high', 'low', 'close' columns.
        swing_order_blocks (list): List of swing order block namedtuples.
        internal_order_blocks (list): List of internal order block namedtuples.
        filename (str): Name of the file to save the chart to.
        mitigation_type (str): 'close' or 'highlow' to determine mitigation.
    """

    # Create the figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot candlesticks
    width = 0.8 / len(df)
    up = df[df['close'] >= df['open']]
    down = df[df['close'] < df['open']]
    ax1.vlines(up['timestamp'], up['low'], up['high'], color='green', linewidth=0.5)
    ax1.vlines(up['timestamp'], up['open'], up['close'], color='green', linewidth=3)
    ax1.vlines(down['timestamp'], down['low'], down['high'], color='red', linewidth=0.5)
    ax1.vlines(down['timestamp'], down['open'], down['close'], color='red', linewidth=3)

    # Plot swing order block zones
    for ob in swing_order_blocks:
        mitigated = is_order_block_mitigated(ob, df, len(df) - 1, mitigation_type)
        if not mitigated:
            if ob.bias == BULLISH:
                color = 'green'
                ax1.fill_betweenx([ob.bar_low, ob.bar_high], df['timestamp'][ob.index], df['timestamp'].iloc[-1], facecolor=color, alpha=0.2, label='Swing Demand')
            elif ob.bias == BEARISH:
                color = 'red'
                ax1.fill_betweenx([ob.bar_low, ob.bar_high], df['timestamp'][ob.index], df['timestamp'].iloc[-1],  facecolor=color, alpha=0.2, label='Swing Supply')

    # Plot internal order block zones
    for ob in internal_order_blocks:
        mitigated = is_order_block_mitigated(ob, df, len(df) - 1, mitigation_type)
        if not mitigated:
            if ob.bias == BULLISH:
                color = 'blue'  # Different color for internal OBs
                ax1.fill_betweenx([ob.bar_low, ob.bar_high], df['timestamp'][ob.index], df['timestamp'].iloc[-1], facecolor=color, alpha=0.2, label='Internal Demand')
            elif ob.bias == BEARISH:
                color = 'orange'  # Different color for internal OBs
                ax1.fill_betweenx([ob.bar_low, ob.bar_high], df['timestamp'][ob.index], df['timestamp'].iloc[-1],  facecolor=color, alpha=0.2, label='Internal Supply')

    # Format the x-axis
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # Add labels and title
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.set_title('Unmitigated Order Blocks Visualization')

    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())

    # Add grid
    ax1.grid(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)