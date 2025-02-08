import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from orderblock_detector import is_order_block_mitigated
from orderblock_detector import BULLISH, BEARISH
import numpy as np

def visualize_order_blocks(df, order_blocks, filename="order_blocks.png", mitigation_type='close'):
    """
    Visualizes order blocks on a candlestick chart.

    Args:
        df (pd.DataFrame): DataFrame containing candlestick data with 'timestamp', 'open', 'high', 'low', 'close' columns.
        order_blocks (list): List of order block namedtuples, each with 'bar_high', 'bar_low', 'bar_time', 'bias', 'index'.
        filename (str): Name of the file to save the chart to.
        mitigation_type (str): 'close' or 'highlow' to determine mitigation.
    """

    # Create the figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twiny()  # Create a twin Axes sharing the y-axis

    # Plot candlesticks on ax1
    width = 0.8 / len(df)  # Width of the candlesticks
    up = df[df['close'] >= df['open']]
    down = df[df['close'] < df['open']]
    ax1.vlines(up['timestamp'], up['low'], up['high'], color='green', linewidth=0.5)
    ax1.vlines(up['timestamp'], up['open'], up['close'], color='green', linewidth=3)
    ax1.vlines(down['timestamp'], down['low'], down['high'], color='red', linewidth=0.5)
    ax1.vlines(down['timestamp'], down['open'], down['close'], color='red', linewidth=3)

    # Create a sequence of numbers for the x-axis
    x = np.arange(len(df))

    # Plot order block zones on ax2
    for ob in order_blocks:
        mitigated = is_order_block_mitigated(ob, df, len(df) - 1, mitigation_type)
        if not mitigated:
            if ob.bias == BULLISH:
                color = 'green'
                # Debugging: Print order block details
                print(f"Plotting BULLISH OB at index {ob.index}, bar_low={ob.bar_low}, bar_high={ob.bar_high}")
                ax2.axvspan(x[ob.index], x[-1], facecolor=color, alpha=0.2, label='demand')

            elif ob.bias == BEARISH:
                color = 'red'
                # Debugging: Print order block details
                print(f"Plotting BEARISH OB at index {ob.index}, bar_low={ob.bar_low}, bar_high={ob.bar_high}")
                ax2.axvspan(x[ob.index], x[-1], facecolor=color, alpha=0.2, label='supply')

    # Format the x-axis of ax1 to show dates
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # Set the x-axis limits of ax2 to the range of the data
    ax2.set_xlim(x[0], x[-1])

    # Hide the x-axis labels and ticks on ax2
    ax2.xaxis.set_visible(False)

    # Align the y-axis limits of ax1 and ax2
    ax2.set_ylim(ax1.get_ylim())

    # Add labels and title
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.set_title('Unmitigated Order Blocks Visualization')

    # Add legend (only show one label per type)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())

    # Add grid
    ax1.grid(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)  # Close the figure to free memory 