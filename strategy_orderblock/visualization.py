import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from orderblock_detector import is_order_block_mitigated
from orderblock_detector import BULLISH, BEARISH

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
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot candlesticks
    width = 0.8 / len(df)  # Width of the candlesticks
    up = df[df['close'] >= df['open']]
    down = df[df['close'] < df['open']]
    ax.vlines(up['timestamp'], up['low'], up['high'], color='green', linewidth=0.5)
    ax.vlines(up['timestamp'], up['open'], up['close'], color='green', linewidth=3)
    ax.vlines(down['timestamp'], down['low'], down['high'], color='red', linewidth=0.5)
    ax.vlines(down['timestamp'], down['open'], down['close'], color='red', linewidth=3)

    # Plot order block zones
    for ob in order_blocks:
        if ob.bias == BULLISH:
            color = 'green'
            # Check for mitigation
            mitigated = False
            for i in range(ob.index + 1, len(df)):
                if mitigation_type == 'close':
                    if df['close'][i] < ob.bar_low:
                        mitigated = True
                        break
                elif mitigation_type == 'highlow':
                    if df['low'][i] < ob.bar_low:
                        mitigated = True
                        break
            if not mitigated:
                ax.axhspan(ob.bar_low, ob.bar_high, facecolor=color, alpha=0.2, label='demand', xmin=mdates.date2num(df['timestamp'].iloc[ob.index]), xmax=mdates.date2num(df['timestamp'].iloc[-1]))

        elif ob.bias == BEARISH:
            color = 'red'
            # Check for mitigation
            mitigated = False
            for i in range(ob.index + 1, len(df)):
                if mitigation_type == 'close':
                    if df['close'][i] > ob.bar_high:
                        mitigated = True
                        break
                elif mitigation_type == 'highlow':
                    if df['high'][i] > ob.bar_high:
                        mitigated = True
                        break
            if not mitigated:
                ax.axhspan(ob.bar_low, ob.bar_high, facecolor=color, alpha=0.2, label='supply', xmin=mdates.date2num(df['timestamp'].iloc[ob.index]), xmax=mdates.date2num(df['timestamp'].iloc[-1]))

    # Format the x-axis to show dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Unmitigated Order Blocks Visualization')

    # Add legend (only show one label per type)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Add grid
    ax.grid(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)  # Close the figure to free memory 