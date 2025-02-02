import os
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime
from config import BASE_DIR

def plot_trade_setup(df, signal, zones):
    """Plot trade setup with visible supply/demand zones."""
    plot_df = df.iloc[-1000:]
    fig, axes = mpf.plot(
        plot_df,
        type='candle',
        style='charles',
        title=f"{signal['Symbol']} Trade Setup",
        ylabel='Price',
        figsize=(18, 9),
        returnfig=True,
        tight_layout=True
    )
    ax = axes[0]
    valid_zones = [z for z in zones if z['valid']]
    if valid_zones:
        min_low = min([z['low'] for z in valid_zones] + [plot_df.low.min()])
        max_high = max([z['high'] for z in valid_zones] + [plot_df.high.max()])
        ax.set_ylim(min_low * 0.98, max_high * 1.02)
    for zone in valid_zones:
        color = '#FF0000' if zone['type'] == 'supply' else '#00FF00'
        ax.axhspan(zone['low'], zone['high'], facecolor=color, alpha=0.2, zorder=0)
        ax.axhline(zone['high'], color=color, alpha=0.5, linestyle='--')
        ax.axhline(zone['low'], color=color, alpha=0.5, linestyle='--')
    plots_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    filename = os.path.join(plots_dir, f"{signal['Symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(filename)
    plt.close(fig) 