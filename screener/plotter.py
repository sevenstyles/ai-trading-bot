import os
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as patches

def plot_trade_setup(BASE_DIR, df, signal, zones):
    """New zone plotting implementation with better visibility"""
    # Take last 1000 candles but ensure proper datetime index
    plot_df = df.iloc[-1000:].copy()
    
    # Create plot figure
    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)  # More space for dates
    
    # Plot zones first (background elements)
    if zones:
        y_min = min([z['low'] for z in zones] + [plot_df.low.min()])
        y_max = max([z['high'] for z in zones] + [plot_df.high.max()])
        ax.set_ylim(y_min * 0.98, y_max * 1.02)
        
        print(f"\nPlotting {len(zones)} zones:")
        for i, zone in enumerate(zones):
            color = 'darkred' if zone['type'] == 'supply' else 'darkgreen'
            linestyle = '-' if zone['valid'] else '--'
            
            # Determine x-range (always full width of chart)
            start_date = plot_df.index[0]
            end_date = plot_df.index[-1]
            
            print(f"Zone {i+1}: {zone['type']} {zone['low']}-{zone['high']}")
            print(f" - Created: {zone['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            print(f" - Valid: {zone['valid']}")
            
            # Plot horizontal lines spanning full width
            ax.hlines(
                y=[zone['high'], zone['low']],
                xmin=start_date,
                xmax=end_date,
                colors=color,
                linestyles=linestyle,
                linewidth=1.5,
                alpha=0.7,
                zorder=1
            )
            
            # Fill between the zone lines
            ax.fill_between(
                x=[start_date, end_date],
                y1=zone['high'],
                y2=zone['low'],
                color=color,
                alpha=0.1,
                zorder=0
            )
            
            # Add zone label
            label_x = start_date + (end_date - start_date) * 0.02
            ax.text(
                label_x, zone['high'],
                f"{zone['type'].upper()} ZONE\n{zone['low']:.5f}-{zone['high']:.5f}",
                color=color,
                fontsize=8,
                verticalalignment='bottom',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                zorder=2
            )
    
    # Plot candlesticks using mplfinance
    mpf.plot(
        plot_df,
        type='candle',
        style='charles',
        ax=ax,
        volume=False,  # Don't show volume
        show_nontrading=False,
        ylabel='Price',
        datetime_format='%Y-%m-%d %H:%M',
        xrotation=45,
        update_width_config=dict(
            candle_linewidth=0.8,
            candle_width=0.6
        ),
        returnfig=False  # Don't return the figure since we're managing it
    )
    
    # Customize x-axis
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    # Set title
    symbol = signal['Symbol']
    ax.set_title(f"{symbol} Trade Setup - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    filename = os.path.join(plots_dir, f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nSaved plot to: {filename}") 