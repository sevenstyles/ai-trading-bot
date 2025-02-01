from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import os

def get_trend_direction(df_4h):
    """Determine trend using 4h data."""
    sma_50 = df_4h.close.rolling(50).mean().iloc[-1]
    current_price = df_4h.close.iloc[-1]
    return 'bullish' if current_price > sma_50 else 'bearish'

def detect_supply_demand_zones(data):
    """Simplified validation - only check current close."""
    # More realistic thresholds
    INSTITUTIONAL_VOLUME_MULTIPLIER = 1.1  # From 1.2
    VOL_PCT_THRESHOLD = 0.7  # From 0.8
    CONSECUTIVE_PUSH_CANDLES = 2  # From 1
    VOLUME_CONFIRMATION = 1.0  # Was 1.2

    df = data['30m']
    zones = []
    
    # Calculate volume benchmarks
    vol_ma = df.volume.rolling(20).mean()
    df['vol_ma'] = vol_ma
    df['vol_pct'] = df.volume / vol_ma
    
    i = 1
    while i < len(df)-2:
        current = df.iloc[i]
        
        # Modified volume check
        if current.vol_pct < VOL_PCT_THRESHOLD or \
           current.volume < (df.volume.rolling(50).mean().iloc[i] * INSTITUTIONAL_VOLUME_MULTIPLIER):
            print("→ Failed volume requirements")
            i += 1
            continue
            
        print("\nLiquidity Check:")
        print(f"- Current High: {current.high:.5f} | 200-period High: {df.high.rolling(200).max().iloc[i]:.5f}")
        print(f"- Current Low: {current.low:.5f} | 200-period Low: {df.low.rolling(200).min().iloc[i]:.5f}")
        
        is_liquidity_area = (
            current.high == df.high.rolling(200).max().iloc[i] or
            current.low == df.low.rolling(200).min().iloc[i]
        )
        print(f"- Liquidity Area: {is_liquidity_area}")
        
        if current.close < current.open and is_liquidity_area:  # Supply
            push_start = i
            while i < len(df)-1 and df.iloc[i].close < df.iloc[i].open:
                i += 1
            push_end = i-1
            
            # Modified push validation
            if (push_end - push_start) < CONSECUTIVE_PUSH_CANDLES:
                continue
                
            extreme_candle = df.iloc[push_start:push_end+1].low.idxmin()
            extreme_candle = df.loc[extreme_candle]
            
            # Modified volume confirmation
            if extreme_candle.vol_pct < VOLUME_CONFIRMATION:
                continue
                
            next_candle = df.iloc[push_end+1]
            if next_candle.high > extreme_candle.high:
                zone_high = extreme_candle.high
                zone_low = extreme_candle.low
            else:
                zone_high = max(extreme_candle.open, extreme_candle.close)
                zone_low = min(extreme_candle.open, extreme_candle.close)
            
            zones.append({
                'type': 'supply', 
                'high': zone_high,
                'low': zone_low,
                'timeframe': '30m',
                'valid': True,
                'timestamp': extreme_candle.name
            })

        elif current.close > current.open and is_liquidity_area:  # Demand
            push_start = i
            while i < len(df)-1 and df.iloc[i].close > df.iloc[i].open:
                i += 1
            push_end = i-1
            
            if (push_end - push_start) < 2:
                continue
                
            extreme_candle = df.iloc[push_start:push_end+1].high.idxmax()
            extreme_candle = df.loc[extreme_candle]
            
            if extreme_candle.vol_pct < 1.2:
                continue
                
            next_candle = df.iloc[push_end+1]
            if next_candle.low < extreme_candle.low:
                zone_high = extreme_candle.high
                zone_low = extreme_candle.low
            else:
                zone_high = max(extreme_candle.open, extreme_candle.close)
                zone_low = min(extreme_candle.open, extreme_candle.close)
            
            zones.append({
                'type': 'demand',
                'high': zone_high,
                'low': zone_low,
                'timeframe': '30m',
                'valid': True,
                'timestamp': extreme_candle.name
            })
        
        i += 1
    
    # Final validation fix - only check closes after zone formation
    for zone in zones:
        zone_df = df[df.index >= zone['timestamp']]
        # Check for ANY 2 consecutive closes inside the zone
        inside_zone = (zone_df['close'] > zone['low']) & (zone_df['close'] < zone['high'])
        zone['valid'] = not any(inside_zone.rolling(2).sum() >= 2)
    
    # Zones expire after 30 days
    for zone in zones:
        if (datetime.now(zone['timestamp'].tzinfo) - zone['timestamp']).days > 30:
            zone['valid'] = False
    
    return zones

def merge_zones(zones):
    """Combine overlapping zones of same type."""
    merged = []
    for zone in sorted(zones, key=lambda x: x['high'], reverse=True):
        if not merged:
            merged.append(zone)
            continue
            
        last = merged[-1]
        if (zone['type'] == last['type'] and 
            zone['timeframe'] == last['timeframe'] and
            abs(zone['high'] - last['high']) < last['high'] * 0.005):
            merged[-1]['high'] = max(last['high'], zone['high'])
            merged[-1]['low'] = min(last['low'], zone['low'])
        else:
            merged.append(zone)
    
    return merged 

def plot_trade_setup(BASE_DIR, df, signal, zones):
    """Plot trade setup with proper datetime handling and zone visualization"""
    # Ensure proper datetime index and take last 200 candles for better visibility
    plot_df = df.iloc[-200:].copy()
    
    # Create figure with specific size
    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 1, 1)
    
    # Calculate price range including both candlesticks and zones
    price_min = plot_df['low'].min()
    price_max = plot_df['high'].max()
    
    if zones:
        price_min = min([z['low'] for z in zones if z['valid']] + [price_min])
        price_max = max([z['high'] for z in zones if z['valid']] + [price_max])
    
    price_padding = (price_max - price_min) * 0.05
    y_min = price_min - price_padding
    y_max = price_max + price_padding
    
    # Plot candlesticks
    mpf.plot(
        plot_df,
        type='candle',
        style='charles',
        ax=ax,
        volume=False,
        ylabel='Price',
        datetime_format='%Y-%m-%d %H:%M',
        xrotation=45,
        update_width_config=dict(
            candle_linewidth=0.8,
            candle_width=0.6
        ),
        returnfig=False
    )
    
    # Plot zones if they exist
    if zones:
        valid_zones = [z for z in zones if z['valid']]
        print(f"\nPlotting {len(valid_zones)} valid zones:")
        
        for zone in valid_zones:
            color = 'darkred' if zone['type'] == 'supply' else 'darkgreen'
            alpha = 0.2
            
            # Create rectangle for zone
            rect = patches.Rectangle(
                xy=(plot_df.index[0], zone['low']),
                width=plot_df.index[-1] - plot_df.index[0],
                height=zone['high'] - zone['low'],
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                zorder=0
            )
            ax.add_patch(rect)
            
            # Add zone label on the right side
            ax.text(
                plot_df.index[-1], 
                zone['high'],
                f"{zone['type'].upper()} ZONE\n{zone['low']:.5f}-{zone['high']:.5f}",
                color=color,
                fontsize=8,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                zorder=2
            )
            
            print(f"Zone: {zone['type']} {zone['low']:.5f}-{zone['high']:.5f}")
    
    # Set y-axis limits
    ax.set_ylim(y_min, y_max)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    # Set title and adjust layout
    symbol = signal['Symbol']
    plt.title(f"{symbol} Trade Setup - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    filename = os.path.join(plots_dir, f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"\nSaved plot to: {filename}") 