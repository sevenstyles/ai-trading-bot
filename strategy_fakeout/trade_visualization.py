#!/usr/bin/env python3
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def visualize_trade_entry(trade, df):
    """
    Visualizes a single trade entry on a given price chart.

    trade: dict with keys 'side', 'entry_time', 'entry', 'stop_loss', 'take_profit'
    df: DataFrame with a datetime index and a 'close' column representing price data
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Close Price', marker='o', linewidth=1)
    
    # Mark the trade entry timestamp
    plt.axvline(trade['entry_time'], color='green', linestyle='--', linewidth=2, label='Entry Time')
    
    # Draw horizontal lines for entry, stop loss, take profit
    plt.axhline(y=trade['entry'], color='green', linestyle='-', linewidth=1.5, label='Entry Price')
    plt.axhline(y=trade['stop_loss'], color='red', linestyle='-', linewidth=1.5, label='Stop Loss')
    plt.axhline(y=trade['take_profit'], color='blue', linestyle='-', linewidth=1.5, label='Take Profit')
    
    plt.title(f"Trade Visualization ({trade['side'].upper()} Trade)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def save_trade_chart_for_symbol(symbol, trades, df):
    """
    Generates and saves a combined candlestick chart for all trades of a given symbol.
    Chart is saved in the 'Charts' folder as '<symbol>_chart.png'.

    symbol: trading pair
    trades: list of trade dictionaries (with keys: 'entry_time', 'entry', 'stop_loss', 'take_profit', 'outcome')
    df: DataFrame with OHLC price data (datetime index and columns: 'open', 'high', 'low', 'close')
    """
    if df.empty:
        print(f"No price data available for {symbol} to plot chart.")
        return

    # Ensure OHLC columns are numeric
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    # Create a copy and rename columns to those expected by mplfinance
    df_plot = df.copy()
    df_plot.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)

    # Plot candlesticks using mplfinance with our new settings
    fig, ax = mpf.plot(df_plot, type='candle', style='yahoo', title=f"Trade Visualization for {symbol}",
                         ylabel="Price", returnfig=True, figsize=(16,8),
                         scale_padding={'left': 0.3, 'right': 0.3, 'top': 0.2, 'bottom': 0.2})
    main_ax = ax[0]

    # Set x-axis limits explicitly using the index
    main_ax.set_xlim(df_plot.index[0], df_plot.index[-1])
    # Rotate x-axis tick labels
    plt.setp(main_ax.get_xticklabels(), rotation=45, ha='right')

    # Overlay each trade on the candlestick chart
    for idx, trade in enumerate(trades):
        entry_time = trade['entry_time']
        entry_price = trade['entry']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        outcome = trade.get('outcome', '')
        # Draw vertical line at the trade entry time
        main_ax.axvline(entry_time, color='purple', linestyle='--', linewidth=1)
        # Overlay markers
        main_ax.scatter(entry_time, entry_price, color='green', marker='o', s=100,
                        label='Entry Price' if idx==0 else "")
        main_ax.scatter(entry_time, stop_loss, color='red', marker='x', s=100,
                        label='Stop Loss' if idx==0 else "")
        main_ax.scatter(entry_time, take_profit, color='blue', marker='^', s=100,
                        label='Take Profit' if idx==0 else "")
        # Annotate the outcome
        main_ax.annotate(f"{outcome}", xy=(entry_time, entry_price), xytext=(5, 5),
                         textcoords='offset points', fontsize=9, color='black')

    # Ensure the 'Charts' folder exists
    charts_folder = 'Charts'
    if not os.path.exists(charts_folder):
        os.makedirs(charts_folder)

    filename = os.path.join(charts_folder, f"{symbol}_chart.png")
    fig.savefig(filename)
    plt.close(fig)
    print(f"Chart saved for {symbol} in {filename}")


if __name__ == "__main__":
    # Create sample price data
    date_range = pd.date_range(start="2023-01-01 09:00", periods=50, freq='15T')
    prices = np.linspace(100, 110, 50)
    df_sample = pd.DataFrame({'open': prices-1, 'high': prices+1, 'low': prices-2, 'close': prices}, index=date_range)
    
    # Example trade signal list (adjust with your actual trade details)
    sample_trades = [
        {
            'side': 'long',
            'entry_time': date_range[20],
            'entry': 105,
            'stop_loss': 103,
            'take_profit': 108,
            'outcome': 'WIN'
        },
        {
            'side': 'short',
            'entry_time': date_range[30],
            'entry': 108,
            'stop_loss': 110,
            'take_profit': 105,
            'outcome': 'LOSS'
        }
    ]
    
    # Call the new function to save the chart
    save_trade_chart_for_symbol('BTCUSDT', sample_trades, df_sample) 