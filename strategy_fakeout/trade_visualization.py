#!/usr/bin/env python3
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
    Generates and saves a combined chart for all trades of a given symbol.
    Chart is saved in the 'Charts' folder as '<symbol>_chart.png'.

    symbol: trading pair
    trades: list of trade dictionaries (with keys: 'entry_time', 'entry', 'stop_loss', 'take_profit', 'outcome')
    df: DataFrame with price data (datetime index and 'close' column)
    """
    if df.empty:
        print(f"No price data available for {symbol} to plot chart.")
        return
    
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Close Price', marker='o', linewidth=1)

    # Plot each trade on the chart
    for idx, trade in enumerate(trades):
        entry_time = trade['entry_time']
        entry_price = trade['entry']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        outcome = trade.get('outcome', '')

        # Vertical line marking the trade entry
        plt.axvline(entry_time, color='green', linestyle='--', linewidth=1)
        # Plot entry, stop loss and take profit markers
        plt.scatter(entry_time, entry_price, color='green', marker='o', s=100, label='Entry Price' if idx==0 else "")
        plt.scatter(entry_time, stop_loss, color='red', marker='x', s=100, label='Stop Loss' if idx==0 else "")
        plt.scatter(entry_time, take_profit, color='blue', marker='^', s=100, label='Take Profit' if idx==0 else "")
        # Annotate outcome near the entry point
        plt.text(entry_time, entry_price, f' {outcome}', fontsize=9, color='black', verticalalignment='bottom')

    plt.title(f"Trade Visualization for {symbol}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    
    # Ensure 'Charts' folder exists
    charts_folder = 'Charts'
    if not os.path.exists(charts_folder):
        os.makedirs(charts_folder)

    filename = os.path.join(charts_folder, f"{symbol}_chart.png")
    plt.savefig(filename)
    plt.close()
    print(f"Chart saved for {symbol} in {filename}")


if __name__ == "__main__":
    # Create sample price data
    date_range = pd.date_range(start="2023-01-01 09:00", periods=50, freq='15T')
    prices = np.linspace(100, 110, 50)
    df_sample = pd.DataFrame({'close': prices}, index=date_range)
    
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