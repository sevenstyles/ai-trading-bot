import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())

from binance.client import Client
import pandas as pd
from config import BINANCE_API_KEY, BINANCE_API_SECRET, OHLCV_TIMEFRAME, CAPITAL
from data_fetch import get_top_volume_pairs
from backtesting import backtest_strategy, analyze_results, analyze_aggregated_results
from datetime import datetime

client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

def load_config(config_file):
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            config[key] = value
    return config

def append_to_csv(data, filename='backtest_results.csv'):
    df = pd.DataFrame(data)
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

def main():
    config = load_config('config.txt')
    start_date_str = config.get('start_date')
    backtest_days = int(config.get('backtest_days', 3))  # Default to 3 days if not specified
    num_pairs = int(config.get('num_pairs', 1))  # Default to 10 pairs if not specified

    symbols = get_top_volume_pairs(limit=num_pairs)
    if not symbols:
        print("No top volume pairs found.")
        return

    all_trades = []
    backtest_results = []
    initial_capital = CAPITAL
    ending_capital = initial_capital
    
    for symbol in symbols:
        print(f"\n=== Backtesting {symbol} ===")
        # backtest_strategy now returns a tuple: (trades, price_df)
        trades, price_df = backtest_strategy(symbol, timeframe=OHLCV_TIMEFRAME, days=backtest_days, client=client, use_random_date=True, start_date_str=start_date_str)
        if trades:
            df_trades = pd.DataFrame(trades)
            long_trades = len(df_trades[df_trades["side"]=="long"])
            short_trades = len(df_trades[df_trades["side"]=="short"])
            print(f"Long trades: {long_trades}, Short trades: {short_trades}")
            analyze_results(trades, symbol)
            all_trades.extend(trades)

            for trade in trades:
                ending_capital *= (1 + trade['profit'])
                backtest_results.append({
                    'pair': symbol,
                    'direction': trade['side'],
                    'entry_date': trade['entry_time'].strftime('%Y-%m-%d'),
                    'entry_time': trade['entry_time'].strftime('%H:%M'),
                    'entry_price': trade['entry'],
                    'stop_loss': trade['stop_loss'],
                    'take_profit': trade['take_profit'],
                    'exit_time': trade['exit_time'].strftime('%H:%M'),
                    'exit_price': trade['exit_price'],
                    'profit_pct': trade['profit'] * 100,
                    'WIN/LOSS': trade['outcome'],
                })
        else:
            print(f"No valid trades generated for {symbol}")
    
    # Call aggregated analysis after processing all symbols
    analyze_aggregated_results(all_trades, initial_capital=CAPITAL, days=backtest_days)
    print("\nAggregated backtesting completed.")

    append_to_csv(backtest_results)
    print(f"Backtesting results appended to backtest_results.csv")

if __name__ == "__main__":
    main()
