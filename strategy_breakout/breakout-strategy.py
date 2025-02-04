from binance.client import Client


import random
import pandas as pd
from config import BINANCE_API_KEY, BINANCE_API_SECRET, OHLCV_TIMEFRAME
from data_fetch import get_top_volume_pairs
from backtesting import backtest_strategy, analyze_results, analyze_aggregated_results

client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

def main():
    top_volume_pairs = get_top_volume_pairs(limit=100)
    if not top_volume_pairs:
        print("No top volume pairs found.")
        return
    symbols = top_volume_pairs
    days = 30
    print(f"Backtesting top 100 symbols: {', '.join(symbols)}\n")
    
    all_trades = []
    for symbol in symbols:
        print(f"\n=== BACKTESTING {symbol} ===")
        signals = backtest_strategy(symbol, timeframe=OHLCV_TIMEFRAME, days=days, client=client, use_random_date=True)
        
        if signals:
            df = pd.DataFrame(signals)
            long_count = len(df[df['direction'] == 'long'])
            short_count = len(df[df['direction'] == 'short'])
            print(f"Long trades: {long_count}, Short trades: {short_count}")
            analyze_results(signals, symbol)
            all_trades.extend(signals)
        else:
            print("No valid trades generated")
    
    print("\n=== AGGREGATED REPORT ===")
    analyze_aggregated_results(all_trades, days=days)

if __name__ == "__main__":
    main()
