from binance.client import Client
import pandas as pd
from config import BINANCE_API_KEY, BINANCE_API_SECRET, OHLCV_TIMEFRAME
from data_fetch import get_top_volume_pairs
from backtesting import backtest_strategy, analyze_results

client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

def main():
    symbols = get_top_volume_pairs(limit=100)
    if not symbols:
        print("No top volume pairs found.")
        return

    days = 30
    all_trades = []
    
    for symbol in symbols:
        print(f"\n=== Backtesting {symbol} ===")
        trades = backtest_strategy(symbol, timeframe=OHLCV_TIMEFRAME, days=days, client=client, use_random_date=True)
        if trades:
            df = pd.DataFrame(trades)
            long_trades = len(df[df["side"]=="long"])
            short_trades = len(df[df["side"]=="short"])
            print(f"Long trades: {long_trades}, Short trades: {short_trades}")
            analyze_results(trades, symbol)
            all_trades.extend(trades)
        else:
            print(f"No valid trades generated for {symbol}")
    
    # Optionally, aggregated analysis for all symbols can be added here.
    print("\nAggregated backtesting completed.")

if __name__ == "__main__":
    main()
