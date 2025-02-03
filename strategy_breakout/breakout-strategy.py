from binance.client import Client
# Updated for backtester settings: OHLCV_TIMEFRAME = 1h, TP target = 4%, SL adjusted accordingly
# NOTE: Backtesting now incorporates fee (FUTURES_FEE) and slippage (SLIPPAGE_RATE) adjustments as configured in config.py for more realistic simulation of live futures trading.
import random
from config import BINANCE_API_KEY, BINANCE_API_SECRET
from data_fetch import get_top_volume_pairs
from backtesting import backtest_strategy, analyze_results, analyze_aggregated_results

client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

def main():
    top_volume_pairs = get_top_volume_pairs(limit=100)
    if not top_volume_pairs:
        print("No top volume pairs found.")
        return
    symbols = top_volume_pairs
    print(f"Testing top 100 symbols: {', '.join(symbols)}\n")
    
    all_trades = []
    for symbol in symbols:
        print(f"\n=== TESTING {symbol} ===")
        signals = backtest_strategy(symbol, client=client)
        
        if signals:
            analyze_results(signals, symbol)
            all_trades.extend(signals)
        else:
            print("No valid trades generated")
    
    print("\n=== AGGREGATED REPORT ===")
    analyze_aggregated_results(all_trades)

if __name__ == "__main__":
    main()
