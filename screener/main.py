import os
import random
from datetime import datetime
from binance.client import Client

from fetcher import fetch_data
from analysis import get_trend_direction, detect_supply_demand_zones
from entry import check_entry_conditions
from plotter import plot_trade_setup

def run_screener():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    STRATEGY_FILE = os.path.join(BASE_DIR, '..', 'strategy_3_prompt.txt')
    
    with open(STRATEGY_FILE, 'r') as f:
        strategy_rules = f.read()

    client = Client()

    all_tickers = client.get_all_tickers()
    usdt_pairs = [t['symbol'] for t in all_tickers if t['symbol'].endswith('USDT')]
    SYMBOLS = random.sample(usdt_pairs, 20)

    for symbol in SYMBOLS:
        print(f"\n{'='*40}")
        print(f"Analyzing {symbol}")
        try:
            data = fetch_data(client, symbol)
            zones = detect_supply_demand_zones(data)
            trend = get_trend_direction(data['4h'])
            signal = check_entry_conditions(data, zones, trend)

            output = {
                'Symbol': symbol,
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Status': 'NO TRADE',
                'Direction': 'NO TRADE',
                'Analysis': []
            }
            output.update(signal)

            print(f"\nStrategy 3 Results for {symbol}:")
            for k, v in output.items():
                if k != 'Symbol':
                    print(f"{k}: {v}")

            plot_trade_setup(BASE_DIR, data['30m'], output, zones)

            print(f"Found {len(zones)} zones")
            for z in zones:
                valid_status = "VALID" if z['valid'] else "INVALID"
                status_reason = ""
                now = datetime.now(z['timestamp'].tzinfo)
                if (now - z['timestamp']).days > 7:
                    status_reason = "(Expired)"
                elif not z['valid']:
                    status_reason = "(Price entered zone)"
                print(f"{z['type']} {valid_status} {status_reason}: {z['low']:.5f}-{z['high']:.5f} | Created: {z['timestamp'].strftime('%Y-%m-%d %H:%M')}")

            if output['Status'] == 'TRADE':
                print(f"Direction: {output['Direction']}")
                print(f"Entry: {output['Entry']:.4f}")
                print(f"SL: {output['SL']:.4f}") 
                print(f"TP1: {output['TP1']:.4f}")
            else:
                print("Status: NO TRADE")
                print("Direction: NO TRADE")

        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
        print(f"{'='*40}\n")

if __name__ == '__main__':
    run_screener() 