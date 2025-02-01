import pandas as pd

def fetch_data(client, symbol):
    """Extended historical data ranges."""
    intervals = {
        '30m': 4032,  # Keep 3 months history
        '4h': 1512,    # 6 months
        '5m': 2016     # Increase from 576 to 7 days (2016 = 5m * 288/day * 7)
    }
    
    data = {}
    for interval, limit in intervals.items():
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                   'close_time', 'quote_asset_volume', 'number_of_trades',
                   'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        # Include volume in the selected columns
        data[interval] = df[['open', 'high', 'low', 'close', 'volume']]
    return data 