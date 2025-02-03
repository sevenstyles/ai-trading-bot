import pandas as pd
from binance_client import client
from logger import log_debug
import state
from data_fetch import get_top_volume_pairs


def preload_candles(interval, limit):
    print("Preloading historical candles for symbols...")
    log_debug("Preloading historical candles for symbols...")
    valid_symbols = []
    invalid_symbols = []
    
    # If state.symbols is empty, fetch top volume pairs
    if not state.symbols:
        state.symbols = get_top_volume_pairs(limit=100)
        if not state.symbols:
            state.symbols = ["BTCUSDT"]

    for symbol in state.symbols:
        try:
            klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                 'close_time', 'quote_asset_volume', 'trades',
                                                 'taker_buy_base', 'taker_buy_quote', 'ignore'])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            state.candles_dict[symbol] = df[numeric_cols]
            msg = f"Preloaded {len(df)} candles for {symbol}"
            print(msg)
            log_debug(msg)
            valid_symbols.append(symbol)
        except Exception as e:
            err_msg = f"Error preloading candles for {symbol}: {e}"
            print(err_msg)
            log_debug(err_msg)
            invalid_symbols.append(symbol)
    if not valid_symbols:
        valid_symbols = ["BTCUSDT"]
    print(f"Successfully preloaded {len(valid_symbols)} symbols; skipped {len(invalid_symbols)} invalid symbols: {invalid_symbols}")
    log_debug(f"Successfully preloaded {len(valid_symbols)} symbols; skipped {len(invalid_symbols)} invalid symbols: {invalid_symbols}")
    state.symbols = valid_symbols 