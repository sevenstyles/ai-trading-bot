import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import random
import requests
from binance.client import Client
from config import BASE_DIR, MIN_LIQUIDITY, BLACKLIST, POSITION_SIZE_CAP

from config import BINANCE_API_KEY, BINANCE_API_SECRET
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

def fetch_data(symbol, interval='4h', limit=1000):
    """Robust data fetching focused on a given timeframe."""
    try:
        data = {}
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines:
            print(f"No {interval} data for {symbol}")
            return None
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                   'close_time', 'quote_asset_volume', 'number_of_trades',
                   'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=columns)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC')
        df.set_index('timestamp', inplace=True)
        data[interval] = df[['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']]
        return data
    except Exception as e:
        print(f"Data fetch failed for {symbol}: {str(e)}")
        return None

def fetch_data_api(symbol, interval="15m", limit=1000):
    """Fetch historical OHLC data (15m timeframe) from Binance API."""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df.index = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df.index = df.index.tz_localize('UTC')
        df = df[['open', 'high', 'low', 'close', 'volume']]
        print(f"Successfully fetched {len(df)} candles of {interval} data for {symbol}")
        return df
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

def get_random_symbols(count=5):
    """Get a random sample of symbols from the top 50 by volume."""
    all_tickers = client.get_ticker()
    usdt_tickers = [t for t in all_tickers if t['symbol'].endswith('USDT')]
    valid_symbols = [t for t in usdt_tickers if float(t['lastPrice']) > 1.0 and t['symbol'] not in BLACKLIST]
    valid_symbols.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
    top_symbols = [t['symbol'] for t in valid_symbols[:50]]
    return random.sample(top_symbols, min(count, len(top_symbols)))

def is_tradable(symbol):
    """Check if a symbol meets a minimal liquidity requirement."""
    ticker = client.get_ticker(symbol=symbol)
    return float(ticker['quoteVolume']) > MIN_LIQUIDITY

def get_top_volume_pairs(limit=50):
    """Retrieve the top-volume USDT pairs from Binance 24hr ticker."""
    try:
        if hasattr(client, "get_ticker_24hr"):
            tickers = client.get_ticker_24hr()
        else:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            response = requests.get(url)
            tickers = response.json()
    except Exception as e:
        print("Failed to fetch ticker data:", e)
        return []
    usdt_pairs = [t for t in tickers if t['symbol'].endswith("USDT") and t['symbol'] not in BLACKLIST]
    usdt_pairs_sorted = sorted(usdt_pairs, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
    top_volume_pairs = [t['symbol'] for t in usdt_pairs_sorted[:limit]]
    return top_volume_pairs

def get_session_data(df, start_time="00:00", end_time="08:00"):
    """Return a subset of the data between specified times (per day)."""
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df.between_time(start_time, end_time)

def get_position_cap(entry_price):
    """Determine an appropriate position cap based on entry price."""
    if entry_price < 0.001:
        return POSITION_SIZE_CAP * 5
    elif entry_price < 0.1:
        return 250000
    else:
        return 10000 