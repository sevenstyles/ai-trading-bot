import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import random
import requests
from binance.client import Client
from config import BASE_DIR, MIN_LIQUIDITY, BLACKLIST, POSITION_SIZE_CAP

from config import BINANCE_API_KEY, BINANCE_API_SECRET
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

def get_top_volume_pairs(limit=100):
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