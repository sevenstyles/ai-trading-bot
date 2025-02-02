import time
import pandas as pd
from datetime import datetime
from binance.client import Client
from binance import ThreadedWebsocketManager
from config import BINANCE_API_KEY, BINANCE_API_SECRET
from data_fetch import get_top_volume_pairs
# Import your signal and indicator functions
from indicators import (
    calculate_market_structure,
    calculate_trend_strength,
    calculate_emas,
    calculate_macd,
    calculate_adx,
    calculate_rsi,
    generate_signal,
    generate_short_signal
)

# Global dictionary to hold candle data for each symbol
candles_dict = {}

# Set interval for live data (e.g., 1m candles)
interval = "1m"

# Get the top 50 coins by volume using our data_fetch function
symbols = get_top_volume_pairs(limit=50)
if not symbols:
    # Fallback to a default symbol if none found
    symbols = ["BTCUSDT"]

# Initialize the Binance Futures Testnet client
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)

## Monkey-patch for Binance WebSocket issue: add a dummy fail_connection if missing.
try:
    from binance.streams import ClientConnection
except ImportError:
    pass
else:
    if not hasattr(ClientConnection, 'fail_connection'):
        ClientConnection.fail_connection = lambda self: None

def process_candle(msg):
    """
    Callback function for each kline message from the multiplex stream.
    It processes the candlestick data, updates the corresponding DataFrame in candles_dict, and checks for trade signals.
    """
    # Handle multiplex socket structure
    if "data" in msg:
        kline = msg["data"]["k"]
    else:
        kline = msg["k"]
    is_candle_closed = kline["x"]
    candle_time = datetime.fromtimestamp(kline["t"] / 1000)
    coin_symbol = kline["s"]

    if is_candle_closed:
        open_price = float(kline["o"])
        high = float(kline["h"])
        low = float(kline["l"])
        close = float(kline["c"])
        volume = float(kline["v"])
        
        new_candle = {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        }
        
        # Initialize DataFrame if this symbol is new
        if coin_symbol not in candles_dict:
            candles_dict[coin_symbol] = pd.DataFrame()
        # Append new candle using pd.concat (since append is deprecated)
        candles_dict[coin_symbol] = pd.concat([
            candles_dict[coin_symbol],
            pd.DataFrame([new_candle], index=[candle_time])
        ])
        # Keep only the most recent 100 candles
        candles_dict[coin_symbol] = candles_dict[coin_symbol].tail(100)
        
        print(f"Candle for {coin_symbol} closed at {candle_time}: close = {close}")
        
        # Check trade signals for this symbol
        check_for_trade(coin_symbol)

def check_for_trade(symbol):
    """
    Checks if a trade signal is generated for the given symbol based on its candle data.
    """
    if symbol not in candles_dict or len(candles_dict[symbol]) < 50:
        return
    df = candles_dict[symbol].copy().sort_index()
    
    # Calculate indicators using existing functions
    df = calculate_market_structure(df)
    df = calculate_trend_strength(df)
    df = calculate_emas(df)
    df = calculate_macd(df)
    df = calculate_adx(df)
    df = calculate_rsi(df)
    
    latest_idx = len(df) - 1
    
    # Check for long signal
    if generate_signal(df, latest_idx):
        print(f"Long signal detected for {symbol}!")
        place_order(symbol, "BUY", df["close"].iloc[-1])
    # Check for short signal
    elif generate_short_signal(df, latest_idx):
        print(f"Short signal detected for {symbol}!")
        place_order(symbol, "SELL", df["close"].iloc[-1])
    else:
        print(f"No trade signal for {symbol}.")

def place_order(symbol, side, price):
    """
    Places a market order for the given symbol on Binance Futures Testnet.
    :param symbol: The trading pair symbol (e.g., BTCUSDT)
    :param side: "BUY" for long, "SELL" for short
    :param price: The current close price (for logging purposes)
    """
    quantity = 0.001  # Example fixed quantity for demonstration
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )
        print(f"Order placed for {symbol}: {order}")
    except Exception as e:
        print(f"Error placing order for {symbol}: {e}")

def main():
    twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET, testnet=True)
    twm.start()
    
    # Build a list of streams for the multiplex socket. Each stream is in the format: "<symbol in lowercase>@kline_<interval>"
    streams = [s.lower() + "@kline_" + interval for s in symbols]
    print(f"Subscribing to streams: {streams}")
    twm.start_multiplex_socket(callback=process_candle, streams=streams)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping live trading...")
    finally:
        twm.stop()

if __name__ == "__main__":
    main() 