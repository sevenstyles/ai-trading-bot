import time
import os
# NOTE: Live trading now mimics backtester signal logic using 4h data. However, live orders are executed on Binance Futures Testnet as real market orders and do NOT apply the fee (FUTURES_FEE) and slippage (SLIPPAGE_RATE) adjustments used in the backtester simulation.
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

DEBUG_MODE = True
DEBUG_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_live_trading.log')

def log_debug(message):
    if DEBUG_MODE:
        from datetime import datetime
        with open(DEBUG_LOG_FILE, 'a') as f:
            f.write(f"{datetime.now()} - {message}\n")

# Global dictionary to hold candle data for each symbol
candles_dict = {}

# Add global active trades dictionary at the top
active_trades = {}  # key: symbol, value: trade dict

# Set interval for live data (now using 5 minute candles)
interval = "5m"

# Get the top 100 coins by volume using our data_fetch function
# Pair selection criteria: Pairs are selected via get_top_volume_pairs(limit=100) which returns the top-volume USDT pairs (excluding blacklisted pairs) based on 24hr ticker data. This selection is done once at startup and is not rechecked for each candlestick update.
symbols = get_top_volume_pairs(limit=100)
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
        
        # Call check_for_trade for the symbol
        check_for_trade(coin_symbol)

def update_active_trade(symbol, df):
    trade = active_trades.get(symbol)
    if not trade:
        return
    trailing_stop_pct = 0.0025
    latest_candle = df.iloc[-1]
    if trade['direction'] == 'long':
        # Initialize new_high if not set
        if 'new_high' not in trade:
            trade['new_high'] = trade['entry_price']
        # Update new high if current candle's high is greater
        if latest_candle['high'] > trade['new_high']:
            trade['new_high'] = latest_candle['high']
            potential_stop = trade['new_high'] * (1 - trailing_stop_pct)
            if potential_stop > trade['stop_loss']:
                trade['stop_loss'] = potential_stop
                print(f"Updated trailing stop for {symbol} long to {trade['stop_loss']}")
        # Check if current candle's low breaches stop loss
        if latest_candle['low'] <= trade['stop_loss']:
            print(f"Long trade stop hit for {symbol}")
            close_trade(symbol, trade, trade['stop_loss'], df.index[-1])
    elif trade['direction'] == 'short':
        if 'new_low' not in trade:
            trade['new_low'] = trade['entry_price']
        if latest_candle['low'] < trade['new_low']:
            trade['new_low'] = latest_candle['low']
            potential_stop = trade['new_low'] * (1 + trailing_stop_pct)
            if potential_stop < trade['stop_loss']:
                trade['stop_loss'] = potential_stop
                print(f"Updated trailing stop for {symbol} short to {trade['stop_loss']}")
        if latest_candle['high'] >= trade['stop_loss']:
            print(f"Short trade stop hit for {symbol}")
            close_trade(symbol, trade, trade['stop_loss'], df.index[-1])

def close_trade(symbol, trade, exit_price, exit_time):
    side = "SELL" if trade['direction'] == 'long' else "BUY"
    print(f"Closing {trade['direction']} trade for {symbol} at {exit_price} on {exit_time} by placing {side} order")
    # For now, we simulate closing trade. In real implementation, you would call client.futures_create_order() to exit order.
    trade['exit_price'] = exit_price
    trade['exit_time'] = exit_time
    trade['status'] = 'closed'
    print(f"Trade closed for {symbol}: {trade}")
    # Remove trade from active_trades
    if symbol in active_trades:
        del active_trades[symbol]

def check_for_trade(symbol):
    """
    Checks if a trade signal is generated for the given symbol based on its candle data.
    """
    if symbol not in candles_dict or len(candles_dict[symbol]) < 50:
        return
    df = candles_dict[symbol].copy().sort_index()
    
    # If there's an active trade, update its trailing stop
    if symbol in active_trades:
        update_active_trade(symbol, df)
        return

    # Calculate indicators using existing functions
    df = calculate_market_structure(df)
    df = calculate_trend_strength(df)
    df = calculate_emas(df)
    df = calculate_macd(df)
    df = calculate_adx(df)
    df = calculate_rsi(df)
    if DEBUG_MODE:
        ds = f"DEBUG: For {symbol} - Latest Candle Time: {df.index[-1]}, Open: {df['open'].iloc[-1]}, High: {df['high'].iloc[-1]}, Low: {df['low'].iloc[-1]}, Close: {df['close'].iloc[-1]}"
        print(ds)
        log_debug(ds)
    
    latest_idx = len(df) - 1
    long_signal = generate_signal(df, latest_idx)
    short_signal = generate_short_signal(df, latest_idx)
    if DEBUG_MODE:
        ds2 = f"DEBUG: For {symbol} at candle index {latest_idx} - long_signal: {long_signal}, short_signal: {short_signal}"
        print(ds2)
        log_debug(ds2)
    
    # Check for long signal
    if long_signal:
        print(f"Long signal detected for {symbol}!")
        price = df["close"].iloc[-1]
        place_order(symbol, "BUY", price)
        # Track the active trade
        active_trades[symbol] = {
            'entry_time': df.index[-1],
            'entry_price': price,
            'stop_loss': price * 0.995,
            'take_profit': price * 1.06,  # using LONG_TAKE_PROFIT_MULTIPLIER from backtesting, adjust as needed
            'direction': 'long',
            'status': 'open'
        }
    # Check for short signal
    elif short_signal:
        print(f"Short signal detected for {symbol}!")
        price = df["close"].iloc[-1]
        place_order(symbol, "SELL", price)
        active_trades[symbol] = {
            'entry_time': df.index[-1],
            'entry_price': price,
            'stop_loss': price * 1.005,
            'take_profit': price * 0.94,  # using SHORT_TAKE_PROFIT_MULTIPLIER, adjust as needed
            'direction': 'short',
            'status': 'open'
        }
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