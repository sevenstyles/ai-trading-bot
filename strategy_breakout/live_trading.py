import time
import os
# NOTE: Live trading now mimics backtester signal logic using 4h data. However, live orders are executed on Binance Futures Testnet as real market orders and do NOT apply the fee (FUTURES_FEE) and slippage (SLIPPAGE_RATE) adjustments used in the backtester simulation.
import pandas as pd
from datetime import datetime
from binance.client import Client
from binance import ThreadedWebsocketManager
from config import BINANCE_API_KEY, BINANCE_API_SECRET, FUTURES_FEE, SLIPPAGE_RATE
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
executed_trades = []  # List to store completed trades

def preload_candles():
    """Preloads the most recent 50 candles (using the current interval) for each symbol, so that technical indicators have sufficient historical data immediately upon starting live trading."""
    print("Preloading historical candles for symbols...")
    log_debug("Preloading historical candles for symbols...")
    for symbol in symbols:
        try:
            klines = client.futures_klines(symbol=symbol, interval=interval, limit=50)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                 'close_time', 'quote_asset_volume', 'trades',
                                                 'taker_buy_base', 'taker_buy_quote', 'ignore'])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            candles_dict[symbol] = df[numeric_cols]
            msg = f"Preloaded {len(df)} candles for {symbol}"
            print(msg)
            log_debug(msg)
        except Exception as e:
            err_msg = f"Error preloading candles for {symbol}: {e}"
            print(err_msg)
            log_debug(err_msg)

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
    ClientConnection.fail_connection = lambda self: None

def process_candle(msg):
    """
    Callback function for each kline message from the multiplex stream.
    It processes the candlestick data, updates the corresponding DataFrame in candles_dict, and checks for trade signals.
    """
    # Handle multiplexed stream structure: extract data if present
    if "data" in msg:
        msg = msg["data"]
    if msg.get('e') != 'kline':
        print('Warning: Received non-kline message, skipping:', msg)
        return
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
        log_debug(f"Candle for {coin_symbol} closed at {candle_time}: close = {close}")
        
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
                update_stop_order(symbol, trade)
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
                update_stop_order(symbol, trade)
        if latest_candle['high'] >= trade['stop_loss']:
            print(f"Short trade stop hit for {symbol}")
            close_trade(symbol, trade, trade['stop_loss'], df.index[-1])

def close_trade(symbol, trade, exit_price, exit_time):
    # Cancel any existing stop order
    if "stop_order_id" in trade:
        try:
            cancel_result = client.futures_cancel_order(symbol=symbol, orderId=trade["stop_order_id"])
            print(f"Cancelled stop order for {symbol}: {cancel_result}")
        except Exception as e:
            print(f"Error cancelling stop order for {symbol} when closing trade: {e}")

    if trade['direction'] == 'long':
        side = "SELL"
    else:
        side = "BUY"

    # Place a market order to close the position on Binance Futures Testnet using the stored quantity
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            closePosition=True
        )
        print(f"Exit order placed for {symbol}: {order}")
        if 'avgPrice' in order and float(order['avgPrice']) > 0:
            effective_exit = float(order['avgPrice'])
        else:
            effective_exit = simulate_fill_price(side, exit_price)
    except Exception as e:
        print(f"Error closing trade for {symbol}: {e}")
        effective_exit = simulate_fill_price(side, exit_price)

    print(f"Closing {trade['direction']} trade for {symbol} at {effective_exit} (actual) on {exit_time} by placing {side} order")
    trade['exit_price'] = effective_exit
    trade['exit_time'] = exit_time
    trade['status'] = 'closed'
    print(f"Trade closed for {symbol}: {trade}")
    
    # Append the closed trade to executed_trades for logging
    trade_record = trade.copy()
    trade_record['symbol'] = symbol
    executed_trades.append(trade_record)
    save_executed_trades_csv()

    if symbol in active_trades:
        del active_trades[symbol]
    
    # Save current active trades after closing one
    save_active_trades_csv()

def check_for_trade(symbol):
    """
    Checks if a trade signal is generated for the given symbol based on its candle data.
    """
    candle_count = len(candles_dict.get(symbol, []))
    log_debug(f"check_for_trade called for {symbol}, candle count: {candle_count}")
    print(f"check_for_trade called for {symbol}, candle count: {candle_count}")

    if symbol not in candles_dict:
        log_debug(f"No candles available yet for {symbol}")
        print(f"No candles available yet for {symbol}")
        return

    if candle_count < 50:
        log_debug(f"Not enough candles for {symbol}. Only {candle_count} available, waiting for at least 50.")
        print(f"Not enough candles for {symbol}. Only {candle_count} available, waiting for at least 50.")
        return

    df = candles_dict[symbol].copy().sort_index()
    
    # If there's an active trade, update its trailing stop
    if symbol in active_trades:
        update_active_trade(symbol, df)
        # Save active trades after update
        save_active_trades_csv()
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
        market_price = df["close"].iloc[-1]
        simulated_entry = simulate_fill_price("BUY", market_price)
        place_order(symbol, "BUY", market_price)
        active_trades[symbol] = {
            'entry_time': df.index[-1],
            'entry_price': simulated_entry,
            'stop_loss': simulated_entry * 0.99,  # LONG_STOP_LOSS_MULTIPLIER
            'take_profit': simulated_entry * 1.03,  # LONG_TAKE_PROFIT_MULTIPLIER
            'direction': 'long',
            'status': 'open',
            'quantity': 0.001
        }
        update_stop_order(symbol, active_trades[symbol])
    # Check for short signal
    elif short_signal:
        print(f"Short signal detected for {symbol}!")
        market_price = df["close"].iloc[-1]
        simulated_entry = simulate_fill_price("SELL", market_price)
        place_order(symbol, "SELL", market_price)
        active_trades[symbol] = {
            'entry_time': df.index[-1],
            'entry_price': simulated_entry,
            'stop_loss': simulated_entry * 1.01,  # SHORT_STOP_LOSS_MULTIPLIER
            'take_profit': simulated_entry * 0.97,  # SHORT_TAKE_PROFIT_MULTIPLIER
            'direction': 'short',
            'status': 'open',
            'quantity': 0.001
        }
        update_stop_order(symbol, active_trades[symbol])
    else:
        print(f"No trade signal for {symbol}.")

    # Save active trades at the end of trade check
    save_active_trades_csv()

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

def simulate_fill_price(side, market_price):
    # For BUY orders, effective fill price is increased by slippage and fee
    # For SELL orders, effective fill price is decreased
    if side == "BUY":
        return market_price * (1 + SLIPPAGE_RATE + FUTURES_FEE)
    elif side == "SELL":
        return market_price * (1 - SLIPPAGE_RATE - FUTURES_FEE)
    return market_price

def save_executed_trades_csv():
    if executed_trades:
        df = pd.DataFrame(executed_trades)
        df.to_csv("executed_trades.csv", index=False)
        print("Executed trades saved to executed_trades.csv")

def save_active_trades_csv():
    if active_trades:
        active_trade_list = []
        for symbol, trade in active_trades.items():
            trade_copy = trade.copy()
            trade_copy['symbol'] = symbol
            active_trade_list.append(trade_copy)
        df = pd.DataFrame(active_trade_list)
        df.to_csv("active_trades.csv", index=False)
        print("Active trades saved to active_trades.csv")

def load_active_positions():
    """Loads currently active positions from Binance Futures Testnet and populates active_trades."""
    try:
        positions = client.futures_position_information()
        for pos in positions:
            position_amt = float(pos.get("positionAmt", 0))
            # Only consider positions that are non-zero (use a small threshold if needed)
            if abs(position_amt) > 0:
                symbol = pos.get("symbol")
                entry_price = float(pos.get("entryPrice", 0))
                # Skip if no valid entry price
                if entry_price == 0:
                    continue
                if position_amt > 0:
                    trade = {
                        'entry_time': datetime.now(),
                        'entry_price': entry_price,
                        'stop_loss': entry_price * 0.99,  # default for long
                        'take_profit': entry_price * 1.03,  # default for long
                        'direction': 'long',
                        'status': 'open',
                        'quantity': abs(position_amt)
                    }
                else:
                    trade = {
                        'entry_time': datetime.now(),
                        'entry_price': entry_price,
                        'stop_loss': entry_price * 1.01,  # default for short
                        'take_profit': entry_price * 0.97,  # default for short
                        'direction': 'short',
                        'status': 'open',
                        'quantity': abs(position_amt)
                    }
                active_trades[symbol] = trade
                print(f"Loaded active position for {symbol}: {trade}")
        save_active_trades_csv()
    except Exception as e:
        print(f"Error loading active positions: {e}")

def update_stop_order(symbol, trade):
    side = "SELL" if trade["direction"] == "long" else "BUY"
    # If there's an existing stop order, cancel it
    if "stop_order_id" in trade:
        try:
            cancel_result = client.futures_cancel_order(symbol=symbol, orderId=trade["stop_order_id"])
            print(f"Cancelled previous stop order for {symbol}: {cancel_result}")
        except Exception as e:
            print(f"Could not cancel previous stop order for {symbol}: {e}")

    # Place a new STOP_MARKET order with the updated stop loss
    try:
        # Determine a simple rounding based on stop_loss value (adjust as needed per asset)
        stop_loss = trade["stop_loss"]
        stop_price_str = ""
        if stop_loss >= 10:
            stop_price_str = f"{stop_loss:.2f}"
        else:
            stop_price_str = f"{stop_loss:.4f}"

        # New check: get the current market price from candles_dict if available
        current_price = None
        if symbol in candles_dict and not candles_dict[symbol].empty:
            current_price = candles_dict[symbol].iloc[-1]["close"]
        
        if current_price is not None:
            stop_price = float(stop_price_str)
            if trade["direction"] == "short" and stop_price <= current_price:
                print(f"Skipping update for {symbol} short: new stop price {stop_price} is not above current price {current_price}")
                return
            if trade["direction"] == "long" and stop_price >= current_price:
                print(f"Skipping update for {symbol} long: new stop price {stop_price} is not below current price {current_price}")
                return

        new_order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="STOP_MARKET",
            stopPrice=stop_price_str,
            closePosition=True
        )
        trade["stop_order_id"] = new_order.get("orderId")
        print(f"Placed new stop order for {symbol} at {stop_price_str}: {new_order}")
    except Exception as e:
        print(f"Error placing new stop order for {symbol}: {e}")

def main():
    preload_candles()
    load_active_positions()
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