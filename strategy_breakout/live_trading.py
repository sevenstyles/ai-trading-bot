import time
import os
# NOTE: Live trading now mimics backtester signal logic using 4h data. However, live orders are executed on Binance Futures Testnet as real market orders and do NOT apply the fee (FUTURES_FEE) and slippage (SLIPPAGE_RATE) adjustments used in the backtester simulation.
import pandas as pd
from datetime import datetime
from binance.client import Client
from binance import ThreadedWebsocketManager
from config import BINANCE_API_KEY, BINANCE_API_SECRET, FUTURES_FEE, SLIPPAGE_RATE, RISK_PER_TRADE
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
import queue
import threading

# Set interval for live data (now using 1m candles)
interval = "1m"

# Get the top 100 coins by volume using our data_fetch function
symbols = get_top_volume_pairs(limit=100)
if not symbols:
    symbols = ["BTCUSDT"]

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

# Keep trade_adjustments_count as it is used for trade updates
trade_adjustments_count = 0

# Add a new message queue
message_queue = queue.Queue(maxsize=2000)

def preload_candles():
    global symbols
    """Preloads the most recent 50 candles (using the current interval) for each symbol so that technical indicators have sufficient historical data."""
    print("Preloading historical candles for symbols...")
    log_debug("Preloading historical candles for symbols...")
    valid_symbols = []
    invalid_symbols = []
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
    symbols = valid_symbols

# Initialize the Binance Futures Testnet client
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)

## Monkey-patch for Binance WebSocket issue: add a dummy fail_connection if missing.
try:
    from binance.streams import ClientConnection
except ImportError:
    pass
else:
    ClientConnection.fail_connection = lambda self: None

def process_candle_message(msg):
    """Processes the full candle message (previously in process_candle)."""
    if "data" in msg:
        msg = msg["data"]
    if msg.get('e') == 'error':
        log_debug(f"Received error message: {msg}")
        return
    if msg.get('e') != 'kline':
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
        
        if coin_symbol not in candles_dict:
            candles_dict[coin_symbol] = pd.DataFrame()
        candles_dict[coin_symbol] = pd.concat([
            candles_dict[coin_symbol],
            pd.DataFrame([new_candle], index=[candle_time])
        ])
        candles_dict[coin_symbol] = candles_dict[coin_symbol].tail(100)
        
        print(f"Candle for {coin_symbol} closed at {candle_time}: close = {close}")
        log_debug(f"Candle for {coin_symbol} closed at {candle_time}: close = {close}")
        
        check_for_trade(coin_symbol)

def process_candle(msg):
    """Lightweight callback that enqueues incoming messages for asynchronous processing."""
    try:
        message_queue.put_nowait(msg)
    except queue.Full:
        print("Local message_queue is full, dropping message")

def update_active_trade(symbol, df):
    global trade_adjustments_count
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
                trade_adjustments_count += 1
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
                trade_adjustments_count += 1
                update_stop_order(symbol, trade)
        # For short trades, if the latest candle's high reaches or exceeds the stop loss level,
        # it indicates adverse price movement against the position, leading to a trade close (as happened for BTCUSDT).
        if (latest_candle["high"] >= trade["stop_loss"]):
            print(f"Short trade stop hit for {symbol}")
            close_trade(symbol, trade, trade["stop_loss"], df.index[-1])

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
            quantity=trade["quantity"],
            reduceOnly=True
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
        quantity = place_order(symbol, "BUY", market_price)
        active_trades[symbol] = {
            'entry_time': df.index[-1],
            'entry_price': simulated_entry,
            'stop_loss': simulated_entry * 0.99,  # LONG_STOP_LOSS_MULTIPLIER
            'take_profit': simulated_entry * 1.03,  # LONG_TAKE_PROFIT_MULTIPLIER
            'direction': 'long',
            'status': 'open',
            'quantity': quantity if quantity is not None else 0.001
        }
        update_stop_order(symbol, active_trades[symbol])
    # Check for short signal
    elif short_signal:
        print(f"Short signal detected for {symbol}!")
        market_price = df["close"].iloc[-1]
        simulated_entry = simulate_fill_price("SELL", market_price)
        quantity = place_order(symbol, "SELL", market_price)
        active_trades[symbol] = {
            'entry_time': df.index[-1],
            'entry_price': simulated_entry,
            'stop_loss': simulated_entry * 1.01,  # SHORT_STOP_LOSS_MULTIPLIER
            'take_profit': simulated_entry * 0.97,  # SHORT_TAKE_PROFIT_MULTIPLIER
            'direction': 'short',
            'status': 'open',
            'quantity': quantity if quantity is not None else 0.001
        }
        update_stop_order(symbol, active_trades[symbol])
    else:
        print(f"No trade signal for {symbol}.")

    # Save active trades at the end of trade check
    save_active_trades_csv()

def place_order(symbol, side, price):
    """
    Places a market order for the given symbol on Binance Futures Testnet using risk percent from config.
    :param symbol: The trading pair symbol (e.g., BTCUSDT)
    :param side: "BUY" for long, "SELL" for short
    :param price: The current close price (for calculating position size and logging purposes)
    """
    try:
        # Retrieve futures account balance and calculate risk percent of available USDT capital
        balance_info = client.futures_account_balance()
        usdt_balance = None
        for b in balance_info:
            if b.get('asset') == 'USDT':
                usdt_balance = float(b.get('balance', 0))
                break
        if usdt_balance is None or usdt_balance <= 0:
            print(f"Unable to retrieve USDT balance or balance is zero. Using default quantity 0.001")
            quantity = 0.001
        else:
            order_value = usdt_balance * RISK_PER_TRADE  # use risk percent from config
            quantity = order_value / price
            # Round quantity to 6 decimal places for demonstration purposes
            quantity = round(quantity, 6)
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )
        print(f"Order placed for {symbol} at calculated quantity {quantity}: {order}")
        return quantity
    except Exception as e:
        print(f"Error placing order for {symbol}: {e}")
        return None

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

    try:
        # Determine a simple rounding based on stop_loss value (adjust as needed per asset)
        stop_loss = trade["stop_loss"]
        stop_price_str = ""
        if stop_loss >= 10:
            stop_price_str = f"{stop_loss:.2f}"
        else:
            stop_price_str = f"{stop_loss:.4f}"

        # Fetch the live market price from Binance ticker
        current_price = None
        try:
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker["price"])
            print(f"Fetched live price for {symbol}: {current_price}")
        except Exception as e:
            print(f"Error fetching ticker for {symbol}, falling back to candles price: {e}")
            if symbol in candles_dict and not candles_dict[symbol].empty:
                current_price = candles_dict[symbol].iloc[-1]["close"]

        if current_price is not None:
            stop_price = float(stop_price_str)
            # Define a minimum gap as 0.25% of current price (adjusted value)
            min_buffer = current_price * 0.0025
            if trade["direction"] == "short":
                if stop_price <= current_price or (stop_price - current_price) < min_buffer:
                    print(f"Skipping update for {symbol} short: new stop price {stop_price} is not sufficiently above live price {current_price} (min gap: {min_buffer})")
                    return
            elif trade["direction"] == "long":
                if stop_price >= current_price or (current_price - stop_price) < min_buffer:
                    print(f"Skipping update for {symbol} long: new stop price {stop_price} is not sufficiently below live price {current_price} (min gap: {min_buffer})")
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

def message_worker():
    while True:
        msg = message_queue.get()
        if msg is None:
            break
        process_candle_message(msg)
        message_queue.task_done()

def main():
    print("Starting live trading with 1m timeframe...")
    preload_candles()
    load_active_positions()
    twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET, testnet=True)
    twm.start()

    # Start a worker thread to process incoming websocket messages asynchronously
    worker_thread = threading.Thread(target=message_worker, daemon=True)
    worker_thread.start()

    streams = [s.lower() + "@kline_" + interval for s in symbols]
    print(f"Subscribing to streams: {streams}")
    socket_id = twm.start_multiplex_socket(callback=process_candle, streams=streams)
    print(f"Multiplex socket started with id: {socket_id}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping live trading...")
    finally:
        twm.stop()

if __name__ == "__main__":
    main() 