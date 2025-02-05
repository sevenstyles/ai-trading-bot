import pandas as pd
from datetime import datetime
import state
from logger import log_debug
from binance_client import client
from config import RISK_PER_TRADE, SLIPPAGE_RATE, LEVERAGE, CANDLESTICK_INTERVAL
from config import CAPITAL, ORDER_SIZE_PCT
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
from trade_logger import log_trade
import math
from decimal import Decimal, ROUND_DOWN
from strategy import get_trend_direction

def apply_trailing_stop(symbol, trade, latest_candle, trailing_stop_pct):
    # For long trades, update new_high and potential trailing stop
    if trade['direction'] == 'long':
        initial_high = trade.get('new_high', trade['entry_price'])
        new_high = max(initial_high, latest_candle['high'])
        if new_high != initial_high:
            trade['new_high'] = new_high
            potential_stop = new_high * (1 - trailing_stop_pct)
            if potential_stop > trade['stop_loss']:
                trade['stop_loss'] = potential_stop
                state.trade_adjustments_count += 1
                update_stop_order(symbol, trade)
    elif trade['direction'] == 'short':
        initial_low = trade.get('new_low', trade['entry_price'])
        new_low = min(initial_low, latest_candle['low'])
        if new_low != initial_low:
            trade['new_low'] = new_low
            potential_stop = new_low * (1 + trailing_stop_pct)
            if potential_stop < trade['stop_loss']:
                trade['stop_loss'] = potential_stop
                state.trade_adjustments_count += 1
                update_stop_order(symbol, trade)

def update_active_trade(symbol, df):
    trade = state.active_trades.get(symbol)
    if not trade:
        return
    trailing_stop_pct = 0.0025
    latest_candle = df.iloc[-1]
    apply_trailing_stop(symbol, trade, latest_candle, trailing_stop_pct)
    # Check if the stop loss has been hit
    if trade['direction'] == 'long' and latest_candle['low'] <= trade['stop_loss']:
        close_trade(symbol, trade, trade['stop_loss'], df.index[-1], close_reason="stop loss hit")
    elif trade['direction'] == 'short' and latest_candle['high'] >= trade['stop_loss']:
        close_trade(symbol, trade, trade['stop_loss'], df.index[-1], close_reason="stop loss hit")


def close_trade(symbol, trade, exit_price, exit_time, close_reason=""):
    if "stop_order_id" in trade:
        try:
            client.futures_cancel_order(symbol=symbol, orderId=trade["stop_order_id"])
        except Exception as e:
            log_debug(f"Error cancelling stop order for {symbol} when closing trade: {e}")
    side = "SELL" if trade['direction'] == 'long' else "BUY"
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=trade["quantity"],
            reduceOnly=True
        )
        if 'avgPrice' in order and float(order['avgPrice']) > 0:
            effective_exit = float(order['avgPrice'])
        else:
            effective_exit = simulate_fill_price(symbol, side, exit_price)
    except Exception as e:
        log_debug(f"Error closing trade for {symbol}: {e}")
        effective_exit = simulate_fill_price(symbol, side, exit_price)
    trade['exit_price'] = effective_exit
    trade['exit_time'] = exit_time
    trade['status'] = 'closed'
    trade['close_reason'] = close_reason
    print(f"Trade closed for {symbol}: {close_reason}")
    trade_record = trade.copy()
    trade_record['symbol'] = symbol
    state.executed_trades.append(trade_record)
    save_executed_trades_csv()
    if symbol in state.active_trades:
        del state.active_trades[symbol]
    save_active_trades_csv()
    log_trade(f"Trade closed for {symbol}: {trade_record}")


def check_for_trade(symbol):
    candle_count = len(state.candles_dict.get(symbol, []))
    log_debug(f"check_for_trade called for {symbol}, candle count: {candle_count}")
    if symbol not in state.candles_dict:
        log_debug(f"No candles available yet for {symbol}")
        return
    if candle_count < 50:
        log_debug(f"Not enough candles for {symbol}. Only {candle_count} available, waiting for at least 50.")
        return
    df = state.candles_dict[symbol].copy().sort_index()
    if symbol in state.active_trades:
        update_active_trade(symbol, df)
        save_active_trades_csv()
        return
    df = calculate_market_structure(df)
    df = calculate_trend_strength(df)
    df = calculate_emas(df)
    df = calculate_macd(df)
    df = calculate_adx(df)
    df = calculate_rsi(df)
    log_debug(f"For {symbol} - Latest Candle Time: {df.index[-1]}, Close: {df['close'].iloc[-1]}")
    latest_idx = len(df) - 1
    long_signal = generate_signal(df, latest_idx)
    short_signal = generate_short_signal(df, latest_idx)
    log_debug(f"For {symbol} at index {latest_idx} - long_signal: {long_signal}, short_signal: {short_signal}")
    
    # Retrieve HTF (4h) candles for filtering (last 30 days)
    import datetime
    start_date = datetime.datetime.now() - datetime.timedelta(days=30)
    end_date = datetime.datetime.now()
    start_str = start_date.strftime("%d %b %Y")
    end_str = end_date.strftime("%d %b %Y")
    htf_klines = client.get_historical_klines(symbol, "4h", start_str, end_str)
    if not htf_klines:
        log_debug(f"Could not retrieve HTF candles for {symbol}")
        return
    df_4h = pd.DataFrame(htf_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df_4h[numeric_cols] = df_4h[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms')
    df_4h = df_4h.sort_values(by='timestamp')
    
    # Filter HTF candles up to the current timeframe's latest timestamp
    current_time = df.index[-1]
    df_4h_current = df_4h[df_4h['timestamp'] <= current_time]
    if df_4h_current.empty:
        log_debug(f"No HTF candles available up to {current_time} for {symbol}")
        return
    
    htf_trend = get_trend_direction(df_4h_current)
    log_debug(f"HTF trend for {symbol}: {htf_trend}")
    
    # Only allow trades if the HTF trend aligns with the lower timeframe signal
    if long_signal and htf_trend != 'bullish':
        log_debug(f"HTF trend is not bullish for {symbol}. Skipping long trade.")
        return
    if short_signal and htf_trend != 'bearish':
        log_debug(f"HTF trend is not bearish for {symbol}. Skipping short trade.")
        return
    
    if long_signal:
        market_price = df["close"].iloc[-1]
        simulated_entry = simulate_fill_price(symbol, "BUY", market_price)
        quantity = place_order(symbol, "BUY", market_price)
        state.active_trades[symbol] = {
            'entry_time': df.index[-1],
            'entry_price': simulated_entry,
            'stop_loss': simulated_entry * 0.995,
            'take_profit': simulated_entry * 1.03,
            'direction': 'long',
            'status': 'open',
            'quantity': quantity if quantity is not None else 0.001
        }
        update_stop_order(symbol, state.active_trades[symbol])
    elif short_signal:
        market_price = df["close"].iloc[-1]
        simulated_entry = simulate_fill_price(symbol, "SELL", market_price)
        quantity = place_order(symbol, "SELL", market_price)
        state.active_trades[symbol] = {
            'entry_time': df.index[-1],
            'entry_price': simulated_entry,
            'stop_loss': simulated_entry * 1.005,
            'take_profit': simulated_entry * 0.97,
            'direction': 'short',
            'status': 'open',
            'quantity': quantity if quantity is not None else 0.001
        }
        update_stop_order(symbol, state.active_trades[symbol])
    save_active_trades_csv()


def dynamic_round_quantity(symbol, quantity):
    try:
        info = client.get_symbol_info(symbol)
        for f in info.get('filters', []):
            if f.get('filterType') == 'LOT_SIZE':
                step = f.get('stepSize')
                dstep = Decimal(str(step))
                qty_decimal = Decimal(str(quantity))
                rounded_qty = qty_decimal.quantize(dstep, rounding=ROUND_DOWN)
                min_qty = Decimal(f.get('minQty', '0'))
                if rounded_qty < min_qty:
                    rounded_qty = min_qty
                return format(rounded_qty, 'f')
        return str(round(quantity, 3))
    except Exception as e:
        return str(round(quantity, 3))


def dynamic_round_price(symbol, price):
    try:
        info = client.get_symbol_info(symbol)
        for f in info.get('filters', []):
            if f.get('filterType') == 'PRICE_FILTER':
                tick = f.get('tickSize')
                dtick = Decimal(str(tick))
                price_decimal = Decimal(str(price))
                rounded_price = price_decimal.quantize(dtick, rounding=ROUND_DOWN)
                return format(rounded_price, 'f')
        return str(round(price, 4))
    except Exception as e:
        return str(round(price, 4))


def place_order(symbol, side, price):
    try:
        # Set the desired leverage for the symbol
        client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
    except Exception as e:
        log_trade(f"Error setting leverage for {symbol}: {e}")
    try:
        # Calculate order value based on capital and order size percentage for live trading
        order_value = CAPITAL * ORDER_SIZE_PCT
        quantity = order_value / price
        quantity = dynamic_round_quantity(symbol, quantity)
     
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )
        log_trade(f"Order placed for {symbol} with quantity {quantity}. Order details: {order}")
        return quantity
    except Exception as e:
        log_trade(f"Error placing order for {symbol}: {e}")
        log_debug(f"Error placing order for {symbol}: {e}")
        return None


def simulate_fill_price(symbol, side, market_price, quantity=1.0, order_book=None):
    """Simulate filling an order using market depth data if provided, otherwise apply default slippage and round the price."""
    if order_book is not None:
        from orderbook_simulator import simulate_market_order
        average_price, filled_qty = simulate_market_order(order_book, quantity, side.lower())
        return float(dynamic_round_price(symbol, average_price))
    else:
        if side.upper() == "BUY":
            simulated = market_price * (1 + SLIPPAGE_RATE)
        elif side.upper() == "SELL":
            simulated = market_price * (1 - SLIPPAGE_RATE)
        else:
            simulated = market_price
        return float(dynamic_round_price(symbol, simulated))


def save_executed_trades_csv():
    if state.executed_trades:
        df = pd.DataFrame(state.executed_trades)
        # Calculate profit percentage if trade details exist
        if all(col in df.columns for col in ['entry_price', 'exit_price', 'direction']):
            def calc_profit(row):
                try:
                    if row['direction'] == 'long':
                        return ((row['exit_price'] - row['entry_price']) / row['entry_price']) * 100
                    elif row['direction'] == 'short':
                        return ((row['entry_price'] - row['exit_price']) / row['entry_price']) * 100
                except Exception as e:
                    return None
            df['profit_pct'] = df.apply(calc_profit, axis=1)
            df['profit_pct'] = df['profit_pct'].round(2)
        filename = f"executed_trades_{CANDLESTICK_INTERVAL}.csv"
        df.to_csv(filename, index=False)
        log_debug(f"Executed trades saved to {filename}")


def save_active_trades_csv():
    if state.active_trades:
        active_trade_list = []
        for symbol, trade in state.active_trades.items():
            trade_copy = trade.copy()
            trade_copy['symbol'] = symbol
            active_trade_list.append(trade_copy)
        df = pd.DataFrame(active_trade_list)
        df.to_csv("active_trades.csv", index=False)
        log_debug("Active trades saved to active_trades.csv")


def load_active_positions():
    try:
        positions = client.futures_position_information()
        for pos in positions:
            position_amt = float(pos.get("positionAmt", 0))
            if abs(position_amt) > 0:
                symbol = pos.get("symbol")
                entry_price = float(pos.get("entryPrice", 0))
                if entry_price == 0:
                    continue
                if position_amt > 0:
                    trade = {
                        'entry_time': datetime.now(),
                        'entry_price': entry_price,
                        'stop_loss': entry_price * 0.99,
                        'take_profit': entry_price * 1.03,
                        'direction': 'long',
                        'status': 'open',
                        'quantity': abs(position_amt)
                    }
                else:
                    trade = {
                        'entry_time': datetime.now(),
                        'entry_price': entry_price,
                        'stop_loss': entry_price * 1.01,
                        'take_profit': entry_price * 0.97,
                        'direction': 'short',
                        'status': 'open',
                        'quantity': abs(position_amt)
                    }
                state.active_trades[symbol] = trade
                log_debug(f"Loaded active position for {symbol}")
        save_active_trades_csv()
    except Exception as e:
        log_debug(f"Error loading active positions: {e}")


def update_stop_order(symbol, trade):
    side = "SELL" if trade["direction"] == "long" else "BUY"
    if "stop_order_id" in trade:
        try:
            client.futures_cancel_order(symbol=symbol, orderId=trade["stop_order_id"])
        except Exception as e:
            log_debug(f"Could not cancel previous stop order for {symbol}: {e}")
    try:
        stop_loss = trade["stop_loss"]
        stop_price = dynamic_round_price(symbol, stop_loss)
        current_price = None
        try:
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker["price"])
        except Exception as e:
            log_debug(f"Error fetching ticker for {symbol}: {e}")
            if symbol in state.candles_dict and not state.candles_dict[symbol].empty:
                current_price = state.candles_dict[symbol].iloc[-1]["close"]
        if current_price is not None:
            # Ensure stop price is sufficiently away from current price
            min_buffer = current_price * 0.0025
            if trade["direction"] == "short":
                if stop_price <= current_price or (stop_price - current_price) < min_buffer:
                    return
            elif trade["direction"] == "long":
                if stop_price >= current_price or (current_price - stop_price) < min_buffer:
                    return

            # Check against PERCENT_PRICE filter if available
            symbol_info = client.get_symbol_info(symbol)
            for f in symbol_info.get("filters", []):
                if f.get("filterType") == "PERCENT_PRICE":
                    multiplier_up = float(f.get("multiplierUp", 0))
                    multiplier_down = float(f.get("multiplierDown", 0))
                    if trade["direction"] == "short" and stop_price > current_price * multiplier_up:
                        log_debug(f"Stop price {stop_price} exceeds allowed multiplier for {symbol}")
                        return
                    if trade["direction"] == "long" and stop_price < current_price * multiplier_down:
                        log_debug(f"Stop price {stop_price} below allowed multiplier for {symbol}")
                        return
                    break

            new_order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                stopPrice=str(stop_price),
                closePosition=True
            )
            trade["stop_order_id"] = new_order.get("orderId")
            log_trade(f"Placed new stop order for {symbol} at {stop_price}: {new_order}")
        else:
            log_debug(f"Could not update stop order for {symbol} due to missing current price.")
    except Exception as e:
        log_debug(f"Error placing new stop order for {symbol}: {e}") 