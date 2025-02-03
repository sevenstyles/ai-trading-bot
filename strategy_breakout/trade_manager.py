import pandas as pd
from datetime import datetime
import state
from logger import log_debug
from binance_client import client
from config import RISK_PER_TRADE, FUTURES_FEE, SLIPPAGE_RATE, LEVERAGE
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

def update_active_trade(symbol, df):
    trade = state.active_trades.get(symbol)
    if not trade:
        return
    trailing_stop_pct = 0.0025
    latest_candle = df.iloc[-1]
    if trade['direction'] == 'long':
        if 'new_high' not in trade:
            trade['new_high'] = trade['entry_price']
        if latest_candle['high'] > trade['new_high']:
            trade['new_high'] = latest_candle['high']
            potential_stop = trade['new_high'] * (1 - trailing_stop_pct)
            if potential_stop > trade['stop_loss']:
                trade['stop_loss'] = potential_stop
                state.trade_adjustments_count += 1
                update_stop_order(symbol, trade)
        if latest_candle['low'] <= trade['stop_loss']:
            close_trade(symbol, trade, trade['stop_loss'], df.index[-1], close_reason="stop loss hit")
    elif trade['direction'] == 'short':
        if 'new_low' not in trade:
            trade['new_low'] = trade['entry_price']
        if latest_candle['low'] < trade['new_low']:
            trade['new_low'] = latest_candle['low']
            potential_stop = trade['new_low'] * (1 + trailing_stop_pct)
            if potential_stop < trade['stop_loss']:
                trade['stop_loss'] = potential_stop
                state.trade_adjustments_count += 1
                update_stop_order(symbol, trade)
        if latest_candle['high'] >= trade['stop_loss']:
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
            effective_exit = simulate_fill_price(side, exit_price)
    except Exception as e:
        log_debug(f"Error closing trade for {symbol}: {e}")
        effective_exit = simulate_fill_price(side, exit_price)
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
    if long_signal:
        market_price = df["close"].iloc[-1]
        simulated_entry = simulate_fill_price("BUY", market_price)
        quantity = place_order(symbol, "BUY", market_price)
        state.active_trades[symbol] = {
            'entry_time': df.index[-1],
            'entry_price': simulated_entry,
            'stop_loss': simulated_entry * 0.99,
            'take_profit': simulated_entry * 1.03,
            'direction': 'long',
            'status': 'open',
            'quantity': quantity if quantity is not None else 0.001
        }
        update_stop_order(symbol, state.active_trades[symbol])
    elif short_signal:
        market_price = df["close"].iloc[-1]
        simulated_entry = simulate_fill_price("SELL", market_price)
        quantity = place_order(symbol, "SELL", market_price)
        state.active_trades[symbol] = {
            'entry_time': df.index[-1],
            'entry_price': simulated_entry,
            'stop_loss': simulated_entry * 1.01,
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
                step_size = float(f.get('stepSize'))
                # Calculate precision based on step_size
                if step_size > 0:
                    precision = int(round(-math.log10(step_size)))
                    return round(quantity, precision)
        return round(quantity, 3)
    except Exception as e:
        return round(quantity, 3)


def place_order(symbol, side, price):
    try:
        # Set the desired leverage for the symbol
        client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
    except Exception as e:
        log_trade(f"Error setting leverage for {symbol}: {e}")
    try:
        balance_info = client.futures_account_balance()
        usdt_balance = None
        for b in balance_info:
            if b.get('asset') == 'USDT':
                usdt_balance = float(b.get('balance', 0))
                break
        if usdt_balance is None or usdt_balance <= 0:
            quantity = 0.001
        else:
            order_value = usdt_balance * RISK_PER_TRADE
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


def simulate_fill_price(side, market_price):
    if side == "BUY":
        return market_price * (1 + SLIPPAGE_RATE + FUTURES_FEE)
    elif side == "SELL":
        return market_price * (1 - SLIPPAGE_RATE - FUTURES_FEE)
    return market_price


def save_executed_trades_csv():
    if state.executed_trades:
        df = pd.DataFrame(state.executed_trades)
        df.to_csv("executed_trades.csv", index=False)
        log_debug("Executed trades saved to executed_trades.csv")


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
        stop_price_str = f"{stop_loss:.2f}" if stop_loss >= 10 else f"{stop_loss:.4f}"
        current_price = None
        try:
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker["price"])
        except Exception as e:
            log_debug(f"Error fetching ticker for {symbol}: {e}")
            if symbol in state.candles_dict and not state.candles_dict[symbol].empty:
                current_price = state.candles_dict[symbol].iloc[-1]["close"]
        if current_price is not None:
            stop_price = float(stop_price_str)
            min_buffer = current_price * 0.0025
            if trade["direction"] == "short":
                if stop_price <= current_price or (stop_price - current_price) < min_buffer:
                    return
            elif trade["direction"] == "long":
                if stop_price >= current_price or (current_price - stop_price) < min_buffer:
                    return
            new_order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                stopPrice=stop_price_str,
                closePosition=True
            )
            trade["stop_order_id"] = new_order.get("orderId")
            log_trade(f"Placed new stop order for {symbol} at {stop_price_str}: {new_order}")
        else:
            log_debug(f"Could not update stop order for {symbol} due to missing current price.")
    except Exception as e:
        log_debug(f"Error placing new stop order for {symbol}: {e}") 