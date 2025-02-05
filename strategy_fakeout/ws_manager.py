import pandas as pd
from datetime import datetime
from logger import log_debug
import state
from trade_manager import check_for_trade


def process_candle_message(msg):
    if "data" in msg:
        msg = msg["data"]
    if msg.get("e") == "error":
        log_debug(f"Received error message: {msg}")
        return
    if msg.get("e") != "kline":
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
        if coin_symbol not in state.candles_dict:
            state.candles_dict[coin_symbol] = pd.DataFrame()
        state.candles_dict[coin_symbol] = pd.concat([
            state.candles_dict[coin_symbol],
            pd.DataFrame([new_candle], index=[candle_time])
        ])
        state.candles_dict[coin_symbol] = state.candles_dict[coin_symbol].tail(100)
        print(f"Candle for {coin_symbol} closed at {candle_time}: close = {close}")
        log_debug(f"Candle for {coin_symbol} closed at {candle_time}: close = {close}")
        check_for_trade(coin_symbol)


def process_candle(msg):
    try:
        state.message_queue.put_nowait(msg)
    except Exception:
        print("Local message_queue is full, dropping message")


def message_worker():
    while True:
        msg = state.message_queue.get()
        if msg is None:
            break
        process_candle_message(msg)
        state.message_queue.task_done() 