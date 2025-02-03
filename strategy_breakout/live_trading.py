import time
import threading

from data_loader import preload_candles
from trade_manager import load_active_positions
from ws_manager import process_candle, message_worker
from binance import ThreadedWebsocketManager
import config
import state


def main():
    print("Starting live trading with 1m timeframe...")
    preload_candles(interval="1m", limit=50)
    load_active_positions()

    twm = ThreadedWebsocketManager(api_key=config.BINANCE_API_KEY, api_secret=config.BINANCE_API_SECRET, testnet=True)
    twm.start()

    worker_thread = threading.Thread(target=message_worker, daemon=True)
    worker_thread.start()

    streams = [s.lower() + "@kline_1m" for s in state.symbols]
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