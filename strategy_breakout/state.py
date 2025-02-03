from queue import Queue

candles_dict = {}
active_trades = {}
executed_trades = []
trade_adjustments_count = 0
message_queue = Queue(maxsize=2000)
symbols = [] 