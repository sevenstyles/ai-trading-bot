import os
from datetime import datetime

TRADE_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_executed.log')


def log_trade(message):
    '''Append a trade event message with timestamp to the trade log file.'''
    with open(TRADE_LOG_FILE, 'a') as f:
        f.write(f"{datetime.now()} - {message}\n") 