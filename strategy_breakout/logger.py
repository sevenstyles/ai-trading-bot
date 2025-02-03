import os
from datetime import datetime

DEBUG_MODE = True
DEBUG_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_live_trading.log')

def log_debug(message):
    if DEBUG_MODE:
        with open(DEBUG_LOG_FILE, 'a') as f:
            f.write(f"{datetime.now()} - {message}\n") 