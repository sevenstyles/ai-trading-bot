import os
from pathlib import Path

# Basic directories and files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_LOG_DIR = Path(BASE_DIR) / 'debug_logs'
DEBUG_LOG_DIR.mkdir(exist_ok=True)
STRATEGY_FILE = os.path.join(BASE_DIR, '..', 'strategy_3_prompt.txt')

# Binance API credentials (update with your keys)
BINANCE_API_KEY = 'YOUR_API_KEY'
BINANCE_API_SECRET = 'YOUR_API_SECRET'

# Trade and risk parameters
CONFIRMATION_WINDOW = 24  # 6 hours (24*15min)
MIN_SESSION_RANGE_PCT = 1.5
MIN_PRICE = 0.000001
POSITION_SIZE_CAP = 5000
MIN_QUOTE_VOLUME = 25000
MIN_TRADE_VALUE = 250

MIN_CONSOL_RANGE_PCT = 3.0
BUFFER_PCT = 0.0025
MIN_CONSOL_BARS = 8
VOL_SPIKE_MULTIPLIER = 1.0
MIN_CONSOL_DAYS = 1
VOL_MA_PERIOD = 20

MAX_HOLD_PERIOD = 72  # in hours
MAX_HOLD_BARS = 72

RISK_REWARD_RATIO = 3.0
ATR_PERIOD = 14
MIN_ATR_PCT = 0.5

CAPITAL = 100000
RISK_PER_TRADE = 0.10
LEVERAGE = 20

BLACKLIST = [
    'EURUSDT', 'GBPUSDT', 'JPYUSDT',
    'TUSDUSDT', 'USDPUSDT', 'BUSDUSDT',
    'AUDUSDT', 'CNYUSDT', 'RUBUSDT'
]

TREND_STRENGTH_THRESHOLD = 0.3
MIN_MOMENTUM_RATIO = 1.3
MIN_ADX = 25
MOMENTUM_WINDOW = 3
BASE_HOLD_BARS = 18

VOL_CONFIRMATION_BARS = 5

EMA_SHORT = 21
EMA_LONG = 50

TAKE_PROFIT_LEVELS = {
    0.5: 0.33,
    0.75: 0.25,
    1.0: 0.42
}

MIN_LIQUIDITY = 500000

# Profit target multipliers adjusted to target an average win of about 4%
LONG_TAKE_PROFIT_MULTIPLIER = 1.04   # 4% profit target for long trades (risk 0.5% yields an 8:1 ratio)
SHORT_TAKE_PROFIT_MULTIPLIER = 0.96    # 4% profit target for short trades (risk 0.5% yields an 8:1 ratio) 