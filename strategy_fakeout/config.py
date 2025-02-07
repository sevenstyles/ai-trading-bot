import os
from pathlib import Path

# Basic directories and files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_FILE = os.path.join(BASE_DIR, '..', 'strategy_3_prompt.txt')

# Binance API credentials (update with your keys)
BINANCE_API_KEY = '65dbbd66d67732a4c601a0031eeb152b37ce9532de11edc857accf41e6087659'
BINANCE_API_SECRET = 'f851e4347572938d9f945c67c8904cece84175d6210d8400b04e3cd44d67e666'

# Trade and risk parameters
CONFIRMATION_WINDOW = 24  # 6 hours (24*15min)
MIN_SESSION_RANGE_PCT = 1.5
MIN_PRICE = 0.000001
MIN_QUOTE_VOLUME = 25000
MIN_TRADE_VALUE = 250
POSITION_SIZE_CAP = 5000  # Maximum position size in quote currency

MIN_CONSOL_RANGE_PCT = 3.0
BUFFER_PCT = 0.0025
MIN_CONSOL_BARS = 8
VOL_SPIKE_MULTIPLIER = 1.0
MIN_CONSOL_DAYS = 1
VOL_MA_PERIOD = 20

MAX_HOLD_PERIOD = 72  # in hours
MAX_HOLD_BARS = 72  # Maximum number of bars to hold a position

RISK_REWARD_RATIO = 3.0

CAPITAL = 100  # Base capital
RISK_PER_TRADE = 0.5  # 50% risk per trade
LEVERAGE = 10  # 10x leverage
ORDER_SIZE_PCT = 0.45  # Use 45% of capital per trade

BLACKLIST = [
    'EURUSDT', 'GBPUSDT', 'JPYUSDT',
    'TUSDUSDT', 'USDPUSDT', 'BUSDUSDT',
    'AUDUSDT', 'CNYUSDT', 'RUBUSDT',
    'USDCUSDT', 'DAIUSDT'
]

TREND_STRENGTH_THRESHOLD = 0.8
MIN_MOMENTUM_RATIO = 1.3
MIN_ADX = 40
MOMENTUM_WINDOW = 3
BASE_HOLD_BARS = 18

VOL_CONFIRMATION_BARS = 5

TAKE_PROFIT_LEVELS = {
    0.5: 0.33,
    0.75: 0.25,
    1.0: 0.42
}

MIN_LIQUIDITY = 500000

# Backtester Settings
OHLCV_TIMEFRAME = "1h"  # Main timeframe for analysis and trading

# Capital and Risk Management
MAX_STOP_DISTANCE = 0.02  # Maximum allowed stop loss distance (2%)
LONG_STOP_LOSS_MULTIPLIER = 0.995  # Default stop loss distance for longs
SHORT_STOP_LOSS_MULTIPLIER = 1.005  # Default stop loss distance for shorts
STOP_LOSS_SLIPPAGE = 0.001  # Additional slippage for stop losses (0.1%)

# Trading Fees and Slippage
FUTURES_MAKER_FEE = 0.0002  # Maker fee for entry (0.02%)
FUTURES_TAKER_FEE = 0.0005  # Taker fee for exit (0.05%)
SLIPPAGE_RATE = 0.0005  # Standard slippage rate (0.05%)
FUNDING_RATE = 0.0001  # Funding rate per funding period (0.01%)

# Trailing Stop Settings
TRAILING_STOP_PCT = 0.003  # Trailing stop distance (0.3%)
TRAILING_START_LONG = 1.002  # Start trailing at 0.2% profit for longs
TRAILING_START_SHORT = 0.998  # Start trailing at 0.2% profit for shorts
MIN_BARS_BEFORE_STOP = 5  # Minimum bars before trailing stop activates

# Removed duplicate profit target multipliers. Previous definitions of 1.06 and 0.94 have been removed.

# New parameters to simulate live futures trading conditions:

# Backtesting and live trading candlestick interval
CANDLESTICK_INTERVAL = "4h" 