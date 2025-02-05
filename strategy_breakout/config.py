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
LEVERAGE = 10
ORDER_SIZE_PCT = 0.02

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

EMA_SHORT = 21
EMA_LONG = 50

TAKE_PROFIT_LEVELS = {
    0.5: 0.33,
    0.75: 0.25,
    1.0: 0.42
}

MIN_LIQUIDITY = 500000

# Backtester Settings
OHLCV_TIMEFRAME = "15m"
LONG_TAKE_PROFIT_MULTIPLIER = 1.02
SHORT_TAKE_PROFIT_MULTIPLIER = 0.98
LONG_STOP_LOSS_MULTIPLIER = 0.995
SHORT_STOP_LOSS_MULTIPLIER = 1.005

# Parameters for consolidation breakout strategy
CONSOLIDATION_THRESHOLD_PCT = 0.5      # Maximum consolidation range as a percent of price (e.g., 0.5 means 0.5%)
CONSOLIDATION_RISK_REWARD_MULTIPLIER = 3.0  # Take profit multiplier for consolidation breakouts

# Removed duplicate profit target multipliers. Previous definitions of 1.06 and 0.94 have been removed.

# New parameters to simulate live futures trading conditions:
FUTURES_MAKER_FEE = 0.0002  # Maker fee for entry (0.02%)
FUTURES_TAKER_FEE = 0.0005  # Taker fee for exit (0.05%)
SLIPPAGE_RATE = 0.0005      # Slippage rate (0.05%)
FUNDING_RATE = 0.0001       # Funding rate per funding period (approx 0.01%), assuming funding applies every 8 hours

# Trailing Stop Settings for Backtesting Strategy
TRAILING_STOP_PCT = 0.003
TRAILING_START_LONG = 1.002
TRAILING_START_SHORT = 0.998

# ATR-based Stop Loss Multipliers
LONG_STOP_LOSS_ATR_MULTIPLIER = 0.55
SHORT_STOP_LOSS_ATR_MULTIPLIER = 0.55

MIN_BARS_BEFORE_STOP = 5

# Backtesting and live trading candlestick interval
CANDLESTICK_INTERVAL = "4h" 