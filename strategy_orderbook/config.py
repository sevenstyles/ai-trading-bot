# config.py

import os

# Binance API Keys
API_KEY = "65dbbd66d67732a4c601a0031eeb152b37ce9532de11edc857accf41e6087659"
API_SECRET = "f851e4347572938d9f945c67c8904cece84175d6210d8400b04e3cd44d67e666"

# Trading Symbol (e.g., BTCUSDT, ETHUSDT)
SYMBOL = "BTCUSDT"

# --- Trading Parameters ---

# Volume Spike Threshold (as a multiple of the moving average)
VOLUME_SPIKE_MULTIPLIER = 2.0

# Order Flow Delta Z-Score Threshold
ORDER_FLOW_DELTA_Z_SCORE_THRESHOLD = 1.5

# Number of data snapshots (ticks) to confirm a signal
CONFIRMATION_TICKS = 3

# Liquidity Parameters
MINIMUM_LIQUIDITY_THRESHOLD = 1000  # Minimum volume to consider for trading
TICK_INTERVAL = 1 #seconds

# Moving Average Window (for volume and order flow)
MOVING_AVERAGE_WINDOW = 5

# Time interval for data snapshots (in seconds) - Adjust based on Binance API limits
TICK_INTERVAL = 1 # Check binance documentation, this might need to be higher.

# --- Order Execution Parameters ---
# Use 'MARKET' or 'LIMIT' orders
ORDER_TYPE = "MARKET"

# If using LIMIT orders, specify the limit price offset (e.g., 0.01 for 1%)
LIMIT_PRICE_OFFSET = 0.01

# --- Risk Management (Simplified - Implement more robust risk management) ---
# Stop-loss percentage (e.g., 0.02 for 2%)
STOP_LOSS_PERCENTAGE = 0.02
# Take-profit percentage
TAKE_PROFIT_PERCENTAGE = 0.04
# Maximum position size as a percentage of total balance
MAX_POSITION_SIZE_PERCENTAGE = 0.01  # 1% of the balance

# --- Confirmation Parameters ---
CONFIRMATION_THRESHOLD = 0.5  # Minimum moving average value for confirmation

# --- ATR Parameters ---
ATR_PERIOD = 14  # Period for ATR calculation 