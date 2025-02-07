# config.py

import os

# Binance API Keys
API_KEY = "65dbbd66d67732a4c601a0031eeb152b37ce9532de11edc857accf41e6087659"  # Directly paste your key here
API_SECRET = "f851e4347572938d9f945c67c8904cece84175d6210d8400b04e3cd44d67e666" # Directly paste your key here

# Trading Symbol (e.g., BTCUSDT, ETHUSDT)
SYMBOL = "BTCUSDT"

# --- Trading Parameters ---

# Volume Spike Threshold (as a multiple of the moving average)
VOLUME_SPIKE_MULTIPLIER = 2.0

# Order Flow Delta Z-Score Threshold
ORDER_FLOW_DELTA_Z_SCORE_THRESHOLD = 2.0  # Trigger when Z-score exceeds this value

# Number of data snapshots (ticks) to confirm a signal
CONFIRMATION_TICKS = 3

# Minimum liquidity threshold (in quote currency, e.g., USDT)
MINIMUM_LIQUIDITY_THRESHOLD = 1000  # Example: At least $1000 volume in the interval

# Moving Average Window (for volume and order flow)
MOVING_AVERAGE_WINDOW = 20  # Number of data points for the moving average

# Time interval for data snapshots (in seconds) - Adjust based on Binance API limits
TICK_INTERVAL = 1 # Check binance documentation, this might need to be higher.

# --- Order Execution Parameters ---
# Use 'MARKET' or 'LIMIT' orders
ORDER_TYPE = 'MARKET'

# If using LIMIT orders, specify the limit price offset (e.g., 0.01 for 1% away from best bid/ask)
LIMIT_PRICE_OFFSET = 0.01  # Only relevant if ORDER_TYPE is 'LIMIT'

# --- Risk Management (Simplified - Implement more robust risk management) ---
# Stop-loss percentage (e.g., 0.02 for 2%)
STOP_LOSS_PERCENTAGE = 0.02
# Take-profit percentage
TAKE_PROFIT_PERCENTAGE = 0.04
# Maximum position size as a percentage of total balance
MAX_POSITION_SIZE_PERCENTAGE = 0.1  # 10% of the balance 