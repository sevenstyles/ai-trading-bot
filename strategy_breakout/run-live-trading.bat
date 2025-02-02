@echo off
cd /d %~dp0
echo "Starting live trading with 5m timeframe..."
python live_trading.py
pause