@echo off
cd /d %~dp0
echo "Starting live trading"
python live_trading.py
pause