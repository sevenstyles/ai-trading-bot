@echo off
cd /d %~dp0
echo Starting live trading with 4h timeframe
python live_trading.py
pause