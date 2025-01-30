@echo off
cd /d %~dp0
pip install -r requirements.txt
python binance_screener.py
pause
