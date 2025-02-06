import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import logging
import sys
sys.path.insert(0, os.getcwd())
logger = logging.getLogger("backtester")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.FileHandler("backtester_detailed.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
logger.propagate = False
from config import MAX_HOLD_BARS, MIN_QUOTE_VOLUME, CAPITAL, RISK_PER_TRADE, LEVERAGE, LONG_TAKE_PROFIT_MULTIPLIER, SHORT_TAKE_PROFIT_MULTIPLIER, SLIPPAGE_RATE, LONG_STOP_LOSS_MULTIPLIER, SHORT_STOP_LOSS_MULTIPLIER, TRAILING_STOP_PCT, TRAILING_START_LONG, TRAILING_START_SHORT, MIN_BARS_BEFORE_STOP, ATR_PERIOD, LONG_STOP_LOSS_ATR_MULTIPLIER, SHORT_STOP_LOSS_ATR_MULTIPLIER, FUTURES_MAKER_FEE, FUTURES_TAKER_FEE, FUNDING_RATE, OHLCV_TIMEFRAME
from strategy import generate_signal

def get_funding_fee(client, symbol, entry_time, exit_time):
    # Convert entry and exit times (assumed to be datetime objects) to milliseconds
    start_ms = int(entry_time.timestamp() * 1000)
    end_ms = int(exit_time.timestamp() * 1000)
    try:
        funding_data = client.futures_funding_rate(symbol=symbol, startTime=start_ms, endTime=end_ms, limit=1000)
        position_value = (CAPITAL * RISK_PER_TRADE) * LEVERAGE
        total_fee = sum(float(event['fundingRate']) * position_value for event in funding_data)
        return total_fee / CAPITAL  # Return as percentage of capital
    except Exception as e:
        print(f"Error fetching funding rate for {symbol}: {e}")
        return 0.0

def calculate_exit_profit_long(entry_price, exit_price):
    # If exit price equals entry price, consider it breakeven
    if abs(exit_price - entry_price) < 1e-8:
        return 0.0
    adjusted_entry = entry_price * (1 + FUTURES_MAKER_FEE + SLIPPAGE_RATE)
    adjusted_exit = exit_price * (1 - FUTURES_TAKER_FEE - SLIPPAGE_RATE)
    return (adjusted_exit - adjusted_entry) / adjusted_entry

def calculate_exit_profit_short(entry_price, exit_price):
    # If exit price equals entry price, consider it breakeven
    if abs(exit_price - entry_price) < 1e-8:
        return 0.0
    adjusted_entry = entry_price * (1 - FUTURES_MAKER_FEE - SLIPPAGE_RATE)
    adjusted_exit = exit_price * (1 + FUTURES_TAKER_FEE + SLIPPAGE_RATE)
    return (adjusted_entry - adjusted_exit) / adjusted_entry

def backtest_strategy(symbol, timeframe=OHLCV_TIMEFRAME, days=7, client=None, use_random_date=False, swing_lookback=20):
    # Determine backtesting date range using the current date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Fetch lower timeframe klines.
    klines = client.get_historical_klines(
        symbol, timeframe,
        start_date.strftime("%d %b %Y"),
        end_date.strftime("%d %b %Y")
    )
    if not klines:
        logger.error(f"No klines fetched for {symbol} on timeframe {timeframe}")
        return [], None
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=numeric_cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Fetch higher timeframe (4h) data.
    klines_4h = client.get_historical_klines(
        symbol, "4h",
        start_date.strftime("%d %b %Y"),
        end_date.strftime("%d %b %Y")
    )
    if not klines_4h:
        logger.error(f"No 4h klines fetched for {symbol}")
        return [], df
    df_htf = pd.DataFrame(klines_4h, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df_htf[numeric_cols] = df_htf[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df_htf = df_htf.dropna(subset=numeric_cols)
    df_htf['timestamp'] = pd.to_datetime(df_htf['timestamp'], unit='ms')
    df_htf.set_index('timestamp', inplace=True)
    df_htf.sort_index(inplace=True)
    
    signals = []
    total_needed = swing_lookback + 2  # lookback candles + breakout + confirmation
    # Loop over the candles starting from the first index where we have total_needed candles.
    for i in range(total_needed - 1, len(df)):
        window = df.iloc[i - (total_needed - 1): i+1]
        current_time = window.index[-1]
        htf_window = df_htf[df_htf.index <= current_time]
        if htf_window.empty:
            continue
        signal = generate_signal(window, lookback=swing_lookback)
        if signal:
            entry_time = current_time
            entry_price = signal["entry"]
            stop_loss = signal["stop_loss"]
            take_profit = signal["take_profit"]
            outcome = None
            exit_price = None
            # Initialize partial booking variables for both long and short
            partial_taken = False
            new_entry = None
            new_take_profit = None
            for j in range(i+1, len(df)):
                candle = df.iloc[j]
                if signal["side"] == "long":
                    if not partial_taken:
                        # Check for partial booking if price moves 2% in favor
                        if candle["high"] >= entry_price * 1.02:
                            partial_taken = True
                            partial_profit_fixed = 0.02
                            new_entry = candle["close"]
                            new_take_profit = new_entry + (take_profit - entry_price)
                    if partial_taken:
                        if candle["low"] <= stop_loss:
                            exit_price = stop_loss
                            profit_remaining = (exit_price - new_entry) / new_entry
                            total_profit = 0.5 * partial_profit_fixed + 0.5 * profit_remaining
                            outcome = "WIN" if total_profit > 0 else ("BREAK EVEN" if abs(total_profit) < 1e-8 else "LOSS")
                            exit_time = df.index[j]
                            signal["profit"] = total_profit
                            break
                        if candle["high"] >= new_take_profit:
                            exit_price = new_take_profit
                            profit_remaining = (exit_price - new_entry) / new_entry
                            total_profit = 0.5 * partial_profit_fixed + 0.5 * profit_remaining
                            outcome = "WIN"
                            exit_time = df.index[j]
                            signal["profit"] = total_profit
                            break
                    else:
                        if candle["close"] >= entry_price * 1.015 and stop_loss < entry_price:
                            stop_loss = entry_price
                        if candle["low"] <= stop_loss:
                            exit_price = stop_loss
                            outcome = "BREAK EVEN" if abs(stop_loss - entry_price) < 1e-8 else "LOSS"
                            exit_time = df.index[j]
                            signal["profit"] = (exit_price - entry_price) / entry_price
                            break
                        if candle["high"] >= take_profit:
                            exit_price = take_profit
                            outcome = "WIN"
                            exit_time = df.index[j]
                            signal["profit"] = (exit_price - entry_price) / entry_price
                            break
                elif signal["side"] == "short":
                    if not partial_taken:
                        if candle["low"] <= entry_price * 0.98:
                            partial_taken = True
                            partial_profit_fixed = 0.02
                            new_entry = candle["close"]
                            new_take_profit = new_entry - (entry_price - take_profit)
                    if partial_taken:
                        if candle["high"] >= stop_loss:
                            exit_price = stop_loss
                            profit_remaining = (new_entry - exit_price) / new_entry
                            total_profit = 0.5 * partial_profit_fixed + 0.5 * profit_remaining
                            outcome = "WIN" if total_profit > 0 else ("BREAK EVEN" if abs(total_profit) < 1e-8 else "LOSS")
                            exit_time = df.index[j]
                            signal["profit"] = total_profit
                            break
                        if candle["low"] <= new_take_profit:
                            exit_price = new_take_profit
                            profit_remaining = (new_entry - exit_price) / new_entry
                            total_profit = 0.5 * partial_profit_fixed + 0.5 * profit_remaining
                            outcome = "WIN"
                            exit_time = df.index[j]
                            signal["profit"] = total_profit
                            break
                    else:
                        if candle["close"] <= entry_price * 0.985 and stop_loss > entry_price:
                            stop_loss = entry_price
                        if candle["high"] >= stop_loss:
                            exit_price = stop_loss
                            outcome = "BREAK EVEN" if abs(stop_loss - entry_price) < 1e-8 else "LOSS"
                            exit_time = df.index[j]
                            signal["profit"] = (entry_price - exit_price) / entry_price
                            break
                        if candle["low"] <= take_profit:
                            exit_price = take_profit
                            outcome = "WIN"
                            exit_time = df.index[j]
                            signal["profit"] = (entry_price - exit_price) / entry_price
                            break
            if outcome is None:
                exit_price = df["close"].iloc[-1]
                exit_time = df.index[-1]
                if signal["side"] == "long":
                    if abs(exit_price - entry_price) < 1e-8:
                        outcome = "BREAK EVEN"
                    else:
                        outcome = "WIN" if exit_price > entry_price else "LOSS"
                else:
                    if abs(exit_price - entry_price) < 1e-8:
                        outcome = "BREAK EVEN"
                    else:
                        outcome = "WIN" if exit_price < entry_price else "LOSS"
                if signal["side"] == "long":
                    signal["profit"] = (exit_price - entry_price) / entry_price
                else:
                    signal["profit"] = (entry_price - exit_price) / entry_price
            if signal["side"] == "long" and outcome != "BREAK EVEN":
                signal["take_profit"] = take_profit  # original take profit remains for reporting
            if signal["side"] == "short" and outcome != "BREAK EVEN":
                signal["take_profit"] = take_profit
            signal["outcome"] = outcome
            signal["entry_time"] = entry_time
            signal["exit_time"] = exit_time
            signal["exit_price"] = exit_price
            # Log the trade (existing logging code remains)
            signals.append(signal)
    logger.debug(f"Completed backtesting for {symbol}. Total trades: {len(signals)}")
    return signals, df

def simulate_capital(signals, initial_capital=1000):
    sorted_signals = sorted(signals, key=lambda t: t['entry_time'])
    capital = initial_capital
    for trade in sorted_signals:
        if 'profit' in trade:
            capital *= (1 + RISK_PER_TRADE * trade['profit'])
            trade['capital_after'] = capital
    return capital

def analyze_results(signals, symbol):
    if not signals:
        print(f"No trades for {symbol}")
        return
    df = pd.DataFrame(signals)
    total_trades = len(df)
    wins = len(df[df["outcome"]=="WIN"])
    losses = len(df[df["outcome"]=="LOSS"])
    win_rate = (wins/(wins+losses)*100) if (wins+losses) > 0 else 0
    # Convert entry_time and exit_time from naive UTC to UTC+11
    from datetime import timezone, timedelta
    utc_plus_11 = timezone(timedelta(hours=11))
    # Assuming the timestamps are in UTC, we localize and then convert
    df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.tz_localize('UTC').dt.tz_convert(utc_plus_11)
    df["exit_time"] = pd.to_datetime(df["exit_time"]).dt.tz_localize('UTC').dt.tz_convert(utc_plus_11)
    
    print(f"--- {symbol} ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate (wins vs losses): {win_rate:.2f}%")
    break_even_trades = len(df[df["outcome"]=="BREAK EVEN"])
    print(f"Break Even Trades: {break_even_trades}")
    df["profit_pct"] = (df["profit"] * 100).round(2)
    # Include the 'side' column in the printed output
    print(df[["side", "entry_time", "entry", "stop_loss", "take_profit", "exit_time", "exit_price", "profit_pct", "outcome"]])

def analyze_aggregated_results(all_signals, initial_capital=1000, days=30):
    if not all_signals:
        print("No aggregated trades generated")
        return
    try:
        import config
        print("\n")  # Added blank line for separation
        print("=== BACKTESTER SETTINGS ===")
        print(f"Timeframe used             : {config.OHLCV_TIMEFRAME}")
        print(f"Backtesting period (days)  : {days}")
        print("="*30)
        
        df = pd.DataFrame(all_signals)
        df["profit_pct"] = (df["profit"] * 100).round(2)
        
        # Rename columns for consistency
        if 'stop_loss' in df.columns:
            df.rename(columns={'stop_loss': 'SL'}, inplace=True)
        if 'take_profit' in df.columns:
            df.rename(columns={'take_profit': 'TP'}, inplace=True)
        if 'entry' in df.columns:
            df.rename(columns={'entry': 'entry_price'}, inplace=True)
        
        # Determine outcome and risk-reward ratio
        df['outcome'] = np.where(abs(df['profit']) < 1e-8, 'BREAK EVEN', np.where(df['profit'] > 0, 'WIN', 'LOSS'))
        df['risk_reward'] = df.apply(lambda row: ((row['TP']-row['entry_price'])/(row['entry_price']-row['SL']) if (row['entry_price']-row['SL']) != 0 else float('inf')) if row['entry_price'] > row['SL'] else ((row['entry_price']-row['TP'])/(row['SL']-row['entry_price']) if (row['SL']-row['entry_price']) != 0 else float('inf')), axis=1)
        
        total_trades = len(df)
        wins = len(df[df['outcome'] == 'WIN'])
        losses = len(df[df['outcome'] == 'LOSS'])
        win_rate = (wins/(wins+losses)*100) if (wins+losses) > 0 else 0
        long_trades = len(df[df['side'] == 'long'])
        short_trades = len(df[df['side'] == 'short'])
        break_even_trades = len(df[df['outcome'] == 'BREAK EVEN'])
        final_capital = simulate_capital(all_signals, initial_capital=initial_capital)
        
        print("=== Aggregated Report ===")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate (wins vs losses): {win_rate:.2f}%")
        print(f"Long Trades: {long_trades}, Short Trades: {short_trades}, Break Even Trades: {break_even_trades}")
        print(f"Simulation: Starting Capital: ${initial_capital}, Final Capital: ${final_capital:.2f}")
        df["profit_pct"] = (df["profit"] * 100).round(2)
        # Include the 'side' column in the printed output
        print(df[["side", "entry_time", "entry_price", "SL", "TP", "exit_time", "exit_price", "profit_pct", "outcome"]])
        print("=== END OF SUMMARY ===\n")
    except Exception as e:
        print(f"Aggregated analysis error: {str(e)}")

def run_sequential_backtest(trades, initial_capital=10000):
    current_capital = initial_capital
    trades_sorted = sorted(trades, key=lambda t: t['entry_time'])
    for i, trade in enumerate(trades_sorted):
        if 'profit' in trade:
            current_capital *= (1 + RISK_PER_TRADE * trade['profit'])
            trade['capital_after'] = current_capital
            print(f"Trade {i+1}: Capital before: {initial_capital:.2f} | Profit: {trade['profit']*100:.2f}% | Capital after: {current_capital:.2f}")
    return current_capital 