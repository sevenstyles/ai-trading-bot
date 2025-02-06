import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import logging
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
from indicators import calculate_market_structure, calculate_trend_strength, calculate_emas, calculate_macd, calculate_adx, calculate_rsi, generate_short_signal, calculate_atr

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
    adjusted_entry = entry_price * (1 + FUTURES_MAKER_FEE + SLIPPAGE_RATE)
    adjusted_exit = exit_price * (1 - FUTURES_TAKER_FEE - SLIPPAGE_RATE)
    return (adjusted_exit - adjusted_entry) / adjusted_entry

def calculate_exit_profit_short(entry_price, exit_price):
    # Adjust entry and exit prices for short trades
    adjusted_entry = entry_price * (1 - FUTURES_MAKER_FEE - SLIPPAGE_RATE)
    adjusted_exit = exit_price * (1 + FUTURES_TAKER_FEE + SLIPPAGE_RATE)
    # For shorts, profit is positive when exit_price is lower than entry_price
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
            for j in range(i+1, len(df)):
                candle = df.iloc[j]
                if signal["side"] == "long":
                    if candle["low"] <= stop_loss:
                        exit_price = stop_loss
                        outcome = "LOSS"
                        exit_time = df.index[j]
                        break
                    if candle["high"] >= take_profit:
                        exit_price = take_profit
                        outcome = "WIN"
                        exit_time = df.index[j]
                        break
                elif signal["side"] == "short":
                    if candle["high"] >= stop_loss:
                        exit_price = stop_loss
                        outcome = "LOSS"
                        exit_time = df.index[j]
                        break
                    if candle["low"] <= take_profit:
                        exit_price = take_profit
                        outcome = "WIN"
                        exit_time = df.index[j]
                        break
            if outcome is None:
                exit_price = df["close"].iloc[-1]
                exit_time = df.index[-1]
                if signal["side"] == "long":
                    outcome = "WIN" if exit_price > entry_price else "LOSS"
                else:
                    outcome = "WIN" if exit_price < entry_price else "LOSS"
            if signal["side"] == "long":
                profit = calculate_exit_profit_long(entry_price, exit_price)
            else:
                profit = calculate_exit_profit_short(entry_price, exit_price)
            trade = {
                "symbol": symbol,
                "side": signal["side"],
                "entry_time": entry_time,
                "entry": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "outcome": outcome,
                "profit": profit
            }
            logger.debug(f"Trade generated for {symbol}: {trade} | Signal debug info: {signal.get('debug_info', {})}")
            signals.append(trade)
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
    win_rate = len(df[df["outcome"]=="WIN"]) / total_trades * 100
    # Convert entry_time and exit_time from naive UTC to UTC+11
    from datetime import timezone, timedelta
    utc_plus_11 = timezone(timedelta(hours=11))
    # Assuming the timestamps are in UTC, we localize and then convert
    df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.tz_localize('UTC').dt.tz_convert(utc_plus_11)
    df["exit_time"] = pd.to_datetime(df["exit_time"]).dt.tz_localize('UTC').dt.tz_convert(utc_plus_11)
    
    print(f"--- {symbol} ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
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
        df['outcome'] = np.where(df['profit'] > 0, 'WIN', np.where(df['profit'] < 0, 'LOSS', 'BREAKEVEN'))
        df['risk_reward'] = df.apply(lambda row: (row['TP']-row['entry_price'])/(row['entry_price']-row['SL']) 
                                     if row['entry_price'] > row['SL'] else (row['entry_price']-row['TP'])/(row['SL']-row['entry_price']), axis=1)
        
        total_trades = len(df)
        win_rate = len(df[df['outcome'] == 'WIN']) / total_trades * 100
        long_trades = len(df[df['side'] == 'long'])
        short_trades = len(df[df['side'] == 'short'])
        
        final_capital = simulate_capital(all_signals, initial_capital=initial_capital)
        
        print("=== Aggregated Report ===")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Long Trades: {long_trades}, Short Trades: {short_trades}")
        print(f"Simulation: Starting Capital: ${initial_capital}, Final Capital: ${final_capital:.2f}")
        
        best_trade = df.loc[df['profit_pct'].idxmax()]
        worst_trade = df.loc[df['profit_pct'].idxmin()]
        print(f"Best Trade: {best_trade['symbol']} - {best_trade['side']} trade on {best_trade['entry_time']}, Profit: {best_trade['profit_pct']}% (Entry: {best_trade['entry_price']}, Exit: {best_trade['exit_price']})")
        print(f"Worst Trade: {worst_trade['symbol']} - {worst_trade['side']} trade on {worst_trade['entry_time']}, Profit: {worst_trade['profit_pct']}% (Entry: {worst_trade['entry_price']}, Exit: {worst_trade['exit_price']})")
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