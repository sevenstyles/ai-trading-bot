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
from config import MAX_HOLD_BARS, MIN_QUOTE_VOLUME, CAPITAL, RISK_PER_TRADE, LEVERAGE, LONG_TAKE_PROFIT_MULTIPLIER, SHORT_TAKE_PROFIT_MULTIPLIER, SLIPPAGE_RATE, LONG_STOP_LOSS_MULTIPLIER, SHORT_STOP_LOSS_MULTIPLIER, TRAILING_STOP_PCT, TRAILING_START_LONG, TRAILING_START_SHORT, MIN_BARS_BEFORE_STOP, ATR_PERIOD, LONG_STOP_LOSS_ATR_MULTIPLIER, SHORT_STOP_LOSS_ATR_MULTIPLIER, FUTURES_MAKER_FEE, FUTURES_TAKER_FEE, FUNDING_RATE

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

def backtest_strategy(symbol, timeframe='1h', days=30, client=None, use_random_date=False):
    from datetime import datetime, timedelta
    if use_random_date:
        start_random = datetime(2021, 1, 1)
        end_random = datetime(2022, 1, 1)
        random_seconds = random.randint(0, int((end_random - start_random).total_seconds()))
        random_end = start_random + timedelta(seconds=random_seconds)
        end_date = random_end
        start_date = end_date - timedelta(days=days)
    else:
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
    klines = client.get_historical_klines(
        symbol, timeframe,
        start_date.strftime("%d %b %Y"),
        end_date.strftime("%d %b %Y")
    )
    logger.debug(f"Fetched {len(klines)} historical klines for symbol {symbol}")
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    logger.debug(f"DataFrame created with shape: {df.shape}")
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=numeric_cols)
    logger.debug(f"DataFrame after dropping NA: shape {df.shape}")
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    logger.debug(f"Timestamps converted. First timestamp: {df['timestamp'].iloc[0] if not df.empty else 'empty'}")
    from indicators import calculate_market_structure, calculate_trend_strength, calculate_emas, calculate_macd, calculate_adx, calculate_rsi, generate_signal, generate_short_signal, calculate_atr
    df = calculate_market_structure(df)
    logger.debug(f"After calculate_market_structure: shape {df.shape}")
    df = calculate_trend_strength(df)
    logger.debug(f"After calculate_trend_strength: shape {df.shape}")
    df = calculate_emas(df)
    logger.debug(f"After calculate_emas: shape {df.shape}")
    df = calculate_macd(df)
    logger.debug(f"After calculate_macd: shape {df.shape}")
    df = calculate_adx(df)
    logger.debug(f"After calculate_adx: shape {df.shape}")
    df = calculate_rsi(df)
    logger.debug(f"After calculate_rsi: shape {df.shape}")
    df['atr'] = calculate_atr(df, period=ATR_PERIOD)
    logger.debug("ATR calculated and added to DataFrame")
    signals = []
    for i in range(len(df)):
        if generate_signal(df, i):
            entry_price = df['close'].iloc[i]
            atr_value = df['atr'].iloc[i]
            if pd.isna(atr_value) or atr_value <= 0:
                atr_value = entry_price * 0.01  # Fallback to 1% of entry price
            stop_loss = entry_price - max(atr_value * LONG_STOP_LOSS_ATR_MULTIPLIER, entry_price * 0.01)
            if stop_loss >= entry_price:
                logger.warning(f"Invalid long SL calculated at index {i}: {stop_loss} >= {entry_price}. Resetting stop loss to fallback value.")
                stop_loss = entry_price * 0.995
            entry = {
                'entry_time': df['timestamp'].iloc[i],
                'entry_price': entry_price,
                'initial_stop_loss': stop_loss,
                'stop_loss': stop_loss,
                'take_profit': df['close'].iloc[i] * LONG_TAKE_PROFIT_MULTIPLIER,
                'exit_price': 0,
                'exit_time': None,
                'profit': 0,
                'status': 'open',
                'drawdown': 0,
                'direction': 'long',
                'symbol': symbol
            }
            signals.append(entry)
        elif generate_short_signal(df, i):
            entry_price = df['close'].iloc[i]
            entry_time = df['timestamp'].iloc[i]
            atr_value = df['atr'].iloc[i]
            if pd.isna(atr_value) or atr_value <= 0:
                atr_value = entry_price * 0.01
            stop_loss = entry_price * SHORT_STOP_LOSS_MULTIPLIER + (atr_value * SHORT_STOP_LOSS_ATR_MULTIPLIER)
            stop_loss = max(stop_loss, entry_price * 1.005)
            if stop_loss <= entry_price:
                logger.warning(f"Invalid short SL calculated at index {i}: {stop_loss} <= {entry_price}. Resetting stop loss to fallback value.")
                stop_loss = entry_price * 1.005
            trade = {
                'symbol': symbol,
                'direction': 'short',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': entry_price * SHORT_TAKE_PROFIT_MULTIPLIER,
                'entry_time': entry_time,
                'status': 'open'
            }
            signals.append(trade)
    trailing_stop_pct = TRAILING_STOP_PCT
    min_bars_before_stop = MIN_BARS_BEFORE_STOP
    for trade in signals:
        entry_idx = df[df['timestamp'] == trade['entry_time']].index[0]
        max_hold = min(entry_idx + MAX_HOLD_BARS, len(df) - 1)
        logger.debug(f"Processing {trade['direction']} trade for {symbol} with entry index {entry_idx} and entry price {trade['entry_price']}")
        if trade['direction'] == 'long':
            new_high = trade['entry_price']
            for j in range(entry_idx, max_hold + 1):
                if df['high'].iloc[j] > trade['entry_price'] * TRAILING_START_LONG:
                    if df['high'].iloc[j] > new_high:
                        old_high = new_high
                        new_high = df['high'].iloc[j]
                        potential_stop = new_high * (1 - TRAILING_STOP_PCT)
                        logger.debug(f"Long trade update at index {j}: new high updated from {old_high} to {new_high}, potential_stop {potential_stop}")
                        if potential_stop > trade['stop_loss'] and potential_stop < trade['entry_price']:
                            trade['stop_loss'] = potential_stop
                if (j - entry_idx) >= MIN_BARS_BEFORE_STOP and df['low'].iloc[j] <= trade['stop_loss']:
                    logger.debug(f"Long trade stop loss triggered at index {j}: low {df['low'].iloc[j]} <= stop_loss {trade['stop_loss']}")
                    exit_price = trade['stop_loss']
                    raw_profit = calculate_exit_profit_long(trade['entry_price'], exit_price)
                    funding_fee = get_funding_fee(client, symbol, trade['entry_time'], df['timestamp'].iloc[j])
                    logger.debug(f"Long trade exit calculated at index {j}: exit_price {exit_price}, raw_profit {raw_profit}, funding_fee {funding_fee}")
                    if trade['stop_loss'] >= trade['entry_price']:
                        logger.warning(f"Stop loss {trade['stop_loss']} is not below entry price {trade['entry_price']}. Resetting stop loss.")
                        trade['stop_loss'] = trade['entry_price'] * 0.995
                    trade.update({
                        'exit_price': exit_price,
                        'exit_time': df['timestamp'].iloc[j],
                        'status': 'stopped',
                        'profit': raw_profit - funding_fee,
                        'outcome': 'WIN' if (raw_profit - funding_fee) >= 0 else 'LOSS'
                    })
                    break
                if df['high'].iloc[j] >= trade['take_profit']:
                    logger.debug(f"Long trade take profit hit at index {j}: high {df['high'].iloc[j]} >= take_profit {trade['take_profit']}")
                    exit_price = trade['take_profit']
                    raw_profit = calculate_exit_profit_long(trade['entry_price'], exit_price)
                    funding_fee = get_funding_fee(client, symbol, trade['entry_time'], df['timestamp'].iloc[j])
                    logger.debug(f"Long trade take profit exit calculated at index {j}: exit_price {exit_price}, raw_profit {raw_profit}, funding_fee {funding_fee}")
                    trade.update({
                        'exit_price': exit_price,
                        'exit_time': df['timestamp'].iloc[j],
                        'status': 'target',
                        'profit': raw_profit - funding_fee,
                        'outcome': 'WIN' if (raw_profit - funding_fee) >= 0 else 'LOSS'
                    })
                    break
            if trade['status'] == 'open':
                final_idx = min(entry_idx + MAX_HOLD_BARS, len(df) - 1) if entry_idx < len(df) - 1 else entry_idx
                exit_price = df['close'].iloc[final_idx]
                raw_profit = calculate_exit_profit_long(trade['entry_price'], exit_price)
                funding_fee = get_funding_fee(client, symbol, trade['entry_time'], df['timestamp'].iloc[final_idx])
                logger.debug(f"Long trade expired at max_hold index {final_idx}: close price {exit_price}, profit {raw_profit - funding_fee}, funding_fee {funding_fee}")
                trade.update({
                    'exit_price': exit_price,
                    'exit_time': df['timestamp'].iloc[final_idx],
                    'status': 'expired',
                    'profit': raw_profit - funding_fee,
                    'outcome': 'WIN' if (raw_profit - funding_fee) >= 0 else 'LOSS'
                })
            lowest_point = df['low'].iloc[entry_idx:j+1].min()
            logger.debug(f"Long trade drawdown: lowest_point {lowest_point}, entry_price {trade['entry_price']}, drawdown {(lowest_point - trade['entry_price']) / trade['entry_price']}")
            trade['drawdown'] = (lowest_point - trade['entry_price']) / trade['entry_price']
        elif trade['direction'] == 'short':
            new_low = trade['entry_price']
            for j in range(entry_idx, max_hold + 1):
                if df['low'].iloc[j] < trade['entry_price'] * TRAILING_START_SHORT:
                    if df['low'].iloc[j] < new_low:
                        old_low = new_low
                        new_low = df['low'].iloc[j]
                        potential_stop = new_low * (1 + TRAILING_STOP_PCT)
                        potential_stop = min(potential_stop, trade['entry_price'] * 1.005)
                        logger.debug(f"Short trade update at index {j}: new low updated from {old_low} to {new_low}, potential_stop {potential_stop}")
                        if potential_stop < trade['stop_loss']:
                            trade['stop_loss'] = potential_stop
                if (j - entry_idx) >= MIN_BARS_BEFORE_STOP and df['low'].iloc[j] <= trade['take_profit']:
                    logger.debug(f"Short trade take profit triggered at index {j}: low {df['low'].iloc[j]} <= take_profit {trade['take_profit']}")
                    exit_price = trade['take_profit']
                    raw_profit = calculate_exit_profit_short(trade['entry_price'], exit_price)
                    funding_fee = get_funding_fee(client, symbol, trade['entry_time'], df['timestamp'].iloc[j])
                    logger.debug(f"Short trade take profit exit calculated at index {j}: exit_price {exit_price}, raw_profit {raw_profit}, funding_fee {funding_fee}")
                    if trade['stop_loss'] <= trade['entry_price']:
                        logger.warning(f"Short stop loss {trade['stop_loss']} is not above entry price {trade['entry_price']}. Resetting stop loss.")
                        trade['stop_loss'] = trade['entry_price'] * 1.005
                    trade.update({
                        'exit_price': exit_price,
                        'exit_time': df['timestamp'].iloc[j],
                        'status': 'target',
                        'profit': raw_profit - funding_fee,
                        'outcome': 'WIN' if (raw_profit - funding_fee) >= 0 else 'LOSS'
                    })
                    break
                if df['high'].iloc[j] >= trade['stop_loss']:
                    logger.debug(f"Short trade stop loss triggered at index {j}: high {df['high'].iloc[j]} >= stop_loss {trade['stop_loss']}")
                    exit_price = trade['stop_loss']
                    raw_profit = calculate_exit_profit_short(trade['entry_price'], exit_price)
                    funding_fee = get_funding_fee(client, symbol, trade['entry_time'], df['timestamp'].iloc[j])
                    logger.debug(f"Short trade stop loss exit calculated at index {j}: exit_price {exit_price}, raw_profit {raw_profit}, funding_fee {funding_fee}")
                    if trade['stop_loss'] <= trade['entry_price']:
                        logger.warning(f"Short stop loss {trade['stop_loss']} is not above entry price {trade['entry_price']}. Resetting stop loss.")
                        trade['stop_loss'] = trade['entry_price'] * 1.005
                    trade.update({
                        'exit_price': exit_price,
                        'exit_time': df['timestamp'].iloc[j],
                        'status': 'stopped',
                        'profit': raw_profit - funding_fee,
                        'outcome': 'WIN' if (raw_profit - funding_fee) >= 0 else 'LOSS'
                    })
                    break
            if trade['status'] == 'open':
                final_idx = min(entry_idx + MAX_HOLD_BARS, len(df) - 1) if entry_idx < len(df) - 1 else entry_idx
                exit_price = df['close'].iloc[final_idx]
                raw_profit = calculate_exit_profit_short(trade['entry_price'], exit_price)
                funding_fee = get_funding_fee(client, symbol, trade['entry_time'], df['timestamp'].iloc[final_idx])
                logger.debug(f"Short trade expired at max_hold index {final_idx}: close price {exit_price}, profit {raw_profit - funding_fee}, funding_fee {funding_fee}")
                trade.update({
                    'exit_price': exit_price,
                    'exit_time': df['timestamp'].iloc[final_idx],
                    'status': 'expired',
                    'profit': raw_profit - funding_fee,
                    'outcome': 'WIN' if (raw_profit - funding_fee) >= 0 else 'LOSS'
                })
            highest_point = df['high'].iloc[entry_idx:j+1].max()
            logger.debug(f"Short trade drawdown: highest_point {highest_point}, entry_price {trade['entry_price']}, drawdown {(trade['entry_price'] - highest_point) / trade['entry_price']}")
            trade['drawdown'] = (trade['entry_price'] - highest_point) / trade['entry_price']
    logger.debug(f"Completed processing trades for symbol {symbol}. Total signals: {len(signals)}")
    return signals

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
        print("No trades to summarize")
        return
    try:
        df = pd.DataFrame(signals)
        df['profit_pct'] = (df['profit'] * 100).round(2)
        if 'stop_loss' in df.columns:
            df.rename(columns={'stop_loss': 'SL'}, inplace=True)
        if 'take_profit' in df.columns:
            df.rename(columns={'take_profit': 'TP'}, inplace=True)
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce', utc=True)
        df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce', utc=True)
        valid_trades = (df['entry_time'].notna() & 
                        df['exit_time'].notna() & 
                        (df['exit_time'] >= df['entry_time']) &
                        df['entry_price'].notna() & 
                        df['exit_price'].notna())
        df = df[valid_trades].copy()
        if df.empty:
            print("No valid trades to summarize")
            return
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['profit_pct'] = (df['profit'] * 100).round(2)
        df['outcome'] = np.where(df['profit'] > 0, 'WIN', np.where(df['profit'] < 0, 'LOSS', 'BREAKEVEN'))
        total_trades = len(df)
        win_trades = df[df['outcome'] == 'WIN']
        loss_trades = df[df['outcome'] == 'LOSS']
        avg_win = win_trades['profit_pct'].mean() if not win_trades.empty else 0
        avg_loss = loss_trades['profit_pct'].mean() if not loss_trades.empty else 0
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        rr_display = f"{risk_reward:.1f}:1" if risk_reward != np.inf else "∞:1"
        print(f"Total Return: {df['profit_pct'].sum():.2f}%")
        print(f"Average Win: {avg_win:.2f}%")
        print(f"Average Loss: {avg_loss:.2f}%")
        print(f"Risk/Reward Ratio: {rr_display}")
    except Exception as e:
        print(f"Analysis error: {str(e)}")

def analyze_aggregated_results(all_signals, initial_capital=1000, days=30):
    if not all_signals:
        print("No aggregated trades generated")
        return
    try:
        import config
        print("=== BACKTESTER SETTINGS ===")
        print(f"Timeframe used             : {config.OHLCV_TIMEFRAME}")
        print(f"Long Take Profit Target    : {config.LONG_TAKE_PROFIT_MULTIPLIER}")
        print(f"Short Take Profit Target   : {config.SHORT_TAKE_PROFIT_MULTIPLIER}")
        print(f"Long Stop Loss Multiplier  : {config.LONG_STOP_LOSS_MULTIPLIER}")
        print(f"Short Stop Loss Multiplier : {config.SHORT_STOP_LOSS_MULTIPLIER}")
        print(f"Futures Maker Fee         : {config.FUTURES_MAKER_FEE}")
        print(f"Futures Taker Fee         : {config.FUTURES_TAKER_FEE}")
        print(f"Slippage Rate              : {config.SLIPPAGE_RATE}")
        print("="*30)
        print("=== TECHNICAL SETTINGS ===")
        print(f"EMA Short Period   : {config.EMA_SHORT}")
        print(f"EMA Long Period    : {config.EMA_LONG}")
        print(f"ATR Period         : {config.ATR_PERIOD}")
        print(f"Minimum ATR %      : {config.MIN_ATR_PCT}")
        print(f"Trend Strength Th. : {config.TREND_STRENGTH_THRESHOLD}")
        print(f"Backtesting period (days)  : {days}")
        print("="*30)
        df = pd.DataFrame(all_signals)
        df['profit_pct'] = (df['profit'] * 100).round(2)
        if 'initial_stop_loss' in df.columns:
            df.rename(columns={'initial_stop_loss': 'SL'}, inplace=True)
        elif 'stop_loss' in df.columns:
            df.rename(columns={'stop_loss': 'SL'}, inplace=True)
        if 'take_profit' in df.columns:
            df.rename(columns={'take_profit': 'TP'}, inplace=True)
        df['outcome'] = np.where(df['profit'] > 0, 'WIN', np.where(df['profit'] < 0, 'LOSS', 'BREAKEVEN'))
        df['risk_reward'] = df.apply(
            lambda row: (row['TP']-row['entry_price'])/(row['entry_price']-row['SL']) if row['entry_price'] > row['SL'] else (row['entry_price']-row['TP'])/(row['SL']-row['entry_price']),
            axis=1
        )
        total_trades = len(df)
        win_trades = df[df['outcome'] == 'WIN']
        loss_trades = df[df['outcome'] == 'LOSS']
        avg_win = win_trades['profit_pct'].mean() if not win_trades.empty else 0
        avg_loss = loss_trades['profit_pct'].mean() if not loss_trades.empty else 0
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        total_wins = df[df['outcome'] == 'WIN']['profit'].sum()
        total_losses = abs(df[df['outcome'] == 'LOSS']['profit'].sum())
        print(f"Total Wins Sum: {total_wins:.2f}%")  # Debug output
        print(f"Total Losses Sum: {total_losses:.2f}%")  # Debug output
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        max_drawdown = df['drawdown'].min() if 'drawdown' in df.columns else 0
        final_capital = simulate_capital(all_signals, initial_capital=initial_capital)
        print("=== Aggregated Report Across All Symbols ===")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {len(win_trades) / total_trades:.1%}")
        print(f"Avg Win: {avg_win:.2f}% | Avg Loss: {avg_loss:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
        print(f"\nStarting Capital: ${initial_capital}")
        print(f"Final Capital: ${final_capital:.2f}")
        wins_count = len(df[df['outcome'] == 'WIN'])
        losses_count = len(df[df['outcome'] == 'LOSS'])
        print(f"Total Wins: {wins_count}")
        print(f"Total Losses: {losses_count}")
        best_trade = df.loc[df['profit_pct'].idxmax()]
        print(f"Best Trade: {best_trade['symbol']} trade on {best_trade['entry_time']} with profit {best_trade['profit_pct']:.2f}% (entry: {best_trade['entry_price']}, exit: {best_trade['exit_price']})")
        pair_profit = df.groupby('symbol')['profit'].sum()
        best_pair = pair_profit.idxmax()
        best_pair_profit = pair_profit.max() * 100
        print(f"Best Performing Pair: {best_pair} with aggregated profit of {best_pair_profit:.2f}%")
        long_count = len(df[df['direction'] == 'long'])
        short_count = len(df[df['direction'] == 'short'])
        print(f"Total Long Trades: {long_count}")
        print(f"Total Short Trades: {short_count}")
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['trade_duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
        df['trade_duration'] = df['trade_duration'].apply(lambda x: max(x, 5))  # Minimum 1 bar duration
        avg_duration = df['trade_duration'].mean()
        max_duration = df['trade_duration'].max()
        min_duration = df['trade_duration'].min()
        duration_std = df['trade_duration'].std()
        avg_risk_reward = df['risk_reward'].mean()
        print(f"Average Trade Duration: {avg_duration:.2f} minutes")
        print(f"Max Trade Duration: {max_duration:.2f} minutes")
        print(f"Min Trade Duration: {min_duration:.2f} minutes")
        print(f"Trade Duration Std Dev: {duration_std:.2f} minutes")
        print(f"Average Risk/Reward Ratio: {avg_risk_reward:.2f}")
        print(f"Duration Percentiles (minutes):")
        print(f"25th: {df['trade_duration'].quantile(0.25):.1f} | 50th: {df['trade_duration'].quantile(0.5):.1f} | 75th: {df['trade_duration'].quantile(0.75):.1f}")
        print("\nSample Aggregated Trades:")
        print(df[['entry_time', 'entry_price', 'SL', 'TP', 'exit_price', 'profit_pct', 'outcome']].head(5))
        if 'drawdown' in df.columns:
            df['drawdown_pct'] = df['drawdown'] * 100
            print("Top 5 Drawdowns:")
            print(df.nsmallest(5, 'drawdown_pct')[['symbol', 'entry_time', 'drawdown_pct']])
        invalid_trades = df[(df['trade_duration'] == 5) | (df['trade_duration'] == 300)]
        print(f"Found {len(invalid_trades)} trades with suspicious durations (5 or 300 minutes)")
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