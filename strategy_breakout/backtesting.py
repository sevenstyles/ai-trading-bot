import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from config import MAX_HOLD_BARS, MIN_QUOTE_VOLUME, CAPITAL, RISK_PER_TRADE, LEVERAGE, LONG_TAKE_PROFIT_MULTIPLIER, SHORT_TAKE_PROFIT_MULTIPLIER

def backtest_strategy(symbol, timeframe='1h', days=180, client=None):
    from datetime import datetime, timedelta
    max_historical_days = 1000
    random_offset = random.randint(days, max_historical_days)
    start_date = datetime.now() - timedelta(days=random_offset)
    end_date = start_date + timedelta(days=days)
    klines = client.get_historical_klines(
        symbol, timeframe,
        start_date.strftime("%d %b %Y"),
        end_date.strftime("%d %b %Y")
    )
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=numeric_cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    from indicators import calculate_market_structure, calculate_trend_strength, calculate_emas, calculate_macd, calculate_adx, calculate_rsi, generate_signal, generate_short_signal
    df = calculate_market_structure(df)
    df = calculate_trend_strength(df)
    df = calculate_emas(df)
    df = calculate_macd(df)
    df = calculate_adx(df)
    df = calculate_rsi(df)
    signals = []
    for i in range(len(df)):
        if generate_signal(df, i):
            entry = {
                'entry_time': df['timestamp'].iloc[i],
                'entry_price': df['close'].iloc[i],
                'stop_loss': df['close'].iloc[i] * 0.995,
                'take_profit': df['close'].iloc[i] * LONG_TAKE_PROFIT_MULTIPLIER,
                'exit_price': 0,
                'exit_time': None,
                'profit': 0,
                'status': 'open',
                'drawdown': 0,
                'direction': 'long'
            }
            signals.append(entry)
        elif generate_short_signal(df, i):
            entry = {
                'entry_time': df['timestamp'].iloc[i],
                'entry_price': df['close'].iloc[i],
                'stop_loss': df['close'].iloc[i] * 1.005,
                'take_profit': df['close'].iloc[i] * SHORT_TAKE_PROFIT_MULTIPLIER,
                'exit_price': 0,
                'exit_time': None,
                'profit': 0,
                'status': 'open',
                'drawdown': 0,
                'direction': 'short'
            }
            signals.append(entry)
    trailing_stop_pct = 0.0025
    min_bars_before_stop = 3
    for trade in signals:
        entry_idx = df[df['timestamp'] == trade['entry_time']].index[0]
        max_hold = min(entry_idx + MAX_HOLD_BARS, len(df) - 1)
        if trade['direction'] == 'long':
            new_high = trade['entry_price']
            for j in range(entry_idx, max_hold + 1):
                if df['high'].iloc[j] > trade['entry_price'] * 1.03:
                    if df['high'].iloc[j] > new_high:
                        new_high = df['high'].iloc[j]
                        potential_stop = new_high * (1 - trailing_stop_pct)
                        if potential_stop > trade['stop_loss']:
                            trade['stop_loss'] = potential_stop
                if (j - entry_idx) >= min_bars_before_stop and df['low'].iloc[j] <= trade['stop_loss']:
                    trade.update({
                        'exit_price': trade['stop_loss'],
                        'exit_time': df['timestamp'].iloc[j],
                        'status': 'stopped',
                        'profit': (trade['stop_loss'] - trade['entry_price']) / trade['entry_price']
                    })
                    break
                if df['high'].iloc[j] >= trade['take_profit']:
                    trade.update({
                        'exit_price': trade['take_profit'],
                        'exit_time': df['timestamp'].iloc[j],
                        'status': 'target',
                        'profit': (trade['take_profit'] - trade['entry_price']) / trade['entry_price']
                    })
                    break
            if trade['status'] == 'open':
                trade.update({
                    'exit_price': df['close'].iloc[max_hold],
                    'exit_time': df['timestamp'].iloc[max_hold],
                    'status': 'expired',
                    'profit': (df['close'].iloc[max_hold] - trade['entry_price']) / trade['entry_price']
                })
            lowest_point = df['low'].iloc[entry_idx:j+1].min()
            trade['drawdown'] = (lowest_point - trade['entry_price']) / trade['entry_price']
        elif trade['direction'] == 'short':
            new_low = trade['entry_price']
            for j in range(entry_idx, max_hold + 1):
                if df['low'].iloc[j] < trade['entry_price'] * 0.97:
                    if df['low'].iloc[j] < new_low:
                        new_low = df['low'].iloc[j]
                        potential_stop = new_low * (1 + trailing_stop_pct)
                        if potential_stop < trade['stop_loss']:
                            trade['stop_loss'] = potential_stop
                if (j - entry_idx) >= min_bars_before_stop and df['low'].iloc[j] <= trade['take_profit']:
                    trade.update({
                        'exit_price': trade['take_profit'],
                        'exit_time': df['timestamp'].iloc[j],
                        'status': 'target',
                        'profit': (trade['entry_price'] - trade['take_profit']) / trade['entry_price']
                    })
                    break
                if df['high'].iloc[j] >= trade['stop_loss']:
                    trade.update({
                        'exit_price': trade['stop_loss'],
                        'exit_time': df['timestamp'].iloc[j],
                        'status': 'stopped',
                        'profit': (trade['entry_price'] - trade['stop_loss']) / trade['entry_price']
                    })
                    break
            if trade['status'] == 'open':
                trade.update({
                    'exit_price': df['close'].iloc[max_hold],
                    'exit_time': df['timestamp'].iloc[max_hold],
                    'status': 'expired',
                    'profit': (trade['entry_price'] - df['close'].iloc[max_hold]) / trade['entry_price']
                })
            highest_point = df['high'].iloc[entry_idx:j+1].max()
            trade['drawdown'] = (trade['entry_price'] - highest_point) / trade['entry_price']
    return signals

def simulate_capital(signals, initial_capital=10000):
    sorted_signals = sorted(signals, key=lambda t: t['entry_time'])
    capital = initial_capital
    for trade in sorted_signals:
        if 'profit' in trade:
            capital *= (1 + trade['profit'])
            trade['capital_after'] = capital
    return capital

def analyze_results(signals, symbol):
    if not signals:
        print("No trades to summarize")
        return
    try:
        df = pd.DataFrame(signals)
        df['profit_pct'] = df['profit'] * 100
        if 'stop_loss' in df.columns:
            df.rename(columns={'stop_loss': 'SL'}, inplace=True)
        if 'take_profit' in df.columns:
            df.rename(columns={'take_profit': 'TP'}, inplace=True)
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce', utc=True)
        df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce', utc=True)
        valid_trades = (df['entry_time'].notna() & 
                        df['exit_time'].notna() & 
                        (df['exit_time'] > df['entry_time']) &
                        df['entry_price'].notna() &
                        df['exit_price'].notna())
        df = df[valid_trades].copy()
        if df.empty:
            print("No valid trades to summarize")
            return
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['profit_pct'] = ((df['exit_price'] - df['entry_price']) / df['entry_price']) * 100
        df['profit_pct'] = df['profit_pct'].round(2)
        print("\n=== TRADE SUMMARY ===")
        print(f"Total Trades: {len(df)}")
        print(f"Time Period: {df['entry_time'].min()} to {df['entry_time'].max()}")
        df['outcome'] = np.where(df['profit'] > 0, 'WIN', np.where(df['profit'] < 0, 'LOSS', 'BREAKEVEN'))
        total_trades = len(df)
        win_rate = len(df[df['outcome'] == 'WIN']) / total_trades
        avg_win = df[df['outcome'] == 'WIN']['profit_pct'].mean() if not df[df['outcome'] == 'WIN'].empty else 0.0
        avg_loss = df[df['outcome'] == 'LOSS']['profit_pct'].mean() if not df[df['outcome'] == 'LOSS'].empty else 0.0
        risk_reward = abs(avg_win / avg_loss) if (avg_win > 0 and avg_loss < 0) else 0.0
        print(f"Total Return: {df['profit_pct'].sum():.2f}%")
        print(f"Average Win: {avg_win:.2f}%")
        print(f"Average Loss: {avg_loss:.2f}%")
        print(f"Risk/Reward Ratio: {risk_reward:.1f}:1")
    except Exception as e:
        print(f"Analysis error: {str(e)}")

def analyze_aggregated_results(all_signals, initial_capital=10000):
    if not all_signals:
        print("No aggregated trades generated")
        return
    try:
        df = pd.DataFrame(all_signals)
        df['profit_pct'] = df['profit'] * 100
        if 'stop_loss' in df.columns:
            df.rename(columns={'stop_loss': 'SL'}, inplace=True)
        if 'take_profit' in df.columns:
            df.rename(columns={'take_profit': 'TP'}, inplace=True)
        df['outcome'] = np.where(df['profit'] > 0, 'WIN', np.where(df['profit'] < 0, 'LOSS', 'BREAKEVEN'))
        df['risk_reward'] = df.apply(
            lambda row: (row['TP']-row['entry_price'])/(row['entry_price']-row['SL']) if row['entry_price'] > row['SL'] else (row['entry_price']-row['TP'])/(row['SL']-row['entry_price']),
            axis=1
        )
        total_trades = len(df)
        win_rate = len(df[df['outcome'] == 'WIN']) / total_trades if total_trades > 0 else 0
        avg_win = df[df['outcome'] == 'WIN']['profit_pct'].mean() if not df[df['outcome']=='WIN'].empty else 0.0
        avg_loss = df[df['outcome'] == 'LOSS']['profit_pct'].mean() if not df[df['outcome']=='LOSS'].empty else 0.0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        max_drawdown = df['drawdown'].min() if 'drawdown' in df.columns else 0
        final_capital = simulate_capital(all_signals, initial_capital=initial_capital)
        print("=== Aggregated Report Across All Symbols ===")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Avg Win: {avg_win:.2f}% | Avg Loss: {avg_loss:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
        print(f"\nStarting Capital: ${initial_capital}")
        print(f"Final Capital: ${final_capital:.2f}")
        print("\nSample Aggregated Trades:")
        print(df[['entry_time', 'entry_price', 'SL', 'TP', 'exit_price', 'profit_pct', 'outcome']].head(5))
    except Exception as e:
        print(f"Aggregated analysis error: {str(e)}")

def run_sequential_backtest(trades, initial_capital=10000):
    current_capital = initial_capital
    trades_sorted = sorted(trades, key=lambda t: t['entry_time'])
    for trade in trades_sorted:
        if 'profit' in trade:
            current_capital *= (1 + trade['profit'])
            trade['capital_after'] = current_capital
    return current_capital 