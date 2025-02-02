import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from config import MAX_HOLD_BARS, MIN_QUOTE_VOLUME, CAPITAL, RISK_PER_TRADE, LEVERAGE

def simulate_trade(data, entry_time, direction, tp_multiplier, sl_multiplier, current_capital):
    entry_price = round(data.iloc[0]['close'], 8)
    if direction == 'long':
        tp_price = entry_price * (1 + tp_multiplier)
        sl_price = entry_price * (1 - sl_multiplier)
    else:
        tp_price = entry_price * (1 - tp_multiplier)
        sl_price = entry_price * (1 + sl_multiplier)
    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit == 0:
        print("Risk per unit is zero, cannot size position")
        return None
    risk_capital = current_capital * RISK_PER_TRADE
    position_size = (risk_capital * LEVERAGE) / risk_per_unit
    position_size = int(position_size)
    if position_size <= 0:
        print("Computed position size is zero, trade aborted")
        return None
    for idx, row in data.iterrows():
        if direction == 'long':
            if row['high'] >= tp_price:
                return {
                    'exit_time': idx,
                    'exit_price': tp_price,
                    'profit': tp_multiplier,
                    'position_size': position_size,
                    'duration': (idx - entry_time).total_seconds()/60
                }
            elif row['low'] <= sl_price:
                return {
                    'exit_time': idx,
                    'exit_price': sl_price,
                    'profit': -sl_multiplier,
                    'position_size': position_size,
                    'duration': (idx - entry_time).total_seconds()/60
                }
        else:
            if row['low'] <= tp_price:
                return {
                    'exit_time': idx,
                    'exit_price': tp_price,
                    'profit': tp_multiplier,
                    'position_size': position_size,
                    'duration': (idx - entry_time).total_seconds()/60
                }
            elif row['high'] >= sl_price:
                return {
                    'exit_time': idx,
                    'exit_price': sl_price,
                    'profit': -sl_multiplier,
                    'position_size': position_size,
                    'duration': (idx - entry_time).total_seconds()/60
                }
    print(f"No exit found within {len(data)} candles")
    return None

def backtest_asian_fakeout(df):
    trades = []
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['date'] = df.index.date
    total_sessions = 0
    sessions_with_range = 0
    sessions_with_volume = 0
    potential_setups = 0
    for day, day_df in df.groupby('date'):
        session_df = day_df.between_time("00:00", "08:00")
        if len(session_df) < 3:
            continue
        total_sessions += 1
        session_high = session_df['high'].max()
        session_low = session_df['low'].min()
        session_range = session_high - session_low
        session_range_pct = (session_range / session_df['close'].mean()) * 100
        print(f"\nAnalyzing session {day}: Range: {session_range_pct:.2f}%")
        daily_range = day_df['high'].max() - day_df['low'].min()
        avg_daily_range = df['high'].rolling(20).max() - df['low'].rolling(20).min()
        if daily_range < (0.3 * avg_daily_range.mean()):
            print("Failed daily range filter")
            continue
        sessions_with_range += 1
        for i in range(2, len(session_df) - 5):
            current = session_df.iloc[i]
            current_vol_ratio = current['volume'] / current['vol_ma20']
            if not (0 < current_vol_ratio < 10):
                continue
            potential_setups += 1
            confirmation_window = 5
            if current['high'] > session_high:
                print(f"\nPotential SHORT setup at {session_df.index[i]}")
                for j in range(1, min(confirmation_window, len(session_df) - i)):
                    future = session_df.iloc[i + j]
                    price_move = (current['high'] - future['close']) / current['high']
                    required_move = max(0.015, 0.0005)
                    if price_move >= required_move:
                        trade = simulate_trade(session_df.iloc[i+j:], session_df.index[i+j],
                                               'short', 0.06, 0.01, session_df['close'].iloc[-1])
                        if trade:
                            trade.update({
                                'entry_time': session_df.index[i+j],
                                'setup_time': session_df.index[i],
                                'direction': 'short',
                                'entry_price': future['close']
                            })
                            trades.append(trade)
                        break
            if current['low'] < session_low:
                print(f"\nPotential LONG setup at {session_df.index[i]}")
                for j in range(1, min(confirmation_window, len(session_df) - i)):
                    future = session_df.iloc[i + j]
                    price_move = (future['close'] - current['low']) / current['low']
                    required_move = max(0.015, 0.0005)
                    if price_move >= required_move:
                        trade = simulate_trade(session_df.iloc[i+j:], session_df.index[i+j],
                                               'long', 0.06, 0.01, session_df['close'].iloc[-1])
                        if trade:
                            trade.update({
                                'entry_time': session_df.index[i+j],
                                'setup_time': session_df.index[i],
                                'direction': 'long',
                                'entry_price': future['close']
                            })
                            trades.append(trade)
                        break
        quote_volume = session_df['volume'].sum() * session_df['close'].mean()
        if quote_volume < MIN_QUOTE_VOLUME:
            print(f"Insufficient liquidity (${quote_volume:.2f} < ${MIN_QUOTE_VOLUME})")
            continue
    print("\nBacktest Statistics:")
    print(f"Total sessions: {total_sessions}, Setup opportunities: {potential_setups}, Actual trades: {len(trades)}")
    return trades

def backtest_strategy(symbol, timeframe='4h', days=180, client=None):
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
    from indicators import calculate_market_structure, calculate_trend_strength, calculate_emas, calculate_macd, calculate_adx, calculate_rsi, generate_signal
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
                'take_profit': df['close'].iloc[i] * 1.02,
                'exit_price': 0,
                'exit_time': None,
                'profit': 0,
                'status': 'open',
                'drawdown': 0
            }
            signals.append(entry)
    trailing_stop_pct = 0.0025
    min_bars_before_stop = 3
    for trade in signals:
        entry_idx = df[df['timestamp'] == trade['entry_time']].index[0]
        max_hold = min(entry_idx + MAX_HOLD_BARS, len(df) - 1)
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
    return signals

def simulate_capital(signals, initial_capital=10000):
    sorted_signals = sorted(signals, key=lambda t: t['entry_time'])
    capital = initial_capital
    for trade in sorted_signals:
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
        trades_dir = os.path.join(os.getcwd(), 'trades')
        os.makedirs(trades_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_trades_{timestamp}.csv"
        filepath = os.path.join(trades_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\nAll trades exported to: {filepath}")
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