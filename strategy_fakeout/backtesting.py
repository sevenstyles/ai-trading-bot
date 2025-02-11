import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import random
import json
import logging
import sys
sys.path.insert(0, os.getcwd())
logger = logging.getLogger("backtester")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Create a handler that outputs to the console
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)

from config import (
    MAX_HOLD_BARS, MIN_QUOTE_VOLUME, CAPITAL, RISK_PER_TRADE, LEVERAGE,
    SLIPPAGE_RATE, STOP_LOSS_SLIPPAGE, LONG_STOP_LOSS_MULTIPLIER,
    SHORT_STOP_LOSS_MULTIPLIER, TRAILING_STOP_PCT, TRAILING_START_LONG,
    TRAILING_START_SHORT, MIN_BARS_BEFORE_STOP, FUTURES_MAKER_FEE,
    FUTURES_TAKER_FEE, FUNDING_RATE, OHLCV_TIMEFRAME, POSITION_SIZE_CAP,
    MIN_TRADE_VALUE
)
from strategy import generate_signal

# Define UTC+11 timezone
UTC11 = timezone(timedelta(hours=11))

def get_funding_fee(client, symbol, entry_time, exit_time):
    """Calculate funding fees for the trade duration."""
    start_ms = int(entry_time.timestamp() * 1000)
    end_ms = int(exit_time.timestamp() * 1000)
    try:
        funding_data = client.futures_funding_rate(symbol=symbol, startTime=start_ms, endBin=end_ms, limit=1000)
        position_value = (CAPITAL * RISK_PER_TRADE) * LEVERAGE
        total_fee = sum(float(event['fundingRate']) * position_value for event in funding_data)
        return total_fee / CAPITAL  # Return as percentage of capital
    except Exception as e:
        logger.error(f"Error fetching funding rate for {symbol}: {e}")
        return 0.0

def calculate_exit_profit_long(entry_price, exit_price, include_funding=True, entry_time=None, exit_time=None, client=None, symbol=None):
    """Calculate profit for a long trade including all fees."""
    # If exit price equals entry price, consider it breakeven
    if abs(exit_price - entry_price) < 1e-8:
        return 0.0
        
    # Add extra slippage for stop loss hits
    actual_slippage = STOP_LOSS_SLIPPAGE if exit_price < entry_price else SLIPPAGE_RATE
    
    adjusted_entry = entry_price * (1 + FUTURES_MAKER_FEE + SLIPPAGE_RATE)
    adjusted_exit = exit_price * (1 - FUTURES_TAKER_FEE - actual_slippage)
    
    # Calculate profit as a percentage of the entry price
    price_profit = (adjusted_exit - adjusted_entry) / entry_price
    
    # Add funding fees if requested
    if include_funding and entry_time and exit_time and client and symbol:
        funding_fee = get_funding_fee(client, symbol, entry_time, exit_time)
        return price_profit - funding_fee
    
    return price_profit

def calculate_exit_profit_short(entry_price, exit_price, include_funding=True, entry_time=None, exit_time=None, client=None, symbol=None):
    """Calculate profit for a short trade including all fees."""
    # If exit price equals entry price, consider it breakeven
    if abs(exit_price - entry_price) < 1e-8:
        return 0.0
        
    # Add extra slippage for stop loss hits
    actual_slippage = STOP_LOSS_SLIPPAGE if exit_price > entry_price else SLIPPAGE_RATE
    
    adjusted_entry = entry_price * (1 - FUTURES_MAKER_FEE - SLIPPAGE_RATE)
    adjusted_exit = exit_price * (1 + FUTURES_TAKER_FEE + actual_slippage)
    
    # Calculate profit as a percentage of the entry price
    price_profit = (adjusted_entry - adjusted_exit) / entry_price
    
    # Add funding fees if requested
    if include_funding and entry_time and exit_time and client and symbol:
        funding_fee = get_funding_fee(client, symbol, entry_time, exit_time)
        return price_profit - funding_fee
    
    return price_profit

def calculate_position_size(entry_price, stop_loss, side="long"):
    """Calculate the position size based on capital, risk per trade, and leverage with limits."""
    risk_amount = CAPITAL * RISK_PER_TRADE  # How much money we're willing to risk
    stop_distance = abs(entry_price - stop_loss)
    
    # Calculate the position size that risks the desired amount
    position_value = (risk_amount / stop_distance) * entry_price if stop_distance != 0 else 0

    # Apply leverage
    leveraged_position = position_value * LEVERAGE
    
    # Apply position size cap
    if leveraged_position > POSITION_SIZE_CAP:
        leveraged_position = POSITION_SIZE_CAP
    
    # Check minimum trade value
    if leveraged_position < MIN_TRADE_VALUE:
        return 0, 0
    
    # Calculate the quantity of contracts/tokens
    quantity = leveraged_position / entry_price
    
    return quantity, leveraged_position

def backtest_strategy(symbol, timeframe=OHLCV_TIMEFRAME, days=3, client=None, use_random_date=False, swing_lookback=20, start_date_str=None, risk_per_trade=None):
    # Determine backtesting date range using the current date in UTC+11
    if start_date_str:
        end_date = datetime.strptime(start_date_str, "%Y-%m-%d") + timedelta(days=days)
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    else:
        end_date = datetime.now(UTC11)
        start_date = end_date - timedelta(days=days)
    
    # Fetch lower timeframe klines
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
    
    i = 0
    signals = []
    while i < len(df):
        # Ensure we have enough candles for the lookback window plus monitoring period
        required_candles = swing_lookback + 2  # lookback + breakout + confirmation
        if i < required_candles or i + 5 >= len(df):  # Need 5 more candles after current position
            i += 1
            continue
            
        # Build sliding window for signal generation
        window = df.iloc[i - required_candles : i + 2]  # Include breakout and confirmation candles
        
        signal = generate_signal(window, lookback=swing_lookback, risk_per_trade=risk_per_trade)
        
        if signal:
            # Calculate position size
            quantity, position_value = calculate_position_size(
                signal['entry'], 
                signal['stop_loss'], 
                signal['side']
            )
            
            # Skip trade if position size is too small
            if quantity == 0:
                i += 1
                signal['quantity'] = 0
                signal['position_value'] = 0
                signal['leverage_used'] = LEVERAGE
                signal['entry_time'] = df.index[i]  # Set entry time
                signal['symbol'] = symbol
                signal['entry'] = 0
                signal['exit_price'] = 0
                signal['profit'] = 0
                signal['outcome'] = 'LOSS'
                signals.append(signal)
                continue
            
            # Add trade details
            signal['quantity'] = quantity
            signal['position_value'] = position_value
            signal['leverage_used'] = LEVERAGE
            signal['entry_time'] = df.index[i]  # Set entry time
            signal['symbol'] = symbol
            
            entry_price = signal["entry"]
            entry_time = df.index[i]
            stop_loss = signal["stop_loss"]
            take_profit = signal["take_profit"]
            
            outcome = None
            max_profit_reached = 0  # Track maximum profit reached
            bars_held = 0  # Track number of bars in trade
            exit_price = None  # Initialize exit price
            exit_time = None  # Initialize exit time
            
            # Process subsequent candles for this trade
            for j in range(i+1, min(i + MAX_HOLD_BARS, len(df))):
                candle = df.iloc[j]
                bars_held += 1
                
                if signal['side'] == 'long':
                    # Calculate current profit in R multiples
                    current_profit = (candle['high'] - entry_price)
                    profit_r = current_profit / abs(entry_price - stop_loss)  # Use absolute distance for R calculation
                    max_profit_reached = max(max_profit_reached, profit_r)
                    
                    # Check stop loss
                    if candle['low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_time = df.index[j]
                        profit = calculate_exit_profit_long(
                            entry_price, exit_price, True,
                            entry_time, exit_time,
                            client, symbol
                        )
                        signal['profit'] = profit
                        outcome = 'LOSS'  # Stop loss hit is always a loss
                        break
                        
                    # Check take profit
                    if candle['high'] >= take_profit:
                        exit_price = take_profit
                        exit_time = df.index[j]
                        profit = calculate_exit_profit_long(
                            entry_price, exit_price, True,
                            entry_time, exit_time,
                            client, symbol
                        )
                        signal['profit'] = profit
                        outcome = 'WIN'  # Take profit hit is always a win
                        break
                        
                else:  # short trade
                    # Calculate current profit in R multiples
                    current_profit = (entry_price - candle['low'])
                    profit_r = current_profit / abs(stop_loss - entry_price) if abs(stop_loss - entry_price) != 0 else 0  # Use absolute distance for R calculation
                    max_profit_reached = max(max_profit_reached, profit_r)
                    
                    # Check stop loss for short trades
                    if candle['high'] >= stop_loss:
                        exit_price = stop_loss
                        exit_time = df.index[j]
                        profit = calculate_exit_profit_short(
                            entry_price, exit_price, True,
                            entry_time, exit_time,
                            client, symbol
                        )
                        signal['profit'] = profit
                        outcome = 'LOSS'  # Stop loss hit is always a loss
                        break
                        
                    # Check take profit
                    if candle['low'] <= take_profit:
                        exit_price = take_profit
                        exit_time = df.index[j]
                        profit = calculate_exit_profit_short(
                            entry_price, exit_price, True,
                            entry_time, exit_time,
                            client, symbol
                        )
                        signal['profit'] = profit
                        outcome = 'WIN'  # Take profit hit is always a win
                        break
            
            # If no exit condition met, close at last candle
            if not outcome:
                exit_price = df.iloc[j]['close']
                exit_time = df.index[j]
                if signal['side'] == 'long':
                    profit = calculate_exit_profit_long(
                        entry_price, exit_price, True,
                        entry_time, exit_time,
                        client, symbol
                    )
                    signal['profit'] = profit
                else:  # short trade
                    profit = calculate_exit_profit_short(
                        entry_price, exit_price, True,
                        entry_time, exit_time,
                        client, symbol
                    )
                    signal['profit'] = profit
            
            # Set outcome based on actual profit ONLY if no outcome was previously determined
            outcome = 'WIN' if signal['profit'] > 0 else 'LOSS'
            
            # Set final trade details
            signal['outcome'] = outcome
            signal['max_profit_reached'] = max_profit_reached
            signal['bars_held'] = bars_held
            signal['exit_price'] = exit_price
            signal['exit_time'] = exit_time
            signal['duration'] = (exit_time - entry_time).total_seconds() / 3600  # Duration in hours
            signal['risk_per_trade'] = risk_per_trade
            signal['entry_price'] = signal["entry"]
            signals.append(signal)
            
            # Skip the processed candles
            i = j + 1
            continue
            
        i += 1
        
    return signals, df

def simulate_capital(signals, initial_capital=1000, risk_per_trade=0.5):
    sorted_signals = sorted(signals, key=lambda t: t['entry_time'])
    capital = initial_capital
    for trade in sorted_signals:
        profit_pct = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) if trade['side'] == 'long' else ((trade['entry_price'] - trade['exit_price']) / trade['entry_price'])
        if 'profit' in trade:
            capital *= (1 + risk_per_trade * profit_pct.item())
            trade['capital_after'] = capital
    return capital

def analyze_results(signals, symbol):
    """Analyze and display trading results for a single symbol."""
    if not signals:
        print(f"No trades for {symbol}")
        return
        
    df = pd.DataFrame(signals)
    
    # Ensure all required columns exist
    required_cols = ['entry_time', 'exit_time', 'profit', 'outcome', 'side', 
                    'entry', 'stop_loss', 'take_profit', 'exit_price']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return
            
    # Calculate basic statistics
    total_trades = len(df)
    wins = len(df[df["outcome"]=="WIN"])
    losses = len(df[df["outcome"]=="LOSS"])
    win_rate = (wins/(wins+losses)*100) if (wins+losses) > 0 else 0
    break_even_trades = len(df[df["outcome"]=="BREAK EVEN"])
    long_trades = len(df[df["side"]=="long"])
    short_trades = len(df[df["side"]=="short"])
    
    # Calculate profit percentage using entry and exit prices
    df['profit_pct'] = ((df['exit_price'] - df['entry']) / df['entry'] * 100).round(2)
    
    # Correct the profit percentage for short positions
    df['profit_pct'] = df.apply(lambda row: -row['profit_pct'] if row['side'] == 'short' else row['profit_pct'], axis=1)
    
    # Calculate average trade duration
    if 'duration' in df.columns:
        avg_duration = df['duration'].mean()
    else:
        avg_duration = None
    
    # Calculate average R multiple
    df['R_multiple'] = df.apply(lambda row: 
        (row['exit_price'] - row['entry']) / (row['entry'] - row['stop_loss'])
        if row['side'] == 'long' else
        0 if row['stop_loss'] == row['entry'] else (row['entry'] - row['exit_price']) / (row['stop_loss'] - row['entry']),
        axis=1
    )
    avg_r_multiple = df['R_multiple'].mean()

    # Print results
    print("\n" + "="*50)
    print(f"=== {symbol} Trading Results ===")
    print("="*50)
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Wins: {wins}, Losses: {losses}, Break Even: {break_even_trades}")
    print(f"Long Trades: {long_trades}, Short Trades: {short_trades}")
    if avg_duration is not None:
        print(f"Average Trade Duration: {avg_duration:.1f} hours")
    print(f"Average R Multiple: {avg_r_multiple:.2f}")
    print("\nDetailed Trade List:")
    print("-"*100)
    
    # Format timestamps for display
    df['entry_time'] = pd.to_datetime(df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
    df['exit_time'] = pd.to_datetime(df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Display trade details
    display_cols = [
        'side', 'entry_time', 'entry', 'stop_loss', 'take_profit',
        'exit_price', 'profit_pct', 'outcome'
    ]
    
    print(df[display_cols].to_string(index=False))
    print("-"*100)
    
    # Print summary statistics
    total_profit_pct = df['profit_pct'].sum()
    avg_profit_pct = df['profit_pct'].mean()
    max_profit_pct = df['profit_pct'].max()
    min_profit_pct = df['profit_pct'].min()
    
    print("\nSummary Statistics:")
    print(f"Total Profit: {total_profit_pct:.2f}%")
    print(f"Average Profit per Trade: {avg_profit_pct:.2f}%")
    print(f"Best Trade: {max_profit_pct:.2f}%")
    print(f"Worst Trade: {min_profit_pct:.2f}%")
    print("="*50 + "\n")

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
        if "profit" not in df.columns:
            df["profit"] = 0.0
        df["profit_pct"] = (df["profit"] * 100).round(2)
        
        # Rename columns for consistency
        if 'stop_loss' in df.columns:
            df.rename(columns={'stop_loss': 'SL'}, inplace=True)
        if 'take_profit' in df.columns:
            df.rename(columns={'take_profit': 'TP'}, inplace=True)
        if 'entry' in df.columns:
            df.rename(columns={'entry': 'entry_price'}, inplace=True)
        
        # Calculate risk-reward ratio
        def calculate_risk_reward(row):
            try:
                entry_price = row['entry_price']
                sl = row['SL']
                tp = row['TP']

                if (entry_price - sl) == 0:
                    return float('inf')  # Avoid division by zero

                if entry_price > sl:  # Long trade
                    return (tp - entry_price) / (entry_price - sl)
                else:  # Short trade
                    if (sl - entry_price) == 0:
                        return float('inf')  # Avoid division by zero
                    return (entry_price - tp) / (sl - entry_price)
            except:
                return float('inf')

        df['risk_reward'] = df.apply(calculate_risk_reward, axis=1)
        
        total_trades = len(df)
        wins = len(df[df['outcome'] == 'WIN'])
        losses = len(df[df['outcome'] == 'LOSS'])
        win_rate = (wins/(wins+losses)*100) if (wins+losses) > 0 else 0
        long_trades = len(df[df['side'] == 'long'])
        short_trades = len(df[df['side'] == 'short'])
        break_even_trades = len(df[df['outcome'] == 'BREAK EVEN'])
        final_capital = simulate_capital(all_signals, initial_capital=initial_capital, risk_per_trade=RISK_PER_TRADE)
        profit = (final_capital - initial_capital) / initial_capital * 100
        
        print("=== Aggregated Report ===")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate (wins vs losses): {win_rate:.2f}%")
        print(f"Long Trades: {long_trades}, Short Trades: {short_trades}, Break Even Trades: {break_even_trades}")
        print(f"Simulation: Starting Capital: ${initial_capital}, Final Capital: ${final_capital:.2f}")
        print(f"Final Profit Percentage: {profit:.2f}%")
        # Remove the trade list from the aggregate section
        # Calculate profit percentage using entry and exit prices
        #df['profit_pct'] = ((df['exit_price'] - df['entry_price']) / df['entry_price'] * 100).round(2)
        # Include the 'side' column in the printed output
        #print(df[["side", "entry_time", "entry_price", "SL", "TP", "exit_time", "exit_price", "profit_pct", "outcome"]])
        print("=== END OF SUMMARY ===\n")
    except Exception as e:
        print(f"Aggregated analysis error: {str(e)}")
