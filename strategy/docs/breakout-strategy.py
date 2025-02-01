def summarize_trades(trades):
    """Print a summary of the backtest statistics."""
    if not trades:
        print("No trades executed.")
        return
    df_trades = pd.DataFrame(trades)
    num_trades = len(df_trades)
    win_rate = len(df_trades[df_trades['pnl'] > 0]) / num_trades * 100 if 'pnl' in df_trades.columns else 0
    avg_pnl = df_trades['pnl'].mean() * 100 if 'pnl' in df_trades.columns else 0
    total_pnl = df_trades['pnl'].sum() * 100 if 'pnl' in df_trades.columns else 0
    print("Number of Trades:", num_trades)
    print("Win Rate: {:.2f}%".format(win_rate))
    print("Average Trade PnL: {:.2f}%".format(avg_pnl))
    print("Total PnL: {:.2f}%".format(total_pnl))
    return df_trades

def simulate_trade(trade, df):
    """
    Simulate a trade outcome using subsequent 4h candles.
    
    For LONG trades:
      - If any subsequent candle's low <= SL, assume trade stops out with pnl = -1.
      - If any subsequent candle's high >= TP, assume trade takes profit with pnl = +3.
    For SHORT trades:
      - If any subsequent candle's high >= SL, assume trade stops out with pnl = -1.
      - If any subsequent candle's low <= TP, assume trade takes profit with pnl = +3.
    If neither is reached, use the final candle's close for pnl calculation.
    """
    entry = trade['entry']
    sl = trade['SL']
    tp = trade['TP']
    direction = trade['direction']
    tstamp = trade['timestamp']
    
    # Get future candles after trade entry
    future = df.loc[df.index > tstamp]
    for _, row in future.iterrows():
        if direction == "LONG":
            if row['low'] <= sl:
                return -1.0
            if row['high'] >= tp:
                return 3.0
        elif direction == "SHORT":
            if row['high'] >= sl:
                return -1.0
            if row['low'] <= tp:
                return 3.0
    # If neither TP nor SL was hit, use the final candle's close
    if not future.empty:
        final_close = future['close'].iloc[-1]
    else:
        final_close = entry
    if direction == "LONG":
        return (final_close - entry) / (entry - sl) if (entry - sl) != 0 else 0
    else:
        return (entry - final_close) / (sl - entry) if (sl - entry) != 0 else 0

def backtest_breakout_strategy(df_4h, df_15m=None):
    trades = []
    df = df_4h.copy()
    df['range_pct'] = (df['high'] - df['low']) / df['low'] * 100
    df['consol_pct'] = df['range_pct'].rolling(3).max()
    df['vol_ma'] = df['volume'].rolling(5).mean()
    
    # Add debug logging
    debug_counter = 0
    for i in range(3, len(df)):
        row = df.iloc[i]
        window = df.iloc[i-3:i]
        consol_range_pct = window['consol_pct'].max()
        # Look only at periods within the allowed consolidation range
        if consol_range_pct <= MIN_CONSOL_RANGE_PCT:
            debug_counter += 1
            vol_ratio = row['volume'] / row['vol_ma']
            print(f"Debug {symbol} | Candle {i}: Consol% {consol_range_pct:.2f} | Vol Ratio {vol_ratio:.2f}")
            consol_high = window['high'].max()
            consol_low = window['low'].min()
            # Check for LONG breakout: close above consolidation high with volume spike
            if row['close'] > consol_high and row['volume'] > VOL_SPIKE_MULTIPLIER * row['vol_ma']:
                entry = row['close']
                sl = consol_low
                risk = entry - sl
                if risk > 0:
                    tp = entry + risk * 3
                    trade = {
                        'timestamp': row.name,
                        'direction': 'LONG',
                        'entry': entry,
                        'SL': sl,
                        'TP': tp,
                        'RR_Ratio': "1:{:.1f}".format(3)
                    }
                    trade['pnl'] = simulate_trade(trade, df)
                    trades.append(trade)
            # Check for SHORT breakout: close below consolidation low with volume spike
            elif row['close'] < consol_low and row['volume'] > VOL_SPIKE_MULTIPLIER * row['vol_ma']:
                entry = row['close']
                sl = consol_high
                risk = sl - entry
                if risk > 0:
                    tp = entry - risk * 3
                    trade = {
                        'timestamp': row.name,
                        'direction': 'SHORT',
                        'entry': entry,
                        'SL': sl,
                        'TP': tp,
                        'RR_Ratio': "1:{:.1f}".format(3)
                    }
                    trade['pnl'] = simulate_trade(trade, df)
                    trades.append(trade)
    print(f"Debug: {symbol} had {debug_counter} potential setup candles")
    return trades 

# Relax parameters further
MIN_CONSOL_RANGE_PCT = 3.0  # Increase from 2.0% to 3.0%
VOL_SPIKE_MULTIPLIER = 1.02  # Lower from 1.05x to 1.02x

def get_binance_server_time():
    """Get current timestamp in milliseconds from Binance server"""
    try:
        server_time = client.get_server_time()
        return server_time['serverTime']
    except Exception as e:
        print(f"Error fetching Binance server time: {e}, using local time")
        return int(datetime.now(timezone.utc).timestamp() * 1000)

def fetch_data(symbol, start_time=None, end_time=None):
    # After getting klines:
    df = pd.DataFrame(klines, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC')
    
    # Add date validation
    now = datetime.now(timezone.utc)
    if df.index.max() > now:
        print(f"Warning: {symbol} has data extending into future ({df.index.max()})")
        df = df[df.index <= now]  # Filter out future dates
    
    return df

def main():
    """
    Main backtest function:
    - Gets 20 random USDT pairs.
    - Fetches data for each and runs the breakout strategy on 4h data over the last 6 months.
    - Prints backtest trade details.
    """
    # Get current time from Binance server
    end_time_ms = get_binance_server_time()
    # Calculate startTime as six months (180 days) ago from Binance server time
    start_time_ms = end_time_ms - (180 * 24 * 60 * 60 * 1000)
    
    all_tickers = client.get_all_tickers()
    usdt_pairs = [t['symbol'] for t in all_tickers if t['symbol'].endswith('USDT')]
    
    # Summary of overall backtest results
    print("\nOVERALL BACKTEST SUMMARY")
    print("=" * 40)
    print(f"Total trades identified across all symbols: {len(overall_trades)}")
    
    if overall_trades:
        summarize_trades(overall_trades)
    else:
        print("No trades to summarize")
    
    STABLECOINS = ['USDCUSDT', 'DAIUSDT', 'BUSDUSDT', 'TUSDUSDT', 'USDPUSDT']

    # In main() loop:
    for symbol in SYMBOLS:
        if symbol in STABLECOINS:
            print(f"Skipping stablecoin {symbol}")
            continue
    
    for symbol in SYMBOLS:
        print(f"\nChecking {symbol} data validity:")
        try:
            data = fetch_data(symbol, start_time=start_time_ms, end_time=end_time_ms)
            if '4h' not in data:
                print(f"- No 4h data for {symbol}")
                continue
            
            df_4h = data['4h']
            print(f"- Data points: {len(df_4h)}")
            print(f"- Date range: {df_4h.index.min()} to {df_4h.index.max()}")
            print(f"- Price range: ${df_4h['low'].min():.8f} - ${df_4h['high'].max():.8f}")
            
        except Exception as e:
            print(f"- Data error: {str(e)}")

# Instead of random.sample(), use known volatile pairs
TEST_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 
                'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT']
SYMBOLS = TEST_SYMBOLS  # Test with top 10 volatile pairs

if __name__ == '__main__':
    main() 