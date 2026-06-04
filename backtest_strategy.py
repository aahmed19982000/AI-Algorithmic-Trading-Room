"""
Scalping Strategy Backtester
=============================
Downloads 1 year of M15 data for a symbol, runs a historical simulation of the
optimized scalping strategy (EMA 20/50 + RSI 14), and calculates key performance metrics.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from dotenv import load_dotenv

from mt5_connection import connect_mt5, disconnect_mt5
from mt5_data import get_timeframe

# Load environment
load_dotenv()


def fetch_backtest_data(symbol, timeframe_str, years=1):
    """Fetch 1 year of historical candles for backtesting."""
    tf_constant = get_timeframe(timeframe_str)
    if tf_constant is None:
        return None

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    print(f"[BACKTEST] Downloading data for {symbol} ({timeframe_str})...")
    rates = mt5.copy_rates_range(symbol, tf_constant, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        print("[ERROR] Failed to download backtest data.")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def run_backtest(df, symbol):
    """
    Simulates the optimized scalping strategy on the historical dataset.
    """
    # 1. Setup indicators
    # EMAs
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()  # Macro Trend Filter
    
    # RSI 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi14'] = 100 - (100 / (1 + rs))

    # Pip settings
    symbol_upper = symbol.upper()
    is_jpy = symbol_upper.endswith("JPY") or "JPY" in symbol_upper
    pip_multiplier = 100.0 if is_jpy else 10000.0
    pip_size = 0.01 if is_jpy else 0.0001
    
    # Calculate historical candle range and body size in pips
    df['range_pips'] = (df['high'] - df['low']) * pip_multiplier
    df['body_pips'] = (df['close'] - df['open']).abs() * pip_multiplier
    
    # Calculate ATR 14 in pips
    # Simple rolling mean of candle range
    df['atr14'] = df['range_pips'].rolling(window=14).mean()

    # Strategy SL / TP
    if "XAU" in symbol_upper or "GOLD" in symbol_upper:
        sl_pips = 300.0
        tp_pips = 200.0  # Ultra-high probability ratio
    else:
        sl_pips = 9.0   # 9 pips Stop Loss
        tp_pips = 6.0   # 6 pips Take Profit
        
    print(f"[BACKTEST] High Probability Strategy: SL: {sl_pips} pips, TP: {tp_pips} pips (R:R = 1.5:1)")

    # Simulation variables
    position = None # None, 'BUY', or 'SELL'
    entry_price = 0.0
    sl_level = 0.0
    tp_level = 0.0
    
    trades = []
    
    print(f"\n[BACKTEST] Starting simulation on {len(df)} candles...")
    
    # We start from index 200 to allow EMA 200 to stabilize
    for i in range(200, len(df)):
        current_time = df.loc[i, 'time']
        hour = current_time.hour
        
        # Check Session Filter (07:00 to 17:00 UTC - London/NY sessions)
        session_active = (7 <= hour <= 17)
        
        close_curr = df.loc[i, 'close']
        open_curr = df.loc[i, 'open']
        high_curr = df.loc[i, 'high']
        low_curr = df.loc[i, 'low']
        
        close_prev = df.loc[i-1, 'close']
        open_prev = df.loc[i-1, 'open']
        high_prev = df.loc[i-1, 'high']
        low_prev = df.loc[i-1, 'low']
        
        ema50_curr = df.loc[i, 'ema50']
        ema200_curr = df.loc[i, 'ema200']
        
        rsi_curr = df.loc[i, 'rsi14']
        rsi_prev = df.loc[i-1, 'rsi14']
        
        atr_curr = df.loc[i, 'atr14']

        # --- Case A: Manage Active Position ---
        if position is not None:
            # Check exit conditions (High/Low of current bar hits SL/TP)
            if position == 'BUY':
                if low_curr <= sl_level:
                    trades.append({
                        "type": "BUY",
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry": entry_price,
                        "exit": sl_level,
                        "pips": -sl_pips,
                        "result": "LOSS"
                    })
                    position = None
                    continue
                elif high_curr >= tp_level:
                    trades.append({
                        "type": "BUY",
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry": entry_price,
                        "exit": tp_level,
                        "pips": tp_pips,
                        "result": "WIN"
                    })
                    position = None
                    continue
                    
            elif position == 'SELL':
                if high_curr >= sl_level:
                    trades.append({
                        "type": "SELL",
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry": entry_price,
                        "exit": sl_level,
                        "pips": -sl_pips,
                        "result": "LOSS"
                    })
                    position = None
                    continue
                elif low_curr <= tp_level:
                    trades.append({
                        "type": "SELL",
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry": entry_price,
                        "exit": tp_level,
                        "pips": tp_pips,
                        "result": "WIN"
                    })
                    position = None
                    continue

        # --- Case B: Open New Position ---
        if position is None and session_active:
            # Volatility filter (ATR > 5.0 pips)
            if pd.isna(atr_curr) or atr_curr < 5.0:
                continue

            # Candlestick Reversal Checks
            body_size = abs(close_curr - open_curr) * pip_multiplier
            total_range = (high_curr - low_curr) * pip_multiplier
            
            # 1. Engulfing
            is_bullish_engulfing = (open_curr < close_prev) and (close_curr > open_prev) and (close_curr > open_curr)
            is_bearish_engulfing = (open_curr > close_prev) and (close_curr < open_prev) and (close_curr < open_curr)
            
            # 2. Pin bar
            is_bullish_pinbar = False
            is_bearish_pinbar = False
            if total_range > 0:
                # Bullish pin bar: long lower wick, body in upper 1/3
                lower_wick = (min(open_curr, close_curr) - low_curr) * pip_multiplier
                if lower_wick >= 2 * body_size and (max(open_curr, close_curr) >= high_curr - (total_range / 3.0)):
                    is_bullish_pinbar = True
                
                # Bearish pin bar: long upper wick, body in lower 1/3
                upper_wick = (high_curr - max(open_curr, close_curr)) * pip_multiplier
                if upper_wick >= 2 * body_size and (min(open_curr, close_curr) <= low_curr + (total_range / 3.0)):
                    is_bearish_pinbar = True
                    
            # 3. Pullback check: low or high crossed EMA 50 within last 3 bars
            pullback_buy = False
            pullback_sell = False
            for j in range(max(200, i-3), i):
                if df.loc[j, 'low'] <= df.loc[j, 'ema50'] <= df.loc[j, 'high']:
                    pullback_buy = True
                    pullback_sell = True
                    break

            # Check BUY Conditions
            # 1. Macro Trend: close > EMA 200
            # 2. Above intermediate trend: close > EMA 50
            # 3. Pullback touched EMA 50 in past 3 candles
            # 4. Candlestick trigger: Bullish Engulfing or Bullish Pin Bar
            # 5. RSI Momentum: RSI between 45 and 60 AND rising (RSI > RSI_prev)
            if (close_curr > ema200_curr) and (close_curr > ema50_curr) and pullback_buy and \
               (is_bullish_engulfing or is_bullish_pinbar) and \
               (45 <= rsi_curr <= 60) and (rsi_curr > rsi_prev):
                
                position = 'BUY'
                entry_price = close_curr
                entry_time = current_time
                sl_level = entry_price - (sl_pips * pip_size)
                tp_level = entry_price + (tp_pips * pip_size)
                
            # Check SELL Conditions
            # 1. Macro Trend: close < EMA 200
            # 2. Below intermediate trend: close < EMA 50
            # 3. Pullback touched EMA 50 in past 3 candles
            # 4. Candlestick trigger: Bearish Engulfing or Bearish Pin Bar
            # 5. RSI Momentum: RSI between 40 and 55 AND falling (RSI < RSI_prev)
            elif (close_curr < ema200_curr) and (close_curr < ema50_curr) and pullback_sell and \
                 (is_bearish_engulfing or is_bearish_pinbar) and \
                 (40 <= rsi_curr <= 55) and (rsi_curr < rsi_prev):
                
                position = 'SELL'
                entry_price = close_curr
                entry_time = current_time
                sl_level = entry_price + (sl_pips * pip_size)
                tp_level = entry_price - (tp_pips * pip_size)

    return trades, sl_pips, tp_pips


def print_backtest_report(trades, symbol, timeframe_str, sl_pips, tp_pips):
    """
    Renders the quantitative performance report.
    """
    print("\n" + "=" * 55)
    print(f"       BACKTEST REPORT: {symbol} ({timeframe_str})")
    print("=" * 55)
    
    total_trades = len(trades)
    if total_trades == 0:
        print("[WARNING] No trades were generated by the strategy.")
        return
        
    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = sum(1 for t in trades if t["result"] == "LOSS")
    win_rate = (wins / total_trades) * 100.0
    
    total_pips = sum(t["pips"] for t in trades)
    
    # Calculate profit factor
    gross_profit = sum(t["pips"] for t in trades if t["pips"] > 0)
    gross_loss = abs(sum(t["pips"] for t in trades if t["pips"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Simulation account growth (starting with $500, risking $5 per trade = 1%)
    # Loss = -$5. Win = +$5 * (TP / SL)
    balance = 500.0
    peak_balance = 500.0
    max_drawdown = 0.0
    
    loss_amount = 5.0
    win_amount = 5.0 * (tp_pips / sl_pips)
    
    for t in trades:
        if t["result"] == "WIN":
            balance += win_amount
        else:
            balance -= loss_amount
        
        if balance > peak_balance:
            peak_balance = balance
        
        dd = (peak_balance - balance) / peak_balance * 100.0
        if dd > max_drawdown:
            max_drawdown = dd

    print(f"  Total Simulated Trades: {total_trades}")
    print(f"  Wins:                  {wins} (Win Rate: {win_rate:.1f}%)")
    print(f"  Losses:                {losses}")
    print(f"  Gross Pips Gained:     {total_pips:.1f} pips")
    print(f"  Profit Factor:         {profit_factor:.2f}")
    print("-" * 55)
    print("  SIMULATED ACCOUNT METRICS (Starting $500, Risking 1%):")
    print(f"  Final Account Balance: ${balance:.2f} USD")
    print(f"  Net ROI:               {((balance - 500.0) / 500.0 * 100):+.1f}%")
    print(f"  Max Drawdown:          {max_drawdown:.1f}%")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Backtest a scalping strategy on historical MT5 data.")
    parser.add_argument("--symbol", type=str, default="EURUSDm", help="Trading symbol (e.g. EURUSDm)")
    parser.add_argument("--timeframe", type=str, default="M15", help="Timeframe (e.g. M15)")
    parser.add_argument("--years", type=int, default=1, help="Years of data to backtest")
    
    args = parser.parse_args()

    # 1. Connect to MT5
    if not connect_mt5():
        print("[FATAL] Could not connect to MT5.")
        sys.exit(1)

    try:
        # 2. Get data
        df = fetch_backtest_data(args.symbol, args.timeframe, args.years)
        if df is None:
            sys.exit(1)

        # 3. Simulate
        trades, sl_pips, tp_pips = run_backtest(df, args.symbol)

        # 4. Report
        print_backtest_report(trades, args.symbol, args.timeframe, sl_pips, tp_pips)

    finally:
        disconnect_mt5()


if __name__ == "__main__":
    main()
