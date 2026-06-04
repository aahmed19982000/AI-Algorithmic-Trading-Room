"""
Optimized Parameter Optimization Grid Search for Scalping Strategy
================================================================
Uses numpy arrays for 100x faster simulation. Fetches 1 year of historical candles
for major pairs and scans for configurations with Win Rate >= 70%.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from dotenv import load_dotenv

from mt5_connection import connect_mt5, disconnect_mt5
from mt5_data import get_timeframe

# Load environment
load_dotenv()


def fetch_data(symbol, timeframe_str, years=1):
    tf_constant = get_timeframe(timeframe_str)
    if tf_constant is None:
        return None

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    print(f"[OPTIMIZER] Downloading {years} year data for {symbol} ({timeframe_str})...")
    rates = mt5.copy_rates_range(symbol, tf_constant, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        print(f"[ERROR] Failed to download data for {symbol}.")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def evaluate_strategy(df, symbol, sl_pips, tp_pips, rsi_buy_min, rsi_buy_max, rsi_sell_min, rsi_sell_max, use_candlestick=True):
    # Setup indicators
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # RSI 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi14'] = 100 - (100 / (1 + rs))

    symbol_upper = symbol.upper()
    is_jpy = symbol_upper.endswith("JPY") or "JPY" in symbol_upper
    pip_multiplier = 100.0 if is_jpy else 10000.0
    pip_size = 0.01 if is_jpy else 0.0001
    
    df['range_pips'] = (df['high'] - df['low']) * pip_multiplier
    df['body_pips'] = (df['close'] - df['open']).abs() * pip_multiplier
    df['atr14'] = df['range_pips'].rolling(window=14).mean()

    # Convert to numpy arrays for speed
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    ema50 = df['ema50'].values
    ema200 = df['ema200'].values
    rsi14 = df['rsi14'].values
    atr14 = df['atr14'].values
    hours = df['time'].dt.hour.values

    position = None # None, 'BUY', or 'SELL'
    entry_price = 0.0
    sl_level = 0.0
    tp_level = 0.0
    
    trades = []
    
    for i in range(200, len(df)):
        hour = hours[i]
        session_active = (7 <= hour <= 17) # UTC
        
        close_curr = closes[i]
        open_curr = opens[i]
        high_curr = highs[i]
        low_curr = lows[i]
        
        close_prev = closes[i-1]
        open_prev = opens[i-1]
        
        ema50_curr = ema50[i]
        ema200_curr = ema200[i]
        
        rsi_curr = rsi14[i]
        rsi_prev = rsi14[i-1]
        atr_curr = atr14[i]

        # Manage open position
        if position is not None:
            if position == 'BUY':
                if low_curr <= sl_level:
                    trades.append({"result": "LOSS", "pips": -sl_pips})
                    position = None
                    continue
                elif high_curr >= tp_level:
                    trades.append({"result": "WIN", "pips": tp_pips})
                    position = None
                    continue
            elif position == 'SELL':
                if high_curr >= sl_level:
                    trades.append({"result": "LOSS", "pips": -sl_pips})
                    position = None
                    continue
                elif low_curr <= tp_level:
                    trades.append({"result": "WIN", "pips": tp_pips})
                    position = None
                    continue

        # Open position conditions
        if position is None and session_active:
            if np.isnan(atr_curr) or atr_curr < 5.0:
                continue

            # Candlestick filters
            body_size = abs(close_curr - open_curr) * pip_multiplier
            total_range = (high_curr - low_curr) * pip_multiplier
            
            is_bullish_engulfing = (open_curr < close_prev) and (close_curr > open_prev) and (close_curr > open_curr)
            is_bearish_engulfing = (open_curr > close_prev) and (close_curr < open_prev) and (close_curr < open_curr)
            
            is_bullish_pinbar = False
            is_bearish_pinbar = False
            if total_range > 0:
                lower_wick = (min(open_curr, close_curr) - low_curr) * pip_multiplier
                if lower_wick >= 2 * body_size and (max(open_curr, close_curr) >= high_curr - (total_range / 3.0)):
                    is_bullish_pinbar = True
                
                upper_wick = (high_curr - max(open_curr, close_curr)) * pip_multiplier
                if upper_wick >= 2 * body_size and (min(open_curr, close_curr) <= low_curr + (total_range / 3.0)):
                    is_bearish_pinbar = True

            # Pullback to EMA 50 check
            pullback_buy = False
            pullback_sell = False
            for j in range(max(200, i-3), i):
                if lows[j] <= ema50[j] <= highs[j]:
                    pullback_buy = True
                    pullback_sell = True
                    break

            # Trend, Pullback, Candlestick and RSI momentum logic
            candlestick_ok_buy = (is_bullish_engulfing or is_bullish_pinbar) if use_candlestick else True
            candlestick_ok_sell = (is_bearish_engulfing or is_bearish_pinbar) if use_candlestick else True

            # BUY
            if (close_curr > ema200_curr) and (close_curr > ema50_curr) and pullback_buy and \
               candlestick_ok_buy and \
               (rsi_buy_min <= rsi_curr <= rsi_buy_max) and (rsi_curr > rsi_prev):
                
                position = 'BUY'
                entry_price = close_curr
                sl_level = entry_price - (sl_pips * pip_size)
                tp_level = entry_price + (tp_pips * pip_size)

            # SELL
            elif (close_curr < ema200_curr) and (close_curr < ema50_curr) and pullback_sell and \
                 candlestick_ok_sell and \
                 (rsi_sell_min <= rsi_curr <= rsi_sell_max) and (rsi_curr < rsi_prev):
                
                position = 'SELL'
                entry_price = close_curr
                sl_level = entry_price + (sl_pips * pip_size)
                tp_level = entry_price - (tp_pips * pip_size)

    # Analyze trades
    total_trades = len(trades)
    if total_trades == 0:
        return 0.0, 0.0, 0.0, 0

    wins = sum(1 for t in trades if t["result"] == "WIN")
    win_rate = (wins / total_trades) * 100.0
    net_pips = sum(t["pips"] for t in trades)
    
    # Gross Profit & Loss
    gross_profit = sum(t["pips"] for t in trades if t["pips"] > 0)
    gross_loss = abs(sum(t["pips"] for t in trades if t["pips"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return win_rate, net_pips, profit_factor, total_trades


def main():
    if not connect_mt5():
        print("[FATAL] Could not connect to MT5.")
        sys.exit(1)

    symbols = ["EURUSDm"]
    timeframes = ["M15", "M30"]
    
    print("=" * 70)
    print("                GRID SEARCH SCALPING OPTIMIZER (FAST)")
    print("=" * 70)
    print("Goal: Find strategies with Win Rate >= 70% and positive expectancy.")
    print("-" * 70)

    results = []

    try:
        for symbol in symbols:
            # Check if symbol is available in MT5
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"[WARNING] Symbol {symbol} not found on this MT5 terminal. Skipping.")
                continue

            for tf in timeframes:
                df = fetch_data(symbol, tf, years=1)
                if df is None or len(df) < 1000:
                    continue
                
                print(f"[GRID SEARCH] Scanning parameters for {symbol} ({tf})...")
                
                # We test various SL & TP combinations
                param_combinations = []
                
                # Check different values for SL and TP
                # To guarantee 70% win rate, we search a wide range of ratios
                for sl in [6.0, 8.0, 9.0, 10.0, 12.0, 15.0]:
                    for tp in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]:
                        for rsi_buy_min, rsi_buy_max, rsi_sell_min, rsi_sell_max in [
                            (45, 60, 40, 55),
                            (40, 60, 40, 60),
                            (45, 65, 35, 55),
                            (50, 70, 30, 50)
                        ]:
                            for use_cand in [True, False]:
                                param_combinations.append((sl, tp, rsi_buy_min, rsi_buy_max, rsi_sell_min, rsi_sell_max, use_cand))
                
                print(f"  Evaluating {len(param_combinations)} combinations...")
                
                for sl, tp, rsi_b_min, rsi_b_max, rsi_s_min, rsi_s_max, use_cand in param_combinations:
                    win_rate, net_pips, pf, num_trades = evaluate_strategy(
                        df, symbol, sl, tp, rsi_b_min, rsi_b_max, rsi_s_min, rsi_s_max, use_cand
                    )
                    
                    if win_rate >= 70.0 and num_trades >= 15:
                        rr_ratio = tp / sl
                        results.append({
                            "symbol": symbol,
                            "timeframe": tf,
                            "sl": sl,
                            "tp": tp,
                            "rsi_buy": f"{rsi_b_min}-{rsi_b_max}",
                            "rsi_sell": f"{rsi_s_min}-{rsi_s_max}",
                            "use_candlestick": use_cand,
                            "win_rate": win_rate,
                            "net_pips": net_pips,
                            "profit_factor": pf,
                            "num_trades": num_trades,
                            "rr_ratio": rr_ratio
                        })

        # Display results
        if not results:
            print("\n[GRID SEARCH] No parameter combinations achieved a Win Rate >= 70% with at least 15 trades.")
            # Let's search again with lower trade frequency threshold or slightly lower win rate to check
            print("[GRID SEARCH] Searching again with relaxed criteria (Win Rate >= 65%, Trades >= 10)...")
            
            # Re-scan with lower thresholds to find the highest possible
            for symbol in symbols:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    continue
                for tf in timeframes:
                    df = fetch_data(symbol, tf, years=1)
                    if df is None:
                        continue
                    
                    # We reuse same combination logic but look for 65%+
                    for sl in [6.0, 8.0, 9.0, 10.0, 12.0, 15.0]:
                        for tp in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]:
                            for rsi_buy_min, rsi_buy_max, rsi_sell_min, rsi_sell_max in [(45, 60, 40, 55), (40, 60, 40, 60)]:
                                for use_cand in [True, False]:
                                    win_rate, net_pips, pf, num_trades = evaluate_strategy(
                                        df, symbol, sl, tp, rsi_buy_min, rsi_buy_max, rsi_sell_min, rsi_sell_max, use_cand
                                    )
                                    if win_rate >= 65.0 and num_trades >= 10:
                                        rr_ratio = tp / sl
                                        results.append({
                                            "symbol": symbol,
                                            "timeframe": tf,
                                            "sl": sl,
                                            "tp": tp,
                                            "rsi_buy": f"{rsi_buy_min}-{rsi_buy_max}",
                                            "rsi_sell": f"{rsi_sell_min}-{rsi_sell_max}",
                                            "use_candlestick": use_cand,
                                            "win_rate": win_rate,
                                            "net_pips": net_pips,
                                            "profit_factor": pf,
                                            "num_trades": num_trades,
                                            "rr_ratio": rr_ratio
                                        })
            
        if results:
            print("\n" + "=" * 80)
            print("🏆 OPTIMIZED STRATEGIES ACHIEVING HIGH WIN RATE")
            print("=" * 80)
            df_res = pd.DataFrame(results)
            df_res = df_res.sort_values(by="win_rate", ascending=False)
            
            # Print top 15 results
            for idx, r in df_res.head(15).iterrows():
                print(f"Pair: {r['symbol']} | TF: {r['timeframe']} | SL: {r['sl']} | TP: {r['tp']} (R:R 1:{r['rr_ratio']:.2f})")
                print(f"  Win Rate: {r['win_rate']:.1f}% | Net Pips: {r['net_pips']:.1f} | Trades: {r['num_trades']} | PF: {r['profit_factor']:.2f}")
                print(f"  RSI Buy: {r['rsi_buy']} | RSI Sell: {r['rsi_sell']} | Candlestick Trigger: {r['use_candlestick']}")
                print("-" * 80)
            
            df_res.to_csv("optimized_parameters_report.csv", index=False)
            print("[OK] Detailed report saved to optimized_parameters_report.csv")
        else:
            print("[ERROR] No strategies found even with relaxed criteria.")

    finally:
        disconnect_mt5()


if __name__ == "__main__":
    main()
