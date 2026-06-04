"""
Martingale Grid Strategy Optimizer for EURUSDm
===============================================
Runs a grid search over timeframes M15, M30, and H1, evaluating a trend-pullback
strategy backed by a Martingale Grid recovery engine. Reports on net ROI,
basket win rates, and drawdowns to find the absolute best settings.
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

    print(f"[DATA] Fetching {years} year historical data for {symbol} ({timeframe_str})...")
    rates = mt5.copy_rates_range(symbol, tf_constant, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        print(f"[ERROR] Failed to download data for {symbol}.")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def simulate_martingale_grid(df, symbol, initial_sl, initial_tp, grid_step, multiplier, max_legs, target_profit, rsi_buy_min, rsi_buy_max, rsi_sell_min, rsi_sell_max, use_candlestick=True):
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

    # Martingale simulation variables
    basket = [] # List of active position dicts: {"type": 'BUY'|'SELL', "entry": price, "lot": size}
    basket_type = None # None, 'BUY', or 'SELL'
    
    # Starting balance
    balance = 500.0
    peak_balance = 500.0
    max_drawdown = 0.0
    
    baskets_closed = [] # List of basket results: "WIN" or "LOSS"

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

        # -------------------------------------------------------------
        # 1. Manage Active Martingale Basket
        # -------------------------------------------------------------
        if basket_type is not None:
            # Calculate total current unrealized profit/loss in pips and USD
            # For simplicity, we calculate basket value based on prices
            # 1 pip on 0.01 lot = $0.10 USD (standard lot sizing for EURUSDm)
            # Basket exit checking:
            # We calculate the average entry price (weighted by lot size)
            total_lot = sum(p["lot"] for p in basket)
            weighted_entry = sum(p["entry"] * p["lot"] for p in basket) / total_lot
            
            # Target TP in absolute price
            # When we have multiple positions, we want to close them at breakeven + target_profit
            # If 1 position: TP is initial_tp
            if len(basket) == 1:
                tp_price = weighted_entry + (initial_tp * pip_size) if basket_type == 'BUY' else weighted_entry - (initial_tp * pip_size)
            else:
                tp_price = weighted_entry + (target_profit * pip_size) if basket_type == 'BUY' else weighted_entry - (target_profit * pip_size)
                
            # Basket Stop Loss:
            # We place a hard Stop Loss at initial_sl pips from the weighted entry (if len == 1)
            # or initial_sl pips from the *last* leg entry (to protect account)
            last_entry = basket[-1]["entry"]
            if basket_type == 'BUY':
                sl_price = last_entry - (initial_sl * pip_size)
            else:
                sl_price = last_entry + (initial_sl * pip_size)

            # Check if current candle triggered exit (TP or SL)
            if basket_type == 'BUY':
                # TP hit
                if high_curr >= tp_price:
                    # Win! Close all positions in profit
                    pips_won = (tp_price - weighted_entry) * pip_multiplier
                    # Compounding profit: risk 3% of balance per lot size (0.01 lot per $500 balance)
                    lot_scale = (balance / 500.0)
                    usd_won = pips_won * 10.0 * total_lot * lot_scale
                    balance += usd_won
                    
                    baskets_closed.append("WIN")
                    basket = []
                    basket_type = None
                    continue
                # SL hit
                elif low_curr <= sl_price:
                    # Loss! Close all positions in loss
                    pips_lost = (weighted_entry - sl_price) * pip_multiplier
                    lot_scale = (balance / 500.0)
                    usd_lost = pips_lost * 10.0 * total_lot * lot_scale
                    balance -= usd_lost
                    
                    baskets_closed.append("LOSS")
                    basket = []
                    basket_type = None
                    continue
            else: # SELL
                # TP hit
                if low_curr <= tp_price:
                    pips_won = (weighted_entry - tp_price) * pip_multiplier
                    lot_scale = (balance / 500.0)
                    usd_won = pips_won * 10.0 * total_lot * lot_scale
                    balance += usd_won
                    
                    baskets_closed.append("WIN")
                    basket = []
                    basket_type = None
                    continue
                # SL hit
                elif high_curr >= sl_price:
                    pips_lost = (sl_price - weighted_entry) * pip_multiplier
                    lot_scale = (balance / 500.0)
                    usd_lost = pips_lost * 10.0 * total_lot * lot_scale
                    balance -= usd_lost
                    
                    baskets_closed.append("LOSS")
                    basket = []
                    basket_type = None
                    continue

            # Check Grid Addition (Next Martingale Leg)
            # If current price went against us by grid_step, and we haven't reached max_legs
            if len(basket) < max_legs:
                last_leg = basket[-1]
                distance = (last_leg["entry"] - close_curr) * pip_multiplier if basket_type == 'BUY' else (close_curr - last_leg["entry"]) * pip_multiplier
                
                if distance >= grid_step:
                    # Open next leg! Lot size is multiplied
                    next_lot = last_leg["lot"] * multiplier
                    basket.append({
                        "type": basket_type,
                        "entry": close_curr,
                        "lot": next_lot
                    })
                    # Adjust weighted entry and recalculate
                    continue

        # -------------------------------------------------------------
        # 2. Open New Basket (Initial Entry Signal)
        # -------------------------------------------------------------
        if basket_type is None and session_active:
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

            candlestick_ok_buy = (is_bullish_engulfing or is_bullish_pinbar) if use_candlestick else True
            candlestick_ok_sell = (is_bearish_engulfing or is_bearish_pinbar) if use_candlestick else True

            # BUY
            if (close_curr > ema200_curr) and (close_curr > ema50_curr) and pullback_buy and \
               candlestick_ok_buy and \
               (rsi_buy_min <= rsi_curr <= rsi_buy_max) and (rsi_curr > rsi_prev):
                
                basket_type = 'BUY'
                basket = [{
                    "type": 'BUY',
                    "entry": close_curr,
                    "lot": 0.01  # Base lot
                }]

            # SELL
            elif (close_curr < ema200_curr) and (close_curr < ema50_curr) and pullback_sell and \
                 candlestick_ok_sell and \
                 (rsi_sell_min <= rsi_curr <= rsi_sell_max) and (rsi_curr < rsi_prev):
                
                basket_type = 'SELL'
                basket = [{
                    "type": 'SELL',
                    "entry": close_curr,
                    "lot": 0.01  # Base lot
                }]

        # Track Drawdown
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100.0
        if dd > max_drawdown:
            max_drawdown = dd

    total_baskets = len(baskets_closed)
    if total_baskets == 0:
        return 500.0, 0.0, 0.0, 0

    wins = sum(1 for b in baskets_closed if b == "WIN")
    win_rate = (wins / total_baskets) * 100.0

    return balance, win_rate, max_drawdown, total_baskets


def main():
    if not connect_mt5():
        print("[FATAL] Could not connect to MT5.")
        sys.exit(1)

    symbol = "EURUSDm"
    timeframes = ["M15", "M30", "H1"]
    
    print("=" * 80)
    print("         MARTINGALE GRID STRATEGY OPTIMIZER (EURUSDm)")
    print("=" * 80)
    print("Evaluating M15, M30, and H1 trend pullback strategies with grid recovery.")
    print("-" * 80)

    results = []

    try:
        # Check symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"[FATAL] Symbol {symbol} not found on this MT5 terminal.")
            return

        for tf in timeframes:
            df = fetch_data(symbol, tf, years=1)
            if df is None or len(df) < 1000:
                continue
            
            print(f"[GRID SEARCH] Scanning martingale parameters for {symbol} ({tf})...")
            
            # Parameters space to evaluate:
            # sl: Stop Loss for each entry/ basket exit
            # tp: Take Profit for 1st entry
            # grid_step: distance in pips to add next martingale leg
            # multiplier: lot size multiplier for next leg
            # max_legs: maximum positions allowed in the basket (2 to 4)
            # target_profit: pips above average entry to close multi-position basket (2 to 5 pips)
            param_combinations = []
            
            # Generate parameter variations
            for sl in [15.0, 20.0, 30.0]:
                for tp in [15.0, 20.0]:
                    for step in [10.0, 15.0, 20.0]:
                        for mult in [1.5, 2.0]:
                            for legs in [3, 4]:
                                for tgt_prof in [2.0, 4.0]:
                                    # RSI Buy: 45-60, RSI Sell: 40-55
                                    param_combinations.append((sl, tp, step, mult, legs, tgt_prof, 45, 60, 40, 55, True))
                                    param_combinations.append((sl, tp, step, mult, legs, tgt_prof, 30, 70, 30, 70, False))

            print(f"  Evaluating {len(param_combinations)} combinations...")
            
            for sl, tp, step, mult, legs, tgt_prof, rsi_b_min, rsi_b_max, rsi_s_min, rsi_s_max, use_cand in param_combinations:
                final_balance, win_rate, max_dd, num_baskets = simulate_martingale_grid(
                    df, symbol, sl, tp, step, mult, legs, tgt_prof, rsi_b_min, rsi_b_max, rsi_s_min, rsi_s_max, use_cand
                )
                
                # Check results: We want positive expectancy and at least 15 baskets traded
                # To prevent extreme drawdowns, we filter out Max Drawdown > 35%
                if final_balance > 500.0 and num_baskets >= 12 and max_dd <= 35.0:
                    net_roi = ((final_balance - 500.0) / 500.0) * 100.0
                    results.append({
                        "timeframe": tf,
                        "sl": sl,
                        "tp": tp,
                        "grid_step": step,
                        "multiplier": mult,
                        "max_legs": legs,
                        "target_profit": tgt_prof,
                        "rsi_buy": f"{rsi_b_min}-{rsi_b_max}",
                        "rsi_sell": f"{rsi_s_min}-{rsi_s_max}",
                        "use_candlestick": use_cand,
                        "win_rate": win_rate,
                        "num_baskets": num_baskets,
                        "final_balance": final_balance,
                        "net_roi": net_roi,
                        "max_drawdown": max_dd
                    })

        # Display results
        if not results:
            print("\n[GRID SEARCH] No martingale strategies met profit, trade count, and drawdown criteria.")
        else:
            print("\n" + "=" * 90)
            print("🏆 TOP MARTINGALE GRID STRATEGIES ORDERED BY SIMULATED COMPOUNDING ROI (STARTING $500):")
            print("=" * 90)
            df_res = pd.DataFrame(results)
            df_res = df_res.sort_values(by="net_roi", ascending=False)
            
            # Print top 15 results
            for idx, r in df_res.head(15).iterrows():
                print(f"TF: {r['timeframe']} | Base SL/TP: {r['sl']}/{r['tp']} pips | Grid Step: {r['grid_step']} pips")
                print(f"  Martingale: Mult={r['multiplier']}x | Max Legs={r['max_legs']} | Recovery Profit={r['target_profit']} pips")
                print(f"  Final Balance: ${r['final_balance']:.2f} | ROI: {r['net_roi']:+.1f}% | Max Drawdown: {r['max_drawdown']:.1f}%")
                print(f"  Basket Win Rate: {r['win_rate']:.1f}% | Total Baskets: {r['num_baskets']} | Candlestick: {r['use_candlestick']}")
                print(f"  RSI Buy: {r['rsi_buy']} | RSI Sell: {r['rsi_sell']}")
                print("-" * 90)
            
            df_res.to_csv("optimized_martingale_report.csv", index=False)
            print("[OK] Detailed report saved to optimized_martingale_report.csv")

    finally:
        disconnect_mt5()


if __name__ == "__main__":
    main()
