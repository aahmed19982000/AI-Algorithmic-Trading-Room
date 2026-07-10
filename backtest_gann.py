"""
W.D. Gann Price Angles + Fibonacci Pivot Breakout Backtester (with Trailing Stop)
================================================================================
Downloads 1 year of historical candles for a symbol,
detects the 1-2-3 pivot correction structure (retracement 50%-75%),
runs the Gann Price Angles breakout simulation with trailing stop, and outputs full metrics.
Supports dynamic contract sizing and pip calculations for Forex & Cryptos.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from dotenv import load_dotenv

from mt5_connection import connect_mt5, disconnect_mt5
from mt5_data import get_timeframe

load_dotenv()

def fetch_backtest_data(symbol, timeframe_str, years=1):
    """Fetch historical candles from MT5."""
    tf_constant = get_timeframe(timeframe_str)
    if tf_constant is None:
        return None

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    print(f"[BACKTEST] Downloading historical data for {symbol} ({timeframe_str})...")
    rates = mt5.copy_rates_range(symbol, tf_constant, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        print("[ERROR] Failed to download backtest data.")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }, inplace=True)
    return df

def get_gann_multiplier(price):
    if price < 10.0:
        return 10000.0
    elif price < 1000.0:
        return 100.0
    else:
        return 1.0

def calculate_gann_levels(base_price, mode='bullish', geometry='square'):
    multiplier = get_gann_multiplier(base_price)
    scaled_price = base_price * multiplier
    sqrt_price = math.sqrt(scaled_price)
    
    if geometry == 'triangle':
        angles = [60, 120, 180, 240, 300, 360]
    elif geometry == 'pentagon':
        angles = [36, 72, 108, 144, 180, 216, 252, 288, 324, 360]
    else: # square
        angles = [45, 90, 135, 180, 225, 270, 315, 360]
        
    levels = []
    for angle in angles:
        if mode == 'bullish':
            level_scaled = (sqrt_price + (angle / 180.0)) ** 2
        else:
            factor = sqrt_price - (angle / 180.0)
            level_scaled = (max(0, factor)) ** 2
            
        level_price = level_scaled / multiplier
        levels.append({
            "angle": angle,
            "price": level_price
        })
    return levels


def check_m5_exit_signal_backtest(current_time, m5_df, pos_type):
    if m5_df is None or m5_df.empty:
        return False
        
    # Filter M5 data up to current simulation time
    sub_df = m5_df[m5_df['time'] <= current_time]
    if len(sub_df) < 15:
        return False
        
    window = 2
    swing_highs = []
    swing_lows = []
    
    # We look at the last 60 candles in sub_df
    sub_df = sub_df.tail(60).reset_index(drop=True)
    
    for i in range(window, len(sub_df) - window):
        # Check swing high
        is_high = True
        for j in range(i - window, i + window + 1):
            if sub_df['High'].iloc[j] > sub_df['High'].iloc[i]:
                is_high = False
                break
        if is_high:
            swing_highs.append((i, float(sub_df['High'].iloc[i])))
            
        # Check swing low
        is_low = True
        for j in range(i - window, i + window + 1):
            if sub_df['Low'].iloc[j] < sub_df['Low'].iloc[i]:
                is_low = False
                break
        if is_low:
            swing_lows.append((i, float(sub_df['Low'].iloc[i])))
            
    if pos_type == 0:  # BUY
        if len(swing_lows) >= 2:
            latest_idx, latest_val = swing_lows[-1]
            prev_idx, prev_val = swing_lows[-2]
            if len(sub_df) - 1 - latest_idx <= 4:
                if latest_val < prev_val:
                    # Correction check: at least one swing high must exist between prev_idx and latest_idx
                    correction_exists = any(prev_idx < sh[0] < latest_idx for sh in swing_highs)
                    if correction_exists:
                        # Candle Close Breakout check
                        close_breakout = False
                        for idx in range(prev_idx + 1, len(sub_df)):
                            if float(sub_df['Close'].iloc[idx]) < prev_val:
                                close_breakout = True
                                break
                        if close_breakout:
                            return True
    else:  # SELL
        if len(swing_highs) >= 2:
            latest_idx, latest_val = swing_highs[-1]
            prev_idx, prev_val = swing_highs[-2]
            if len(sub_df) - 1 - latest_idx <= 4:
                if latest_val > prev_val:
                    # Correction check: at least one swing low must exist between prev_idx and latest_idx
                    correction_exists = any(prev_idx < sl[0] < latest_idx for sl in swing_lows)
                    if correction_exists:
                        # Candle Close Breakout check
                        close_breakout = False
                        for idx in range(prev_idx + 1, len(sub_df)):
                            if float(sub_df['Close'].iloc[idx]) > prev_val:
                                close_breakout = True
                                break
                        if close_breakout:
                            return True
    return False


def run_backtest_gann(df, symbol, geometry='square', lookback=100, use_grid=True):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[ERROR] Symbol {symbol} not found in MT5.")
        return []
        
    contract_size = symbol_info.trade_contract_size
    point = symbol_info.point
    
    # Standardize pip/point size
    if point == 0.00001:
        pip_size = 0.0001
    elif point == 0.001:
        pip_size = 0.01
    else:
        pip_size = point
        
    pip_multiplier = 1.0 / pip_size
    
    # Fetch M5 candles for M5 swing exits simulation
    end_date = df['time'].max()
    import pandas as pd
    import datetime
    end_dt = datetime.datetime.fromtimestamp(pd.Timestamp(end_date).timestamp())
    m5_rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, end_dt, 80000)
    if m5_rates is not None and len(m5_rates) > 0:
        m5_df = pd.DataFrame(m5_rates)
        m5_df['time'] = pd.to_datetime(m5_df['time'], unit='s')
        m5_df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
        }, inplace=True)
    else:
        m5_df = pd.DataFrame()
        
    # Grid parameters
    grid_step = 10.0 # pips
    grid_mult = 2.0
    grid_max_legs = 4
    grid_target_profit = 2.0 # pips
    grid_sl = 20.0 # pips
    
    # Initial simulated balance for dynamic sizing
    sim_balance = 500.0
    
    # Scale pip sizes for cryptos / CFDs
    if contract_size < 1000.0: 
        avg_price = df['Close'].mean()
        pip_size = (0.01 * avg_price) / 10.0
        pip_multiplier = 1.0 / pip_size
        print(f"[INFO] Crypto/CFD Asset detected. Adjusted pip_size to {pip_size:.4f} (10 pips = {10.0*pip_size:.2f} USD)")

    # 1. Identify local pivots
    window = 5
    pivots_high = []
    pivots_low = []
    
    for i in range(window, len(df) - window):
        is_high = True
        for j in range(i - window, i + window + 1):
            if df.loc[j, 'High'] > df.loc[i, 'High']:
                is_high = False
                break
        if is_high:
            pivots_high.append((i, df.loc[i, 'High']))
            
        is_low = True
        for j in range(i - window, i + window + 1):
            if df.loc[j, 'Low'] < df.loc[i, 'Low']:
                is_low = False
                break
        if is_low:
            pivots_low.append((i, df.loc[i, 'Low']))
            
    basket = [] # active legs: [{"entry": price, "vol": volume}]
    basket_type = None # 'BUY' or 'SELL'
    basket_sl = 0.0
    basket_initial_sl = 0.0
    basket_tp = 0.0
    basket_open_time = None
    
    # Trailing Stop variables
    basket_targets = []
    basket_highest_reached = -1
    
    active_setup = None
    trades = []
    
    for i in range(200, len(df)):
        current_time = df.loc[i, 'time']
        close_curr = df.loc[i, 'Close']
        open_curr = df.loc[i, 'Open']
        high_curr = df.loc[i, 'High']
        low_curr = df.loc[i, 'Low']
        
        # --- Case A: Manage Active Trade Basket ---
        if len(basket) > 0:
            if basket_type == 'BUY':
                total_vol = sum(leg["vol"] for leg in basket)
                avg_entry = sum(leg["entry"] * leg["vol"] for leg in basket) / total_vol
                # Check Stop Loss
                if low_curr <= basket_sl:
                    profit_usd = sum((basket_sl - leg["entry"]) * leg["vol"] for leg in basket) * contract_size
                    total_pips = sum((basket_sl - leg["entry"]) * leg["vol"] for leg in basket) * pip_multiplier
                    total_vol = sum(leg["vol"] for leg in basket)
                    trades.append({
                        "type": "BUY",
                        "entry_time": basket_open_time,
                        "exit_time": current_time,
                        "legs": len(basket),
                        "volume": total_vol,
                        "pips": total_pips / total_vol,
                        "profit_usd_raw": profit_usd,
                        "result": "LOSS",
                        "sl_price": basket_sl,
                        "tp_price": basket_tp,
                        "gann_data": active_setup.copy() if active_setup else None
                    })
                    sim_balance += profit_usd
                    basket = []
                    basket_type = None
                    active_setup = None
                    continue
                    
                # Check Take Profit
                elif high_curr >= basket_tp:
                    profit_usd = sum((basket_tp - leg["entry"]) * leg["vol"] for leg in basket) * contract_size
                    total_pips = sum((basket_tp - leg["entry"]) * leg["vol"] for leg in basket) * pip_multiplier
                    total_vol = sum(leg["vol"] for leg in basket)
                    trades.append({
                        "type": "BUY",
                        "entry_time": basket_open_time,
                        "exit_time": current_time,
                        "legs": len(basket),
                        "volume": total_vol,
                        "pips": total_pips / total_vol,
                        "profit_usd_raw": profit_usd,
                        "result": "WIN",
                        "sl_price": basket_sl,
                        "tp_price": basket_tp,
                        "gann_data": active_setup.copy() if active_setup else None
                    })
                    sim_balance += profit_usd
                    basket = []
                    basket_type = None
                    active_setup = None
                    continue
                
                # Check Take Profit (90% of TP distance as Close Early)
                elif high_curr >= avg_entry + (basket_tp - avg_entry) * 0.90:
                    tp_early = avg_entry + (basket_tp - avg_entry) * 0.90
                    profit_usd = sum((tp_early - leg["entry"]) * leg["vol"] for leg in basket) * contract_size
                    total_pips = sum((tp_early - leg["entry"]) * leg["vol"] for leg in basket) * pip_multiplier
                    trades.append({
                        "type": "BUY",
                        "entry_time": basket_open_time,
                        "exit_time": current_time,
                        "legs": len(basket),
                        "volume": total_vol,
                        "pips": total_pips / total_vol,
                        "profit_usd_raw": profit_usd,
                        "result": "WIN",
                        "sl_price": basket_sl,
                        "tp_price": tp_early,
                        "gann_data": active_setup.copy() if active_setup else None
                    })
                    sim_balance += profit_usd
                    basket = []
                    basket_type = None
                    active_setup = None
                    continue
                    
                # Check M5 swing exit
                elif check_m5_exit_signal_backtest(current_time, m5_df, 0):
                    profit_usd = sum((close_curr - leg["entry"]) * leg["vol"] for leg in basket) * contract_size
                    total_pips = sum((close_curr - leg["entry"]) * leg["vol"] for leg in basket) * pip_multiplier
                    trades.append({
                        "type": "BUY",
                        "entry_time": basket_open_time,
                        "exit_time": current_time,
                        "legs": len(basket),
                        "volume": total_vol,
                        "pips": total_pips / total_vol,
                        "profit_usd_raw": profit_usd,
                        "result": "WIN" if profit_usd >= 0 else "LOSS",
                        "sl_price": basket_sl,
                        "tp_price": close_curr,
                        "gann_data": active_setup.copy() if active_setup else None
                    })
                    sim_balance += profit_usd
                    basket = []
                    basket_type = None
                    active_setup = None
                    continue
                    
                # Check Breakeven (at 50% TP distance)
                else:
                    halfway_target = avg_entry + (basket_tp - avg_entry) * 0.5
                    if high_curr >= halfway_target:
                        if basket_highest_reached < 0:
                            basket_highest_reached = 0
                            basket_sl = avg_entry # Move SL to average entry (breakeven)
                
                # Check Grid Recovery addition
                if use_grid and len(basket) < grid_max_legs:
                    last_leg = basket[-1]
                    drawdown_pips = (last_leg["entry"] - low_curr) * pip_multiplier
                    if drawdown_pips >= grid_step:
                        # Deactivate Trailing stop once grid recovery starts
                        next_entry = last_leg["entry"] - (grid_step * pip_size)
                        next_vol = round(last_leg["vol"] * grid_mult, 2)
                        basket.append({"entry": next_entry, "vol": next_vol})
                        
                        total_vol = sum(leg["vol"] for leg in basket)
                        avg_entry = sum(leg["entry"] * leg["vol"] for leg in basket) / total_vol
                        basket_tp = avg_entry + (grid_target_profit * pip_size)
                        basket_sl = basket_initial_sl # Stop Loss remains at Point A
                        
            elif basket_type == 'SELL':
                total_vol = sum(leg["vol"] for leg in basket)
                avg_entry = sum(leg["entry"] * leg["vol"] for leg in basket) / total_vol
                # Check Stop Loss
                if high_curr >= basket_sl:
                    profit_usd = sum((leg["entry"] - basket_sl) * leg["vol"] for leg in basket) * contract_size
                    total_pips = sum((leg["entry"] - basket_sl) * leg["vol"] for leg in basket) * pip_multiplier
                    total_vol = sum(leg["vol"] for leg in basket)
                    trades.append({
                        "type": "SELL",
                        "entry_time": basket_open_time,
                        "exit_time": current_time,
                        "legs": len(basket),
                        "volume": total_vol,
                        "pips": total_pips / total_vol,
                        "profit_usd_raw": profit_usd,
                        "result": "LOSS",
                        "sl_price": basket_sl,
                        "tp_price": basket_tp,
                        "gann_data": active_setup.copy() if active_setup else None
                    })
                    sim_balance += profit_usd
                    basket = []
                    basket_type = None
                    active_setup = None
                    continue
                    
                # Check Take Profit
                elif low_curr <= basket_tp:
                    profit_usd = sum((leg["entry"] - basket_tp) * leg["vol"] for leg in basket) * contract_size
                    total_pips = sum((leg["entry"] - basket_tp) * leg["vol"] for leg in basket) * pip_multiplier
                    total_vol = sum(leg["vol"] for leg in basket)
                    trades.append({
                        "type": "SELL",
                        "entry_time": basket_open_time,
                        "exit_time": current_time,
                        "legs": len(basket),
                        "volume": total_vol,
                        "pips": total_pips / total_vol,
                        "profit_usd_raw": profit_usd,
                        "result": "WIN",
                        "sl_price": basket_sl,
                        "tp_price": basket_tp,
                        "gann_data": active_setup.copy() if active_setup else None
                    })
                    sim_balance += profit_usd
                    basket = []
                    basket_type = None
                    active_setup = None
                    continue
                
                # Check Take Profit (90% of TP distance as Close Early)
                elif low_curr <= avg_entry - (avg_entry - basket_tp) * 0.90:
                    tp_early = avg_entry - (avg_entry - basket_tp) * 0.90
                    profit_usd = sum((leg["entry"] - tp_early) * leg["vol"] for leg in basket) * contract_size
                    total_pips = sum((leg["entry"] - tp_early) * leg["vol"] for leg in basket) * pip_multiplier
                    trades.append({
                        "type": "SELL",
                        "entry_time": basket_open_time,
                        "exit_time": current_time,
                        "legs": len(basket),
                        "volume": total_vol,
                        "pips": total_pips / total_vol,
                        "profit_usd_raw": profit_usd,
                        "result": "WIN",
                        "sl_price": basket_sl,
                        "tp_price": tp_early,
                        "gann_data": active_setup.copy() if active_setup else None
                    })
                    sim_balance += profit_usd
                    basket = []
                    basket_type = None
                    active_setup = None
                    continue
                    
                # Check M5 swing exit
                elif check_m5_exit_signal_backtest(current_time, m5_df, 1):
                    profit_usd = sum((leg["entry"] - close_curr) * leg["vol"] for leg in basket) * contract_size
                    total_pips = sum((leg["entry"] - close_curr) * leg["vol"] for leg in basket) * pip_multiplier
                    trades.append({
                        "type": "SELL",
                        "entry_time": basket_open_time,
                        "exit_time": current_time,
                        "legs": len(basket),
                        "volume": total_vol,
                        "pips": total_pips / total_vol,
                        "profit_usd_raw": profit_usd,
                        "result": "WIN" if profit_usd >= 0 else "LOSS",
                        "sl_price": basket_sl,
                        "tp_price": close_curr,
                        "gann_data": active_setup.copy() if active_setup else None
                    })
                    sim_balance += profit_usd
                    basket = []
                    basket_type = None
                    active_setup = None
                    continue
                    
                # Check Breakeven (at 50% TP distance)
                else:
                    halfway_target = avg_entry - (avg_entry - basket_tp) * 0.5
                    if low_curr <= halfway_target:
                        if basket_highest_reached < 0:
                            basket_highest_reached = 0
                            basket_sl = avg_entry # Move SL to average entry (breakeven)
                    
                # Check Grid Recovery addition
                if use_grid and len(basket) < grid_max_legs:
                    last_leg = basket[-1]
                    drawdown_pips = (high_curr - last_leg["entry"]) * pip_multiplier
                    if drawdown_pips >= grid_step:
                        next_entry = last_leg["entry"] + (grid_step * pip_size)
                        next_vol = round(last_leg["vol"] * grid_mult, 2)
                        basket.append({"entry": next_entry, "vol": next_vol})
                        
                        total_vol = sum(leg["vol"] for leg in basket)
                        avg_entry = sum(leg["entry"] * leg["vol"] for leg in basket) / total_vol
                        basket_tp = avg_entry - (grid_target_profit * pip_size)
                        basket_sl = basket_initial_sl # Stop Loss remains at Point A

        # --- Case B: Scan for structural setup or entry ---
        if len(basket) == 0:
            lookback_start = max(0, i - lookback)
            window_highs = [p for p in pivots_high if lookback_start <= p[0] < i]
            window_lows = [p for p in pivots_low if lookback_start <= p[0] < i]
            
            buy_setup = None
            if len(window_lows) >= 2 and len(window_highs) >= 1:
                for c_pivot in reversed(window_lows):
                    idx_C, val_C = c_pivot
                    b_pivots = [p for p in window_highs if p[0] < idx_C]
                    if len(b_pivots) > 0:
                        idx_B, val_B = b_pivots[-1]
                        a_pivots = [p for p in window_lows if p[0] < idx_B]
                        if len(a_pivots) > 0:
                            idx_A, val_A = a_pivots[-1]
                            if val_C > val_A and val_B > val_C:
                                ratio = (val_B - val_C) / (val_B - val_A)
                                if 0.50 <= ratio <= 0.75:
                                    buy_setup = {
                                        "A": val_A,
                                        "B": val_B,
                                        "C": val_C,
                                        "idx_B": idx_B,
                                        "time_A": df.loc[idx_A, 'time'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(df.loc[idx_A, 'time'], 'strftime') else str(df.loc[idx_A, 'time']),
                                        "time_B": df.loc[idx_B, 'time'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(df.loc[idx_B, 'time'], 'strftime') else str(df.loc[idx_B, 'time']),
                                        "time_C": df.loc[idx_C, 'time'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(df.loc[idx_C, 'time'], 'strftime') else str(df.loc[idx_C, 'time'])
                                    }
                                    break
                                    
            sell_setup = None
            if len(window_highs) >= 2 and len(window_lows) >= 1:
                for c_pivot in reversed(window_highs):
                    idx_C, val_C = c_pivot
                    b_pivots = [p for p in window_lows if p[0] < idx_C]
                    if len(b_pivots) > 0:
                        idx_B, val_B = b_pivots[-1]
                        a_pivots = [p for p in window_highs if p[0] < idx_B]
                        if len(a_pivots) > 0:
                            idx_A, val_A = a_pivots[-1]
                            if val_C < val_A and val_B < val_C:
                                ratio = (val_C - val_B) / (val_A - val_B)
                                if 0.50 <= ratio <= 0.75:
                                    sell_setup = {
                                        "A": val_A,
                                        "B": val_B,
                                        "C": val_C,
                                        "idx_B": idx_B,
                                        "time_A": df.loc[idx_A, 'time'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(df.loc[idx_A, 'time'], 'strftime') else str(df.loc[idx_A, 'time']),
                                        "time_B": df.loc[idx_B, 'time'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(df.loc[idx_B, 'time'], 'strftime') else str(df.loc[idx_B, 'time']),
                                        "time_C": df.loc[idx_C, 'time'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(df.loc[idx_C, 'time'], 'strftime') else str(df.loc[idx_C, 'time'])
                                    }
                                    break
            
            if buy_setup and sell_setup:
                if buy_setup["idx_B"] > sell_setup["idx_B"]:
                    active_setup = {"type": "BUY", **buy_setup}
                else:
                    active_setup = {"type": "SELL", **sell_setup}
            elif buy_setup:
                active_setup = {"type": "BUY", **buy_setup}
            elif sell_setup:
                active_setup = {"type": "SELL", **sell_setup}
            else:
                active_setup = None
                
            if active_setup:
                if active_setup["type"] == "BUY":
                    if close_curr > active_setup["B"] and open_curr <= active_setup["B"]:
                        basket_type = 'BUY'
                        basket_open_time = current_time
                        
                        from gann_helper import detect_dynamic_gann_levels
                        dyn = detect_dynamic_gann_levels(active_setup["A"], active_setup["B"], mode='bullish')
                        basket_tp = dyn["target_price"]
                        basket_sl = dyn["sl_price"]
                        basket_initial_sl = dyn["sl_price"]
                        basket_targets = [dyn["entry_price"], dyn["target_price"]]
                        basket_highest_reached = -1
                        
                        # Dynamic lot sizing: risk 3.0% of current simulated balance
                        risk_percent = 3.0
                        risk_amount = sim_balance * (risk_percent / 100.0)
                        sl_price_diff = abs(close_curr - basket_sl)
                        
                        if sl_price_diff > 0:
                            vol = risk_amount / (sl_price_diff * contract_size)
                            vol_step = symbol_info.volume_step if (symbol_info and symbol_info.volume_step > 0) else 0.01
                            vol = round(vol / vol_step) * vol_step
                            vol_min = symbol_info.volume_min if symbol_info else 0.01
                            vol_max = symbol_info.volume_max if symbol_info else 100.0
                            vol = max(vol_min, min(vol, vol_max))
                            vol = round(vol, 2)
                        else:
                            vol = 0.01
                            
                        basket = [{"entry": close_curr, "vol": vol}]
                        
                        sl_dist = (close_curr - basket_sl) * pip_multiplier
                        tp_dist = (basket_tp - close_curr) * pip_multiplier
                        if sl_dist <= 0 or tp_dist <= 0:
                            basket = []
                            basket_type = None
                            
                elif active_setup["type"] == "SELL":
                    if close_curr < active_setup["B"] and open_curr >= active_setup["B"]:
                        basket_type = 'SELL'
                        basket_open_time = current_time
                        
                        from gann_helper import detect_dynamic_gann_levels
                        dyn = detect_dynamic_gann_levels(active_setup["A"], active_setup["B"], mode='bearish')
                        basket_tp = dyn["target_price"]
                        basket_sl = dyn["sl_price"]
                        basket_initial_sl = dyn["sl_price"]
                        basket_targets = [dyn["entry_price"], dyn["target_price"]]
                        basket_highest_reached = -1
                        
                        # Dynamic lot sizing: risk 3.0% of current simulated balance
                        risk_percent = 3.0
                        risk_amount = sim_balance * (risk_percent / 100.0)
                        sl_price_diff = abs(basket_sl - close_curr)
                        
                        if sl_price_diff > 0:
                            vol = risk_amount / (sl_price_diff * contract_size)
                            vol_step = symbol_info.volume_step if (symbol_info and symbol_info.volume_step > 0) else 0.01
                            vol = round(vol / vol_step) * vol_step
                            vol_min = symbol_info.volume_min if symbol_info else 0.01
                            vol_max = symbol_info.volume_max if symbol_info else 100.0
                            vol = max(vol_min, min(vol, vol_max))
                            vol = round(vol, 2)
                        else:
                            vol = 0.01
                            
                        basket = [{"entry": close_curr, "vol": vol}]
                        
                        sl_dist = (basket_sl - close_curr) * pip_multiplier
                        tp_dist = (close_curr - basket_tp) * pip_multiplier
                        if sl_dist <= 0 or tp_dist <= 0:
                            basket = []
                            basket_type = None

    return trades

def print_gann_report(trades, symbol, timeframe_str):
    print("\n" + "=" * 65)
    print(f"       GANN + FIBONACCI BACKTEST REPORT: {symbol} ({timeframe_str})")
    print("=" * 65)
    
    total_baskets = len(trades)
    if total_baskets == 0:
        print("[WARNING] No trades were generated by the Gann strategy.")
        return
        
    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = sum(1 for t in trades if t["result"] == "LOSS")
    win_rate = (wins / total_baskets) * 100.0
    
    total_pips = sum(t["pips"] for t in trades)
    avg_legs = sum(t["legs"] for t in trades) / total_baskets
    max_legs_used = max(t["legs"] for t in trades)
    
    gross_win_pips = sum(t["pips"] for t in trades if t["pips"] > 0)
    gross_loss_pips = abs(sum(t["pips"] for t in trades if t["pips"] < 0))
    profit_factor = gross_win_pips / gross_loss_pips if gross_loss_pips > 0 else float('inf')
    
    # Account Balance Simulation (Starting with $500)
    balance = 500.0
    peak_balance = 500.0
    max_drawdown = 0.0
    
    for t in trades:
        profit = t["profit_usd_raw"]
        balance += profit
        
        if balance > peak_balance:
            peak_balance = balance
        
        dd = (peak_balance - balance) / peak_balance * 100.0
        if dd > max_drawdown:
            max_drawdown = dd

    print(f"  Total Trades (Baskets): {total_baskets}")
    print(f"  Wins (Baskets):         {wins} (Win Rate: {win_rate:.1f}%)")
    print(f"  Losses (Baskets):       {losses}")
    print(f"  Average Legs per Trade: {avg_legs:.2f}")
    print(f"  Max Legs Used:          {max_legs_used}")
    print(f"  Gross Pips Gained:      {total_pips:.1f} pips")
    print(f"  Profit Factor:          {profit_factor:.2f}")
    print("-" * 65)
    print("  SIMULATED ACCOUNT METRICS (Starting $500, Risking 3%):")
    print(f"  Final Account Balance:  ${balance:.2f} USD")
    print(f"  Net ROI:                {((balance - 500.0) / 500.0 * 100):+.1f}%")
    print(f"  Max Drawdown:           {max_drawdown:.1f}%")
    print("=" * 65)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest W.D. Gann Price Angles on historical MT5 data.")
    parser.add_argument("--symbol", type=str, default="EURUSDm", help="Trading symbol (e.g. EURUSDm, or comma-separated list e.g. EURUSDm,GBPUSDm)")
    parser.add_argument("--timeframe", type=str, default="M30", help="Timeframe (e.g. M30)")
    parser.add_argument("--years", type=int, default=1, help="Years of data to backtest")
    parser.add_argument("--geometry", type=str, default="square", help="Geometry (square, triangle, pentagon)")
    parser.add_argument("--lookback", type=int, default=100, help="Lookback window size")
    parser.add_argument("--no-grid", action="store_true", help="Disable Martingale grid recovery")
    
    args = parser.parse_args()
    
    if not connect_mt5():
        print("[FATAL] Could not connect to MT5.")
        sys.exit(1)
        
    try:
        symbols = [s.strip() for s in args.symbol.split(",") if s.strip()]
        results = {}
        
        for sym in symbols:
            df = fetch_backtest_data(sym, args.timeframe, args.years)
            if df is None or len(df) == 0:
                print(f"[WARNING] Skipping symbol {sym} due to lack of historical data.")
                continue
                
            trades = run_backtest_gann(df, sym, geometry=args.geometry, lookback=args.lookback, use_grid=not args.no_grid)
            results[sym] = trades
            print_gann_report(trades, sym, args.timeframe)
            
        if len(results) > 1:
            print("\n" + "=" * 95)
            print("                        COMPARATIVE PORTFOLIO BACKTEST SUMMARY")
            print("=" * 95)
            print(f"{'Symbol':<12} | {'Trades':<6} | {'Wins/Losses':<12} | {'Win Rate':<8} | {'Net ROI (%)':<12} | {'Max DD (%)':<11} | {'Profit Factor':<13}")
            print("-" * 95)
            for sym, trades in results.items():
                total = len(trades)
                if total == 0:
                    print(f"{sym:<12} | {'0':<6} | {'0/0':<12} | {'0.0%':<8} | {'+0.0%':<12} | {'0.0%':<11} | {'0.00':<13}")
                    continue
                wins = sum(1 for t in trades if t["result"] == "WIN")
                losses = sum(1 for t in trades if t["result"] == "LOSS")
                win_rate = (wins / total) * 100.0
                
                gross_win_pips = sum(t["pips"] for t in trades if t["pips"] > 0)
                gross_loss_pips = abs(sum(t["pips"] for t in trades if t["pips"] < 0))
                pf = gross_win_pips / gross_loss_pips if gross_loss_pips > 0 else (99.99 if gross_win_pips > 0 else 1.0)
                
                balance = 500.0
                peak = 500.0
                max_dd = 0.0
                for t in trades:
                    balance += t["profit_usd_raw"]
                    if balance > peak:
                        peak = balance
                    dd = (peak - balance) / peak * 100.0
                    if dd > max_dd:
                        max_dd = dd
                roi = ((balance - 500.0) / 500.0) * 100.0
                
                pf_str = f"{pf:.2f}" if pf < 99.99 else "99.99"
                print(f"{sym:<12} | {total:<6} | {f'{wins}/{losses}':<12} | {f'{win_rate:.1f}%':<8} | {f'{roi:+.1f}%':<12} | {f'{max_dd:.1f}%':<11} | {pf_str:<13}")
            print("=" * 95)
            
    finally:
        disconnect_mt5()

if __name__ == "__main__":
    main()
