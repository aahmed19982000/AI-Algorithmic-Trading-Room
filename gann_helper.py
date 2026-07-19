"""
W.D. Gann Price Angles & Fibonacci Pivot Structure Calculator
==============================================================
Calculates Gann Price Angles (Square of 9) for square, triangle,
and pentagon geometries, and detects the 1-2-3 pivot correction structure.
"""

import math
import pandas as pd

def get_gann_multiplier(price):
    """Detect appropriate scaling factor based on the price of the asset."""
    if price < 10.0:
        return 10000.0
    elif price < 1000.0:
        return 100.0
    else:
        return 1.0

def calculate_gann_levels(base_price, mode='bullish', geometry='square'):
    """
    Project Gann Price Angles starting from a base price.
    
    Args:
        base_price: The starting price (Low A for Bullish, High A for Bearish)
        mode: 'bullish' (project resistance levels) or 'bearish' (project support levels)
        geometry: 'square' (45°), 'triangle' (60°), 'pentagon' (36°)
        
    Returns:
        list of dicts: [{"angle": angle, "price": price}, ...]
    """
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

def detect_market_structure(df, lookback=100):
    """
    Detect the latest valid A-B-C pivot setup in the historical candles.
    
    Args:
        df: candles DataFrame containing 'High', 'Low', 'Close'
        lookback: number of recent candles to search
        
    Returns:
        dict or None: {
            "type": "BUY" | "SELL",
            "A": base_price,
            "B": trigger_price,
            "C": correction_price,
            "idx_B": index_of_B
        }
    """
    if df is None or len(df) < 20:
        return None

    # 1. Identify swing points (pivots)
    # A point is a pivot if it's the peak/trough in a window of 5 candles on each side
    window = 5
    pivots_high = []
    pivots_low = []
    
    for i in range(window, len(df) - window):
        # High pivot
        is_high = True
        for j in range(i - window, i + window + 1):
            if df.loc[j, 'High'] > df.loc[i, 'High']:
                is_high = False
                break
        if is_high:
            pivots_high.append((i, df.loc[i, 'High']))
            
        # Low pivot
        is_low = True
        for j in range(i - window, i + window + 1):
            if df.loc[j, 'Low'] < df.loc[i, 'Low']:
                is_low = False
                break
        if is_low:
            pivots_low.append((i, df.loc[i, 'Low']))

    # Filter pivots to the active lookback window
    last_idx = len(df) - 1
    lookback_start = max(0, last_idx - lookback)
    
    window_highs = [p for p in pivots_high if lookback_start <= p[0] <= last_idx]
    window_lows = [p for p in pivots_low if lookback_start <= p[0] <= last_idx]
    
    # 2. Trace BUY setup (Low A -> High B -> Low C)
    buy_setup = None
    if len(window_lows) >= 2 and len(window_highs) >= 1:
        # Scan from newest low pivot (C) backwards
        for c_pivot in reversed(window_lows):
            idx_C, val_C = c_pivot
            # Find the most recent high pivot (B) before C
            b_pivots = [p for p in window_highs if p[0] < idx_C]
            if b_pivots:
                idx_B, val_B = b_pivots[-1]
                # Find the most recent low pivot (A) before B
                a_pivots = [p for p in window_lows if p[0] < idx_B]
                if a_pivots:
                    idx_A, val_A = a_pivots[-1]
                    
                    # Verify model structures: C > A and B > C
                    if val_C > val_A and val_B > val_C:
                        # Retracement percentage
                        ratio = (val_B - val_C) / (val_B - val_A)
                        if 0.50 <= ratio <= 0.75:
                            buy_setup = {
                                "type": "BUY",
                                "A": float(val_A),
                                "B": float(val_B),
                                "C": float(val_C),
                                "idx_B": int(idx_B),
                                "time_A": df.loc[idx_A, 'time'].strftime('%Y-%m-%d %H:%M:%S') if 'time' in df.columns else str(df.index[idx_A]),
                                "time_B": df.loc[idx_B, 'time'].strftime('%Y-%m-%d %H:%M:%S') if 'time' in df.columns else str(df.index[idx_B]),
                                "time_C": df.loc[idx_C, 'time'].strftime('%Y-%m-%d %H:%M:%S') if 'time' in df.columns else str(df.index[idx_C])
                            }
                            break

    # 3. Trace SELL setup (High A -> Low B -> High C)
    sell_setup = None
    if len(window_highs) >= 2 and len(window_lows) >= 1:
        for c_pivot in reversed(window_highs):
            idx_C, val_C = c_pivot
            b_pivots = [p for p in window_lows if p[0] < idx_C]
            if b_pivots:
                idx_B, val_B = b_pivots[-1]
                a_pivots = [p for p in window_highs if p[0] < idx_B]
                if a_pivots:
                    idx_A, val_A = a_pivots[-1]
                    
                    # Verify structures: C < A and B < C
                    if val_C < val_A and val_B < val_C:
                        ratio = (val_C - val_B) / (val_A - val_B)
                        if 0.50 <= ratio <= 0.75:
                            sell_setup = {
                                "type": "SELL",
                                "A": float(val_A),
                                "B": float(val_B),
                                "C": float(val_C),
                                "idx_B": int(idx_B),
                                "time_A": df.loc[idx_A, 'time'].strftime('%Y-%m-%d %H:%M:%S') if 'time' in df.columns else str(df.index[idx_A]),
                                "time_B": df.loc[idx_B, 'time'].strftime('%Y-%m-%d %H:%M:%S') if 'time' in df.columns else str(df.index[idx_B]),
                                "time_C": df.loc[idx_C, 'time'].strftime('%Y-%m-%d %H:%M:%S') if 'time' in df.columns else str(df.index[idx_C])
                            }
                            break

    # Return the fresher setup (based on the index of B)
    if buy_setup and sell_setup:
        if buy_setup["idx_B"] > sell_setup["idx_B"]:
            return buy_setup
        else:
            return sell_setup
    elif buy_setup:
        return buy_setup
    elif sell_setup:
        return sell_setup

    return None


def calculate_fibonacci_extension(a, b, ratio=1.618):
    """
    Projects a Fibonacci extension target from a two-point move A->B.
    Works for both directions via a single formula: for a bullish move
    (B > A) this projects upward past B; for a bearish move (B < A) it
    projects downward past B by the same proportion.
    """
    return b + ratio * (b - a)


def _check_wave1_impulse_quality(window_highs, window_lows, idx_A, val_A, idx_B, val_B,
                                  max_internal_retracement, is_buy):
    """
    Checks whether the A->B move "looks" impulsive rather than corrective by
    scanning for any interim pullback between A and B and verifying it never
    retraces more than `max_internal_retracement` of the move made so far.
    A 3-wave corrective structure misread as an impulsive Wave 1 typically
    shows a much deeper interim retracement than a genuine impulse would.
    """
    if is_buy:
        interim_highs = sorted(p for p in window_highs if idx_A < p[0] < idx_B)
        interim_lows = sorted(p for p in window_lows if idx_A < p[0] < idx_B)
        for low_idx, low_val in interim_lows:
            preceding_highs = [h for h in interim_highs if h[0] < low_idx]
            ref_val = preceding_highs[-1][1] if preceding_highs else val_A
            move_up = ref_val - val_A
            if move_up <= 0:
                continue
            retracement = (ref_val - low_val) / move_up
            if retracement > max_internal_retracement:
                return False
        return True
    else:
        interim_lows = sorted(p for p in window_lows if idx_A < p[0] < idx_B)
        interim_highs = sorted(p for p in window_highs if idx_A < p[0] < idx_B)
        for high_idx, high_val in interim_highs:
            preceding_lows = [l for l in interim_lows if l[0] < high_idx]
            ref_val = preceding_lows[-1][1] if preceding_lows else val_A
            move_down = val_A - ref_val
            if move_down <= 0:
                continue
            retracement = (high_val - ref_val) / move_down
            if retracement > max_internal_retracement:
                return False
        return True


def detect_elliott_wave2_setup(df, lookback=100, min_retracement=0.382, max_retracement=0.786,
                                max_wave1_internal_retracement=0.618):
    """
    Detects a completed Elliott Wave 2 retracement following an impulsive Wave 1,
    using the same swing-pivot scan as detect_market_structure() (A-B-C shape:
    A=Wave 0 start, B=Wave 1 extreme, C=Wave 2 retracement).

    Two things distinguish this from a plain Gann A-B-C structure:
    - The retracement band matches Elliott's Wave 2 convention (38.2%-78.6% by
      default) rather than Gann's 50-75%. A ratio >= 1.0 (Wave 2 fully
      retracing Wave 1) is automatically excluded by this band.
    - A Wave 1 "impulse quality" check (_check_wave1_impulse_quality) rejects
      setups where the A->B move itself looks like a 3-wave correction rather
      than a genuine impulse — a real, entry-time-checkable piece of Elliott
      rigor beyond a bare 3-point retracement count.

    Returns:
        dict or None: {
            "type": "BUY" | "SELL", "A": wave0_price, "B": wave1_price, "C": wave2_price,
            "idx_B": index_of_B, "retracement_pct": ratio*100,
            "time_A"/"time_B"/"time_C": timestamps (if available)
        }
    """
    if df is None or len(df) < 20:
        return None

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

    last_idx = len(df) - 1
    lookback_start = max(0, last_idx - lookback)

    window_highs = [p for p in pivots_high if lookback_start <= p[0] <= last_idx]
    window_lows = [p for p in pivots_low if lookback_start <= p[0] <= last_idx]

    def _make_context(idx_A, idx_B, idx_C):
        return {
            "time_A": df.loc[idx_A, 'time'].strftime('%Y-%m-%d %H:%M:%S') if 'time' in df.columns else str(df.index[idx_A]),
            "time_B": df.loc[idx_B, 'time'].strftime('%Y-%m-%d %H:%M:%S') if 'time' in df.columns else str(df.index[idx_B]),
            "time_C": df.loc[idx_C, 'time'].strftime('%Y-%m-%d %H:%M:%S') if 'time' in df.columns else str(df.index[idx_C]),
        }

    # Trace bullish Wave 0-1-2 (Low A -> High B -> Low C, expecting Wave 3 up)
    buy_setup = None
    if len(window_lows) >= 2 and len(window_highs) >= 1:
        for c_pivot in reversed(window_lows):
            idx_C, val_C = c_pivot
            b_pivots = [p for p in window_highs if p[0] < idx_C]
            if not b_pivots:
                continue
            idx_B, val_B = b_pivots[-1]
            a_pivots = [p for p in window_lows if p[0] < idx_B]
            if not a_pivots:
                continue
            idx_A, val_A = a_pivots[-1]

            if val_C > val_A and val_B > val_C:
                ratio = (val_B - val_C) / (val_B - val_A)
                if min_retracement <= ratio <= max_retracement:
                    if _check_wave1_impulse_quality(window_highs, window_lows, idx_A, val_A,
                                                     idx_B, val_B, max_wave1_internal_retracement, is_buy=True):
                        buy_setup = {
                            "type": "BUY", "A": float(val_A), "B": float(val_B), "C": float(val_C),
                            "idx_B": int(idx_B), "idx_C": int(idx_C),
                            "retracement_pct": ratio * 100.0,
                            **_make_context(idx_A, idx_B, idx_C)
                        }
                        break

    # Trace bearish Wave 0-1-2 (High A -> Low B -> High C, expecting Wave 3 down)
    sell_setup = None
    if len(window_highs) >= 2 and len(window_lows) >= 1:
        for c_pivot in reversed(window_highs):
            idx_C, val_C = c_pivot
            b_pivots = [p for p in window_lows if p[0] < idx_C]
            if not b_pivots:
                continue
            idx_B, val_B = b_pivots[-1]
            a_pivots = [p for p in window_highs if p[0] < idx_B]
            if not a_pivots:
                continue
            idx_A, val_A = a_pivots[-1]

            if val_C < val_A and val_B < val_C:
                ratio = (val_C - val_B) / (val_A - val_B)
                if min_retracement <= ratio <= max_retracement:
                    if _check_wave1_impulse_quality(window_highs, window_lows, idx_A, val_A,
                                                     idx_B, val_B, max_wave1_internal_retracement, is_buy=False):
                        sell_setup = {
                            "type": "SELL", "A": float(val_A), "B": float(val_B), "C": float(val_C),
                            "idx_B": int(idx_B), "idx_C": int(idx_C),
                            "retracement_pct": ratio * 100.0,
                            **_make_context(idx_A, idx_B, idx_C)
                        }
                        break

    if buy_setup and sell_setup:
        return buy_setup if buy_setup["idx_B"] > sell_setup["idx_B"] else sell_setup
    return buy_setup or sell_setup


def detect_dynamic_gann_levels(base_price, trigger_price, mode='bullish'):
    """
    Detects which standard Gann angle B (trigger_price) corresponds to relative to A (base_price).
    Returns the entry angle, target price (double the entry angle), and stop loss price (0° angle = base_price).
    """
    multiplier = get_gann_multiplier(base_price)
    scaled_base = base_price * multiplier
    sqrt_base = math.sqrt(scaled_base)
    
    # Check all standard angles across Square, Triangle, and Pentagon geometries
    standard_angles = [36, 45, 60, 72, 90, 120, 135, 144, 180, 216, 225, 240, 270, 288, 300, 315, 324, 360]
        
    best_angle = None
    best_diff = float('inf')
    best_price = None
    
    for angle in standard_angles:
        if mode == 'bullish':
            price_scaled = (sqrt_base + (angle / 180.0)) ** 2
        else:
            factor = sqrt_base - (angle / 180.0)
            price_scaled = (max(0, factor)) ** 2
            
        price_level = price_scaled / multiplier
        diff = abs(price_level - trigger_price)
        if diff < best_diff:
            best_diff = diff
            best_angle = angle
            best_price = price_level
            
    # Target angle is double the entry angle
    target_angle = best_angle * 2
    if mode == 'bullish':
        target_scaled = (sqrt_base + (target_angle / 180.0)) ** 2
    else:
        factor = sqrt_base - (target_angle / 180.0)
        target_scaled = (max(0, factor)) ** 2
        
    target_price = target_scaled / multiplier
    
    # Stop Loss is at 0° (which is base_price A)
    sl_price = base_price
    
    return {
        "entry_angle": best_angle,
        "entry_price": best_price,
        "target_angle": target_angle,
        "target_price": target_price,
        "sl_price": sl_price
    }

