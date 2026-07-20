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

def _find_swing_pivots(df, window=5):
    """
    Shared fractal swing-pivot detector: a candle is a pivot high/low if it's
    the extreme within `window` candles on each side. Extracted from
    detect_market_structure() so detect_range_zone() (and any future
    pivot-based detector) can reuse the exact same scan rather than
    duplicating it.

    Requires positional 0-based integer labels (matching every other pivot
    scanner in this module) — pass a freshly reset_index()'d DataFrame if
    slicing a larger frame first.

    Uses numpy arrays (df['High'].values / .values) for the neighbor
    comparisons rather than scalar df.loc[j, ...] access in a nested Python
    loop: the latter is correct but roughly 500x slower in practice (each
    .loc[] scalar lookup carries real overhead, and this function is called
    once per bar by every strategy's backtester) — measured at ~560ms per
    call on a 200-bar window with .loc, vs. sub-millisecond with numpy
    slicing, the difference between a backtest finishing in minutes and one
    that would take the better part of a day.

    Returns (pivots_high, pivots_low), each a list of (index, price) tuples
    in ascending index order.
    """
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)

    pivots_high = []
    pivots_low = []

    for i in range(window, n - window):
        if highs[i] >= highs[i - window:i + window + 1].max():
            pivots_high.append((i, highs[i]))
        if lows[i] <= lows[i - window:i + window + 1].min():
            pivots_low.append((i, lows[i]))

    return pivots_high, pivots_low


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

    # 1. Identify swing points (pivots) — a point is a pivot if it's the
    # peak/trough in a window of 5 candles on each side
    window = 5
    pivots_high, pivots_low = _find_swing_pivots(df, window=window)

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


def _wilder_running_sum(series, period):
    """
    Wilder's original running-sum smoothing, for raw accumulating quantities
    (True Range, +DM, -DM): seed = sum of the first `period` values (skipping
    index 0, which is always NaN from a .diff()/.shift()), then each next
    value is smoothed - smoothed/period + current. This is NOT the same as
    the DX->ADX smoothing below — DX is already a 0-100 percentage, and
    running this recursion on it would compound it like a raw sum and blow
    the result far outside 0-100.
    """
    result = pd.Series(float('nan'), index=series.index)
    values = series.values
    n = len(values)
    if n <= period:
        return result

    window = values[1:period + 1]
    if any(pd.isna(v) for v in window):
        return result

    seed = float(sum(window))
    result.iloc[period] = seed
    smoothed = seed
    for i in range(period + 1, n):
        current = values[i]
        if pd.isna(current):
            smoothed = float('nan')
        elif not pd.isna(smoothed):
            smoothed = smoothed - (smoothed / period) + current
        result.iloc[i] = smoothed
    return result


def _wilder_running_average(series, period):
    """
    Wilder's running-AVERAGE smoothing, used only for DX->ADX: DX is already
    a 0-100 ratio, so seeding with a simple mean of the first `period` DX
    values and then recursing smoothed = (smoothed*(period-1) + current)/period
    keeps ADX bounded in 0-100. Using the running-sum recursion above on DX
    instead would be wrong — that recursion is for raw accumulating
    quantities, not an already-normalized percentage.
    """
    result = pd.Series(float('nan'), index=series.index)
    first_valid = series.first_valid_index()
    if first_valid is None:
        return result

    first_pos = series.index.get_loc(first_valid)
    values = series.values
    n = len(values)
    if first_pos + period > n:
        return result

    window = values[first_pos:first_pos + period]
    if any(pd.isna(v) for v in window):
        return result

    seed = float(sum(window)) / period
    seed_pos = first_pos + period - 1
    result.iloc[seed_pos] = seed
    smoothed = seed
    for i in range(seed_pos + 1, n):
        current = values[i]
        if pd.isna(current):
            smoothed = float('nan')
        elif not pd.isna(smoothed):
            smoothed = (smoothed * (period - 1) + current) / period
        result.iloc[i] = smoothed
    return result


def calculate_adx(df, period=14):
    """
    Average Directional Index (Wilder, 1978) — measures trend strength
    regardless of direction, 0-100. Used by the Range Trading strategy as a
    regime filter: low ADX supports fading a range's boundaries; high ADX
    means the market is trending, which invalidates a fade premise but
    supports treating a level break as a genuine breakout.

    Proper Wilder smoothing requires TWO different recursions (see
    _wilder_running_sum / _wilder_running_average above) — using the same
    one for both TR/DM and DX is a common implementation bug that produces
    an ADX far outside the valid 0-100 range.

    Returns a pandas Series aligned to df's index; NaN during warmup
    (~2*period bars). Callers must treat pd.isna(adx) as an explicit HOLD —
    never let a raw comparison against NaN silently evaluate False.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_mask = (up_move > down_move) & (up_move > 0)
    minus_mask = (down_move > up_move) & (down_move > 0)
    plus_dm[plus_mask] = up_move[plus_mask]
    minus_dm[minus_mask] = down_move[minus_mask]
    plus_dm.iloc[0] = float('nan')
    minus_dm.iloc[0] = float('nan')

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    smoothed_tr = _wilder_running_sum(tr, period)
    smoothed_plus_dm = _wilder_running_sum(plus_dm, period)
    smoothed_minus_dm = _wilder_running_sum(minus_dm, period)

    plus_di = pd.Series(float('nan'), index=df.index)
    minus_di = pd.Series(float('nan'), index=df.index)
    has_tr = smoothed_tr.notna()
    valid_tr = has_tr & (smoothed_tr > 0)

    # Guard: a flat market (TR smoothed to exactly 0) must read as "no
    # directional movement" (DI = 0), not a division by zero.
    plus_di[has_tr] = 0.0
    minus_di[has_tr] = 0.0
    plus_di[valid_tr] = 100.0 * smoothed_plus_dm[valid_tr] / smoothed_tr[valid_tr]
    minus_di[valid_tr] = 100.0 * smoothed_minus_dm[valid_tr] / smoothed_tr[valid_tr]

    di_sum = plus_di + minus_di
    dx = pd.Series(float('nan'), index=df.index)
    has_sum = di_sum.notna()
    valid_sum = has_sum & (di_sum > 0)

    # Guard: if +DI and -DI are both 0 (flat market), DX = 0, not NaN/inf.
    dx[has_sum] = 0.0
    dx[valid_sum] = 100.0 * (plus_di[valid_sum] - minus_di[valid_sum]).abs() / di_sum[valid_sum]

    adx = _wilder_running_average(dx, period)
    return adx


def detect_range_zone(df, lookback=100, window=5, peak_tolerance_pct=0.15, trough_tolerance_pct=0.15,
                       min_range_pct=0.3, max_range_pct=3.0):
    """
    Detects a horizontal consolidation range from the two most recent swing
    highs (range top) and two most recent swing lows (range bottom), reusing
    the same fractal pivot scan as detect_market_structure(). Unlike a
    rolling min/max window, this requires the boundary to be confirmed by
    TWO separate touches within tolerance — a rolling min/max would happily
    fit a single one-off spike, which isn't a genuine range.

    Args:
        df: candles DataFrame containing 'High', 'Low' (0-based positional index)
        lookback: number of recent candles to search for pivots
        window: swing-pivot detection window (candles on each side)
        peak_tolerance_pct: max allowed %% difference between the two most
            recent swing highs for them to count as "the same" resistance
        trough_tolerance_pct: same, for the two most recent swing lows
        min_range_pct/max_range_pct: sanity bounds on range width as a %% of
            price — too tight and spread eats any edge, too wide and it's
            not a real consolidation

    Returns:
        dict or None: {
            "range_top":, "range_bottom":, "peak_a":, "peak_b":,
            "trough_a":, "trough_b":
        }
        (peak_a/trough_a is the older of the two confirming pivots, _b the
        more recent — same "A before B" convention as detect_market_structure)
    """
    if df is None or len(df) < 20:
        return None

    pivots_high, pivots_low = _find_swing_pivots(df, window=window)

    last_idx = len(df) - 1
    lookback_start = max(0, last_idx - lookback)

    window_highs = [p for p in pivots_high if lookback_start <= p[0] <= last_idx]
    window_lows = [p for p in pivots_low if lookback_start <= p[0] <= last_idx]

    if len(window_highs) < 2 or len(window_lows) < 2:
        return None

    peak_a_idx, peak_a = window_highs[-2]
    peak_b_idx, peak_b = window_highs[-1]
    peak_min = min(peak_a, peak_b)
    if peak_min <= 0:
        return None
    if abs(peak_a - peak_b) / peak_min * 100.0 > peak_tolerance_pct:
        return None
    range_top = max(peak_a, peak_b)

    trough_a_idx, trough_a = window_lows[-2]
    trough_b_idx, trough_b = window_lows[-1]
    trough_min = min(trough_a, trough_b)
    if trough_min <= 0:
        return None
    if abs(trough_a - trough_b) / trough_min * 100.0 > trough_tolerance_pct:
        return None
    range_bottom = min(trough_a, trough_b)

    if range_top <= range_bottom:
        return None

    range_width = range_top - range_bottom
    mid_price = (range_top + range_bottom) / 2.0
    if mid_price <= 0:
        return None
    range_width_pct = range_width / mid_price * 100.0
    if range_width_pct < min_range_pct or range_width_pct > max_range_pct:
        return None

    return {
        "range_top": float(range_top),
        "range_bottom": float(range_bottom),
        "peak_a": float(peak_a),
        "peak_b": float(peak_b),
        "trough_a": float(trough_a),
        "trough_b": float(trough_b),
    }


# Classical harmonic pattern Fibonacci ratio bands. Single-value tuples (e.g.
# Gartley's ad_xa) get a shared tolerance applied when matching — same
# tolerance-band idiom as detect_range_zone's peak_tolerance_pct.
HARMONIC_PATTERNS = {
    "Gartley":   {"ab_xa": (0.618, 0.618), "bc_ab": (0.382, 0.886), "ad_xa": (0.786, 0.786), "cd_bc": (1.13, 1.618)},
    "Bat":       {"ab_xa": (0.382, 0.500), "bc_ab": (0.382, 0.886), "ad_xa": (0.886, 0.886), "cd_bc": (1.618, 2.618)},
    "Butterfly": {"ab_xa": (0.786, 0.786), "bc_ab": (0.382, 0.886), "ad_xa": (1.27, 1.618), "cd_bc": (1.618, 2.24)},
    "Crab":      {"ab_xa": (0.382, 0.618), "bc_ab": (0.382, 0.886), "ad_xa": (1.618, 1.618), "cd_bc": (2.24, 3.618)},
}


def _harmonic_ratio_matches(value, band, tolerance_pct):
    """Checks `value` against a (lo, hi) ratio band, widened by tolerance_pct."""
    lo, hi = band
    span = (hi - lo) if hi != lo else abs(lo)
    tol = span * (tolerance_pct / 100.0)
    return (lo - tol) <= value <= (hi + tol)


def detect_harmonic_pattern(df, lookback=100, window=5, patterns=None,
                            ratio_tolerance_pct=5.0, prz_confluence_pct=10.0):
    """
    Detects a classical harmonic X-A-B-C-D price structure via Fibonacci
    ratio confluence, reusing the same fractal pivot scan as
    detect_market_structure()/detect_elliott_wave2_setup() — just one link
    deeper (X-A-B-C instead of A-B-C). D itself is intentionally NOT a
    confirmed historical pivot here: it's a projected zone (the "Potential
    Reversal Zone", PRZ) computed from X-A-B-C, since the entire point of a
    harmonic pattern is to enter as price is completing D, not after it has
    already reversed away from it.

    Two independent Fibonacci projections of D must agree (within
    prz_confluence_pct of each other) before a PRZ is accepted — this
    "confluence" is harmonic trading's own rigor-adding requirement, playing
    the same role _check_wave1_impulse_quality plays for Elliott Wave:
    - ad_xa: D as a retracement of the whole X->A move
    - cd_bc: D as an extension of the B->C move

    Args:
        df: candles DataFrame containing 'High', 'Low' (0-based positional index)
        lookback: number of recent candles to search for pivots
        window: swing-pivot detection window (candles on each side)
        patterns: iterable of pattern names to check (default: all of HARMONIC_PATTERNS)
        ratio_tolerance_pct: tolerance applied to each ratio band
        prz_confluence_pct: max allowed %% difference between the two D
            projections for them to count as a valid PRZ

    Returns:
        dict or None: {
            "pattern": name, "type": "BUY"/"SELL",
            "X":, "A":, "B":, "C":, "idx_C":,
            "d_zone_low":, "d_zone_high":, "prz_price": (midpoint of the two projections)
        }
    """
    if df is None or len(df) < 20:
        return None

    if patterns is None:
        patterns = list(HARMONIC_PATTERNS.keys())

    pivots_high, pivots_low = _find_swing_pivots(df, window=window)

    last_idx = len(df) - 1
    lookback_start = max(0, last_idx - lookback)

    window_highs = [p for p in pivots_high if lookback_start <= p[0] <= last_idx]
    window_lows = [p for p in pivots_low if lookback_start <= p[0] <= last_idx]

    def _trace(c_pivots, b_pool, a_pool, x_pool, is_bullish):
        for idx_C, val_C in reversed(c_pivots):
            b_candidates = [p for p in b_pool if p[0] < idx_C]
            if not b_candidates:
                continue
            idx_B, val_B = b_candidates[-1]
            a_candidates = [p for p in a_pool if p[0] < idx_B]
            if not a_candidates:
                continue
            idx_A, val_A = a_candidates[-1]
            x_candidates = [p for p in x_pool if p[0] < idx_A]
            if not x_candidates:
                continue
            idx_X, val_X = x_candidates[-1]

            if is_bullish:
                # X low -> A high -> B low (retraces down, stays above X) -> C high (retraces up, stays below A)
                if not (val_A > val_X and val_B > val_X and val_B < val_A and val_C > val_B and val_C < val_A):
                    continue
                xa = val_A - val_X
                ab = val_A - val_B
                bc = val_C - val_B
            else:
                # X high -> A low -> B high (retraces up, stays below X) -> C low (retraces down, stays above A)
                if not (val_X > val_A and val_B > val_A and val_B < val_X and val_C < val_B and val_C > val_A):
                    continue
                xa = val_X - val_A
                ab = val_B - val_A
                bc = val_B - val_C

            if xa <= 0 or ab <= 0 or bc <= 0:
                continue

            ab_xa_ratio = ab / xa
            bc_ab_ratio = bc / ab

            for name in patterns:
                bands = HARMONIC_PATTERNS.get(name)
                if not bands:
                    continue
                if not _harmonic_ratio_matches(ab_xa_ratio, bands["ab_xa"], ratio_tolerance_pct):
                    continue
                if not _harmonic_ratio_matches(bc_ab_ratio, bands["bc_ab"], ratio_tolerance_pct):
                    continue

                ad_lo, ad_hi = bands["ad_xa"]
                cd_lo, cd_hi = bands["cd_bc"]
                # Midpoint of each band for the projection itself — tolerance
                # is for ratio *matching* above, not for widening the
                # projected price beyond the pattern's own definition.
                ad_ratio = (ad_lo + ad_hi) / 2.0
                cd_ratio = (cd_lo + cd_hi) / 2.0

                if is_bullish:
                    d_via_ad = val_A - ad_ratio * xa
                    d_via_cd = val_C - cd_ratio * bc
                else:
                    d_via_ad = val_A + ad_ratio * xa
                    d_via_cd = val_C + cd_ratio * bc

                d_min = min(d_via_ad, d_via_cd)
                if d_min == 0:
                    continue
                if abs(d_via_ad - d_via_cd) / abs(d_min) * 100.0 > prz_confluence_pct:
                    continue

                return {
                    "pattern": name,
                    "type": "BUY" if is_bullish else "SELL",
                    "X": float(val_X), "A": float(val_A), "B": float(val_B), "C": float(val_C),
                    "idx_C": int(idx_C),
                    "d_zone_low": float(min(d_via_ad, d_via_cd)),
                    "d_zone_high": float(max(d_via_ad, d_via_cd)),
                    "prz_price": float((d_via_ad + d_via_cd) / 2.0),
                }
        return None

    bullish = None
    if len(window_lows) >= 2 and len(window_highs) >= 2:
        bullish = _trace(window_highs, window_lows, window_highs, window_lows, is_bullish=True)

    bearish = None
    if bullish is None and len(window_highs) >= 2 and len(window_lows) >= 2:
        bearish = _trace(window_lows, window_highs, window_lows, window_highs, is_bullish=False)

    return bullish or bearish


def _fit_trendline(pivots):
    """
    Least-squares line through a list of (index, price) pivot tuples.
    Returns (slope, intercept) such that price ≈ slope*index + intercept,
    or None if fewer than 2 pivots are given.
    """
    import numpy as np
    if len(pivots) < 2:
        return None
    xs = np.array([p[0] for p in pivots], dtype=float)
    ys = np.array([p[1] for p in pivots], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def detect_continuation_pattern(df, pole_lookback=20, pole_min_move_pct=1.5, pole_min_efficiency=0.6,
                                 consolidation_max_bars=40, swing_window=3,
                                 flat_slope_threshold_pct=0.02):
    """
    Detects a continuation chart pattern: a strong directional "pole" move
    followed by a consolidation whose upper/lower boundaries (swing-pivot
    trendlines fitted via _fit_trendline) classify it as one of Rectangle,
    Ascending/Descending Triangle, Rising/Falling Wedge, Pennant, or Flag.

    The pole is checked for both magnitude AND cleanliness via an
    "efficiency ratio" (net move / sum of absolute bar-to-bar moves — a
    clean directional run scores near 1, a choppy one near 0) — the
    rigor-adding check here, playing the same role
    _check_wave1_impulse_quality plays for Elliott Wave.

    Pennant and Symmetrical Triangle are the same geometric shape
    (converging trendlines, opposite-sign slopes) — this always labels it
    "Pennant" since a qualifying pole is required to reach this branch at
    all. Rising/Falling Wedges are the one shape traded counter to the pole
    direction (textbook convention: a rising wedge is bearish, a falling
    wedge bullish, regardless of what preceded them) — every other shape
    here trades in the pole's own direction (true continuation).

    Args:
        df: candles DataFrame containing 'High', 'Low', 'Close' (0-based positional index)

    Returns:
        dict or None: {
            "pattern":, "type": "BUY"/"SELL",
            "pole_start":, "pole_end":, "pole_height":,
            "upper_slope":, "upper_intercept":, "lower_slope":, "lower_intercept":,
            "breakout_level": (relevant trendline's value at the last bar in the window)
        }
    """
    total_window = pole_lookback + consolidation_max_bars
    if df is None or len(df) < total_window + swing_window * 2 + 1:
        return None

    last_idx = len(df) - 1
    consol_start = last_idx - consolidation_max_bars + 1
    pole_start_idx = consol_start - pole_lookback
    if pole_start_idx < 0:
        return None

    pole_slice = df.iloc[pole_start_idx:consol_start]
    if len(pole_slice) < pole_lookback:
        return None

    pole_start_price = float(pole_slice['Close'].iloc[0])
    pole_end_price = float(pole_slice['Close'].iloc[-1])
    if pole_start_price == 0:
        return None
    net_move = pole_end_price - pole_start_price
    net_move_pct = net_move / pole_start_price * 100.0

    bar_moves = pole_slice['Close'].diff().dropna()
    sum_abs_moves = bar_moves.abs().sum()
    efficiency = abs(net_move) / sum_abs_moves if sum_abs_moves > 0 else 0.0

    if abs(net_move_pct) < pole_min_move_pct or efficiency < pole_min_efficiency:
        return None

    pole_direction = "BUY" if net_move > 0 else "SELL"
    pole_height = abs(net_move)

    consol_df = df.iloc[consol_start:last_idx + 1].reset_index(drop=True)
    pivots_high, pivots_low = _find_swing_pivots(consol_df, window=swing_window)
    if len(pivots_high) < 2 or len(pivots_low) < 2:
        return None

    upper_fit = _fit_trendline(pivots_high)
    lower_fit = _fit_trendline(pivots_low)
    if upper_fit is None or lower_fit is None:
        return None
    upper_slope, upper_intercept = upper_fit
    lower_slope, lower_intercept = lower_fit

    avg_price = float(consol_df['Close'].mean())
    if avg_price == 0:
        return None
    upper_slope_pct = upper_slope / avg_price * 100.0
    lower_slope_pct = lower_slope / avg_price * 100.0

    upper_flat = abs(upper_slope_pct) <= flat_slope_threshold_pct
    lower_flat = abs(lower_slope_pct) <= flat_slope_threshold_pct
    converging = upper_slope_pct < lower_slope_pct

    pattern_name = None
    pattern_type = pole_direction

    if upper_flat and lower_flat:
        pattern_name = "Rectangle"
    elif upper_flat and lower_slope_pct > flat_slope_threshold_pct:
        pattern_name = "Ascending Triangle"
    elif lower_flat and upper_slope_pct < -flat_slope_threshold_pct:
        pattern_name = "Descending Triangle"
    elif not upper_flat and not lower_flat and upper_slope_pct > 0 and lower_slope_pct > 0 and converging:
        pattern_name = "Rising Wedge"
        pattern_type = "SELL"
    elif not upper_flat and not lower_flat and upper_slope_pct < 0 and lower_slope_pct < 0 and converging:
        pattern_name = "Falling Wedge"
        pattern_type = "BUY"
    elif not upper_flat and not lower_flat and upper_slope_pct < 0 and lower_slope_pct > 0:
        pattern_name = "Pennant"
    elif not upper_flat and not lower_flat and (upper_slope_pct > 0) == (lower_slope_pct > 0):
        # Same-sign, non-flat, non-converging-wedge -> Flag, valid only when
        # sloping AGAINST the pole direction (the textbook flag definition)
        slopes_negative = upper_slope_pct < 0 and lower_slope_pct < 0
        if (pole_direction == "BUY" and slopes_negative) or (pole_direction == "SELL" and not slopes_negative):
            pattern_name = "Flag"

    if pattern_name is None:
        return None

    x_last = len(consol_df) - 1
    upper_value_last = upper_slope * x_last + upper_intercept
    lower_value_last = lower_slope * x_last + lower_intercept
    breakout_level = upper_value_last if pattern_type == "BUY" else lower_value_last

    return {
        "pattern": pattern_name, "type": pattern_type,
        "pole_start": pole_start_price, "pole_end": pole_end_price, "pole_height": pole_height,
        "upper_slope": float(upper_slope), "upper_intercept": float(upper_intercept),
        "lower_slope": float(lower_slope), "lower_intercept": float(lower_intercept),
        "breakout_level": float(breakout_level)
    }


def _tolerance_match(a, b, tolerance_pct):
    """Checks two price levels are within tolerance_pct of each other (relative to the smaller magnitude)."""
    m = min(abs(a), abs(b))
    if m == 0:
        return a == b
    return abs(a - b) / m * 100.0 <= tolerance_pct


def detect_reversal_pattern(df, lookback=100, window=5, shoulder_tolerance_pct=2.0,
                            neckline_tolerance_pct=1.0, patterns=None):
    """
    Detects Head & Shoulders (+ inverse) or Double Top/Bottom reversal
    patterns, reusing the same fractal pivot scan and "reversed scan, chain
    to nearest prior opposite pivot" idiom as detect_elliott_wave2_setup /
    detect_harmonic_pattern, extended to 3 (H&S) or 2 (Double Top/Bottom)
    same-type pivots.

    Head & Shoulders (bearish): three peaks — left shoulder, head (the
    highest), right shoulder — with the two shoulders within
    shoulder_tolerance_pct of each other (same tolerance-band idiom as
    detect_range_zone's peak_tolerance_pct), connected by two troughs (the
    neckline) within neckline_tolerance_pct of each other. Inverse (bullish)
    is the mirror image using troughs as the pattern and peaks as the neckline.

    Double Top/Bottom: two peaks/troughs within shoulder_tolerance_pct, with
    one connecting trough/peak forming the neckline.

    Args:
        df: candles DataFrame containing 'High', 'Low' (0-based positional index)
        patterns: iterable of pattern names to check (default: all four)

    Returns:
        dict or None: {
            "pattern":, "type": "BUY"/"SELL",
            "head":, "shoulder_a":, "shoulder_b":, "neckline_level":, "idx_head":
        }
        (Double Top/Bottom has no distinct shoulders — "head" and
        "shoulder_a"/"shoulder_b" all reference the two matched peaks/troughs.)
    """
    if patterns is None:
        patterns = ["Head and Shoulders", "Inverse Head and Shoulders", "Double Top", "Double Bottom"]

    if df is None or len(df) < 20:
        return None

    pivots_high, pivots_low = _find_swing_pivots(df, window=window)
    last_idx = len(df) - 1
    lookback_start = max(0, last_idx - lookback)
    window_highs = [p for p in pivots_high if lookback_start <= p[0] <= last_idx]
    window_lows = [p for p in pivots_low if lookback_start <= p[0] <= last_idx]

    if "Head and Shoulders" in patterns and len(window_highs) >= 3 and len(window_lows) >= 2:
        for idx_sb, val_sb in reversed(window_highs):
            head_candidates = [p for p in window_highs if p[0] < idx_sb]
            if not head_candidates:
                continue
            idx_head, val_head = head_candidates[-1]
            sa_candidates = [p for p in window_highs if p[0] < idx_head]
            if not sa_candidates:
                continue
            idx_sa, val_sa = sa_candidates[-1]

            troughs_between = [p for p in window_lows if idx_sa < p[0] < idx_sb]
            trough1 = [p for p in troughs_between if p[0] < idx_head]
            trough2 = [p for p in troughs_between if p[0] > idx_head]
            if not trough1 or not trough2:
                continue
            idx_t1, val_t1 = trough1[-1]
            idx_t2, val_t2 = trough2[0]

            if (val_head > val_sa and val_head > val_sb and
                    _tolerance_match(val_sa, val_sb, shoulder_tolerance_pct) and
                    _tolerance_match(val_t1, val_t2, neckline_tolerance_pct)):
                return {
                    "pattern": "Head and Shoulders", "type": "SELL",
                    "head": float(val_head), "shoulder_a": float(val_sa), "shoulder_b": float(val_sb),
                    "neckline_level": float((val_t1 + val_t2) / 2.0), "idx_head": int(idx_head)
                }

    if "Inverse Head and Shoulders" in patterns and len(window_lows) >= 3 and len(window_highs) >= 2:
        for idx_sb, val_sb in reversed(window_lows):
            head_candidates = [p for p in window_lows if p[0] < idx_sb]
            if not head_candidates:
                continue
            idx_head, val_head = head_candidates[-1]
            sa_candidates = [p for p in window_lows if p[0] < idx_head]
            if not sa_candidates:
                continue
            idx_sa, val_sa = sa_candidates[-1]

            peaks_between = [p for p in window_highs if idx_sa < p[0] < idx_sb]
            peak1 = [p for p in peaks_between if p[0] < idx_head]
            peak2 = [p for p in peaks_between if p[0] > idx_head]
            if not peak1 or not peak2:
                continue
            idx_p1, val_p1 = peak1[-1]
            idx_p2, val_p2 = peak2[0]

            if (val_head < val_sa and val_head < val_sb and
                    _tolerance_match(val_sa, val_sb, shoulder_tolerance_pct) and
                    _tolerance_match(val_p1, val_p2, neckline_tolerance_pct)):
                return {
                    "pattern": "Inverse Head and Shoulders", "type": "BUY",
                    "head": float(val_head), "shoulder_a": float(val_sa), "shoulder_b": float(val_sb),
                    "neckline_level": float((val_p1 + val_p2) / 2.0), "idx_head": int(idx_head)
                }

    if "Double Top" in patterns and len(window_highs) >= 2 and len(window_lows) >= 1:
        for idx_p2, val_p2 in reversed(window_highs):
            p1_candidates = [p for p in window_highs if p[0] < idx_p2]
            if not p1_candidates:
                continue
            idx_p1, val_p1 = p1_candidates[-1]
            troughs_between = [p for p in window_lows if idx_p1 < p[0] < idx_p2]
            if not troughs_between:
                continue
            if _tolerance_match(val_p1, val_p2, shoulder_tolerance_pct):
                idx_neck, val_neck = min(troughs_between, key=lambda p: p[1])
                return {
                    "pattern": "Double Top", "type": "SELL",
                    "head": float(val_p1), "shoulder_a": float(val_p1), "shoulder_b": float(val_p2),
                    "neckline_level": float(val_neck), "idx_head": int(idx_p2)
                }

    if "Double Bottom" in patterns and len(window_lows) >= 2 and len(window_highs) >= 1:
        for idx_t2, val_t2 in reversed(window_lows):
            t1_candidates = [p for p in window_lows if p[0] < idx_t2]
            if not t1_candidates:
                continue
            idx_t1, val_t1 = t1_candidates[-1]
            peaks_between = [p for p in window_highs if idx_t1 < p[0] < idx_t2]
            if not peaks_between:
                continue
            if _tolerance_match(val_t1, val_t2, shoulder_tolerance_pct):
                idx_neck, val_neck = max(peaks_between, key=lambda p: p[1])
                return {
                    "pattern": "Double Bottom", "type": "BUY",
                    "head": float(val_t1), "shoulder_a": float(val_t1), "shoulder_b": float(val_t2),
                    "neckline_level": float(val_neck), "idx_head": int(idx_t2)
                }

    return None

