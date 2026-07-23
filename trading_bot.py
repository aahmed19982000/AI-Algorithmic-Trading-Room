"""
Trading Bot Orchestrator (Settings & Database Integrated)
==========================================================
Coordinates MT5 terminal, fetches prices, calls Gemini AI engine with custom strategy prompt,
verifies risk rules (SL, R:R, Max positions), blocks on economic news events, and logs everything to SQLite.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import os
import argparse
import MetaTrader5 as mt5
from dotenv import load_dotenv

from mt5_connection import connect_mt5, disconnect_mt5
from mt5_data import get_candles, get_current_price, get_account_info, get_timeframe
from mt5_orders import open_trade, get_open_positions, close_position, modify_position_sl_tp
from ai_engine import AITradingEngine, format_candles_for_ai
from news_filter import is_news_time
from db_manager import (
    get_settings, 
    log_trade_open, 
    log_trade_close, 
    get_active_trades, 
    log_balance_snapshot
)
from gann_helper import (
    detect_market_structure, calculate_gann_levels, detect_elliott_wave2_setup,
    calculate_fibonacci_extension, detect_range_zone, calculate_adx,
    detect_harmonic_pattern, HARMONIC_PATTERNS,
    detect_continuation_pattern, detect_reversal_pattern
)

# Load configuration
load_dotenv()
MAGIC_NUMBER = int(os.getenv("BOT_MAGIC", "20260604"))
DEFAULT_TIMEFRAME = os.getenv("TRADING_TIMEFRAME", "H4")

# The swing-reversal exit check (check_m5_exit_signal) needs a timeframe
# proportionally faster than whatever timeframe a position was actually
# entered on — a fixed M5 check made sense back when every strategy entered
# on one shared default timeframe, but is pure noise for a position entered
# on an H4 or D1 setup now that every strategy can enter on any of
# scan_timeframes. Falls back to M5 (the original behavior) for anything unmapped.
EXIT_CHECK_TIMEFRAME_MAP = {
    "D1": "H4",
    "H4": "H1",
    "H1": "M15",
    "M30": "M5",
}

# Global dict to store the results of the last scanning cycle per symbol
last_scan_reports = {}
last_ai_scan_times = {}
consecutive_ai_failures = {}  # per-symbol consecutive AI failure counts
last_exit_check_candle = {}  # per-symbol M5 candle time of the last AI exit-check call
last_exit_check_result = {}  # per-symbol cached AI exit-check result for that candle
last_range_check_candle = {}  # per-ticket completed-candle time of the last Range Trading invalidation check
last_range_check_result = {}  # per-ticket cached Range Trading invalidation result for that candle
last_classical_check_candle = {}  # per-ticket completed-candle time of the last Classical Patterns invalidation check
last_classical_check_result = {}  # per-ticket cached Classical Patterns invalidation result for that candle


def get_db_connection():
    import sqlite3
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn


def is_setup_stopped_out(symbol, gann_context, decision):
    """
    Checks if the last closed trade for this symbol was closed due to Stop Loss,
    and prevents re-entry. Also blocks re-entry if the current Gann setup (time_B & time_C)
    has already been traded (opened or closed) in the database.
    """
    import json
    from datetime import datetime
    
    # 1. Block re-entry on the same Gann setup
    if gann_context:
        curr_time_B = gann_context.get("time_B")
        curr_time_C = gann_context.get("time_C")
        
        if curr_time_B and curr_time_C:
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT ticket, gann_data FROM trades 
                    WHERE symbol = ?
                """, (symbol,))
                rows = cursor.fetchall()
                for row in rows:
                    ticket, db_gann_str = row
                    if db_gann_str:
                        db_gann = json.loads(db_gann_str)
                        if db_gann:
                            if db_gann.get("time_B") == curr_time_B and db_gann.get("time_C") == curr_time_C:
                                print(f"[RE-ENTRY BLOCK] Gann setup (B: {curr_time_B}, C: {curr_time_C}) already traded on ticket #{ticket}.")
                                return True
            except Exception as db_ex:
                print(f"[ERROR] Failed to query Gann setup trades: {db_ex}")
            finally:
                conn.close()

    # 2. General Cooldown / Stop Loss check for non-matching setups
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT ticket, action, profit, close_time, reason 
            FROM trades 
            WHERE symbol = ? AND status = 'CLOSED' 
            ORDER BY close_time DESC LIMIT 1
        """, (symbol,))
        row = cursor.fetchone()
    except Exception as db_ex:
        print(f"[ERROR] Failed to query last closed trade: {db_ex}")
        row = None
    finally:
        conn.close()
        
    if not row:
        return False
        
    ticket, action, profit, close_time, reason = row
    
    is_stop_loss = False
    if reason and "Hit Stop Loss" in reason:
        is_stop_loss = True
    elif profit < 0:
        is_stop_loss = True
        
    if not is_stop_loss:
        return False
        
    if action != decision:
        return False
        
    if close_time:
        try:
            close_dt = datetime.strptime(close_time, '%Y-%m-%d %H:%M:%S')
            time_diff = (datetime.now() - close_dt).total_seconds() / 3600.0
            if time_diff < 4.0:
                print(f"[RE-ENTRY BLOCK] Cooldown active for {symbol} ({decision}). Last trade #{ticket} hit SL {time_diff:.2f} hours ago.")
                return True
        except Exception as e:
            print(f"[ERROR] Calculating Stop Loss cooldown: {e}")
            
    return False


def check_m5_exit_signal(symbol, pos_type, timeframe=None):
    """
    Checks if there is a swing-reversal exit signal on `timeframe` (defaults
    to M5, its original single-timeframe design):
    - For BUY (0): returns True if a Lower Low (LL) is formed,
      with a correction separating them, and a candle Close below the previous low.
    - For SELL (1): returns True if a Higher High (HH) is formed,
      with a correction separating them, and a candle Close above the previous high.

    Despite the name (kept for compatibility — every call site still refers
    to "M5 exit"), this now runs on whatever timeframe is passed in. The
    caller resolves a check timeframe proportional to the position's own
    entry timeframe (see EXIT_CHECK_TIMEFRAME_MAP in manage_active_positions_grid)
    — a fixed M5 check made sense when every strategy entered on one shared
    default timeframe, but is far too fast/noisy for a position entered on
    an H4 or D1 setup now that every strategy can enter on any of
    scan_timeframes.
    """
    tf = timeframe if timeframe is not None else mt5.TIMEFRAME_M5
    df = get_candles(symbol=symbol, timeframe=tf, count=60)
    if df is None or len(df) < 15:
        return False

    # Find swing highs and lows with window = 2, storing (index, price)
    window = 2
    swing_highs = []
    swing_lows = []

    for i in range(window, len(df) - window):
        # Check for swing high
        is_high = True
        for j in range(i - window, i + window + 1):
            if df['High'].iloc[j] > df['High'].iloc[i]:
                is_high = False
                break
        if is_high:
            swing_highs.append((i, float(df['High'].iloc[i])))

        # Check for swing low
        is_low = True
        for j in range(i - window, i + window + 1):
            if df['Low'].iloc[j] < df['Low'].iloc[i]:
                is_low = False
                break
        if is_low:
            swing_lows.append((i, float(df['Low'].iloc[i])))

    if pos_type == 0:  # BUY trade -> Check for Lower Low (LL)
        if len(swing_lows) >= 2:
            latest_idx, latest_val = swing_lows[-1]
            prev_idx, prev_val = swing_lows[-2]
            
            # 1. Recency check (Latest low must have formed within the last 4 completed candles)
            if len(df) - 1 - latest_idx <= 4:
                # 2. Lower Low check
                if latest_val < prev_val:
                    # 3. Correction check: at least one swing high must exist between prev_idx and latest_idx
                    correction_exists = any(prev_idx < sh[0] < latest_idx for sh in swing_highs)
                    if correction_exists:
                        # 4. Candle Close Breakout check: at least one candle from prev_idx + 1 onwards must close below prev_val
                        close_breakout = False
                        for idx in range(prev_idx + 1, len(df)):
                            if float(df['Close'].iloc[idx]) < prev_val:
                                close_breakout = True
                                break
                        if close_breakout:
                            print(f"[M5 EXIT] BUY trade exit triggered: Recent Lower Low detected on M5. "
                                  f"Latest Low: {latest_val:.5f}, Prev Low: {prev_val:.5f}.")
                            return True
                            
    elif pos_type == 1:  # SELL trade -> Check for Higher High (HH)
        if len(swing_highs) >= 2:
            latest_idx, latest_val = swing_highs[-1]
            prev_idx, prev_val = swing_highs[-2]
            
            # 1. Recency check (Latest high must have formed within the last 4 completed candles)
            if len(df) - 1 - latest_idx <= 4:
                # 2. Higher High check
                if latest_val > prev_val:
                    # 3. Correction check: at least one swing low must exist between prev_idx and latest_idx
                    correction_exists = any(prev_idx < sl[0] < latest_idx for sl in swing_lows)
                    if correction_exists:
                        # 4. Candle Close Breakout check: at least one candle from prev_idx + 1 onwards must close above prev_val
                        close_breakout = False
                        for idx in range(prev_idx + 1, len(df)):
                            if float(df['Close'].iloc[idx]) > prev_val:
                                close_breakout = True
                                break
                        if close_breakout:
                            print(f"[M5 EXIT] SELL trade exit triggered: Recent Higher High detected on M5. "
                                  f"Latest High: {latest_val:.5f}, Prev High: {prev_val:.5f}.")
                            return True

    return False


def check_elliott_wave4_overlap(ticket, pos_type, current_price, pos=None):
    """
    Elliott Wave invalidation check for an open position: Wave 4 must never
    overlap Wave 1's price territory (a core, non-negotiable Elliott rule).
    Looks up the Wave 0/1/2 context stored at entry time (in the trade's
    gann_data column — reused generically, not Gann-specific) and checks
    whether price has moved back past the stored Wave 1 (B) level.

    Returns True if the count is invalidated (position should be closed now).
    """
    from db_manager import get_trade_by_ticket
    import json
    import datetime as _dt

    trade = get_trade_by_ticket(ticket)
    if not trade or not trade.get("gann_data"):
        return False

    try:
        context = json.loads(trade["gann_data"])
        wave1_price = context.get("B")
    except Exception:
        return False

    if wave1_price is None:
        return False

    symbol = trade.get("symbol")
    if not symbol:
        return False

    # Re-check on the same timeframe the setup was actually detected on (each
    # strategy now scans all of scan_timeframes, not one shared default) —
    # falls back to DEFAULT_TIMEFRAME for trades opened before this was tracked.
    position_timeframe = context.get("timeframe", DEFAULT_TIMEFRAME)

    # Wave 4 non-overlap only becomes a meaningful invalidation AFTER price has
    # actually confirmed Wave 3 by moving beyond Wave 1's extreme (B) at least
    # once since entry. Checking this from the moment of entry — while price is
    # still near Wave 2 (C), which sits between A and B by construction — would
    # flag almost every fresh trade as "invalidated" on its very first tick;
    # that's Wave 3 not having started yet, not a genuine overlap.
    try:
        open_dt = None
        if pos is not None and hasattr(pos, 'time') and pos.time > 0:
            open_dt = _dt.datetime.utcfromtimestamp(pos.time)
        else:
            try:
                positions = mt5.positions_get(ticket=ticket)
                if positions and len(positions) > 0 and positions[0].time > 0:
                    open_dt = _dt.datetime.utcfromtimestamp(positions[0].time)
            except Exception:
                pass

        if open_dt is None:
            open_time_str = trade.get("open_time")
            if open_time_str:
                open_dt = _dt.datetime.strptime(open_time_str, "%Y-%m-%d %H:%M:%S")
            else:
                return False

        candles = get_candles(symbol=symbol, timeframe=get_timeframe(position_timeframe), count=200)
        if candles is None or len(candles) == 0:
            return False
        since_entry = candles[candles['Time'] >= open_dt]
        if len(since_entry) == 0:
            return False
    except Exception:
        return False

    if pos_type == 0:  # BUY
        wave3_confirmed = max(since_entry['High'].max(), current_price) > wave1_price
        if not wave3_confirmed:
            return False
        return current_price < wave1_price
    else:  # SELL
        wave3_confirmed = min(since_entry['Low'].min(), current_price) < wave1_price
        if not wave3_confirmed:
            return False
        return current_price > wave1_price


def calculate_lot_size(symbol, sl_price, entry_price, risk_percent=1.0):
    """
    Calculate position size (lots) based on risk percentage of account balance
    and Stop Loss distance in points.
    """
    account_info = get_account_info()
    if not account_info:
        print("[WARNING] Could not fetch account info for lot size calculation. Defaulting to 0.01 lots.")
        return 0.01

    balance = account_info["balance"]
    risk_amount = balance * (risk_percent / 100.0)

    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"[WARNING] Could not get symbol info for {symbol}. Defaulting to 0.01 lots.")
        return 0.01

    point = symbol_info.point
    tick_value = symbol_info.trade_tick_value

    # Calculate SL distance in points
    sl_distance_points = abs(entry_price - sl_price) / point
    if sl_distance_points <= 0:
        print("[WARNING] Stop Loss price is identical to entry price. Defaulting to 0.01 lots.")
        return 0.01

    # Calculate lot size: lot = risk / (points * tick_value)
    lot = risk_amount / (sl_distance_points * tick_value)

    # Round to volume step
    volume_step = symbol_info.volume_step
    lot = round(lot / volume_step) * volume_step

    # Enforce volume limits
    min_volume = symbol_info.volume_min
    max_volume = symbol_info.volume_max
    
    lot = max(min_volume, min(lot, max_volume))
    lot = round(lot, 2)

    return lot


def check_technical_strategy_signals(df, symbol):
    """
    Computes technical indicators (EMA 50, EMA 200, RSI 14) and candlestick patterns,
    returning a proposed trade direction ('BUY', 'SELL', or 'HOLD') and reason details.
    """
    import pandas as pd
    
    if len(df) < 200:
        return "HOLD", f"Not enough candle data ({len(df)} candles, minimum 200 required for EMA 200)."

    # 1. Calculate EMAs
    df = df.copy()
    df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # 2. Calculate RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi14'] = 100 - (100 / (1 + rs))

    # We evaluate completed candle triggers on index -2 (completed candle)
    # df.iloc[-1] is the current open/forming candle.
    c_close = df['Close'].iloc[-2]
    c_open = df['Open'].iloc[-2]
    c_high = df['High'].iloc[-2]
    c_low = df['Low'].iloc[-2]

    prev_close = df['Close'].iloc[-3]
    prev_open = df['Open'].iloc[-3]

    # Candlestick Pattern Detection
    # Engulfing
    is_bullish_engulfing = (c_close > c_open) and (prev_close < prev_open) and (c_close >= prev_open) and (c_open <= prev_close)
    is_bearish_engulfing = (c_close < c_open) and (prev_close > prev_open) and (c_close <= prev_open) and (c_open >= prev_close)

    # Pin Bar (body in upper or lower 1/3 of the candle range, shadow is at least 50% of the range)
    candle_range = c_high - c_low
    body_size = abs(c_close - c_open)
    upper_wick = c_high - max(c_open, c_close)
    lower_wick = min(c_open, c_close) - c_low

    is_bullish_pinbar = (candle_range > 0) and (lower_wick > candle_range * 0.5) and (upper_wick < candle_range / 3) and (body_size < candle_range * 0.35)
    is_bearish_pinbar = (candle_range > 0) and (upper_wick > candle_range * 0.5) and (lower_wick < candle_range / 3) and (body_size < candle_range * 0.35)

    # 3. Pullback Confirmation: Touch of EMA 50 within the last 3 completed candles (index -2, -3, -4)
    pullback_buy = False
    pullback_sell = False
    for i in [-2, -3, -4]:
        low_val = df['Low'].iloc[i]
        high_val = df['High'].iloc[i]
        ema50_val = df['ema50'].iloc[i]
        if low_val <= ema50_val <= high_val:
            pullback_buy = True
            pullback_sell = True
            break

    # 4. RSI Range & Trend Filter
    rsi_curr = df['rsi14'].iloc[-2]
    rsi_prev = df['rsi14'].iloc[-3]

    # For BUY: RSI must be between 45 and 60 AND rising
    rsi_buy_ok = (45 <= rsi_curr <= 60) and (rsi_curr > rsi_prev)
    # For SELL: RSI must be between 40 and 55 AND falling
    rsi_sell_ok = (40 <= rsi_curr <= 55) and (rsi_curr < rsi_prev)

    # 5. Trend Rules
    macro_bullish = c_close > df['ema200'].iloc[-2]
    macro_bearish = c_close < df['ema200'].iloc[-2]

    inter_bullish = c_close > df['ema50'].iloc[-2]
    inter_bearish = c_close < df['ema50'].iloc[-2]

    # Evaluate BUY Setup
    if macro_bullish and inter_bullish and pullback_buy and (is_bullish_engulfing or is_bullish_pinbar) and rsi_buy_ok:
        pattern = "Bullish Engulfing" if is_bullish_engulfing else "Bullish Pin Bar"
        return "BUY", f"Technical BUY setup triggered via {pattern}. EMA200 uptrend confirmed. EMA50 touch confirmed. RSI 14: {rsi_curr:.1f}."

    # Evaluate SELL Setup
    elif macro_bearish and inter_bearish and pullback_sell and (is_bearish_engulfing or is_bearish_pinbar) and rsi_sell_ok:
        pattern = "Bearish Engulfing" if is_bearish_engulfing else "Bearish Pin Bar"
        return "SELL", f"Technical SELL setup triggered via {pattern}. EMA200 downtrend confirmed. EMA50 touch confirmed. RSI 14: {rsi_curr:.1f}."

    # Otherwise HOLD
    reasons = []
    if not (macro_bullish and inter_bullish) and not (macro_bearish and inter_bearish):
        reasons.append("EMA trend alignment failed")
    if not pullback_buy and not pullback_sell:
        reasons.append("EMA50 pullback touch not detected in last 3 candles")
    if not (is_bullish_engulfing or is_bullish_pinbar or is_bearish_engulfing or is_bearish_pinbar):
        reasons.append("no engulfing or pin-bar pattern")
    if not rsi_buy_ok and not rsi_sell_ok:
        reasons.append(f"RSI 14 ({rsi_curr:.1f}) out of range/trend")

    return "HOLD", "Technical Hold - " + " & ".join(reasons)


def check_sma5_reversion_signal(df, current_price, threshold_pct=0.3, rsi_overbought=65.0, rsi_oversold=35.0):
    """
    Mean-reversion signal: compares the live current price against the SMA(5)
    of the last 5 completed candles. A large enough deviation is read as price
    being "stretched" away from its short-term average, betting it reverts
    back toward that average. Confirmed with RSI(14): only take the trade if
    momentum is genuinely exhausted (overbought/oversold), not just mid-trend.

    Returns:
        (decision, reason, sma5): decision is "BUY"/"SELL"/"HOLD"; sma5 is the
        computed SMA(5) value (or None if not enough data), used as the TP target.
    """
    if len(df) < 20:
        return "HOLD", f"Not enough candle data for SMA5+RSI14 ({len(df)} candles, minimum 20 required).", None

    sma5 = df['Close'].iloc[-6:-1].mean()  # last 5 completed candles
    deviation_pct = (current_price - sma5) / sma5 * 100.0

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi14 = (100 - (100 / (1 + gain / loss))).iloc[-2]  # last completed candle

    if deviation_pct >= threshold_pct:
        if rsi14 < rsi_overbought:
            return "HOLD", f"Deviation {deviation_pct:.2f}% above SMA5 but RSI14 ({rsi14:.1f}) not overbought (>{rsi_overbought}) — momentum not exhausted.", sma5
        return "SELL", f"Price {deviation_pct:.2f}% above SMA5 ({sma5:.5f}), RSI14 {rsi14:.1f} confirms overbought — expecting reversion down.", sma5
    elif deviation_pct <= -threshold_pct:
        if rsi14 > rsi_oversold:
            return "HOLD", f"Deviation {abs(deviation_pct):.2f}% below SMA5 but RSI14 ({rsi14:.1f}) not oversold (<{rsi_oversold}) — momentum not exhausted.", sma5
        return "BUY", f"Price {abs(deviation_pct):.2f}% below SMA5 ({sma5:.5f}), RSI14 {rsi14:.1f} confirms oversold — expecting reversion up.", sma5
    else:
        return "HOLD", f"Deviation {deviation_pct:.2f}% below {threshold_pct}% threshold.", sma5


def check_range_trading_signal(df, current_price, range_info, adx_period=14, adx_threshold=25.0,
                                entry_zone_pct=0.15, rsi_overbought=65.0, rsi_oversold=35.0, atr_period=14):
    """
    Two entry modes for an already-detected consolidation range (detect_range_zone),
    checked in this order:

    1. Breakout: the last COMPLETED candle's close is beyond range_top (BUY) or
       range_bottom (SELL) — same "close beyond the level" breakout-confirmation
       convention the Gann loop already uses. No ADX/RSI gating here: a breakout
       is the market starting to trend, so "not trending"/"exhausted" filters
       would be self-contradictory.
    2. Fade (only checked if no breakout): requires ADX < adx_threshold (still
       ranging), live price within entry_zone_pct% of a boundary, and RSI(14)
       confirms exhaustion (near bottom + oversold -> BUY, near top +
       overbought -> SELL) — same RSI-gating convention as
       check_sma5_reversion_signal.

    Also computes ATR (simple rolling mean of True Range, same convention
    already used in the Gann loop's volatility filter) as a noise buffer for
    the SL.

    Returns:
        (decision, reason, context): context always includes range_top,
        range_bottom, adx (None if NaN — NaN isn't valid JSON and this gets
        stored in the trade's gann_data column), atr; "mode" ("breakout" or
        "fade") is present only when decision != "HOLD".
    """
    import pandas as pd

    range_top = range_info["range_top"]
    range_bottom = range_info["range_bottom"]
    range_width = range_top - range_bottom

    min_bars_needed = max(2 * adx_period + 5, atr_period + 5)
    if len(df) < min_bars_needed:
        return "HOLD", f"Not enough candle data for ADX/ATR warmup ({len(df)} candles, minimum {min_bars_needed}).", None

    # ATR: simple rolling mean of True Range — a noise buffer for the SL.
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_raw = tr.rolling(window=atr_period).mean().iloc[-2]  # last completed candle
    atr = None if pd.isna(atr_raw) else float(atr_raw)

    adx_series = calculate_adx(df, period=adx_period)
    adx_raw = adx_series.iloc[-2]  # last completed candle
    adx = None if pd.isna(adx_raw) else float(adx_raw)

    base_context = {"range_top": range_top, "range_bottom": range_bottom, "adx": adx, "atr": atr}

    last_close = df['Close'].iloc[-2]  # last completed candle

    # 1. Breakout mode
    if last_close > range_top:
        return "BUY", (
            f"Range Trading: last close {last_close:.5f} broke above range top {range_top:.5f} "
            f"(range {range_bottom:.5f}-{range_top:.5f})."
        ), {**base_context, "mode": "breakout"}
    if last_close < range_bottom:
        return "SELL", (
            f"Range Trading: last close {last_close:.5f} broke below range bottom {range_bottom:.5f} "
            f"(range {range_bottom:.5f}-{range_top:.5f})."
        ), {**base_context, "mode": "breakout"}

    # 2. Fade mode — requires a valid (non-warmup) ADX confirming the range is still intact
    if adx is None:
        return "HOLD", "ADX not yet available (warmup) — treating as HOLD rather than risking a bad fade entry.", base_context
    if adx >= adx_threshold:
        return "HOLD", f"ADX ({adx:.1f}) >= threshold ({adx_threshold}) — market trending, not a safe range to fade.", base_context
    if range_width <= 0:
        return "HOLD", "Range width is zero or negative — invalid range.", base_context

    dist_from_bottom_pct = abs(current_price - range_bottom) / range_bottom * 100.0
    dist_from_top_pct = abs(current_price - range_top) / range_top * 100.0

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi14 = (100 - (100 / (1 + gain / loss))).iloc[-2]

    if dist_from_bottom_pct <= entry_zone_pct:
        if pd.isna(rsi14) or rsi14 > rsi_oversold:
            rsi_str = f"{rsi14:.1f}" if not pd.isna(rsi14) else "N/A"
            return "HOLD", f"Price near range bottom but RSI14 ({rsi_str}) not oversold (<{rsi_oversold}) — momentum not exhausted.", base_context
        return "BUY", (
            f"Range Trading: price within {entry_zone_pct}% of range bottom ({range_bottom:.5f}), "
            f"ADX {adx:.1f} confirms ranging, RSI14 {rsi14:.1f} confirms oversold."
        ), {**base_context, "mode": "fade"}

    if dist_from_top_pct <= entry_zone_pct:
        if pd.isna(rsi14) or rsi14 < rsi_overbought:
            rsi_str = f"{rsi14:.1f}" if not pd.isna(rsi14) else "N/A"
            return "HOLD", f"Price near range top but RSI14 ({rsi_str}) not overbought (>{rsi_overbought}) — momentum not exhausted.", base_context
        return "SELL", (
            f"Range Trading: price within {entry_zone_pct}% of range top ({range_top:.5f}), "
            f"ADX {adx:.1f} confirms ranging, RSI14 {rsi14:.1f} confirms overbought."
        ), {**base_context, "mode": "fade"}

    return "HOLD", (
        f"Price mid-range ({dist_from_bottom_pct:.2f}% from bottom, {dist_from_top_pct:.2f}% from top) — "
        f"no breakout, not near a boundary for fade."
    ), base_context


def check_range_trading_invalidation(symbol, ticket, mode, range_top, range_bottom, adx_exit_threshold, adx_period=14, timeframe_str=None):
    """
    Post-entry invalidation check for a Range Trading position, cached per
    completed candle — ADX/price only change on a new bar, so recomputing on
    every ~30s management tick is pure waste (same idiom as the AI M5-exit
    cache: last_exit_check_candle/last_exit_check_result above, keyed here
    per-ticket instead of per-symbol since a symbol could in principle hold
    both a fade and a breakout Range position at once).

    Fade-mode: if ADX has risen above adx_exit_threshold (higher than the
    entry adx_threshold), the "market is still ranging" premise behind the
    fade is now false — invalidate. Gated to fade-mode positions only: a
    rising ADX during a breakout trade is confirmation, not invalidation.
    Breakout-mode: if a completed candle closes back INSIDE the range, the
    breakout failed — invalidate immediately rather than waiting for the far
    (opposite-side) SL.

    Returns True if the position should be closed now.
    """
    import pandas as pd
    global last_range_check_candle, last_range_check_result

    # Re-check on the same timeframe the range was actually detected on (each
    # strategy now scans all of scan_timeframes, not one shared default) —
    # falls back to DEFAULT_TIMEFRAME for trades opened before this was tracked.
    resolved_timeframe = timeframe_str or DEFAULT_TIMEFRAME
    df = get_candles(symbol=symbol, timeframe=get_timeframe(resolved_timeframe), count=max(adx_period * 3, 50))
    if df is None or len(df) < 20:
        return False
    df = df.reset_index(drop=True)

    current_candle_time = str(df['Time'].iloc[-1])
    cache_key = f"{ticket}_{mode}"
    if last_range_check_candle.get(cache_key) == current_candle_time:
        return last_range_check_result.get(cache_key, False)

    invalidated = False
    if mode == "fade":
        adx_raw = calculate_adx(df, period=adx_period).iloc[-2]
        if not pd.isna(adx_raw) and float(adx_raw) > adx_exit_threshold:
            invalidated = True
    elif mode == "breakout":
        last_close = df['Close'].iloc[-2]
        if range_bottom < last_close < range_top:
            invalidated = True

    last_range_check_candle[cache_key] = current_candle_time
    last_range_check_result[cache_key] = invalidated
    return invalidated


def check_harmonic_pattern_signal(df, current_price, pattern_info, entry_zone_pct=0.15,
                                   rsi_overbought=65.0, rsi_oversold=35.0, atr_period=14):
    """
    Given an already-detected X-A-B-C-D harmonic pattern (detect_harmonic_pattern),
    checks whether the current live price has reached the projected PRZ
    (Potential Reversal Zone) and confirms exhaustion via RSI(14) — the
    "D-completion + RSI confirmation" convention decided for this strategy,
    matching check_sma5_reversion_signal / Range Trading's fade-mode gating.

    Also computes ATR (same simple rolling True Range convention already
    used by Range Trading) as the SL noise buffer beyond point X.

    Returns:
        (decision, reason, context): context includes pattern, X, A, B, C,
        prz_price, atr (None if NaN — NaN isn't valid JSON and this gets
        stored in the trade's gann_data column).
    """
    import pandas as pd

    prz_price = pattern_info["prz_price"]
    pattern_type = pattern_info["type"]  # "BUY" or "SELL"
    pattern_name = pattern_info["pattern"]

    min_bars_needed = atr_period + 5
    if len(df) < min_bars_needed:
        return "HOLD", f"Not enough candle data for ATR warmup ({len(df)} candles, minimum {min_bars_needed}).", None

    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_raw = tr.rolling(window=atr_period).mean().iloc[-2]  # last completed candle
    atr = None if pd.isna(atr_raw) else float(atr_raw)

    base_context = {
        "pattern": pattern_name, "X": pattern_info["X"], "A": pattern_info["A"],
        "B": pattern_info["B"], "C": pattern_info["C"], "prz_price": prz_price, "atr": atr
    }

    # Entry trigger: current live price has reached the PRZ — D is completing right now.
    if prz_price == 0:
        return "HOLD", "PRZ price is zero — invalid projection.", base_context
    dist_from_prz_pct = abs(current_price - prz_price) / prz_price * 100.0
    if dist_from_prz_pct > entry_zone_pct:
        return "HOLD", f"Price not yet within {entry_zone_pct}% of the {pattern_name} PRZ ({prz_price:.5f}) — D hasn't completed yet.", base_context

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi14 = (100 - (100 / (1 + gain / loss))).iloc[-2]  # last completed candle

    if pd.isna(rsi14):
        return "HOLD", "RSI14 not yet available (warmup).", base_context

    if pattern_type == "BUY":
        if rsi14 > rsi_oversold:
            return "HOLD", f"Price at {pattern_name} PRZ but RSI14 ({rsi14:.1f}) not oversold (<{rsi_oversold}) — momentum not exhausted.", base_context
        return "BUY", (
            f"Harmonic {pattern_name}: D completing at PRZ {prz_price:.5f} "
            f"(X={pattern_info['X']:.5f}, A={pattern_info['A']:.5f}, B={pattern_info['B']:.5f}, C={pattern_info['C']:.5f}), "
            f"RSI14 {rsi14:.1f} confirms oversold."
        ), base_context
    else:
        if rsi14 < rsi_overbought:
            return "HOLD", f"Price at {pattern_name} PRZ but RSI14 ({rsi14:.1f}) not overbought (>{rsi_overbought}) — momentum not exhausted.", base_context
        return "SELL", (
            f"Harmonic {pattern_name}: D completing at PRZ {prz_price:.5f} "
            f"(X={pattern_info['X']:.5f}, A={pattern_info['A']:.5f}, B={pattern_info['B']:.5f}, C={pattern_info['C']:.5f}), "
            f"RSI14 {rsi14:.1f} confirms overbought."
        ), base_context


def check_classical_pattern_signal(df, pattern_info, atr_period=14):
    """
    Given an already-detected classical chart pattern — continuation
    (detect_continuation_pattern: Rectangle/Triangle-variants/Wedge-variants/
    Pennant/Flag) or reversal (detect_reversal_pattern: Head & Shoulders +
    inverse, Double Top/Bottom) — checks whether the last completed candle's
    close has broken beyond the pattern's defining level (breakout_level for
    continuation, neckline_level for reversal) — the same "close beyond the
    level" breakout-confirmation convention the Gann loop already uses. No
    RSI gate: a breakout is the market starting to move, so an exhaustion
    filter would be self-contradictory (same principle as Gann/Range
    Trading's breakout mode).

    Also computes ATR (same convention as Range Trading/Harmonic Patterns)
    as the SL noise buffer.

    Returns:
        (decision, reason, context): context includes pattern, type, level
        (the breakout/neckline price), atr (None if NaN).
    """
    import pandas as pd

    pattern_name = pattern_info["pattern"]
    pattern_type = pattern_info["type"]
    is_continuation = "breakout_level" in pattern_info
    level = pattern_info["breakout_level"] if is_continuation else pattern_info["neckline_level"]
    level_label = "breakout level" if is_continuation else "neckline"

    min_bars_needed = atr_period + 5
    if len(df) < min_bars_needed:
        return "HOLD", f"Not enough candle data for ATR warmup ({len(df)} candles, minimum {min_bars_needed}).", None

    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_raw = tr.rolling(window=atr_period).mean().iloc[-2]  # last completed candle
    atr = None if pd.isna(atr_raw) else float(atr_raw)

    base_context = {"pattern": pattern_name, "type": pattern_type, "level": level, "atr": atr}

    last_close = df['Close'].iloc[-2]  # last completed candle

    if pattern_type == "BUY" and last_close > level:
        return "BUY", f"Classical {pattern_name}: last close {last_close:.5f} broke above the {level_label} {level:.5f}.", base_context
    if pattern_type == "SELL" and last_close < level:
        return "SELL", f"Classical {pattern_name}: last close {last_close:.5f} broke below the {level_label} {level:.5f}.", base_context
    return "HOLD", f"Price has not yet broken the {pattern_name} {level_label} ({level:.5f}).", base_context


def check_classical_pattern_invalidation(symbol, ticket, pos_type, level, timeframe_str=None):
    """
    Post-entry invalidation check for a Classical Patterns position, cached
    per completed candle — same idiom as check_range_trading_invalidation
    (level/price only changes on a new bar, so recomputing every ~30s
    management tick is pure waste).

    If a completed candle closes back on the wrong side of the breakout/
    neckline level, the breakout failed — invalidate now rather than wait
    for the far SL (the opposite trendline or the pattern's head/peak, which
    sits much further away than this level).

    Returns True if the position should be closed now.
    """
    global last_classical_check_candle, last_classical_check_result

    # Re-check on the same timeframe the pattern was actually detected on (each
    # strategy now scans all of scan_timeframes, not one shared default) —
    # falls back to DEFAULT_TIMEFRAME for trades opened before this was tracked.
    resolved_timeframe = timeframe_str or DEFAULT_TIMEFRAME
    df = get_candles(symbol=symbol, timeframe=get_timeframe(resolved_timeframe), count=20)
    if df is None or len(df) < 3:
        return False

    current_candle_time = str(df['Time'].iloc[-1])
    cache_key = f"{ticket}"
    if last_classical_check_candle.get(cache_key) == current_candle_time:
        return last_classical_check_result.get(cache_key, False)

    last_close = df['Close'].iloc[-2]
    if pos_type == "BUY":
        invalidated = last_close < level
    else:
        invalidated = last_close > level

    last_classical_check_candle[cache_key] = current_candle_time
    last_classical_check_result[cache_key] = invalidated
    return invalidated


def sync_db_with_mt5_positions():
    """
    Sync open positions from MT5 to SQLite database.
    Detects if any trades were closed outside the bot (e.g., hit SL/TP or closed manually)
    and logs them as closed in the SQLite database.
    """
    # 1. Fetch live open positions from MT5
    mt5_positions = get_open_positions()
    mt5_tickets = [pos.ticket for pos in mt5_positions]

    # 2. Fetch what our database thinks are open positions
    db_open_trades = get_active_trades()

    # 3. For any position in DB that is no longer in MT5, it must have been closed
    for db_trade in db_open_trades:
        ticket = db_trade["ticket"]
        if ticket not in mt5_tickets:
            # Trade was closed! Let's find details from MT5 history
            close_price = db_trade["entry_price"]
            profit = 0.0
            
            # Fetch from MT5 history to get actual closing stats
            history_deals = mt5.history_deals_get(position=ticket)
            close_comment = ""
            close_reason_code = 0
            if history_deals and len(history_deals) > 0:
                # Find the deal that closed the position (usually DEAL_ENTRY_OUT)
                for deal in history_deals:
                    if deal.entry == mt5.DEAL_ENTRY_OUT or deal.position_id == ticket:
                        # Use deal price and profit
                        close_price = deal.price
                        profit = deal.profit
                        close_comment = getattr(deal, 'comment', '')
                        close_reason_code = getattr(deal, 'reason', 0)
            
            exit_reason = "Unknown Close (إغلاق لسبب غير معروف)"
            if close_comment == "Gann Reversal Close":
                exit_reason = "Gann Reversal Close (إغلاق بسبب انعكاس الاتجاه)"
            elif close_comment == "Manual Web Close":
                exit_reason = "Manual Web Close (إغلاق يدوي من لوحة التحكم للويب)"
            elif close_reason_code == 3:
                exit_reason = "Hit Stop Loss (ضرب الاستوب)"
            elif close_reason_code == 4:
                exit_reason = "Hit Take Profit (ضرب الهدف)"
            elif close_reason_code == 0:
                exit_reason = "Manual MT5 Close (إغلاق يدوي مباشر من تطبيق/منصة MT5)"
            elif close_comment:
                exit_reason = f"Closed: {close_comment}"

            # Log as closed in database
            log_trade_close(ticket, close_price, profit)
            print(f"[SYNC] Closed position ticket {ticket} detected and logged in DB (Profit: ${profit})")
            
            # Send Telegram closed alert and update screenshot
            result = "WIN" if profit >= 0 else "LOSS"
            close_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
            screenshot_url = None
            try:
                from chart_generator import generate_chart_screenshot
                from db_manager import update_trade_screenshot
                import json
                
                gann_context = json.loads(db_trade["gann_data"]) if db_trade.get("gann_data") else None
                
                screenshot_url = generate_chart_screenshot(
                    symbol=db_trade["symbol"],
                    ticket=ticket,
                    entry_price=db_trade["entry_price"],
                    sl_price=db_trade["sl"],
                    tp_price=db_trade["tp"],
                    gann_data=gann_context,
                    entry_time_str=db_trade["open_time"],
                    exit_time_str=close_time_str,
                    exit_price=close_price,
                    result=result,
                    profit_usd=profit
                )
                update_trade_screenshot(ticket, screenshot_url)
                print(f"[SCREENSHOT] Updated closed trade chart to {screenshot_url}")
            except Exception as ex:
                print(f"[SCREENSHOT] [ERROR] Failed to update closed screenshot: {ex}")
                
            try:
                from telegram_notifier import notify_trade_close
                notify_trade_close(
                    ticket=ticket,
                    symbol=db_trade["symbol"],
                    action=db_trade["action"],
                    volume=db_trade["volume"],
                    entry_price=db_trade["entry_price"],
                    close_price=close_price,
                    profit=profit,
                    result=result,
                    open_time=db_trade["open_time"],
                    close_time=close_time_str,
                    screenshot_url=screenshot_url,
                    original_msg_id=db_trade.get("telegram_msg_id"),
                    exit_reason=exit_reason
                )
                print(f"[TELEGRAM] Closed notification sent for ticket #{ticket}")
            except Exception as tg_ex:
                print(f"[TELEGRAM] [ERROR] Failed to send closed alert: {tg_ex}")


def manage_active_positions_grid():
    """
    Monitors active positions in MetaTrader 5.
    If a position goes into drawdown by grid_step, opens the next Martingale grid leg
    and modifies the TP/SL of all active legs of that symbol.
    Also handles general trade protections (50% Breakeven, 90% Early Close) regardless of grid settings.
    """
    global last_exit_check_candle, last_exit_check_result
    settings = get_settings()
    grid_enabled = int(settings.get("grid_enabled", 1)) == 1

    grid_step = float(settings.get("grid_step", 10.0))
    multiplier = float(settings.get("grid_multiplier", 2.0))
    max_legs = int(settings.get("grid_max_legs", 4))
    target_profit = float(settings.get("grid_target_profit", 2.0))
    grid_sl = float(settings.get("grid_sl", 20.0))

    # Fetch active positions from MT5
    active_positions = get_open_positions()
    if not active_positions:
        return

    # Group positions by symbol
    from collections import defaultdict
    grouped = defaultdict(list)
    for pos in active_positions:
        if pos.magic == MAGIC_NUMBER:
            grouped[pos.symbol].append(pos)

    for symbol, pos_list in grouped.items():
        if len(pos_list) == 0:
            continue

        # Sort positions by open time
        pos_list = sorted(pos_list, key=lambda p: p.time)
        basket_type = 'BUY' if pos_list[0].type == 0 else 'SELL'
        
        # Get current price info
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            continue
            
        current_price = tick.bid if basket_type == 'BUY' else tick.ask
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            continue
            
        point = symbol_info.point
        contract_size = symbol_info.trade_contract_size
        if contract_size < 1000.0:
            pip_size = (0.01 * current_price) / 10.0
            pip_multiplier = 1.0 / pip_size
        else:
            is_jpy = symbol.upper().endswith("JPY") or "JPY" in symbol.upper()
            pip_size = 0.01 if is_jpy else 0.0001
            pip_multiplier = 100.0 if is_jpy else 10000.0
        
        # Find drawdown from the last opened leg
        last_leg = pos_list[-1]
        last_entry = last_leg.price_open
        
        if basket_type == 'BUY':
            drawdown_pips = (last_entry - tick.bid) * pip_multiplier
        else:
            drawdown_pips = (tick.ask - last_entry) * pip_multiplier

        print(f"[PROTECTION INFO] Symbol {symbol} has {len(pos_list)} active trades. Current profit/drawdown pips from last trade: {-drawdown_pips:.1f} pips.")

        # Check if we need to open the next leg (ONLY if Grid is explicitly enabled, and not a
        # Gann, SMA5-reversion, Elliott Wave, Range Trading, Harmonic Patterns, or Classical
        # Patterns trade — all six manage their own SL/TP and shouldn't get martingale legs)
        is_gann_trade = any("gann" in (pos.comment or "").lower() for pos in pos_list)
        is_sma5_trade = any("sma5" in (pos.comment or "").lower() for pos in pos_list)
        is_elliott_trade = any("elliott" in (pos.comment or "").lower() for pos in pos_list)
        is_range_trade = any("range" in (pos.comment or "").lower() for pos in pos_list)
        is_harmonic_trade = any("harmonic" in (pos.comment or "").lower() for pos in pos_list)
        is_classical_trade = any("classical" in (pos.comment or "").lower() for pos in pos_list)
        if grid_enabled and not is_gann_trade and not is_sma5_trade and not is_elliott_trade and not is_range_trade and not is_harmonic_trade and not is_classical_trade and drawdown_pips >= grid_step:
            if len(pos_list) < max_legs:
                print(f"[GRID TRIGGER] Drawdown ({drawdown_pips:.1f} pips) >= Grid Step ({grid_step} pips). Opening next leg!")
                
                # Sizing: last position lot size * multiplier
                next_lot = round(last_leg.volume * multiplier, 2)
                next_lot = max(symbol_info.volume_min, min(next_lot, symbol_info.volume_max))
                next_lot = round(next_lot, 2)
                
                # Execute Trade leg
                trade_res = open_trade(
                    action=basket_type,
                    symbol=symbol,
                    volume=next_lot,
                    sl=0.0,
                    tp=0.0,
                    magic=MAGIC_NUMBER,
                    comment=f"Leg {len(pos_list)+1} Recovery"[:28]
                )
                
                if trade_res and trade_res["success"]:
                    log_trade_open(
                        ticket=trade_res["ticket"],
                        symbol=symbol,
                        action=basket_type,
                        volume=next_lot,
                        entry_price=trade_res["price"],
                        sl=0.0,
                        tp=0.0,
                        reason=f"Martingale Leg {len(pos_list)+1} added"
                    )
                    
                    time.sleep(0.5)
                    pos_list = get_open_positions(symbol=symbol)
                    pos_list = [p for p in pos_list if p.magic == MAGIC_NUMBER]
                    pos_list = sorted(pos_list, key=lambda p: p.time)
            else:
                print(f"[GRID INFO] Drawdown is {drawdown_pips:.1f} pips but Max Legs ({max_legs}) reached. Waiting.")

        # Recalculate average entry price, TP and SL for the whole basket
        if len(pos_list) > 0:
            total_vol = sum(p.volume for p in pos_list)
            weighted_entry = sum(p.price_open * p.volume for p in pos_list) / total_vol
            
            is_gann_trade = any("gann" in (pos.comment or "").lower() for pos in pos_list)
            is_sma5_trade = any("sma5" in (pos.comment or "").lower() for pos in pos_list)
            is_elliott_trade = any("elliott" in (pos.comment or "").lower() for pos in pos_list)
            is_range_trade = any("range" in (pos.comment or "").lower() for pos in pos_list)
            is_harmonic_trade = any("harmonic" in (pos.comment or "").lower() for pos in pos_list)
            is_classical_trade = any("classical" in (pos.comment or "").lower() for pos in pos_list)
            preserve_original_tp_sl = is_gann_trade or is_sma5_trade or is_elliott_trade or is_range_trade or is_harmonic_trade or is_classical_trade
            strategy_label = "Gann" if is_gann_trade else ("SMA5 reversion" if is_sma5_trade else ("Elliott Wave" if is_elliott_trade else ("Range Trading" if is_range_trade else ("Harmonic Patterns" if is_harmonic_trade else "Classical Patterns"))))

            if preserve_original_tp_sl and pos_list[0].tp > 0:
                tp_price = pos_list[0].tp
                print(f"[GRID INFO] {strategy_label} trade detected. Preserving oldest TP price: {tp_price}")
            else:
                if len(pos_list) == 1:
                    grid_tp = float(settings.get("grid_tp", 20.0))
                    tp_price = weighted_entry + (grid_tp * pip_size) if basket_type == 'BUY' else weighted_entry - (grid_tp * pip_size)
                else:
                    tp_price = weighted_entry + (target_profit * pip_size) if basket_type == 'BUY' else weighted_entry - (target_profit * pip_size)
                
            last_entry_price = pos_list[-1].price_open
            
            if preserve_original_tp_sl and pos_list[0].sl > 0:
                sl_price = pos_list[0].sl
                print(f"[GRID INFO] {strategy_label} trade detected. Preserving oldest SL price: {sl_price}")
            else:
                sl_price = last_entry_price - (grid_sl * pip_size) if basket_type == 'BUY' else last_entry_price + (grid_sl * pip_size)
            
            # --- APPLY GRID PROTECTIONS ---
            # 1. Calculate percentage distance reached towards TP
            tp_dist = abs(tp_price - weighted_entry)
            current_dist = (current_price - weighted_entry) if basket_type == 'BUY' else (weighted_entry - current_price)
            pct_reached = current_dist / tp_dist if tp_dist > 0 else 0.0
            
            # 2. Check 90% Early Close
            if pct_reached >= 0.90:
                print(f"[TP CLOSE EARLY] Basket for {symbol} reached {pct_reached*100:.1f}% of TP. Closing basket early.")
                for pos in pos_list:
                    close_res = close_position(ticket=pos.ticket, comment="TP 90% Close Early")
                    if close_res:
                        log_trade_close(pos.ticket, current_price, pos.profit, exit_reason="Take Profit 90% Close Early")
                continue

            # 2.5 Elliott Wave invalidation check: Wave 4 must never overlap Wave 1's
            # price territory. If price has moved back past the stored Wave 1 (B)
            # level, the wave count is proven wrong — exit now instead of waiting
            # for the fixed SL (which sits further back at Wave 0/A).
            if is_elliott_trade:
                any_closed = False
                for pos in pos_list:
                    if "elliott" not in (pos.comment or "").lower():
                        continue
                    if check_elliott_wave4_overlap(pos.ticket, 0 if basket_type == 'BUY' else 1, current_price, pos=pos):
                        print(f"[ELLIOTT INVALIDATION] Wave 4 overlap detected on {symbol} (ticket #{pos.ticket}). Closing — wave count invalidated.")
                        close_res = close_position(ticket=pos.ticket, comment="Elliott Wave 4 Overlap")
                        if close_res:
                            log_trade_close(pos.ticket, current_price, pos.profit, exit_reason="Elliott Wave 4 overlap invalidation")
                            any_closed = True
                if any_closed:
                    continue

            # 2.6 Range Trading invalidation checks: fade-mode ADX regime-break
            # (market started trending, the "still ranging" premise behind the
            # fade is now false) and breakout-mode failed-breakout (a completed
            # candle closed back inside the range). Both are cached per
            # completed candle inside check_range_trading_invalidation.
            if is_range_trade:
                import json
                from db_manager import get_trade_by_ticket

                any_closed = False
                range_adx_period = int(settings.get("range_trading_adx_period", 14))
                range_adx_exit_threshold = float(settings.get("range_trading_adx_exit_threshold", 30.0))
                for pos in pos_list:
                    if "range" not in (pos.comment or "").lower():
                        continue
                    trade_row = get_trade_by_ticket(pos.ticket)
                    if not trade_row or not trade_row.get("gann_data"):
                        continue
                    try:
                        range_context = json.loads(trade_row["gann_data"])
                    except Exception:
                        continue
                    pos_mode = range_context.get("mode")
                    pos_range_top = range_context.get("range_top")
                    pos_range_bottom = range_context.get("range_bottom")
                    pos_timeframe = range_context.get("timeframe")
                    if pos_mode is None or pos_range_top is None or pos_range_bottom is None:
                        continue
                    if check_range_trading_invalidation(symbol, pos.ticket, pos_mode, pos_range_top, pos_range_bottom,
                                                         range_adx_exit_threshold, adx_period=range_adx_period,
                                                         timeframe_str=pos_timeframe):
                        reason_label = "ADX regime-break (no longer ranging)" if pos_mode == "fade" else "Failed breakout (closed back inside range)"
                        print(f"[RANGE INVALIDATION] {reason_label} on {symbol} (ticket #{pos.ticket}). Closing.")
                        close_res = close_position(ticket=pos.ticket, comment="Range Trading Invalidation")
                        if close_res:
                            log_trade_close(pos.ticket, current_price, pos.profit, exit_reason=f"Range Trading invalidation: {reason_label}")
                            any_closed = True
                if any_closed:
                    continue

            # 2.7 Classical Patterns invalidation check: if a completed candle
            # closes back on the wrong side of the breakout/neckline level, the
            # breakout failed — close now rather than wait for the far SL (the
            # opposite trendline or the pattern's head/peak).
            if is_classical_trade:
                import json
                from db_manager import get_trade_by_ticket

                any_closed = False
                for pos in pos_list:
                    if "classical" not in (pos.comment or "").lower():
                        continue
                    trade_row = get_trade_by_ticket(pos.ticket)
                    if not trade_row or not trade_row.get("gann_data"):
                        continue
                    try:
                        classical_context = json.loads(trade_row["gann_data"])
                    except Exception:
                        continue
                    pos_type = classical_context.get("type")
                    pos_level = classical_context.get("level")
                    pos_timeframe = classical_context.get("timeframe")
                    if pos_type is None or pos_level is None:
                        continue
                    if check_classical_pattern_invalidation(symbol, pos.ticket, pos_type, pos_level, timeframe_str=pos_timeframe):
                        print(f"[CLASSICAL INVALIDATION] Failed breakout on {symbol} (ticket #{pos.ticket}). Closing.")
                        close_res = close_position(ticket=pos.ticket, comment="Classical Pattern Invalidation")
                        if close_res:
                            log_trade_close(pos.ticket, current_price, pos.profit, exit_reason="Classical pattern failed breakout invalidation")
                            any_closed = True
                if any_closed:
                    continue

            # 2.8 SMA5 Reversion: Dynamic Trailing TP to SMA5 (with spread adjustment) & Entry Point Exit
            if is_sma5_trade:
                import json
                from db_manager import get_trade_by_ticket
                from mt5_orders import modify_position_sl_tp

                any_closed = False
                for pos in pos_list:
                    if "sma5" not in (pos.comment or "").lower():
                        continue

                    trade_row = get_trade_by_ticket(pos.ticket)
                    pos_tf = DEFAULT_TIMEFRAME
                    if trade_row and trade_row.get("gann_data"):
                        try:
                            gdata = json.loads(trade_row["gann_data"])
                            pos_tf = gdata.get("timeframe", DEFAULT_TIMEFRAME)
                        except Exception:
                            pass

                    candles_df = get_candles(symbol=symbol, timeframe=get_timeframe(pos_tf), count=30)
                    if candles_df is None or len(candles_df) < 5:
                        continue

                    price_info = get_current_price(symbol)
                    if not price_info:
                        continue

                    # Current SMA5 of last 5 completed candles
                    current_sma5 = candles_df['Close'].iloc[-6:-1].mean()
                    spread = price_info['ask'] - price_info['bid']
                    symbol_info = mt5.symbol_info(symbol)
                    digits = symbol_info.digits if symbol_info else 5
                    point = symbol_info.point if symbol_info else (10 ** -digits)

                    # Dynamic TP adjustment considering spread:
                    # BUY: closes at Bid -> target is current_sma5
                    # SELL: closes at Ask -> target is current_sma5 + spread
                    if pos.type == 0:  # BUY
                        target_tp = round(current_sma5, digits)
                    else:  # SELL
                        target_tp = round(current_sma5 + spread, digits)

                    # Update MT5 order TP if SMA5 shifted by more than 1 point
                    if abs(pos.tp - target_tp) >= point:
                        print(f"[SMA5 TRAILING TP] Symbol {symbol} (ticket #{pos.ticket}) SMA5 moved. Updating TP from {pos.tp} to {target_tp}")
                        modify_position_sl_tp(pos.ticket, pos.sl, target_tp)

                    # 3. Manual target reach check (fallback if MT5 TP hasn't executed yet)
                    target_reached = (current_price >= current_sma5) if is_buy else (current_price <= current_sma5)

                    # Entry Point Exit (Break-even / Return to Entry):
                    entry_price = pos.price_open
                    import datetime as _dt
                    open_dt = _dt.datetime.utcfromtimestamp(pos.time)
                    since_entry = candles_df[candles_df['Time'] >= open_dt]

                    went_into_profit = False
                    if len(since_entry) > 0:
                        if is_buy:
                            went_into_profit = since_entry['High'].max() > (entry_price + 2 * point)
                        else:
                            went_into_profit = since_entry['Low'].min() < (entry_price - 2 * point)

                    # 1. Target SMA5 reached or crossed entry price (reversion potential gone)
                    # 2. Price moved towards SMA5 target and returned to entry price
                    sma5_crossed_entry = (current_sma5 <= entry_price) if is_buy else (current_sma5 >= entry_price)
                    returned_to_entry = went_into_profit and ((current_price <= entry_price) if is_buy else (current_price >= entry_price))

                    if target_reached or sma5_crossed_entry or returned_to_entry:
                        if target_reached:
                            reason_msg = f"Price reached SMA5 target ({current_sma5:.5f})"
                        elif sma5_crossed_entry:
                            reason_msg = "SMA5 target reached entry price"
                        else:
                            reason_msg = "Price returned to entry point (Break-even)"

                        print(f"[SMA5 EXIT] {reason_msg} on {symbol} (ticket #{pos.ticket}). Closing position.")
                        close_res = close_position(ticket=pos.ticket, comment="SMA5 Reversion Exit")
                        if close_res:
                            log_trade_close(pos.ticket, current_price, pos.profit, exit_reason=f"SMA5 Reversion exit: {reason_msg}")
                            any_closed = True

                if any_closed:
                    continue

            # 3. Check Swing Exit — on a timeframe proportional to the position's
            # own entry timeframe (stored in gann_data by every strategy), not a
            # fixed M5 regardless of whether this basket is an M30 scalp or a D1
            # swing trade. Falls back to M5 if no stored timeframe is found
            # (e.g. a Gann grid-recovery leg, which doesn't store gann_data).
            entry_timeframe_str = None
            try:
                import json
                from db_manager import get_trade_by_ticket
                first_trade_row = get_trade_by_ticket(pos_list[0].ticket)
                if first_trade_row and first_trade_row.get("gann_data"):
                    entry_timeframe_str = json.loads(first_trade_row["gann_data"]).get("timeframe")
            except Exception:
                entry_timeframe_str = None
            exit_check_timeframe_str = EXIT_CHECK_TIMEFRAME_MAP.get(entry_timeframe_str, "M5")
            exit_check_tf = get_timeframe(exit_check_timeframe_str) or mt5.TIMEFRAME_M5

            if check_m5_exit_signal(symbol, 0 if basket_type == 'BUY' else 1, timeframe=exit_check_tf):
                # Call AI to verify the exit signal, but only once per candle to avoid
                # re-querying the AI on every 30s position-management tick for the same signal.
                exit_cache_key = f"{symbol}_{basket_type}"
                ai_exit_confirmed = False
                try:
                    m5_df = get_candles(symbol=symbol, timeframe=exit_check_tf, count=15)
                    if m5_df is not None and len(m5_df) >= 5:
                        current_m5_candle_time = str(m5_df['Time'].iloc[-1])
                        if last_exit_check_candle.get(exit_cache_key) == current_m5_candle_time:
                            ai_exit_confirmed = last_exit_check_result.get(exit_cache_key, False)
                            print(f"[AI EXIT CHECK] Reusing cached decision for {symbol} on candle {current_m5_candle_time}: "
                                  f"{'EXIT' if ai_exit_confirmed else 'HOLD'}")
                        else:
                            m5_candles_text = format_candles_for_ai(m5_df, last_n=15)
                            engine = AITradingEngine(model_name="gemini-2.5-flash")
                            ai_exit_confirmed = engine.analyze_exit(symbol, 0 if basket_type == 'BUY' else 1, m5_candles_text)
                            last_exit_check_candle[exit_cache_key] = current_m5_candle_time
                            last_exit_check_result[exit_cache_key] = ai_exit_confirmed
                    else:
                        print(f"[AI EXIT CHECK] Not enough {exit_check_timeframe_str} candles to verify. Skipping AI check.")
                except Exception as ai_ex:
                    print(f"[AI EXIT CHECK] [ERROR] Failed to run AI exit verification: {ai_ex}")

                if ai_exit_confirmed:
                    exit_reason_ar = f"تأكيد انعكاس الاتجاه على {exit_check_timeframe_str} بواسطة الذكاء الاصطناعي"
                    exit_reason_en = f"AI Verified {exit_check_timeframe_str} Exit"
                    print(f"[SWING EXIT] AI CONFIRMED exit for {symbol} on {exit_check_timeframe_str}. Closing basket.")
                    for pos in pos_list:
                        close_res = close_position(ticket=pos.ticket, comment=exit_reason_en[:28])
                        if close_res:
                            log_trade_close(pos.ticket, current_price, pos.profit, exit_reason=f"{exit_reason_en} ({exit_reason_ar})")
                    continue
                else:
                    print(f"[SWING EXIT BLOCK] {exit_check_timeframe_str} exit trigger detected, but AI rejected it as noise. Keeping position open.")
                
            # 4. Check 50% Breakeven (Move SL to weighted average entry price)
            if pct_reached >= 0.50:
                sl_price = weighted_entry
                print(f"[BREAKEVEN] Basket for {symbol} reached {pct_reached*100:.1f}% of TP. Moving SL to Entry Breakeven ({sl_price}).")
                
            digits = symbol_info.digits
            tp_price = round(tp_price, digits)
            sl_price = round(sl_price, digits)
            
            for pos in pos_list:
                if abs(pos.sl - sl_price) > 0.5 * point or abs(pos.tp - tp_price) > 0.5 * point:
                    print(f"[GRID UPDATE] Modifying SL/TP for position #{pos.ticket}. New SL: {sl_price}, New TP: {tp_price}")
                    modify_position_sl_tp(pos.ticket, sl_price, tp_price)


def manage_sma5_reversion_strategy(settings, active_news_events, risk_percent, auto_trade_enabled):
    """
    Independent, parallel strategy: bets on price reverting to its SMA(5) once
    it strays far enough away. Runs its own signal check, pre-trade filters,
    and SL/TP/lot-size pipeline — deliberately decoupled from the Gann loop
    above so neither strategy's `continue`s or state can interfere with the other.

    Trades its own symbol list (`sma5_reversion_symbols`), independent of the
    Gann strategy's `symbols` setting — backtesting showed several symbols are
    a poor fit for this specific strategy (see sma5_reversion_symbols default).

    Scans every symbol across every configured timeframe (`scan_timeframes`)
    each cycle — a position is still capped at one per symbol per strategy
    regardless of which timeframe triggers it (the duplicate-direction guard
    below is timeframe-agnostic by design).
    """
    if int(settings.get("sma5_reversion_enabled", 1)) != 1:
        return

    sma5_symbols = settings.get("sma5_reversion_symbols", [])
    threshold_pct = float(settings.get("sma5_reversion_threshold_pct", 0.3))
    sl_multiplier = float(settings.get("sma5_reversion_sl_multiplier", 1.5))
    min_rr_ratio = float(settings.get("sma5_reversion_min_rr_ratio", 0.5))
    rsi_overbought = float(settings.get("sma5_reversion_rsi_overbought", 65.0))
    rsi_oversold = float(settings.get("sma5_reversion_rsi_oversold", 35.0))
    min_tp_spread_multiple = float(settings.get("sma5_reversion_min_tp_spread_multiple", 2.0))
    # SMA5 reversion strategy is strictly restricted to H1 (1 Hour) timeframe
    scan_timeframes = ["H1"]

    for symbol in sma5_symbols:
        for timeframe_str in scan_timeframes:
            try:
                # News freeze check (reuses the same events computed once per cycle)
                if any(event["currency"].upper() in symbol.upper() for event in active_news_events):
                    continue

                candles_df = get_candles(symbol=symbol, timeframe=get_timeframe(timeframe_str), count=50)
                if candles_df is None or len(candles_df) < 20:
                    continue

                price_info = get_current_price(symbol)
                if not price_info:
                    continue

                mid_price = (price_info['bid'] + price_info['ask']) / 2.0
                decision, reason, sma5 = check_sma5_reversion_signal(candles_df, mid_price, threshold_pct, rsi_overbought, rsi_oversold)

                if decision == "HOLD":
                    continue

                # Duplicate same-direction entry guard
                symbol_positions = [
                    pos for pos in get_open_positions()
                    if pos.magic == MAGIC_NUMBER and pos.symbol == symbol and "sma5" in (pos.comment or "").lower()
                ]
                pos_type_str_map = {0: "BUY", 1: "SELL"}
                if any(pos_type_str_map.get(pos.type) == decision for pos in symbol_positions):
                    continue

                # Stop-loss cooldown reuse (no Gann-specific setup context here)
                if is_setup_stopped_out(symbol, None, decision):
                    continue

                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    continue

                entry_price = price_info['ask'] if decision == "BUY" else price_info['bid']
                spread = price_info['ask'] - price_info['bid']
                deviation_distance = abs(entry_price - sma5)
                sl_distance = sl_multiplier * deviation_distance
                sl = entry_price - sl_distance if decision == "BUY" else entry_price + sl_distance
                # Account for spread: BUY closes at Bid (TP=sma5), SELL closes at Ask (TP=sma5+spread)
                tp = sma5 if decision == "BUY" else (sma5 + spread)

                sl = round(sl, symbol_info.digits)
                tp = round(tp, symbol_info.digits)

                sl_dist = abs(entry_price - sl)
                tp_dist = abs(tp - entry_price)
                if sl_dist <= 0:
                    print(f"[SMA5 REVERSION] {symbol} ({timeframe_str}) rejected: Stop Loss at or ahead of entry price.")
                    continue

                # Spread guard: MT5 closes a BUY at Bid and a SELL at Ask, not at the mid-price
                # the TP was computed from — so the spread eats directly into this strategy's
                # small reversion target. If the target isn't comfortably bigger than the current
                # spread, the position may sit open long past "reaching" its nominal TP, or never
                # close at all. Require the target to be at least a configurable multiple of spread.
                spread = price_info['ask'] - price_info['bid']
                if tp_dist < spread * min_tp_spread_multiple:
                    print(f"[SMA5 REVERSION] {symbol} ({timeframe_str}) rejected: TP distance ({tp_dist:.5f}) too small relative to spread ({spread:.5f}) — would rarely/never close at target.")
                    continue

                rr_ratio = tp_dist / sl_dist
                if rr_ratio < min_rr_ratio:
                    print(f"[SMA5 REVERSION] {symbol} ({timeframe_str}) rejected: R:R 1:{rr_ratio:.2f} below strategy floor 1:{min_rr_ratio:.2f}.")
                    continue

                volume = calculate_lot_size(symbol, sl, entry_price, risk_percent=risk_percent)
                reason_msg = f"SMA5 Reversion ({timeframe_str}): {reason}"

                if not auto_trade_enabled:
                    print(f"[SMA5 REVERSION] Signal-Only Mode: {decision} on {symbol} ({timeframe_str}) at {entry_price} (SL: {sl}, TP: {tp}). Not executed.")
                    last_scan_reports[f"{symbol}_sma5_{timeframe_str}"] = {
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "Signal Only (SMA5)",
                        "details": reason_msg,
                        "structure": None
                    }
                    continue

                trade_res = open_trade(
                    action=decision,
                    symbol=symbol,
                    volume=volume,
                    sl=sl,
                    tp=tp,
                    magic=MAGIC_NUMBER,
                    comment=f"SMA5 Reversion {decision}"[:28]
                )

                if trade_res and trade_res["success"]:
                    log_trade_open(
                        ticket=trade_res["ticket"],
                        symbol=symbol,
                        action=decision,
                        volume=volume,
                        entry_price=trade_res["price"],
                        sl=sl,
                        tp=tp,
                        reason=reason_msg,
                        gann_data={"timeframe": timeframe_str}
                    )
                    print(f"[SMA5 REVERSION] Executed {decision} {volume} lots on {symbol} ({timeframe_str}) at {trade_res['price']} (SL: {sl}, TP: {tp}). Ticket: {trade_res['ticket']}.")
                    last_scan_reports[f"{symbol}_sma5_{timeframe_str}"] = {
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": f"Executed {decision} (SMA5)",
                        "details": reason_msg,
                        "structure": None
                    }

                    entry_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                    screenshot_url = None
                    try:
                        from chart_generator import generate_chart_screenshot
                        from db_manager import update_trade_screenshot

                        screenshot_url = generate_chart_screenshot(
                            symbol=symbol,
                            ticket=trade_res["ticket"],
                            entry_price=trade_res["price"],
                            sl_price=sl,
                            tp_price=tp,
                            timeframe_str=timeframe_str,
                            entry_time_str=entry_time_str
                        )
                        update_trade_screenshot(trade_res["ticket"], screenshot_url)
                        print(f"[SCREENSHOT] Saved SMA5 trade chart to {screenshot_url}")
                    except Exception as ex:
                        print(f"[SCREENSHOT] [ERROR] Failed to save SMA5 screenshot: {ex}")

                    try:
                        from telegram_notifier import notify_trade_open
                        from db_manager import update_trade_telegram_msg_id
                        msg_id = notify_trade_open(
                            ticket=trade_res["ticket"],
                            symbol=symbol,
                            action=decision,
                            volume=volume,
                            entry_price=trade_res["price"],
                            sl=sl,
                            tp=tp,
                            reason=reason_msg,
                            screenshot_url=screenshot_url,
                            timeframe=timeframe_str
                        )
                        if msg_id:
                            update_trade_telegram_msg_id(trade_res["ticket"], msg_id)
                            print(f"[TELEGRAM] SMA5 reversion trade open notification sent (Msg ID: {msg_id})")
                    except Exception as tg_ex:
                        print(f"[TELEGRAM] [ERROR] Failed to send SMA5 reversion open alert: {tg_ex}")
                else:
                    print(f"[SMA5 REVERSION] [ERROR] Failed to execute trade order for {symbol} ({timeframe_str}).")
            except Exception as sma5_ex:
                print(f"[SMA5 REVERSION] [ERROR] {symbol} ({timeframe_str}): {sma5_ex}")


def manage_elliott_wave_strategy(settings, active_news_events, risk_percent, auto_trade_enabled):
    """
    Independent, parallel strategy: enters after a completed Elliott Wave 2
    retracement, targeting the Wave 3 Fibonacci extension. Structured exactly
    like manage_sma5_reversion_strategy() — its own symbol list, its own
    signal source, its own SL/TP/lot-size pipeline — so none of the three
    strategies' `continue`s or state can interfere with each other.

    SL sits at Wave 0 (the point A price) — this is not an arbitrary choice,
    it is the exact Elliott invalidation level ("Wave 2 never retraces beyond
    Wave 1's start"), so the stop-loss and the rule are the same price.

    Scans every symbol across every configured timeframe (`scan_timeframes`)
    each cycle — a position is still capped at one per symbol per strategy
    regardless of which timeframe triggers it (the duplicate-direction guard
    below is timeframe-agnostic by design).
    """
    if int(settings.get("elliott_wave_enabled", 1)) != 1:
        return

    elliott_symbols = settings.get("elliott_wave_symbols", [])
    lookback = int(settings.get("elliott_wave_lookback", 100))
    min_retracement = float(settings.get("elliott_wave_min_retracement", 0.382))
    max_retracement = float(settings.get("elliott_wave_max_retracement", 0.786))
    max_wave1_internal_retracement = float(settings.get("elliott_wave_max_wave1_internal_retracement", 0.618))
    extension_ratio = float(settings.get("elliott_wave_extension_ratio", 1.618))
    min_rr_ratio = float(settings.get("elliott_wave_min_rr_ratio", 1.0))
    scan_timeframes = settings.get("scan_timeframes", [DEFAULT_TIMEFRAME])

    for symbol in elliott_symbols:
        for timeframe_str in scan_timeframes:
            try:
                # News freeze check (reuses the same events computed once per cycle)
                if any(event["currency"].upper() in symbol.upper() for event in active_news_events):
                    continue

                candles_df = get_candles(symbol=symbol, timeframe=get_timeframe(timeframe_str), count=lookback + 50)
                if candles_df is None or len(candles_df) < 20:
                    continue

                setup = detect_elliott_wave2_setup(
                    candles_df, lookback=lookback,
                    min_retracement=min_retracement, max_retracement=max_retracement,
                    max_wave1_internal_retracement=max_wave1_internal_retracement
                )
                if not setup:
                    continue

                # Freshness: Wave 2 (point C) must have completed recently, not ancient history
                last_idx = len(candles_df) - 1
                if last_idx - setup["idx_C"] > 5:
                    continue

                price_info = get_current_price(symbol)
                if not price_info:
                    continue
                mid_price = (price_info['bid'] + price_info['ask']) / 2.0

                decision = setup["type"]
                wave0, wave1, wave2 = setup["A"], setup["B"], setup["C"]

                # Entry confirmation: price must already be reversing away from C in the
                # Wave 1 direction (Wave 2 has bottomed/topped and Wave 3 is beginning)
                if decision == "BUY" and mid_price <= wave2:
                    continue
                if decision == "SELL" and mid_price >= wave2:
                    continue

                # Duplicate same-direction entry guard
                symbol_positions = [
                    pos for pos in get_open_positions()
                    if pos.magic == MAGIC_NUMBER and pos.symbol == symbol and "elliott" in (pos.comment or "").lower()
                ]
                pos_type_str_map = {0: "BUY", 1: "SELL"}
                if any(pos_type_str_map.get(pos.type) == decision for pos in symbol_positions):
                    continue

                # Stop-loss cooldown reuse (no Gann-specific setup context here)
                if is_setup_stopped_out(symbol, None, decision):
                    continue

                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    continue

                entry_price = price_info['ask'] if decision == "BUY" else price_info['bid']
                sl = wave0  # Wave 0 = the exact Elliott invalidation level
                tp = calculate_fibonacci_extension(wave0, wave1, extension_ratio)  # Wave 3 target

                sl = round(sl, symbol_info.digits)
                tp = round(tp, symbol_info.digits)

                sl_dist = abs(entry_price - sl)
                tp_dist = abs(tp - entry_price)
                if sl_dist <= 0:
                    print(f"[ELLIOTT WAVE] {symbol} ({timeframe_str}) rejected: Stop Loss at or ahead of entry price.")
                    continue

                rr_ratio = tp_dist / sl_dist
                if rr_ratio < min_rr_ratio:
                    print(f"[ELLIOTT WAVE] {symbol} ({timeframe_str}) rejected: R:R 1:{rr_ratio:.2f} below strategy floor 1:{min_rr_ratio:.2f}.")
                    continue

                volume = calculate_lot_size(symbol, sl, entry_price, risk_percent=risk_percent)
                reason_msg = (
                    f"Elliott Wave ({timeframe_str}): Wave 2 retracement {setup['retracement_pct']:.1f}% of Wave 1 "
                    f"(A={wave0:.5f}, B={wave1:.5f}, C={wave2:.5f}) — targeting Wave 3 extension at {tp:.5f}."
                )
                wave_context = {
                    "type": decision, "A": wave0, "B": wave1, "C": wave2,
                    "retracement_pct": setup["retracement_pct"], "extension_ratio": extension_ratio,
                    "time_A": setup.get("time_A"), "time_B": setup.get("time_B"), "time_C": setup.get("time_C"),
                    "timeframe": timeframe_str
                }

                if not auto_trade_enabled:
                    print(f"[ELLIOTT WAVE] Signal-Only Mode: {decision} on {symbol} ({timeframe_str}) at {entry_price} (SL: {sl}, TP: {tp}). Not executed.")
                    last_scan_reports[f"{symbol}_elliott_{timeframe_str}"] = {
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "Signal Only (Elliott)",
                        "details": reason_msg,
                        "structure": wave_context
                    }
                    continue

                trade_res = open_trade(
                    action=decision,
                    symbol=symbol,
                    volume=volume,
                    sl=sl,
                    tp=tp,
                    magic=MAGIC_NUMBER,
                    comment=f"Elliott Wave {decision}"[:28]
                )

                if trade_res and trade_res["success"]:
                    log_trade_open(
                        ticket=trade_res["ticket"],
                        symbol=symbol,
                        action=decision,
                        volume=volume,
                        entry_price=trade_res["price"],
                        sl=sl,
                        tp=tp,
                        reason=reason_msg,
                        gann_data=wave_context
                    )
                    print(f"[ELLIOTT WAVE] Executed {decision} {volume} lots on {symbol} ({timeframe_str}) at {trade_res['price']} (SL: {sl}, TP: {tp}). Ticket: {trade_res['ticket']}.")
                    last_scan_reports[f"{symbol}_elliott_{timeframe_str}"] = {
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": f"Executed {decision} (Elliott)",
                        "details": reason_msg,
                        "structure": wave_context
                    }

                    entry_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                    screenshot_url = None
                    try:
                        from chart_generator import generate_chart_screenshot
                        from db_manager import update_trade_screenshot

                        screenshot_url = generate_chart_screenshot(
                            symbol=symbol,
                            ticket=trade_res["ticket"],
                            entry_price=trade_res["price"],
                            sl_price=sl,
                            tp_price=tp,
                            gann_data=wave_context,
                            timeframe_str=timeframe_str,
                            entry_time_str=entry_time_str,
                            strategy_label="Elliott Wave"
                        )
                        update_trade_screenshot(trade_res["ticket"], screenshot_url)
                        print(f"[SCREENSHOT] Saved Elliott Wave trade chart to {screenshot_url}")
                    except Exception as ex:
                        print(f"[SCREENSHOT] [ERROR] Failed to save Elliott Wave screenshot: {ex}")

                    try:
                        from telegram_notifier import notify_trade_open
                        from db_manager import update_trade_telegram_msg_id
                        msg_id = notify_trade_open(
                            ticket=trade_res["ticket"],
                            symbol=symbol,
                            action=decision,
                            volume=volume,
                            entry_price=trade_res["price"],
                            sl=sl,
                            tp=tp,
                            reason=reason_msg,
                            screenshot_url=screenshot_url,
                            timeframe=timeframe_str
                        )
                        if msg_id:
                            update_trade_telegram_msg_id(trade_res["ticket"], msg_id)
                            print(f"[TELEGRAM] Elliott Wave trade open notification sent (Msg ID: {msg_id})")
                    except Exception as tg_ex:
                        print(f"[TELEGRAM] [ERROR] Failed to send Elliott Wave open alert: {tg_ex}")
                else:
                    print(f"[ELLIOTT WAVE] [ERROR] Failed to execute trade order for {symbol} ({timeframe_str}).")
            except Exception as elliott_ex:
                print(f"[ELLIOTT WAVE] [ERROR] {symbol} ({timeframe_str}): {elliott_ex}")


def manage_range_trading_strategy(settings, active_news_events, risk_percent, auto_trade_enabled):
    """
    Independent, parallel strategy: detects a horizontal consolidation range
    from confirmed swing-pivot boundaries (detect_range_zone — not a rolling
    min/max), then either fades the boundaries while ADX confirms the market
    is still ranging, or trades a confirmed breakout of either boundary.
    Structured exactly like the other three strategies — its own symbol
    list, its own signal source, its own SL/TP/lot-size pipeline, so none of
    the four strategies' `continue`s or state can interfere with each other.

    Left disabled by default (range_trading_enabled defaults to "0") — this
    is a new, unvalidated strategy on a live account; see backtest_range_trading.py.

    Scans every symbol across every configured timeframe (`scan_timeframes`)
    each cycle — a position is still capped at one per symbol per strategy
    regardless of which timeframe triggers it (the duplicate-direction guard
    below is timeframe-agnostic by design).
    """
    if int(settings.get("range_trading_enabled", 0)) != 1:
        return

    range_symbols = settings.get("range_trading_symbols", [])
    lookback = int(settings.get("range_trading_lookback", 100))
    swing_window = int(settings.get("range_trading_swing_window", 5))
    peak_tolerance_pct = float(settings.get("range_trading_peak_tolerance_pct", 0.15))
    trough_tolerance_pct = float(settings.get("range_trading_trough_tolerance_pct", 0.15))
    adx_period = int(settings.get("range_trading_adx_period", 14))
    adx_threshold = float(settings.get("range_trading_adx_threshold", 25.0))
    entry_zone_pct = float(settings.get("range_trading_entry_zone_pct", 0.15))
    min_range_pct = float(settings.get("range_trading_min_range_pct", 0.3))
    max_range_pct = float(settings.get("range_trading_max_range_pct", 3.0))
    atr_period = int(settings.get("range_trading_atr_period", 14))
    atr_sl_multiplier = float(settings.get("range_trading_atr_sl_multiplier", 1.0))
    tp_buffer_pct = float(settings.get("range_trading_tp_buffer_pct", 10.0))
    breakout_tp_multiplier = float(settings.get("range_trading_breakout_tp_multiplier", 1.0))
    rsi_overbought = float(settings.get("range_trading_rsi_overbought", 65.0))
    rsi_oversold = float(settings.get("range_trading_rsi_oversold", 35.0))
    min_rr_ratio = float(settings.get("range_trading_min_rr_ratio", 1.0))
    min_tp_spread_multiple = float(settings.get("range_trading_min_tp_spread_multiple", 2.0))
    scan_timeframes = settings.get("scan_timeframes", [DEFAULT_TIMEFRAME])

    for symbol in range_symbols:
        for timeframe_str in scan_timeframes:
            try:
                # News freeze check (reuses the same events computed once per cycle)
                if any(event["currency"].upper() in symbol.upper() for event in active_news_events):
                    continue

                candles_df = get_candles(symbol=symbol, timeframe=get_timeframe(timeframe_str), count=lookback + 100)
                if candles_df is None or len(candles_df) < 20:
                    continue
                candles_df = candles_df.reset_index(drop=True)

                range_info = detect_range_zone(
                    candles_df, lookback=lookback, window=swing_window,
                    peak_tolerance_pct=peak_tolerance_pct, trough_tolerance_pct=trough_tolerance_pct,
                    min_range_pct=min_range_pct, max_range_pct=max_range_pct
                )
                if not range_info:
                    continue

                price_info = get_current_price(symbol)
                if not price_info:
                    continue
                mid_price = (price_info['bid'] + price_info['ask']) / 2.0

                decision, reason, context = check_range_trading_signal(
                    candles_df, mid_price, range_info,
                    adx_period=adx_period, adx_threshold=adx_threshold, entry_zone_pct=entry_zone_pct,
                    rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold, atr_period=atr_period
                )
                if decision == "HOLD":
                    continue

                mode = context["mode"]
                mode_tag = "BO" if mode == "breakout" else "Fade"

                # Duplicate same-direction entry guard
                symbol_positions = [
                    pos for pos in get_open_positions()
                    if pos.magic == MAGIC_NUMBER and pos.symbol == symbol and "range" in (pos.comment or "").lower()
                ]
                pos_type_str_map = {0: "BUY", 1: "SELL"}
                if any(pos_type_str_map.get(pos.type) == decision for pos in symbol_positions):
                    continue

                # Stop-loss cooldown reuse (no Gann-specific setup context here)
                if is_setup_stopped_out(symbol, None, decision):
                    continue

                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    continue

                entry_price = price_info['ask'] if decision == "BUY" else price_info['bid']
                range_top = context["range_top"]
                range_bottom = context["range_bottom"]
                range_width = range_top - range_bottom
                atr = context.get("atr") or 0.0

                # SL is unified by direction regardless of mode — a breakout BUY's
                # SL sits at the far/opposite side of the range (deliberately wide),
                # the exact same level a fade BUY near that boundary would use anyway.
                if decision == "BUY":
                    sl = range_bottom - atr * atr_sl_multiplier
                else:
                    sl = range_top + atr * atr_sl_multiplier

                # TP is mode-dependent: fade targets the opposite boundary pulled in
                # by a fill-probability buffer (MT5 fills at bid/ask, not the
                # mid-price the range was computed from); breakout targets a
                # measured-move projection of the range width.
                if mode == "fade":
                    if decision == "BUY":
                        tp = range_top - range_width * (tp_buffer_pct / 100.0)
                    else:
                        tp = range_bottom + range_width * (tp_buffer_pct / 100.0)
                else:
                    if decision == "BUY":
                        tp = range_top + range_width * breakout_tp_multiplier
                    else:
                        tp = range_bottom - range_width * breakout_tp_multiplier

                sl = round(sl, symbol_info.digits)
                tp = round(tp, symbol_info.digits)

                # TP must actually sit on the profitable side of entry — a breakout
                # entry can occur well beyond range_top/range_bottom on a strong move,
                # so a measured-move TP computed purely from the boundary can land
                # behind entry (abs(tp-entry) would still look like a positive R:R).
                if decision == "BUY" and tp <= entry_price:
                    print(f"[RANGE TRADING] {symbol} ({timeframe_str}) rejected: TP ({tp}) is not ahead of entry ({entry_price}) for a BUY.")
                    continue
                if decision == "SELL" and tp >= entry_price:
                    print(f"[RANGE TRADING] {symbol} ({timeframe_str}) rejected: TP ({tp}) is not ahead of entry ({entry_price}) for a SELL.")
                    continue

                sl_dist = abs(entry_price - sl)
                tp_dist = abs(tp - entry_price)
                if sl_dist <= 0:
                    print(f"[RANGE TRADING] {symbol} ({timeframe_str}) rejected: Stop Loss at or ahead of entry price.")
                    continue

                # Spread guard: same rationale as SMA5 — MT5 closes a BUY at Bid and
                # a SELL at Ask, not the mid-price the range/TP levels were computed from.
                spread = price_info['ask'] - price_info['bid']
                if tp_dist < spread * min_tp_spread_multiple:
                    print(f"[RANGE TRADING] {symbol} ({timeframe_str}) rejected: TP distance ({tp_dist:.5f}) too small relative to spread ({spread:.5f}).")
                    continue

                rr_ratio = tp_dist / sl_dist
                if rr_ratio < min_rr_ratio:
                    print(f"[RANGE TRADING] {symbol} ({timeframe_str}) rejected: R:R 1:{rr_ratio:.2f} below strategy floor 1:{min_rr_ratio:.2f}.")
                    continue

                volume = calculate_lot_size(symbol, sl, entry_price, risk_percent=risk_percent)
                reason_msg = f"Range Trading ({mode.capitalize()}, {timeframe_str}): {reason}"
                range_context = {
                    "type": decision, "mode": mode,
                    "range_top": range_top, "range_bottom": range_bottom,
                    "adx": context.get("adx"), "timeframe": timeframe_str
                }

                if not auto_trade_enabled:
                    print(f"[RANGE TRADING] Signal-Only Mode: {decision} on {symbol} ({timeframe_str}) at {entry_price} (SL: {sl}, TP: {tp}). Not executed.")
                    last_scan_reports[f"{symbol}_range_{timeframe_str}"] = {
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "Signal Only (Range)",
                        "details": reason_msg,
                        "structure": range_context
                    }
                    continue

                trade_res = open_trade(
                    action=decision,
                    symbol=symbol,
                    volume=volume,
                    sl=sl,
                    tp=tp,
                    magic=MAGIC_NUMBER,
                    comment=f"Range {mode_tag} {decision}"[:28]
                )

                if trade_res and trade_res["success"]:
                    log_trade_open(
                        ticket=trade_res["ticket"],
                        symbol=symbol,
                        action=decision,
                        volume=volume,
                        entry_price=trade_res["price"],
                        sl=sl,
                        tp=tp,
                        reason=reason_msg,
                        gann_data=range_context
                    )
                    print(f"[RANGE TRADING] Executed {decision} {volume} lots on {symbol} ({timeframe_str}) at {trade_res['price']} (SL: {sl}, TP: {tp}). Ticket: {trade_res['ticket']}.")
                    last_scan_reports[f"{symbol}_range_{timeframe_str}"] = {
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": f"Executed {decision} (Range {mode_tag})",
                        "details": reason_msg,
                        "structure": range_context
                    }

                    entry_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                    screenshot_url = None
                    try:
                        from chart_generator import generate_chart_screenshot
                        from db_manager import update_trade_screenshot

                        screenshot_url = generate_chart_screenshot(
                            symbol=symbol,
                            ticket=trade_res["ticket"],
                            entry_price=trade_res["price"],
                            sl_price=sl,
                            tp_price=tp,
                            timeframe_str=timeframe_str,
                            entry_time_str=entry_time_str
                        )
                        update_trade_screenshot(trade_res["ticket"], screenshot_url)
                        print(f"[SCREENSHOT] Saved Range Trading chart to {screenshot_url}")
                    except Exception as ex:
                        print(f"[SCREENSHOT] [ERROR] Failed to save Range Trading screenshot: {ex}")

                    try:
                        from telegram_notifier import notify_trade_open
                        from db_manager import update_trade_telegram_msg_id
                        msg_id = notify_trade_open(
                            ticket=trade_res["ticket"],
                            symbol=symbol,
                            action=decision,
                            volume=volume,
                            entry_price=trade_res["price"],
                            sl=sl,
                            tp=tp,
                            reason=reason_msg,
                            screenshot_url=screenshot_url,
                            timeframe=timeframe_str
                        )
                        if msg_id:
                            update_trade_telegram_msg_id(trade_res["ticket"], msg_id)
                            print(f"[TELEGRAM] Range Trading trade open notification sent (Msg ID: {msg_id})")
                    except Exception as tg_ex:
                        print(f"[TELEGRAM] [ERROR] Failed to send Range Trading open alert: {tg_ex}")
                else:
                    print(f"[RANGE TRADING] [ERROR] Failed to execute trade order for {symbol} ({timeframe_str}).")
            except Exception as range_ex:
                print(f"[RANGE TRADING] [ERROR] {symbol} ({timeframe_str}): {range_ex}")


def manage_harmonic_patterns_strategy(settings, active_news_events, risk_percent, auto_trade_enabled):
    """
    Independent, parallel strategy: detects classical harmonic X-A-B-C-D
    price structures (Gartley/Bat/Butterfly/Crab, via Fibonacci ratio
    confluence — detect_harmonic_pattern) and enters as price completes the
    projected D point (the "Potential Reversal Zone"), confirmed by RSI(14)
    exhaustion. Structured exactly like the other four strategies — its own
    symbol list, its own signal source, its own SL/TP/lot-size pipeline.

    SL sits beyond point X — the standard harmonic invalidation level — and
    (unlike Elliott Wave) needs no separate post-entry invalidation check:
    entry happens near D, which sits on the opposite side of the whole
    pattern from X, so "price beyond X" cannot be true at the moment of
    entry, and the SL alone already captures the real invalidation rule.

    Left disabled by default (harmonic_patterns_enabled defaults to "0") —
    this is a new, unvalidated strategy on a live account; see
    backtest_harmonic_patterns.py.

    Scans every symbol across every configured timeframe (`scan_timeframes`)
    each cycle — a position is still capped at one per symbol per strategy
    regardless of which timeframe triggers it (the duplicate-direction guard
    below is timeframe-agnostic by design).
    """
    if int(settings.get("harmonic_patterns_enabled", 0)) != 1:
        return

    harmonic_symbols = settings.get("harmonic_patterns_symbols", [])
    lookback = int(settings.get("harmonic_patterns_lookback", 100))
    swing_window = int(settings.get("harmonic_patterns_swing_window", 5))
    pattern_types = settings.get("harmonic_patterns_types", list(HARMONIC_PATTERNS.keys()))
    ratio_tolerance_pct = float(settings.get("harmonic_patterns_ratio_tolerance_pct", 5.0))
    prz_confluence_pct = float(settings.get("harmonic_patterns_prz_confluence_pct", 10.0))
    entry_zone_pct = float(settings.get("harmonic_patterns_entry_zone_pct", 0.15))
    atr_period = int(settings.get("harmonic_patterns_atr_period", 14))
    atr_sl_multiplier = float(settings.get("harmonic_patterns_atr_sl_multiplier", 1.0))
    tp_retracement_ratio = float(settings.get("harmonic_patterns_tp_retracement_ratio", 0.618))
    rsi_overbought = float(settings.get("harmonic_patterns_rsi_overbought", 65.0))
    rsi_oversold = float(settings.get("harmonic_patterns_rsi_oversold", 35.0))
    min_rr_ratio = float(settings.get("harmonic_patterns_min_rr_ratio", 1.0))
    min_tp_spread_multiple = float(settings.get("harmonic_patterns_min_tp_spread_multiple", 2.0))
    scan_timeframes = settings.get("scan_timeframes", [DEFAULT_TIMEFRAME])

    for symbol in harmonic_symbols:
        for timeframe_str in scan_timeframes:
            try:
                if any(event["currency"].upper() in symbol.upper() for event in active_news_events):
                    continue

                candles_df = get_candles(symbol=symbol, timeframe=get_timeframe(timeframe_str), count=lookback + 100)
                if candles_df is None or len(candles_df) < 20:
                    continue
                candles_df = candles_df.reset_index(drop=True)

                pattern_info = detect_harmonic_pattern(
                    candles_df, lookback=lookback, window=swing_window, patterns=pattern_types,
                    ratio_tolerance_pct=ratio_tolerance_pct, prz_confluence_pct=prz_confluence_pct
                )
                if not pattern_info:
                    continue

                # Freshness: C must have completed recently, not ancient history
                # (same convention as Elliott Wave's idx_C freshness gate, given a
                # slightly longer leash since D can take a bit longer to complete)
                last_idx = len(candles_df) - 1
                if last_idx - pattern_info["idx_C"] > 10:
                    continue

                price_info = get_current_price(symbol)
                if not price_info:
                    continue
                mid_price = (price_info['bid'] + price_info['ask']) / 2.0

                decision, reason, context = check_harmonic_pattern_signal(
                    candles_df, mid_price, pattern_info,
                    entry_zone_pct=entry_zone_pct, rsi_overbought=rsi_overbought,
                    rsi_oversold=rsi_oversold, atr_period=atr_period
                )
                if decision == "HOLD":
                    continue

                pattern_name = context["pattern"]

                # Duplicate same-direction entry guard
                symbol_positions = [
                    pos for pos in get_open_positions()
                    if pos.magic == MAGIC_NUMBER and pos.symbol == symbol and "harmonic" in (pos.comment or "").lower()
                ]
                pos_type_str_map = {0: "BUY", 1: "SELL"}
                if any(pos_type_str_map.get(pos.type) == decision for pos in symbol_positions):
                    continue

                # Stop-loss cooldown reuse (no Gann-specific setup context here)
                if is_setup_stopped_out(symbol, None, decision):
                    continue

                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    continue

                entry_price = price_info['ask'] if decision == "BUY" else price_info['bid']
                point_X = context["X"]
                point_C = context["C"]
                prz_price = context["prz_price"]
                atr = context.get("atr") or 0.0

                # SL sits beyond X — the standard harmonic invalidation level, ATR-buffered
                if decision == "BUY":
                    sl = point_X - atr * atr_sl_multiplier
                else:
                    sl = point_X + atr * atr_sl_multiplier

                # TP: a single Fibonacci retracement of the C->D leg back toward C
                # (harmonic trading often ladders 3 partial targets; this bot has
                # no partial-close infrastructure anywhere, so — like every other
                # strategy here — this uses one single TP)
                if decision == "BUY":
                    tp = prz_price + tp_retracement_ratio * (point_C - prz_price)
                else:
                    tp = prz_price - tp_retracement_ratio * (prz_price - point_C)

                sl = round(sl, symbol_info.digits)
                tp = round(tp, symbol_info.digits)

                # TP must actually sit on the profitable side of entry (defense in
                # depth — same class of bug found and fixed in Range Trading/Classical
                # Patterns; lower risk here since entry is anchored close to the PRZ,
                # but cheap to guard against regardless).
                if decision == "BUY" and tp <= entry_price:
                    print(f"[HARMONIC] {symbol} ({timeframe_str}) rejected: TP ({tp}) is not ahead of entry ({entry_price}) for a BUY.")
                    continue
                if decision == "SELL" and tp >= entry_price:
                    print(f"[HARMONIC] {symbol} ({timeframe_str}) rejected: TP ({tp}) is not ahead of entry ({entry_price}) for a SELL.")
                    continue

                sl_dist = abs(entry_price - sl)
                tp_dist = abs(tp - entry_price)
                if sl_dist <= 0:
                    print(f"[HARMONIC] {symbol} ({timeframe_str}) rejected: Stop Loss at or ahead of entry price.")
                    continue

                # Spread guard: same rationale as SMA5/Range Trading — MT5 closes a
                # BUY at Bid and a SELL at Ask, not the mid-price the PRZ/TP were computed from.
                spread = price_info['ask'] - price_info['bid']
                if tp_dist < spread * min_tp_spread_multiple:
                    print(f"[HARMONIC] {symbol} ({timeframe_str}) rejected: TP distance ({tp_dist:.5f}) too small relative to spread ({spread:.5f}).")
                    continue

                rr_ratio = tp_dist / sl_dist
                if rr_ratio < min_rr_ratio:
                    print(f"[HARMONIC] {symbol} ({timeframe_str}) rejected: R:R 1:{rr_ratio:.2f} below strategy floor 1:{min_rr_ratio:.2f}.")
                    continue

                volume = calculate_lot_size(symbol, sl, entry_price, risk_percent=risk_percent)
                reason_msg = f"Harmonic {pattern_name} ({timeframe_str}): {reason}"
                harmonic_context = {
                    "type": decision, "pattern": pattern_name,
                    "X": point_X, "A": context["A"], "B": context["B"], "C": point_C,
                    "prz_price": prz_price, "timeframe": timeframe_str
                }

                if not auto_trade_enabled:
                    print(f"[HARMONIC] Signal-Only Mode: {decision} on {symbol} ({timeframe_str}) at {entry_price} (SL: {sl}, TP: {tp}). Not executed.")
                    last_scan_reports[f"{symbol}_harmonic_{timeframe_str}"] = {
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "Signal Only (Harmonic)",
                        "details": reason_msg,
                        "structure": harmonic_context
                    }
                    continue

                trade_res = open_trade(
                    action=decision,
                    symbol=symbol,
                    volume=volume,
                    sl=sl,
                    tp=tp,
                    magic=MAGIC_NUMBER,
                    comment=f"Harmonic {pattern_name} {decision}"[:28]
                )

                if trade_res and trade_res["success"]:
                    log_trade_open(
                        ticket=trade_res["ticket"],
                        symbol=symbol,
                        action=decision,
                        volume=volume,
                        entry_price=trade_res["price"],
                        sl=sl,
                        tp=tp,
                        reason=reason_msg,
                        gann_data=harmonic_context
                    )
                    print(f"[HARMONIC] Executed {decision} {volume} lots on {symbol} ({timeframe_str}) at {trade_res['price']} (SL: {sl}, TP: {tp}). Ticket: {trade_res['ticket']}.")
                    last_scan_reports[f"{symbol}_harmonic_{timeframe_str}"] = {
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": f"Executed {decision} (Harmonic {pattern_name})",
                        "details": reason_msg,
                        "structure": harmonic_context
                    }

                    entry_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                    screenshot_url = None
                    try:
                        from chart_generator import generate_chart_screenshot
                        from db_manager import update_trade_screenshot

                        screenshot_url = generate_chart_screenshot(
                            symbol=symbol,
                            ticket=trade_res["ticket"],
                            entry_price=trade_res["price"],
                            sl_price=sl,
                            tp_price=tp,
                            gann_data=harmonic_context,
                            timeframe_str=timeframe_str,
                            entry_time_str=entry_time_str,
                            strategy_label="Harmonic Patterns"
                        )
                        update_trade_screenshot(trade_res["ticket"], screenshot_url)
                        print(f"[SCREENSHOT] Saved Harmonic Patterns chart to {screenshot_url}")
                    except Exception as ex:
                        print(f"[SCREENSHOT] [ERROR] Failed to save Harmonic Patterns screenshot: {ex}")

                    try:
                        from telegram_notifier import notify_trade_open
                        from db_manager import update_trade_telegram_msg_id
                        msg_id = notify_trade_open(
                            ticket=trade_res["ticket"],
                            symbol=symbol,
                            action=decision,
                            volume=volume,
                            entry_price=trade_res["price"],
                            sl=sl,
                            tp=tp,
                            reason=reason_msg,
                            screenshot_url=screenshot_url,
                            timeframe=timeframe_str
                        )
                        if msg_id:
                            update_trade_telegram_msg_id(trade_res["ticket"], msg_id)
                            print(f"[TELEGRAM] Harmonic Patterns trade open notification sent (Msg ID: {msg_id})")
                    except Exception as tg_ex:
                        print(f"[TELEGRAM] [ERROR] Failed to send Harmonic Patterns open alert: {tg_ex}")
                else:
                    print(f"[HARMONIC] [ERROR] Failed to execute trade order for {symbol} ({timeframe_str}).")
            except Exception as harmonic_ex:
                print(f"[HARMONIC] [ERROR] {symbol} ({timeframe_str}): {harmonic_ex}")


def manage_classical_patterns_strategy(settings, active_news_events, risk_percent, auto_trade_enabled):
    """
    Independent, parallel strategy: detects classical technical chart
    patterns — continuation (Rectangle/Triangle-variants/Wedge-variants/
    Pennant/Flag, via detect_continuation_pattern) or reversal (Head &
    Shoulders + inverse, Double Top/Bottom, via detect_reversal_pattern) —
    and enters on a confirmed breakout of the pattern's defining level
    (trendline or neckline). Structured exactly like the other five
    strategies — its own symbol list, its own signal source, its own
    SL/TP/lot-size pipeline.

    Pure breakout entry, no RSI gate (see check_classical_pattern_signal's
    docstring) — this is the strategy's second pure-breakout strategy
    alongside Gann/Range Trading's breakout mode, in contrast to the
    fade-style strategies (SMA5, Range-fade, Harmonic Patterns).

    Left disabled by default (classical_patterns_enabled defaults to "0") —
    this is a new, unvalidated strategy on a live account; see
    backtest_classical_patterns.py.

    Scans every symbol across every configured timeframe (`scan_timeframes`)
    each cycle — a position is still capped at one per symbol per strategy
    regardless of which timeframe triggers it (the duplicate-direction guard
    below is timeframe-agnostic by design).
    """
    if int(settings.get("classical_patterns_enabled", 0)) != 1:
        return

    classical_symbols = settings.get("classical_patterns_symbols", [])
    pattern_types = settings.get("classical_patterns_types", None)
    pole_lookback = int(settings.get("classical_patterns_pole_lookback", 20))
    pole_min_move_pct = float(settings.get("classical_patterns_pole_min_move_pct", 0.5))
    pole_min_efficiency = float(settings.get("classical_patterns_pole_min_efficiency", 0.4))
    consolidation_max_bars = int(settings.get("classical_patterns_consolidation_max_bars", 40))
    swing_window = int(settings.get("classical_patterns_swing_window", 3))
    flat_slope_threshold_pct = float(settings.get("classical_patterns_flat_slope_threshold_pct", 0.02))
    shoulder_tolerance_pct = float(settings.get("classical_patterns_shoulder_tolerance_pct", 3.0))
    neckline_tolerance_pct = float(settings.get("classical_patterns_neckline_tolerance_pct", 2.0))
    atr_period = int(settings.get("classical_patterns_atr_period", 14))
    atr_sl_multiplier = float(settings.get("classical_patterns_atr_sl_multiplier", 1.0))
    tp_multiplier = float(settings.get("classical_patterns_tp_multiplier", 1.0))
    min_rr_ratio = float(settings.get("classical_patterns_min_rr_ratio", 1.0))
    min_tp_spread_multiple = float(settings.get("classical_patterns_min_tp_spread_multiple", 2.0))
    scan_timeframes = settings.get("scan_timeframes", [DEFAULT_TIMEFRAME])

    for symbol in classical_symbols:
        for timeframe_str in scan_timeframes:
            try:
                if any(event["currency"].upper() in symbol.upper() for event in active_news_events):
                    continue

                fetch_count = pole_lookback + consolidation_max_bars + 100
                candles_df = get_candles(symbol=symbol, timeframe=get_timeframe(timeframe_str), count=fetch_count)
                if candles_df is None or len(candles_df) < 20:
                    continue
                candles_df = candles_df.reset_index(drop=True)

                pattern_info = detect_continuation_pattern(
                    candles_df, pole_lookback=pole_lookback, pole_min_move_pct=pole_min_move_pct,
                    pole_min_efficiency=pole_min_efficiency, consolidation_max_bars=consolidation_max_bars,
                    swing_window=swing_window, flat_slope_threshold_pct=flat_slope_threshold_pct
                )
                is_continuation = pattern_info is not None
                if not pattern_info:
                    pattern_info = detect_reversal_pattern(
                        candles_df, lookback=pole_lookback + consolidation_max_bars, window=swing_window,
                        shoulder_tolerance_pct=shoulder_tolerance_pct, neckline_tolerance_pct=neckline_tolerance_pct,
                        patterns=pattern_types
                    )
                if not pattern_info:
                    continue

                decision, reason, context = check_classical_pattern_signal(candles_df, pattern_info, atr_period=atr_period)
                if decision == "HOLD":
                    continue

                pattern_name = context["pattern"]
                level = context["level"]

                # Duplicate same-direction entry guard
                symbol_positions = [
                    pos for pos in get_open_positions()
                    if pos.magic == MAGIC_NUMBER and pos.symbol == symbol and "classical" in (pos.comment or "").lower()
                ]
                pos_type_str_map = {0: "BUY", 1: "SELL"}
                if any(pos_type_str_map.get(pos.type) == decision for pos in symbol_positions):
                    continue

                # Stop-loss cooldown reuse (no Gann-specific setup context here)
                if is_setup_stopped_out(symbol, None, decision):
                    continue

                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    continue

                price_info = get_current_price(symbol)
                if not price_info:
                    continue
                entry_price = price_info['ask'] if decision == "BUY" else price_info['bid']
                atr = context.get("atr") or 0.0

                if is_continuation:
                    x_last = consolidation_max_bars - 1
                    if decision == "BUY":
                        far_level = pattern_info["lower_slope"] * x_last + pattern_info["lower_intercept"]
                        sl = far_level - atr * atr_sl_multiplier
                        tp = level + pattern_info["pole_height"] * tp_multiplier
                    else:
                        far_level = pattern_info["upper_slope"] * x_last + pattern_info["upper_intercept"]
                        sl = far_level + atr * atr_sl_multiplier
                        tp = level - pattern_info["pole_height"] * tp_multiplier
                else:
                    head = pattern_info["head"]
                    if decision == "BUY":
                        sl = head - atr * atr_sl_multiplier
                        tp = level + (level - head) * tp_multiplier
                    else:
                        sl = head + atr * atr_sl_multiplier
                        tp = level - (head - level) * tp_multiplier

                sl = round(sl, symbol_info.digits)
                tp = round(tp, symbol_info.digits)

                # TP must actually sit on the profitable side of entry. Entry can occur
                # well beyond the breakout level/neckline on a strong move, so a target
                # computed purely from level +/- a fixed offset can land behind entry —
                # abs(tp - entry) would still look like a positive R:R, silently letting
                # a losing trade through as if its target were ahead of it.
                if decision == "BUY" and tp <= entry_price:
                    print(f"[CLASSICAL] {symbol} ({timeframe_str}) rejected: TP ({tp}) is not ahead of entry ({entry_price}) for a BUY.")
                    continue
                if decision == "SELL" and tp >= entry_price:
                    print(f"[CLASSICAL] {symbol} ({timeframe_str}) rejected: TP ({tp}) is not ahead of entry ({entry_price}) for a SELL.")
                    continue

                sl_dist = abs(entry_price - sl)
                tp_dist = abs(tp - entry_price)
                if sl_dist <= 0:
                    print(f"[CLASSICAL] {symbol} ({timeframe_str}) rejected: Stop Loss at or ahead of entry price.")
                    continue

                # Spread guard: same rationale as the other strategies — MT5 closes a
                # BUY at Bid and a SELL at Ask, not the price the level was computed from.
                spread = price_info['ask'] - price_info['bid']
                if tp_dist < spread * min_tp_spread_multiple:
                    print(f"[CLASSICAL] {symbol} ({timeframe_str}) rejected: TP distance ({tp_dist:.5f}) too small relative to spread ({spread:.5f}).")
                    continue

                rr_ratio = tp_dist / sl_dist
                if rr_ratio < min_rr_ratio:
                    print(f"[CLASSICAL] {symbol} ({timeframe_str}) rejected: R:R 1:{rr_ratio:.2f} below strategy floor 1:{min_rr_ratio:.2f}.")
                    continue

                volume = calculate_lot_size(symbol, sl, entry_price, risk_percent=risk_percent)
                reason_msg = f"Classical {pattern_name} ({timeframe_str}): {reason}"
                classical_context = {
                    "type": decision,
                    "pattern": pattern_name,
                    "level": level,
                    "timeframe": timeframe_str,
                    "is_continuation": is_continuation,
                    "pattern_info": pattern_info,
                    "consol_start_time": candles_df.iloc[candles_df.shape[0] - consolidation_max_bars]['Time'].strftime('%Y-%m-%d %H:%M:%S') if 'Time' in candles_df.columns else str(candles_df.index[candles_df.shape[0] - consolidation_max_bars])
                }

                if not auto_trade_enabled:
                    print(f"[CLASSICAL] Signal-Only Mode: {decision} on {symbol} ({timeframe_str}) at {entry_price} (SL: {sl}, TP: {tp}). Not executed.")
                    last_scan_reports[f"{symbol}_classical_{timeframe_str}"] = {
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "Signal Only (Classical)",
                        "details": reason_msg,
                        "structure": classical_context
                    }
                    continue

                trade_res = open_trade(
                    action=decision,
                    symbol=symbol,
                    volume=volume,
                    sl=sl,
                    tp=tp,
                    magic=MAGIC_NUMBER,
                    comment=f"Classical {pattern_name} {decision}"[:28]
                )

                if trade_res and trade_res["success"]:
                    log_trade_open(
                        ticket=trade_res["ticket"],
                        symbol=symbol,
                        action=decision,
                        volume=volume,
                        entry_price=trade_res["price"],
                        sl=sl,
                        tp=tp,
                        reason=reason_msg,
                        gann_data=classical_context
                    )
                    print(f"[CLASSICAL] Executed {decision} {volume} lots on {symbol} ({timeframe_str}) at {trade_res['price']} (SL: {sl}, TP: {tp}). Ticket: {trade_res['ticket']}.")
                    last_scan_reports[f"{symbol}_classical_{timeframe_str}"] = {
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": f"Executed {decision} (Classical {pattern_name})",
                        "details": reason_msg,
                        "structure": classical_context
                    }

                    entry_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                    screenshot_url = None
                    try:
                        from chart_generator import generate_chart_screenshot
                        from db_manager import update_trade_screenshot

                        screenshot_url = generate_chart_screenshot(
                            symbol=symbol,
                            ticket=trade_res["ticket"],
                            entry_price=trade_res["price"],
                            sl_price=sl,
                            tp_price=tp,
                            gann_data=classical_context,
                            timeframe_str=timeframe_str,
                            entry_time_str=entry_time_str,
                            strategy_label="Classical Patterns"
                        )
                        update_trade_screenshot(trade_res["ticket"], screenshot_url)
                        print(f"[SCREENSHOT] Saved Classical Patterns chart to {screenshot_url}")
                    except Exception as ex:
                        print(f"[SCREENSHOT] [ERROR] Failed to save Classical Patterns screenshot: {ex}")

                    try:
                        from telegram_notifier import notify_trade_open
                        from db_manager import update_trade_telegram_msg_id
                        msg_id = notify_trade_open(
                            ticket=trade_res["ticket"],
                            symbol=symbol,
                            action=decision,
                            volume=volume,
                            entry_price=trade_res["price"],
                            sl=sl,
                            tp=tp,
                            reason=reason_msg,
                            screenshot_url=screenshot_url,
                            timeframe=timeframe_str
                        )
                        if msg_id:
                            update_trade_telegram_msg_id(trade_res["ticket"], msg_id)
                            print(f"[TELEGRAM] Classical Patterns trade open notification sent (Msg ID: {msg_id})")
                    except Exception as tg_ex:
                        print(f"[TELEGRAM] [ERROR] Failed to send Classical Patterns open alert: {tg_ex}")
                else:
                    print(f"[CLASSICAL] [ERROR] Failed to execute trade order for {symbol} ({timeframe_str}).")
            except Exception as classical_ex:
                print(f"[CLASSICAL] [ERROR] {symbol} ({timeframe_str}): {classical_ex}")


def check_and_execute_trading_cycle():
    """
    Executes a single scanning cycle across all symbols stored in the SQLite settings database.
    """
    global last_scan_reports, consecutive_ai_failures
    print("\n" + "=" * 60)
    print(f"🔄 STARTING SCANNING CYCLE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    last_scan_reports.clear()
    ai_failure_count = 0
    cycle_in_tokens = 0
    cycle_out_tokens = 0
    cycle_cost = 0.0

    # 1. Load Settings from SQLite Database
    settings = get_settings()
    symbols_to_trade = settings.get("symbols", ["EURUSDm"])
    scan_timeframes = settings.get("scan_timeframes", [DEFAULT_TIMEFRAME])
    risk_percent = float(settings.get("risk_percent", 1.0))
    max_positions = int(settings.get("max_positions", 2))
    auto_trade_enabled = int(settings.get("auto_trade", 1)) == 1
    news_filter_enabled = int(settings.get("news_filter", 1)) == 1
    ai_evaluation_enabled = int(settings.get("ai_evaluation", 1)) == 1
    fallback_to_technical = int(settings.get("fallback_to_technical", 1)) == 1

    print(f"[CONFIG] Symbols: {symbols_to_trade}")
    print(f"[CONFIG] Risk: {risk_percent}% per trade | Max Positions: {max_positions}")
    print(f"[CONFIG] Mode: {'AUTO-TRADE' if auto_trade_enabled else 'SIGNAL-ONLY (MANUAL)'}")
    print(f"[CONFIG] News Filter: {'ENABLED' if news_filter_enabled else 'DISABLED'}")
    print(f"[CONFIG] AI Risk Evaluation: {'ENABLED' if ai_evaluation_enabled else 'DISABLED'}")
    print(f"[CONFIG] Technical Fallback: {'ENABLED' if fallback_to_technical else 'DISABLED'}")

    # 2. Sync Positions & Log Balance Snapshot
    account_info = get_account_info()
    if not account_info:
        print("[ERROR] Could not fetch account info. Skipping cycle.")
        for symbol in symbols_to_trade:
            last_scan_reports[symbol] = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "Error",
                "details": "Could not fetch MetaTrader 5 account info.",
                "structure": None
            }
        return

    print(f"\n[ACCOUNT] Balance: {account_info['balance']} {account_info['currency']} | Equity: {account_info['equity']}")
    
    # Log balance snapshot to DB
    log_balance_snapshot(account_info['balance'], account_info['equity'])
    
    # Sync DB positions
    sync_db_with_mt5_positions()

    # Manage active martingale grids (adds legs and recalculates TP/SL dynamically)
    try:
        manage_active_positions_grid()
    except Exception as e:
        print(f"[ERROR] Grid management failed: {e}")

    # Get active positions to check limits
    active_positions = get_open_positions()
    print(f"[ACCOUNT] Active Positions: {len(active_positions)} / {max_positions}")

    # --- PORTFOLIO RISK CONTROL CHECKS (Local Python implementation) ---
    balance = float(account_info.get("balance", 0.0))
    equity = float(account_info.get("equity", 0.0))
    floating_dd_pct = ((balance - equity) / balance) * 100.0 if balance > 0 else 0.0
    margin_level = float(account_info.get("margin_level", 1000.0))

    portfolio_risk_blocked = False
    portfolio_block_reason = ""

    if floating_dd_pct > 5.0:
        portfolio_risk_blocked = True
        portfolio_block_reason = f"Total portfolio floating drawdown ({floating_dd_pct:.2f}%) exceeds 5.0% limit."
        print(f"[PORTFOLIO RISK BLOCK] {portfolio_block_reason}")
    elif len(active_positions) > 0 and margin_level < 300.0:
        portfolio_risk_blocked = True
        portfolio_block_reason = f"Margin Level ({margin_level:.1f}%) is below 300.0% safety threshold."
        print(f"[PORTFOLIO RISK BLOCK] {portfolio_block_reason}")

    # Check Max Positions Limit (Flag instead of immediate return to allow report delivery)
    max_positions_reached = len(active_positions) >= max_positions or portfolio_risk_blocked
    if max_positions_reached:
        details_msg = portfolio_block_reason if portfolio_risk_blocked else f"Maximum active positions reached ({len(active_positions)} / {max_positions})."
        print(f"[RISK] Skipping analysis to protect margin: {details_msg}")
        for symbol in symbols_to_trade:
            last_scan_reports[symbol] = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "Skipped (Risk/Max Positions)",
                "details": details_msg,
                "structure": None
            }

    # Fetch active news events if filter is enabled
    active_news_events = []
    if news_filter_enabled and not max_positions_reached:
        _, active_news_events = is_news_time(minutes_before=30, minutes_after=30)
        if active_news_events:
            print(f"\n[NEWS FILTER] Detected {len(active_news_events)} active high-impact events. Checking symbol exposure...")

    # Keep track of symbols we already have open positions for
    open_symbols = [pos.symbol for pos in active_positions]

    # Initialize AI Engine (it will dynamically load the strategy prompt from DB)
    if not max_positions_reached:
        try:
            engine = AITradingEngine(model_name="gemini-2.5-flash")
        except Exception as e:
            print(f"[ERROR] Failed to initialize AI Engine: {e}")
            return

    # 3. Analyze each symbol on each configured timeframe (scan_timeframes) —
    # iterating (symbol, timeframe) pairs directly here avoids re-indenting the
    # whole per-symbol block below into a nested loop.
    scan_pairs = [(s, tf) for s in symbols_to_trade for tf in scan_timeframes]
    for symbol, timeframe_str in ([] if max_positions_reached else scan_pairs):
        print(f"\n--------------------------------------------------")
        print(f"🔎 Scanning symbol: {symbol} ({timeframe_str})")
        print(f"--------------------------------------------------")

        # Note: We do not skip here anymore to allow detection of opposite reversal signals
        # and closing of active trades. Double entry is prevented later before execution.

        # Check if this specific symbol contains any of the currently frozen news currencies
        symbol_frozen = False
        symbol_news_events = []
        for event in active_news_events:
            curr = event["currency"]
            # e.g., if currency is "EUR", check if "EUR" is in symbol name "EURUSDm"
            if curr.upper() in symbol.upper():
                symbol_frozen = True
                symbol_news_events.append(event)

        if symbol_frozen:
            print(f"\n🚫 [NEWS FILTER] Trading on {symbol} is currently FROZEN due to upcoming economic news:")
            for event in symbol_news_events:
                print(f"  - [{event['importance']}] {event['currency']} - {event['title']} at {event['time']} (in {event['time_diff_minutes']} mins)")
            print(f"Skipping scan for {symbol}.")
            events_desc = ", ".join([f"{e['currency']}: {e['title']}" for e in symbol_news_events])
            last_scan_reports[symbol] = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "News Frozen",
                "details": f"Trading is frozen due to upcoming news events: {events_desc}",
                "structure": None
            }
            continue

        # Get current price
        price_info = get_current_price(symbol)
        if not price_info:
            print(f"[ERROR] Could not fetch price for {symbol}. Skipping.")
            last_scan_reports[symbol] = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "Error",
                "details": "Could not fetch bid/ask price from MetaTrader 5.",
                "structure": None
            }
            continue

        # Fetch candles based on lookback
        gann_enabled = int(settings.get("gann_enabled", 1)) == 1
        gann_lookback = int(settings.get("gann_lookback", 100))
        gann_geometry = settings.get("gann_geometry", "square")
        grid_enabled = int(settings.get("grid_enabled", 1)) == 1

        tf_constant = get_timeframe(timeframe_str)
        if tf_constant is None:
            print(f"[ERROR] Invalid timeframe configuration: {timeframe_str}")
            last_scan_reports[symbol] = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "Error",
                "details": f"Invalid timeframe configuration: {timeframe_str}",
                "structure": None
            }
            continue

        # Fetch count
        count_to_fetch = max(300, gann_lookback + 10)
        candles_df = get_candles(symbol=symbol, timeframe=tf_constant, count=count_to_fetch)
        if candles_df is None or len(candles_df) < 10:
            print(f"[ERROR] Not enough candle data for {symbol}. Skipping.")
            last_scan_reports[symbol] = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "Error",
                "details": "Failed to fetch historical candle data from MetaTrader 5.",
                "structure": None
            }
            continue

        # Format candles for the AI model
        candles_text = format_candles_for_ai(candles_df, last_n=20)

        # Determine technical signal in Python
        proposed_action = "HOLD"
        technical_reason = "No trade setup detected"
        gann_context = None

        if gann_enabled:
            setup = detect_market_structure(candles_df, lookback=gann_lookback)
            if not setup:
                print(f"[INFO] No valid Gann + Fibonacci structure detected on {symbol}. Skipping analysis.")
                last_scan_reports[symbol] = {
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "status": "No Setup",
                    "details": f"No valid swing structure (Point A-B-C with 50-75% correction) detected within the last {gann_lookback} candles.",
                    "structure": None
                }
                continue

            # Retracement percentage
            if setup["type"] == "BUY":
                retr_pct = ((setup["B"] - setup["C"]) / (setup["B"] - setup["A"])) * 100.0
            else:
                retr_pct = ((setup["C"] - setup["B"]) / (setup["A"] - setup["B"])) * 100.0

            from gann_helper import detect_dynamic_gann_levels
            dyn = detect_dynamic_gann_levels(setup["A"], setup["B"], mode='bullish' if setup["type"] == 'BUY' else 'bearish')

            gann_context = {
                "type": setup["type"],
                "A": setup["A"],
                "B": setup["B"],
                "C": setup["C"],
                "time_A": setup.get("time_A"),
                "time_B": setup.get("time_B"),
                "time_C": setup.get("time_C"),
                "retracement_pct": retr_pct,
                "entry_angle": dyn["entry_angle"],
                "target_angle": dyn["target_angle"],
                "target_price": dyn["target_price"],
                "sl_price": dyn["sl_price"],
                "timeframe": timeframe_str
            }
            print(f"[GANN SETUP] Valid {setup['type']} structure: A={setup['A']}, B={setup['B']}, C={setup['C']} (Retracement: {retr_pct:.1f}%)")

            # --- ADVANCED TECHNICAL FILTERS (LIKE GEMINI) ---
            # Calculate ATR for Volatility filter
            import pandas as pd
            df_temp = candles_df.copy()
            high_low = df_temp['High'] - df_temp['Low']
            high_close = (df_temp['High'] - df_temp['Close'].shift()).abs()
            low_close = (df_temp['Low'] - df_temp['Close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df_temp['atr'] = true_range.rolling(14).mean()
            
            # Calculate RSI for Momentum/Divergence filter
            delta = df_temp['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_temp['rsi14'] = 100 - (100 / (1 + rs))

            current_atr = df_temp['atr'].iloc[-2]
            avg_atr = df_temp['atr'].tail(100).mean()
            current_rsi = df_temp['rsi14'].iloc[-2]
            prev_rsi = df_temp['rsi14'].iloc[-3]
            
            last_close = df_temp['Close'].iloc[-2]
            last_open = df_temp['Open'].iloc[-2]
            last_high = df_temp['High'].iloc[-2]
            last_low = df_temp['Low'].iloc[-2]
            
            # 1. Volatility Filter Check
            if pd.notna(current_atr) and pd.notna(avg_atr):
                if current_atr > 2.5 * avg_atr:
                    filter_msg = f"ATR Volatility filter: High volatility warning (ATR: {current_atr:.5f} > 2.5 * Avg: {avg_atr:.5f})."
                    print(f"[FILTER] {symbol} rejected: {filter_msg}")
                    last_scan_reports[symbol] = {
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "No Setup",
                        "details": filter_msg,
                        "structure": gann_context
                    }
                    continue
                elif current_atr < 0.3 * avg_atr:
                    filter_msg = f"ATR Volatility filter: Flat market warning (ATR: {current_atr:.5f} < 0.3 * Avg: {avg_atr:.5f})."
                    print(f"[FILTER] {symbol} rejected: {filter_msg}")
                    last_scan_reports[symbol] = {
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "No Setup",
                        "details": filter_msg,
                        "structure": gann_context
                    }
                    continue

            # 2. Breakout and False Breakout Confirmation Check
            is_breakout = False
            breakout_reason = ""
            if setup["type"] == "BUY":
                if last_close > setup["B"]:
                    is_breakout = True
                else:
                    breakout_reason = f"Breakout filter: Close ({last_close:.5f}) is not above High B ({setup['B']:.5f}) yet."
            else: # SELL
                if last_close < setup["B"]:
                    is_breakout = True
                else:
                    breakout_reason = f"Breakout filter: Close ({last_close:.5f}) is not below Low B ({setup['B']:.5f}) yet."

            if not is_breakout:
                print(f"[FILTER] {symbol} rejected: {breakout_reason}")
                last_scan_reports[symbol] = {
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "status": "No Setup",
                    "details": breakout_reason,
                    "structure": gann_context
                }
                continue

            # 3. Momentum & Divergence Check (RSI)
            is_rsi_valid = True
            rsi_reason = ""
            if pd.notna(current_rsi) and pd.notna(prev_rsi):
                if setup["type"] == "BUY":
                    if current_rsi > 70:
                        is_rsi_valid = False
                        rsi_reason = f"RSI filter: Overbought warning (RSI {current_rsi:.1f} > 70)."
                    elif current_rsi <= prev_rsi:
                        is_rsi_valid = False
                        rsi_reason = f"RSI filter: Falling momentum (RSI {current_rsi:.1f} <= {prev_rsi:.1f})."
                else: # SELL
                    if current_rsi < 30:
                        is_rsi_valid = False
                        rsi_reason = f"RSI filter: Oversold warning (RSI {current_rsi:.1f} < 30)."
                    elif current_rsi >= prev_rsi:
                        is_rsi_valid = False
                        rsi_reason = f"RSI filter: Rising momentum (RSI {current_rsi:.1f} >= {prev_rsi:.1f})."

            if not is_rsi_valid:
                print(f"[FILTER] {symbol} rejected: {rsi_reason}")
                last_scan_reports[symbol] = {
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "status": "No Setup",
                    "details": rsi_reason,
                    "structure": gann_context
                }
                continue

            # 4. Price Action confirmation check
            candle_body = abs(last_close - last_open)
            candle_range = last_high - last_low
            is_valid_pa = False
            pa_detail = ""
            
            if candle_range > 0:
                upper_wick = last_high - max(last_close, last_open)
                lower_wick = min(last_close, last_open) - last_low
                
                prev_close = df_temp['Close'].iloc[-3]
                prev_open = df_temp['Open'].iloc[-3]
                
                if setup["type"] == "BUY":
                    is_bullish_engulfing = (last_close > last_open) and (prev_close < prev_open) and (last_close >= prev_open) and (last_open <= prev_close)
                    is_bullish_pinbar = (lower_wick >= 2.0 * candle_body) and (upper_wick <= 0.5 * lower_wick)
                    is_strong_breakout = (last_close > last_open) and ((last_high - last_close) / candle_range <= 0.3)
                    
                    if is_bullish_engulfing:
                        is_valid_pa = True
                        pa_detail = "Bullish Engulfing pattern"
                    elif is_bullish_pinbar:
                        is_valid_pa = True
                        pa_detail = "Bullish Pin Bar pattern"
                    elif is_strong_breakout:
                        is_valid_pa = True
                        pa_detail = "Strong bullish breakout candle"
                else: # SELL
                    is_bearish_engulfing = (last_close < last_open) and (prev_close > prev_open) and (last_close <= prev_open) and (last_open >= prev_close)
                    is_bearish_pinbar = (upper_wick >= 2.0 * candle_body) and (lower_wick <= 0.5 * upper_wick)
                    is_strong_breakout = (last_close < last_open) and ((last_close - last_low) / candle_range <= 0.3)
                    
                    if is_bearish_engulfing:
                        is_valid_pa = True
                        pa_detail = "Bearish Engulfing pattern"
                    elif is_bearish_pinbar:
                        is_valid_pa = True
                        pa_detail = "Bearish Pin Bar pattern"
                    elif is_strong_breakout:
                        is_valid_pa = True
                        pa_detail = "Strong bearish breakout candle"

            if not is_valid_pa:
                filter_msg = "Price Action filter: Breakout candle lacks bullish/bearish pattern strength."
                print(f"[FILTER] {symbol} rejected: {filter_msg}")
                last_scan_reports[symbol] = {
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "status": "No Setup",
                    "details": filter_msg,
                    "structure": gann_context
                }
                continue

            proposed_action = setup["type"]
            technical_reason = f"Gann breakout {proposed_action} pivot setup with {pa_detail} confirmation."
        else:
            proposed_action, technical_reason = check_technical_strategy_signals(candles_df, symbol)

        # If technical strategy indicates HOLD, do not execute
        if proposed_action == "HOLD":
            print(f"[INFO] {symbol}: {technical_reason}")
            last_scan_reports[symbol] = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "No Setup",
                "details": technical_reason,
                "structure": gann_context
            }
            continue

        print(f"🎯 [TECHNICAL TRIGGER] Generated {proposed_action} signal for {symbol}. Reason: {technical_reason}")

        # --- PRE-AI VALIDATION CHECKS (Token Savers) ---
        # 1. Check active positions for this symbol to prevent double entries in same direction
        # Excludes the other five strategies' own positions (tagged in their comment)
        # — otherwise Gann's duplicate-entry guard and reversal-close logic below
        # would treat any position on this symbol as its own, closing or blocking
        # around trades that SMA5/Elliott/Range/Harmonic/Classical opened.
        symbol_positions = [
            pos for pos in active_positions
            if pos.magic == MAGIC_NUMBER and pos.symbol == symbol
            and not any(tag in (pos.comment or "").lower() for tag in ("sma5", "elliott", "range", "harmonic", "classical"))
        ]
        has_same_direction = False
        if symbol_positions:
            for pos in symbol_positions:
                pos_type_str = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                if pos_type_str == proposed_action:
                    has_same_direction = True

        if has_same_direction:
            skip_msg = f"Already have an active {proposed_action} position on {symbol}. Skipping AI call and entry."
            print(f"[SKIP] {skip_msg}")
            last_scan_reports[symbol] = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "Already Open",
                "details": skip_msg,
                "structure": gann_context
            }
            continue

        # 1.5. Correlation & Drawdown Safety Check
        # Resolve base currencies to check for correlation (e.g. EUR, GBP, USD, CHF, JPY)
        base_currencies = ["EUR", "GBP", "USD", "CHF", "JPY", "AUD", "CAD", "NZD"]
        symbol_base_currencies = [c for c in base_currencies if c in symbol.upper()]
        
        has_correlated_drawdown = False
        correlated_symbol_name = ""
        
        for pos in active_positions:
            if pos.magic == MAGIC_NUMBER and pos.profit < 0: # Active trade in drawdown
                # Check base currency overlap
                pos_base_currencies = [c for c in base_currencies if c in pos.symbol.upper()]
                overlap = set(symbol_base_currencies).intersection(set(pos_base_currencies))
                
                # Check if it shares a key currency and is in the same direction (e.g. USD exposure drawdown)
                pos_type_str = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                if len(overlap) > 0 and pos_type_str == proposed_action:
                    has_correlated_drawdown = True
                    correlated_symbol_name = pos.symbol
                    break

        if has_correlated_drawdown:
            corr_msg = f"Correlation Block: There is an active position on {correlated_symbol_name} in drawdown. Skipping entry on {symbol} to control risk."
            print(f"[CORRELATION BLOCK] {corr_msg}")
            last_scan_reports[symbol] = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "Correlation Blocked",
                "details": corr_msg,
                "structure": gann_context
            }
            continue

        # 2. Check if the last closed trade was a Stop Loss and blocks re-entry
        if is_setup_stopped_out(symbol, gann_context, proposed_action):
            block_msg = f"Blocked re-entry on {symbol} in direction {proposed_action}. The setup/symbol recently hit Stop Loss or was already traded."
            print(f"[BLOCKED] {block_msg}")
            last_scan_reports[symbol] = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "Blocked (Stop Loss / Traded)",
                "details": block_msg,
                "structure": gann_context
            }
            continue

        # Run AI analysis (if enabled) or bypass
        decision = "HOLD"
        trade_params = None
        reasoning = ""

        if ai_evaluation_enabled:
            # Check if we already did an AI scan for this symbol+timeframe on the
            # current candle to save API usage — keyed by (symbol, timeframe)
            # since each symbol is now scanned on every configured timeframe, and
            # a candle-time cache-hit on one timeframe must not suppress a scan
            # on another.
            current_candle_time = str(candles_df['Time'].iloc[-1])
            ai_scan_cache_key = f"{symbol}_{timeframe_str}"
            if last_ai_scan_times.get(ai_scan_cache_key) == current_candle_time:
                print(f"[AI CACHE] Already evaluated {symbol} ({timeframe_str}) on the current candle ({current_candle_time}). Skipping AI call to save tokens.")
                decision = "HOLD"
                last_scan_reports[symbol] = {
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "status": "AI Cached",
                    "details": "AI analysis already performed for this candle. Skipping to save API quota.",
                    "structure": gann_context
                }
                continue

            try:
                ai_result = engine.analyze_market(
                    symbol=symbol,
                    timeframe=timeframe_str,
                    candles_data=candles_text,
                    account_info=account_info,
                    current_price=price_info,
                    gann_context=gann_context,
                    active_positions=active_positions,
                    proposed_action=proposed_action
                )
                # Accumulate token usage
                if "usage" in ai_result:
                    cycle_in_tokens += ai_result["usage"].get("in_tokens", 0)
                    cycle_out_tokens += ai_result["usage"].get("out_tokens", 0)
                    cycle_cost += ai_result["usage"].get("cost", 0.0)

                decision = ai_result.get("decision", "HOLD")
                if decision == "ERROR":
                    raise Exception(ai_result.get("analysis", "Gemini API call failed"))
                if decision in ["APPROVE", "BUY", "SELL"]:
                    decision = proposed_action
                else:
                    decision = "HOLD"
                trade_params = ai_result.get("trade_params")
                reasoning = ai_result.get("analysis", "HOLD - AI rejected proposed trade")

                # Record scan timestamp in cache
                last_ai_scan_times[ai_scan_cache_key] = current_candle_time
                # Reset this symbol's consecutive failure counter on success
                consecutive_ai_failures[symbol] = 0
            except Exception as e:
                print(f"[ERROR] AI analysis failed for {symbol}: {e}")
                ai_failure_count += 1
                consecutive_ai_failures[symbol] = consecutive_ai_failures.get(symbol, 0) + 1
                symbol_failures = consecutive_ai_failures[symbol]

                # Check if auto_trade was active, and send alert only when disabling it (on 3rd consecutive failure for this symbol)
                try:
                    from db_manager import save_settings
                    current_settings = get_settings()
                    auto_trade_was_active = int(current_settings.get("auto_trade", 1)) == 1

                    if auto_trade_was_active and symbol_failures >= 3:
                        save_settings({"auto_trade": "0"})

                        ai_provider_name = "Ollama" if current_settings.get("ai_provider", "gemini").lower() == "ollama" else "Gemini"

                        from telegram_notifier import get_telegram_config, send_telegram_message
                        enabled, token, chat_id = get_telegram_config()
                        if enabled and token and chat_id:
                            err_alert = (
                                f"🚨 *فشل الاتصال بـ {ai_provider_name} ({ai_provider_name} Connection Failed)*\n\n"
                                f"• *الزوج:* `{symbol}`\n"
                                f"• *عدد الإخفاقات المتتالية:* `{symbol_failures}/3`\n"
                                f"• *نوع الخطأ:* `{str(e)[:150]}`\n\n"
                                f"🛑 *قرار الحماية:* تم إيقاف التداول التلقائي تلقائياً لحماية حسابك من التداول الفني العشوائي.\n"
                                f"💬 لتفعيل التداول الفني البديل يدوياً أرسل `/start_trade`."
                            )
                            send_telegram_message(token, chat_id, err_alert)
                    elif auto_trade_was_active:
                        ai_provider_name = "Ollama" if current_settings.get("ai_provider", "gemini").lower() == "ollama" else "Gemini"
                        print(f"[WARNING] AI Engine ({ai_provider_name}) failed for {symbol}. Consecutive failures: {symbol_failures}/3. Continuing with HOLD for this cycle.")
                except Exception as db_err:
                    print(f"[ERROR] Failed to save disabled trade state or send alert: {db_err}")

                last_scan_reports[symbol] = {
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "status": "Error (Skip)",
                    "details": f"AI API call failed ({symbol_failures}/3): {str(e)}. Skip current cycle.",
                    "structure": gann_context
                }

                # If this symbol hasn't reached the failure threshold, default to HOLD and let other pairs scan
                if symbol_failures < 3:
                    decision = "HOLD"
                    trade_params = None
                    reasoning = f"AI call failed ({symbol_failures}/3): {str(e)}. Defaulting to HOLD."
                else:
                    # Force max_positions_reached to True to skip other pairs in this cycle
                    max_positions_reached = True
                    continue
        else:
            print("[INFO] AI Risk Evaluation disabled. Executing purely based on technical strategy.")
            decision = proposed_action
            
            # Calculate default SL/TP (20 pips / 30 pips) if not overridden later
            symbol_info = mt5.symbol_info(symbol)
            entry_price = price_info['ask'] if proposed_action == "BUY" else price_info['bid']
            pip_size = 0.01 if (symbol.upper().endswith("JPY") or "JPY" in symbol.upper()) else 0.0001
            sl = entry_price - (20.0 * pip_size) if proposed_action == "BUY" else entry_price + (20.0 * pip_size)
            tp = entry_price + (30.0 * pip_size) if proposed_action == "BUY" else entry_price - (30.0 * pip_size)
            if symbol_info:
                sl = round(sl, symbol_info.digits)
                tp = round(tp, symbol_info.digits)
                
            trade_params = {
                "action": proposed_action,
                "symbol": symbol,
                "volume": 0.01,
                "sl": sl,
                "tp": tp,
                "reason": f"Pure Technical: {technical_reason}"
            }
            reasoning = f"Executed pure technical signal: {proposed_action}. Details: {technical_reason}"

        # Initialize default report status
        if decision in ["BUY", "SELL"]:
            status_str = f"AI Approved ({decision})" if ai_evaluation_enabled else f"Technical ({decision})"
        else:
            status_str = f"AI Rejected (HOLD)" if ai_evaluation_enabled else f"Technical (HOLD)"

        last_scan_reports[symbol] = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "status": status_str,
            "details": reasoning,
            "structure": gann_context
        }

        # Check active positions for this symbol to manage reversals and prevent double entries
        # Excludes the other five strategies' own positions (tagged in their comment)
        # — otherwise Gann's duplicate-entry guard and reversal-close logic below
        # would treat any position on this symbol as its own, closing or blocking
        # around trades that SMA5/Elliott/Range/Harmonic/Classical opened.
        symbol_positions = [
            pos for pos in active_positions
            if pos.magic == MAGIC_NUMBER and pos.symbol == symbol
            and not any(tag in (pos.comment or "").lower() for tag in ("sma5", "elliott", "range", "harmonic", "classical"))
        ]
        has_same_direction = False
        has_opposite_direction = False
        
        if symbol_positions:
            for pos in symbol_positions:
                pos_type_str = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                if pos_type_str == decision:
                    has_same_direction = True
                else:
                    has_opposite_direction = True
                    
        # If opposite positions are active, close them (Reversal Close)
        if decision in ["BUY", "SELL"] and has_opposite_direction:
            print(f"[REVERSAL] Opposite signal ({decision}) detected on {symbol}. Closing opposite positions...")
            for pos in symbol_positions:
                pos_type_str = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                if pos_type_str != decision:
                    print(f"[REVERSAL] Closing position #{pos.ticket} ({pos_type_str})")
                    close_res = close_position(ticket=pos.ticket, comment="Gann Reversal Close")
                    if close_res:
                        # Log closure to DB
                        log_trade_close(pos.ticket, pos.price_current, pos.profit)
                        print(f"[REVERSAL] Position #{pos.ticket} closed successfully.")
                        
            # Sync database / wait briefly to ensure MT5 updates active positions
            time.sleep(0.5)
            active_positions = get_open_positions()
            open_symbols = [pos.symbol for pos in active_positions]
            # Excludes the other five strategies' own positions (tagged in their
            # comment) — same scoping as above, re-applied after the reversal-close
            # since active_positions was just re-fetched.
            symbol_positions = [
                pos for pos in active_positions
                if pos.magic == MAGIC_NUMBER and pos.symbol == symbol
                and not any(tag in (pos.comment or "").lower() for tag in ("sma5", "elliott", "range", "harmonic", "classical"))
            ]
            has_opposite_direction = False
            
        # (Re-entry and duplicate entry checks are now handled pre-AI to conserve tokens)

        # 4. Handle decision
        if decision in ["BUY", "SELL"] and trade_params:
            print(f"\n🎯 [AI SIGNAL] Triggered {decision} on {symbol}")
            
            reason = trade_params.get("reason", "AI Strategy Entry")
            if grid_enabled and not gann_enabled:
                grid_sl = float(settings.get("grid_sl", 20.0))
                grid_tp = float(settings.get("grid_tp", 20.0))
                symbol_info = mt5.symbol_info(symbol)
                entry_price = price_info['ask'] if decision == "BUY" else price_info['bid']
                contract_size = symbol_info.trade_contract_size if symbol_info else 100000.0
                if contract_size < 1000.0:
                    pip_size = (0.01 * entry_price) / 10.0
                else:
                    pip_size = 0.01 if (symbol.upper().endswith("JPY") or "JPY" in symbol.upper()) else 0.0001
                
                # Override SL and TP from database settings
                sl = entry_price - (grid_sl * pip_size) if decision == "BUY" else entry_price + (grid_sl * pip_size)
                tp = entry_price + (grid_tp * pip_size) if decision == "BUY" else entry_price - (grid_tp * pip_size)
                sl = round(sl, symbol_info.digits)
                tp = round(tp, symbol_info.digits)
                print(f"[GRID OVERRIDE] Enforcing grid parameters. SL: {sl} ({grid_sl} pips), TP: {tp} ({grid_tp} pips)")
            elif gann_enabled and 'dyn' in locals() and dyn:
                sl = dyn["sl_price"]
                tp = dyn["target_price"]
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    sl = round(sl, symbol_info.digits)
                    tp = round(tp, symbol_info.digits)
                print(f"[GANN OVERRIDE] Enforcing dynamic Gann parameters. SL: {sl} (Angle 0° / A), TP: {tp} (Target Angle: {dyn['target_angle']}°)")
                reason = "Gann: " + reason
            else:
                sl = float(trade_params.get("sl", 0.0))
                tp = float(trade_params.get("tp", 0.0))
                
            entry_price = price_info['ask'] if decision == "BUY" else price_info['bid']
            
            if sl <= 0:
                reject_msg = f"Trade rejected: Stop Loss is mandatory but was not specified (SL: {sl})"
                print(f"[REJECT] {reject_msg}")
                last_scan_reports[symbol]["status"] = "AI Rejection"
                last_scan_reports[symbol]["details"] = reject_msg
                continue

            # --- RISK/REWARD (R:R) RATIO CHECK ---
            sl_distance = abs(entry_price - sl)
            tp_distance = abs(tp - entry_price)
            
            if sl_distance <= 0:
                reject_msg = "Trade rejected: Stop Loss is at or ahead of entry price."
                print(f"[REJECT] {reject_msg}")
                last_scan_reports[symbol]["status"] = "AI Rejection"
                last_scan_reports[symbol]["details"] = reject_msg
                continue
                
            rr_ratio = tp_distance / sl_distance
            min_rr_ratio = float(settings.get("min_rr_ratio", 1.0) if gann_enabled else settings.get("min_rr_ratio", 1.5))
            if rr_ratio < min_rr_ratio:
                reject_msg = f"Trade rejected: Risk/Reward ratio is 1:{rr_ratio:.2f} (must be at least 1:{min_rr_ratio:.2f})"
                print(f"[REJECT] {reject_msg}")
                last_scan_reports[symbol]["status"] = "AI Rejection"
                last_scan_reports[symbol]["details"] = reject_msg
                continue
                
            print(f"[OK] Risk validation passed: Risk/Reward ratio is 1:{rr_ratio:.2f}")

            # --- DYNAMIC LOT SIZE CALCULATION ---
            volume = calculate_lot_size(symbol, sl, entry_price, risk_percent=risk_percent)
            print(f"[LOTS] Calculated Dynamic Lot: {volume} lots (risking {risk_percent}% of balance)")

            if not auto_trade_enabled:
                manual_msg = f"Signal-Only Mode: Trade was NOT executed. Signal: {decision} on {symbol} at {entry_price} (SL: {sl}, TP: {tp})"
                print(f"[MANUAL] {manual_msg}")
                last_scan_reports[symbol]["status"] = "Signal Only"
                last_scan_reports[symbol]["details"] = manual_msg
                continue

            # Execute Trade
            trade_res = open_trade(
                action=decision,
                symbol=symbol,
                volume=volume,
                sl=sl,
                tp=tp,
                magic=MAGIC_NUMBER,
                comment=reason[:28]
            )

            # Log to SQLite DB
            if trade_res and trade_res["success"]:
                entry_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                log_trade_open(
                    ticket=trade_res["ticket"],
                    symbol=symbol,
                    action=decision,
                    volume=volume,
                    entry_price=trade_res["price"],
                    sl=sl,
                    tp=tp,
                    reason=reason,
                    gann_data=gann_context
                )
                print(f"[DB] Trade ticket {trade_res['ticket']} logged successfully in SQLite.")
                
                exec_msg = f"Trade executed: {decision} {volume} lots at {trade_res['price']} (SL: {sl}, TP: {tp}). Ticket: {trade_res['ticket']}."
                last_scan_reports[symbol]["status"] = f"Executed {decision}"
                last_scan_reports[symbol]["details"] = exec_msg
                
                # Generate and save chart screenshot
                screenshot_url = None
                try:
                    from chart_generator import generate_chart_screenshot
                    from db_manager import update_trade_screenshot
                    
                    screenshot_url = generate_chart_screenshot(
                        symbol=symbol,
                        ticket=trade_res["ticket"],
                        entry_price=trade_res["price"],
                        sl_price=sl,
                        tp_price=tp,
                        gann_data=gann_context,
                        timeframe_str=timeframe_str,
                        entry_time_str=entry_time_str
                    )
                    update_trade_screenshot(trade_res["ticket"], screenshot_url)
                    print(f"[SCREENSHOT] Saved trade chart to {screenshot_url}")
                except Exception as ex:
                    print(f"[SCREENSHOT] [ERROR] Failed to save screenshot: {ex}")
                    
                # Telegram notification
                try:
                    from telegram_notifier import notify_trade_open
                    from db_manager import update_trade_telegram_msg_id
                    
                    msg_id = notify_trade_open(
                        ticket=trade_res["ticket"],
                        symbol=symbol,
                        action=decision,
                        volume=volume,
                        entry_price=trade_res["price"],
                        sl=sl,
                        tp=tp,
                        reason=reason,
                        screenshot_url=screenshot_url,
                        gann_context=gann_context,
                        timeframe=timeframe_str
                    )
                    if msg_id:
                        update_trade_telegram_msg_id(trade_res["ticket"], msg_id)
                        print(f"[TELEGRAM] Trade open notification sent (Msg ID: {msg_id})")
                except Exception as tg_ex:
                    print(f"[TELEGRAM] [ERROR] Failed to send open alert: {tg_ex}")
            else:
                fail_msg = f"Failed to execute trade order in MetaTrader 5."
                print(f"[ERROR] {fail_msg}")
                last_scan_reports[symbol]["status"] = "Execution Fail"
                last_scan_reports[symbol]["details"] = fail_msg
        else:
            print(f"\n➡️ [AI DECISION] {decision}")
            print(f"  Reasoning: {reasoning[:300]}")

    # 4. Run the SMA(5) mean-reversion strategy — fully independent of the Gann
    # loop above, its own signal source, its own pre-trade filters, its own SL/TP.
    if not max_positions_reached:
        try:
            manage_sma5_reversion_strategy(settings, active_news_events, risk_percent, auto_trade_enabled)
        except Exception as sma5_cycle_ex:
            print(f"[ERROR] SMA5 reversion strategy pass failed: {sma5_cycle_ex}")

    # 5. Run the Elliott Wave strategy — same independence guarantee as SMA5 above.
    if not max_positions_reached:
        try:
            manage_elliott_wave_strategy(settings, active_news_events, risk_percent, auto_trade_enabled)
        except Exception as elliott_cycle_ex:
            print(f"[ERROR] Elliott Wave strategy pass failed: {elliott_cycle_ex}")

    # 6. Run the Range Trading strategy — same independence guarantee as the other three.
    if not max_positions_reached:
        try:
            manage_range_trading_strategy(settings, active_news_events, risk_percent, auto_trade_enabled)
        except Exception as range_cycle_ex:
            print(f"[ERROR] Range Trading strategy pass failed: {range_cycle_ex}")

    # 7. Run the Harmonic Patterns strategy — same independence guarantee as the other four.
    if not max_positions_reached:
        try:
            manage_harmonic_patterns_strategy(settings, active_news_events, risk_percent, auto_trade_enabled)
        except Exception as harmonic_cycle_ex:
            print(f"[ERROR] Harmonic Patterns strategy pass failed: {harmonic_cycle_ex}")

    # 8. Run the Classical Chart Patterns strategy — same independence guarantee as the other five.
    if not max_positions_reached:
        try:
            manage_classical_patterns_strategy(settings, active_news_events, risk_percent, auto_trade_enabled)
        except Exception as classical_cycle_ex:
            print(f"[ERROR] Classical Patterns strategy pass failed: {classical_cycle_ex}")

    # Fetch latest open positions for the report
    try:
        latest_active = get_open_positions()
    except Exception:
        latest_active = []

    # Print Live Portfolio Summary Report
    print("\n" + "=" * 90)
    print("                       LIVE PORTFOLIO SUMMARY REPORT")
    print("=" * 90)
    
    # 1. Active Positions Table
    active_by_symbol = {}
    for pos in latest_active:
        sym = pos.symbol
        if sym not in active_by_symbol:
            active_by_symbol[sym] = {"count": 0, "lots": 0.0, "profit": 0.0}
        active_by_symbol[sym]["count"] += 1
        active_by_symbol[sym]["lots"] += pos.volume
        active_by_symbol[sym]["profit"] += pos.profit

    # 2. Historical stats from SQLite
    try:
        from db_manager import get_trade_history
        all_history = get_trade_history(limit=500) # get recent 500 trades
    except Exception:
        all_history = []
        
    hist_by_symbol = {}
    for trade in all_history:
        if trade.get("status") == "CLOSED":
            sym = trade.get("symbol")
            if sym not in hist_by_symbol:
                hist_by_symbol[sym] = {"wins": 0, "losses": 0, "profit": 0.0}
            profit_val = float(trade.get("profit") or 0.0)
            if profit_val >= 0:
                hist_by_symbol[sym]["wins"] += 1
            else:
                hist_by_symbol[sym]["losses"] += 1
            hist_by_symbol[sym]["profit"] += profit_val

    print(f"{'Symbol':<12} | {'Active Trades':<13} | {'Total Lots':<10} | {'Floating P/L':<12} | {'Closed Wins/Losses':<18} | {'Total Closed P/L':<16}")
    print("-" * 90)
    
    for sym in symbols_to_trade:
        active = active_by_symbol.get(sym, {"count": 0, "lots": 0.0, "profit": 0.0})
        hist = hist_by_symbol.get(sym, {"wins": 0, "losses": 0, "profit": 0.0})
        
        active_str = f"{active['count']} trades"
        lots_str = f"{active['lots']:.2f}"
        floating_str = f"${active['profit']:+.2f}"
        win_loss_str = f"{hist['wins']}W / {hist['losses']}L"
        closed_profit_str = f"${hist['profit']:+.2f}"
        
        print(f"{sym:<12} | {active_str:<13} | {lots_str:<10} | {floating_str:<12} | {win_loss_str:<18} | {closed_profit_str:<16}")
    print("=" * 90)

    # 2.5 Daily AI cost cap: accumulate paid-API spend and auto-disable AI evaluation
    # if it crosses the configured daily ceiling (Ollama calls cost $0.00, so this
    # only ever engages when ai_provider is a paid API such as Gemini).
    if cycle_cost > 0:
        try:
            from db_manager import save_settings
            today_str = time.strftime('%Y-%m-%d')
            cost_date = settings.get("ai_cost_today_date", today_str)
            accumulated_cost = float(settings.get("ai_cost_today", 0.0))
            if cost_date != today_str:
                accumulated_cost = 0.0
                cost_date = today_str
            accumulated_cost += cycle_cost

            save_settings({
                "ai_cost_today": str(accumulated_cost),
                "ai_cost_today_date": cost_date
            })
            print(f"[AI COST] Accumulated AI API cost today ({cost_date}): ${accumulated_cost:.4f}")

            daily_cap = float(settings.get("ai_cost_cap_daily", 1.0))
            if accumulated_cost >= daily_cap and int(settings.get("ai_evaluation", 1)) == 1:
                save_settings({"ai_evaluation": "0"})
                print(f"[COST CAP] Daily AI cost cap (${daily_cap:.2f}) reached. AI evaluation disabled for the rest of the day; falling back to pure technical strategy.")
                try:
                    from telegram_notifier import get_telegram_config, send_telegram_message
                    cap_enabled, cap_token, cap_chat_id = get_telegram_config()
                    if cap_enabled and cap_token and cap_chat_id:
                        cap_alert = (
                            f"\U0001F4B0 *تم بلوغ سقف تكلفة الذكاء الاصطناعي اليومي*\n\n"
                            f"• التكلفة المتراكمة اليوم: `${accumulated_cost:.4f}`\n"
                            f"• السقف المحدد: `${daily_cap:.2f}`\n\n"
                            f"\U0001F6D1 تم تعطيل طبقة تقييم الذكاء الاصطناعي تلقائياً لبقية اليوم، والتنفيذ سيعتمد على الاستراتيجية الفنية البحتة.\n"
                            f"\U0001F4AC لإعادة التفعيل يدوياً أرسل `/toggle_ai`."
                        )
                        send_telegram_message(cap_token, cap_chat_id, cap_alert)
                except Exception as cap_alert_ex:
                    print(f"[ERROR] Failed to send AI cost cap alert: {cap_alert_ex}")
        except Exception as cost_cap_ex:
            print(f"[ERROR] Failed to track/enforce daily AI cost cap: {cost_cap_ex}")

    # 3. Send Telegram Cycle Report
    try:
        from telegram_notifier import get_telegram_config, send_telegram_message
        enabled, token, chat_id = get_telegram_config()
        if enabled and token and chat_id:
            # Active positions summary
            active_count = len(latest_active)
            total_lots = sum(pos.volume for pos in latest_active)
            total_profit = sum(pos.profit for pos in latest_active)

            # Classify active positions by strategy
            open_by_strat_str = ""
            if latest_active:
                try:
                    from db_manager import get_trade_by_ticket, classify_trade_strategy
                    strat_groups = {}
                    for pos in latest_active:
                        db_t = get_trade_by_ticket(pos.ticket)
                        reason_str = db_t.get("reason") if db_t else (pos.comment or "")
                        st_name = classify_trade_strategy(reason_str)
                        if st_name not in strat_groups:
                            strat_groups[st_name] = {"count": 0, "volume": 0.0, "profit": 0.0, "symbols": []}
                        strat_groups[st_name]["count"] += 1
                        strat_groups[st_name]["volume"] += pos.volume
                        strat_groups[st_name]["profit"] += pos.profit
                        strat_groups[st_name]["symbols"].append(pos.symbol)

                    lines_active = []
                    for st_name, gdata in strat_groups.items():
                        unique_syms = list(dict.fromkeys(gdata["symbols"]))
                        syms_text = ", ".join(unique_syms[:3])
                        if len(unique_syms) > 3:
                            syms_text += f" +{len(unique_syms)-3}"
                        lines_active.append(
                            f"  • *{st_name}:* `{gdata['count']}` صفقات (`{gdata['volume']:.2f}l`) — `{gdata['profit']:+.2f} USD` [{syms_text}]"
                        )
                    if lines_active:
                        open_by_strat_str = "• *تفصيل الصفقات المفتوحة حسب الاستراتيجية:*\n" + "\n".join(lines_active) + "\n"
                except Exception as open_strat_ex:
                    print(f"[ERROR] Failed to breakdown active positions by strategy: {open_strat_ex}")
            
            # Find signals triggered in this cycle
            triggered_signals = []
            for sym, report in last_scan_reports.items():
                status = report.get("status", "")
                details = report.get("details", "")
                if "Executed" in status or "AI Approved" in status or "AI Rejected" in status or "Technical" in status or "Approved" in status:
                    # Clean up details to be a short reasoning message
                    clean_reason = details.replace("HOLD - AI rejected proposed trade", "").replace("HOLD - ", "").strip()
                    if len(clean_reason) > 60:
                        clean_reason = clean_reason[:57] + "..."
                    reason_suffix = f" ({clean_reason})" if clean_reason else ""
                    triggered_signals.append(f"• `{sym}`: {status}{reason_suffix}")
                elif "Blocked" in status:
                    triggered_signals.append(f"• `{sym}`: 🚫 {status}")
                elif status == "News Frozen":
                    triggered_signals.append(f"• `{sym}`: 📰 *مجمّد بسبب الأخبار (News Frozen)*")
            
            signals_str = "\n".join(triggered_signals) if triggered_signals else "• لا يوجد إشارات جديدة منفذة"
            
            # Format account info
            balance_val = account_info.get("balance", 0.0)
            equity_val = account_info.get("equity", 0.0)
            free_margin = account_info.get("margin_free", 0.0)
            currency = account_info.get("currency", "USD")
            
            ai_provider_name = "Ollama" if settings.get("ai_provider", "gemini").lower() == "ollama" else "Gemini"
            ai_err_str = f"\n⚠️ *خطأ اتصال الذكاء الاصطناعي ({ai_provider_name}):* فشل الاتصال لـ `{ai_failure_count}` أزواج (تم تفعيل البديل الفني)." if ai_failure_count > 0 else ""

            # Token usage details
            usage_str = ""
            if cycle_in_tokens > 0 or cycle_out_tokens > 0:
                usage_str = (
                    f"📊 *استهلاك الذكاء الاصطناعي (AI Token Usage):*\n"
                    f"• الـ Tokens المرسلة (Input): `{cycle_in_tokens}` Tokens\n"
                    f"• الـ Tokens المستلمة (Output): `{cycle_out_tokens}` Tokens\n"
                    f"• التكلفة التقديرية للدورة: `${cycle_cost:.5f} USD`\n\n"
                )

            # Per-strategy performance breakdown (all closed trades, attributed via reason text)
            strategy_perf_str = ""
            try:
                from db_manager import get_strategy_performance_summary
                strategy_summary = get_strategy_performance_summary()
                if strategy_summary:
                    lines = []
                    for strat_name, stats in strategy_summary.items():
                        total = stats["wins"] + stats["losses"]
                        wr = (stats["wins"] / total * 100.0) if total > 0 else 0.0
                        lines.append(f"• *{strat_name}:* `{stats['wins']}W/{stats['losses']}L` ({wr:.0f}%) — `${stats['profit']:+.2f}`")
                    strategy_perf_str = "📊 *أداء كل استراتيجية (Strategy Performance):*\n" + "\n".join(lines) + "\n\n"
            except Exception as strat_perf_ex:
                print(f"[ERROR] Failed to build strategy performance summary: {strat_perf_ex}")

            report_msg = (
                f"🔄 *تقرير دورة الفحص (Scanning Cycle Report)*\n"
                f"⏰ *الوقت:* `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n\n"
                f"💳 *حالة الحساب (Account Details):*\n"
                f"• الرصيد (Balance): `{balance_val:.2f} {currency}`\n"
                f"• السيولة (Equity): `{equity_val:.2f} {currency}`\n"
                f"• الهامش الحر (Free Margin): `{free_margin:.2f} {currency}`\n\n"
                f"📈 *الصفقات المفتوحة (Portfolio Status):*\n"
                f"• إجمالي الصفقات: `{active_count}` صفقات (`{total_lots:.2f} lots`) — الأرباح: `{total_profit:+.2f} {currency}`\n"
                f"{open_by_strat_str}\n"
                f"{strategy_perf_str}"
                f"{usage_str}"
                f"🎯 *أحداث الدورة الحالية (Current Signals):*\n"
                f"{signals_str}\n{ai_err_str}\n"
                 f"⚙️ تم فحص {len(symbols_to_trade) * len(scan_timeframes)} زوج (Gann) + "
                 f"{len(settings.get('sma5_reversion_symbols', [])) * 1} (SMA5 على H1 فقط) + "
                 f"{len(settings.get('elliott_wave_symbols', [])) * len(scan_timeframes)} (Elliott) + "
                 f"{len(settings.get('range_trading_symbols', [])) * len(scan_timeframes)} (Range) + "
                 f"{len(settings.get('harmonic_patterns_symbols', [])) * len(scan_timeframes)} (Harmonic) + "
                 f"{len(settings.get('classical_patterns_symbols', [])) * len(scan_timeframes)} (Classical) "
                 f"عبر أطر زمنية ({', '.join(scan_timeframes)}) بنجاح."
            )
            send_telegram_message(token, chat_id, report_msg)
            print("[TELEGRAM] Scanning cycle report sent successfully.")
    except Exception as tg_report_ex:
        print(f"[ERROR] Failed to send Telegram cycle report: {tg_report_ex}")

    print("\n" + "=" * 60)
    print("🔄 SCANNING CYCLE COMPLETED")
    print("=" * 60)


def is_market_closed(reference_symbol="EURUSDm", stale_minutes=15):
    """
    Detect real forex market closure (weekend) by checking how fresh the broker's
    last tick is for a reference symbol. During active trading, ticks update within
    seconds; a tick older than `stale_minutes` means prices aren't moving (closed).

    Uses a fixed major FX pair rather than the configured symbol list — several
    configured symbols are crypto (trades 24/7), which would never read as closed.
    """
    try:
        tick = mt5.symbol_info_tick(reference_symbol)
        if tick is None or tick.time == 0:
            return True
        age_seconds = time.time() - tick.time
        return age_seconds > (stale_minutes * 60)
    except Exception as e:
        print(f"[WARNING] Could not determine market status: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="AI Algorithmic Trading Bot Daemon")
    parser.add_argument("--loop", action="store_true", help="Run in a continuous loop")
    parser.add_argument("--interval", type=int, default=300, help="Interval between scanning cycles in seconds")
    args = parser.parse_args()

    print("=" * 60)
    print("          AI ALGORITHMIC TRADING BOT")
    print(f"  Mode: {'Continuous Loop' if args.loop else 'Single Scan'}")
    if args.loop:
        print(f"  Interval: {args.interval} seconds")
    print("============================================================\n")

    # Connect to MT5
    if not connect_mt5():
        print("[FATAL] Could not connect to MT5 Terminal. Exiting.")
        sys.exit(1)

    # Send startup message
    try:
        from telegram_notifier import get_telegram_config, send_telegram_message
        enabled, token, chat_id = get_telegram_config()
        if enabled and token and chat_id:
            startup_msg = (
                f"🚀 *تم تشغيل بوت التداول بنجاح (Bot Started)*\n\n"
                f"• *السيرفر:* `ويندوز VPS`\n"
                f"• *وضع التشغيل:* `مستمر (Continuous Loop)`\n"
                f"• *دورة الفحص:* كل `{args.interval}` ثانية ({args.interval // 60} دقيقة)"
            )
            send_telegram_message(token, chat_id, startup_msg)
    except Exception as tg_startup_ex:
        print(f"[ERROR] Failed to send startup TG alert: {tg_startup_ex}")

    try:
        if args.loop:
            last_full_scan_time = 0.0
            market_was_closed = False
            print(f"[LOOP] Starting loop. Positions checked every 30s. Full scan every {args.interval}s.")

            while True:
                # Ensure connection is active
                if mt5.terminal_info() is None or mt5.account_info() is None:
                    print("[WARNING] Connection lost. Attempting to reconnect...")
                    if not connect_mt5():
                        print("[ERROR] Reconnection failed. Retrying next cycle.")
                        time.sleep(30)
                        continue

                # 1. Frequently manage active positions and sync positions (every 30 seconds)
                try:
                    sync_db_with_mt5_positions()
                    manage_active_positions_grid()
                except Exception as grid_ex:
                    print(f"[ERROR] Frequent grid management failed: {grid_ex}")

                # 2. Skip the heavier scanning cycle while the real market is closed
                # (weekend) — the loop itself, position management, and reconnection
                # checks above keep running as usual.
                if is_market_closed():
                    if not market_was_closed:
                        print("[MARKET] Market appears closed (stale quotes detected). Pausing scanning cycles until it reopens.")
                        market_was_closed = True
                        try:
                            from telegram_notifier import get_telegram_config, send_telegram_message
                            mkt_enabled, mkt_token, mkt_chat_id = get_telegram_config()
                            if mkt_enabled and mkt_token and mkt_chat_id:
                                send_telegram_message(
                                    mkt_token, mkt_chat_id,
                                    "\U0001F634 *السوق مغلق حالياً*\n\nتم إيقاف دورات الفحص مؤقتاً (لا توجد أسعار جديدة). سيتم استئناف الفحص تلقائياً فور فتح السوق."
                                )
                        except Exception as mkt_tg_ex:
                            print(f"[ERROR] Failed to send market-closed Telegram alert: {mkt_tg_ex}")
                else:
                    if market_was_closed:
                        print("[MARKET] Market appears open again (fresh quotes detected). Resuming scanning.")
                        market_was_closed = False
                        last_full_scan_time = 0.0  # force an immediate scan on reopen
                        try:
                            from telegram_notifier import get_telegram_config, send_telegram_message
                            mkt_enabled, mkt_token, mkt_chat_id = get_telegram_config()
                            if mkt_enabled and mkt_token and mkt_chat_id:
                                send_telegram_message(
                                    mkt_token, mkt_chat_id,
                                    "\U0001F514 *تم فتح السوق من جديد*\n\nاستُؤنف الفحص التلقائي للفرص."
                                )
                        except Exception as mkt_tg_ex:
                            print(f"[ERROR] Failed to send market-reopened Telegram alert: {mkt_tg_ex}")

                    # 3. Run full scanning cycle only if args.interval seconds have elapsed
                    current_time = time.time()
                    if current_time - last_full_scan_time >= args.interval:
                        check_and_execute_trading_cycle()
                        last_full_scan_time = current_time

                time.sleep(30)
        else:
            check_and_execute_trading_cycle()
        
    except KeyboardInterrupt:
        print("\n[INFO] Bot stopped by user.")
        # Send user shutdown message
        try:
            from telegram_notifier import get_telegram_config, send_telegram_message
            enabled, token, chat_id = get_telegram_config()
            if enabled and token and chat_id:
                send_telegram_message(token, chat_id, "⏹️ *تم إيقاف بوت التداول يدوياً بواسطة المستخدم (Bot Stopped).*")
        except Exception:
            pass
    except Exception as e:
        print(f"[ERROR] Unexpected error in main loop: {e}")
        # Send crash message
        try:
            from telegram_notifier import get_telegram_config, send_telegram_message
            enabled, token, chat_id = get_telegram_config()
            if enabled and token and chat_id:
                send_telegram_message(token, chat_id, f"⚠️ *توقف بوت التداول بسبب خطأ فني مفاجئ (Bot Crashed):*\n`{str(e)}`")
        except Exception:
            pass
    finally:
        # Disconnect safely
        disconnect_mt5()

if __name__ == "__main__":
    main()
