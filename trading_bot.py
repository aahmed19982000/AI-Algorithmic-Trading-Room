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
    log_balance_snapshot,
    get_db_connection
)
from gann_helper import detect_market_structure, calculate_gann_levels

# Load configuration
load_dotenv()
MAGIC_NUMBER = int(os.getenv("BOT_MAGIC", "20260604"))
DEFAULT_TIMEFRAME = os.getenv("TRADING_TIMEFRAME", "H4")

# Global dict to store the results of the last scanning cycle per symbol
last_scan_reports = {}


def is_setup_stopped_out(symbol, gann_context, decision):
    """
    Checks if the last closed trade for this symbol was closed due to Stop Loss,
    and if so, prevents re-entry if either:
    1. The Gann setup (time_B & time_C) matches the stopped-out trade.
    2. Or a 4-hour cooldown has not passed since the Stop Loss hit (for technical/general safety).
    """
    import json
    from datetime import datetime
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT ticket, action, profit, gann_data, close_time, reason 
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
        
    ticket, action, profit, db_gann_str, close_time, reason = row
    
    # Check if the close reason indicated hitting Stop Loss
    is_stop_loss = False
    if reason and "Hit Stop Loss" in reason:
        is_stop_loss = True
    elif profit < 0:
        # Fallback: if profit is negative, treat as stopped out/loss
        is_stop_loss = True
        
    if not is_stop_loss:
        return False
        
    # If the last trade direction was different, we don't block
    if action != decision:
        return False
        
    # Case 1: Check Gann swing structure match (to block re-entry on the same structure)
    if gann_context and db_gann_str:
        try:
            db_gann = json.loads(db_gann_str)
            if db_gann:
                curr_time_B = gann_context.get("time_B")
                curr_time_C = gann_context.get("time_C")
                db_time_B = db_gann.get("time_B")
                db_time_C = db_gann.get("time_C")
                
                if curr_time_B == db_time_B and curr_time_C == db_time_C:
                    print(f"[RE-ENTRY BLOCK] Same Gann setup (B: {curr_time_B}, C: {curr_time_C}) already stopped out on ticket #{ticket}.")
                    return True
        except Exception as e:
            print(f"[ERROR] Comparing Gann data: {e}")
            
    # Case 2: Cooldown check (prevent any re-entry on same symbol & direction for 4 hours)
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


def calculate_lot_size(symbol, sl_price, entry_price, risk_percent=1.0):
    """
    Calculate position size (lots) based on risk percentage of account balance
    and Stop Loss distance in points.
    """
    # Hard cap risk_percent to 3% max to protect the account
    risk_percent = min(risk_percent, 3.0)
    
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
            elif "TP 90% Close Early" in close_comment or "Take Profit 90%" in close_comment:
                exit_reason = "Take Profit 90% Close Early (إغلاق مبكر عند 90% من الهدف)"
            elif "M5" in close_comment:
                exit_reason = f"M5 Swing Close ({close_comment})"
            elif close_reason_code == 4: # DEAL_REASON_SL
                exit_reason = "Hit Stop Loss (ضرب الاستوب)"
            elif close_reason_code == 5: # DEAL_REASON_TP
                exit_reason = "Hit Take Profit (ضرب الهدف)"
            elif close_reason_code == 6: # DEAL_REASON_SO
                exit_reason = "Stop Out (تسييل الحساب - مارجن كول)"
            elif close_reason_code == 0: # DEAL_REASON_CLIENT
                exit_reason = "Manual MT5 Close (إغلاق يدوي مباشر من تطبيق/منصة MT5)"
            else:
                # Fallback based on profit
                if profit < 0:
                    exit_reason = "Hit Stop Loss (ضرب الاستوب - تلقائي)"
                elif profit > 0:
                    exit_reason = "Hit Take Profit (ضرب الهدف - تلقائي)"
                elif close_comment:
                    exit_reason = f"Closed: {close_comment}"

            # Log as closed in database with exit reason
            log_trade_close(ticket, close_price, profit, exit_reason=exit_reason)
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


def check_h1_exit_signal(symbol, pos_type):
    """
    Checks if there is an exit signal on H1 timeframe:
    - For BUY (0): returns True if a Lower Low (LL) is formed on H1, 
      where the new downward wave length is >= 25% larger than the previous, and the low is recent.
    - For SELL (1): returns True if a Higher High (HH) is formed on H1,
      where the new upward wave length is >= 25% larger than the previous, and the high is recent.
    """
    df = get_candles(symbol=symbol, timeframe=mt5.TIMEFRAME_H1, count=60)
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
                    # 3. Wave size check (latest downward wave vs previous downward wave)
                    preceding_highs_latest = [sh for sh in swing_highs if sh[0] < latest_idx]
                    preceding_highs_prev = [sh for sh in swing_highs if sh[0] < prev_idx]
                    
                    if preceding_highs_latest and preceding_highs_prev:
                        latest_wave = preceding_highs_latest[-1][1] - latest_val
                        prev_wave = preceding_highs_prev[-1][1] - prev_val
                        
                        if prev_wave > 0 and latest_wave >= 1.25 * prev_wave:
                            print(f"[H1 EXIT] BUY trade exit triggered: Recent Lower Low detected. "
                                  f"Latest Low: {latest_val:.5f} (Wave: {latest_wave:.5f}), "
                                  f"Prev Low: {prev_val:.5f} (Wave: {prev_wave:.5f}). "
                                  f"Wave length increased by {(latest_wave/prev_wave - 1)*100:.1f}%.")
                            return True
                            
    elif pos_type == 1:  # SELL trade -> Check for Higher High (HH)
        if len(swing_highs) >= 2:
            latest_idx, latest_val = swing_highs[-1]
            prev_idx, prev_val = swing_highs[-2]
            
            # 1. Recency check (Latest high must have formed within the last 4 completed candles)
            if len(df) - 1 - latest_idx <= 4:
                # 2. Higher High check
                if latest_val > prev_val:
                    # 3. Wave size check (latest upward wave vs previous upward wave)
                    preceding_lows_latest = [sl for sl in swing_lows if sl[0] < latest_idx]
                    preceding_lows_prev = [sl for sl in swing_lows if sl[0] < prev_idx]
                    
                    if preceding_lows_latest and preceding_lows_prev:
                        latest_wave = latest_val - preceding_lows_latest[-1][1]
                        prev_wave = prev_val - preceding_lows_prev[-1][1]
                        
                        if prev_wave > 0 and latest_wave >= 1.25 * prev_wave:
                            print(f"[H1 EXIT] SELL trade exit triggered: Recent Higher High detected. "
                                  f"Latest High: {latest_val:.5f} (Wave: {latest_wave:.5f}), "
                                  f"Prev High: {prev_val:.5f} (Wave: {prev_wave:.5f}). "
                                  f"Wave length increased by {(latest_wave/prev_wave - 1)*100:.1f}%.")
                            return True

    return False


def manage_active_positions_tp_sl():
    """
    Monitors active positions and applies:
    1. Moves SL to entry price (breakeven) when 50% of the target (TP) distance is reached.
    2. Closes the trade immediately when 90% of the target is reached (only 10% remaining).
    3. Closes the trade if M5 forms Higher High (for SELL) or Lower Low (for BUY).
    """
    active_positions = get_open_positions()
    if not active_positions:
        return

    for pos in active_positions:
        if pos.magic != MAGIC_NUMBER:
            continue

        symbol = pos.symbol
        ticket = pos.ticket
        entry_price = pos.price_open
        current_price = pos.price_current
        tp = pos.tp
        sl = pos.sl
        pos_type = pos.type # 0 = BUY, 1 = SELL

        # If there is no TP, we cannot calculate percentages
        if tp <= 0:
            continue

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            continue

        digits = symbol_info.digits

        # Calculate distances
        if pos_type == 0:  # BUY
            tp_dist = tp - entry_price
            current_dist = current_price - entry_price
        else:  # SELL
            tp_dist = entry_price - tp
            current_dist = entry_price - current_price

        if tp_dist <= 0:
            continue

        # 1. Check H1 swing exit signal
        if check_h1_exit_signal(symbol, pos_type):
            exit_reason_ar = "تكوين قمة أعلى (للبيع) أو قاع أدنى (للشراء) على H1"
            exit_reason_en = f"H1 {'Lower Low' if pos_type == 0 else 'Higher High'} Exit"
            print(f"[H1 EXIT] Closing Position #{ticket} ({symbol}) due to H1 exit signal.")
            
            # Close the trade on MT5
            close_res = close_position(ticket=ticket, comment=exit_reason_en[:28])
            if close_res:
                # Log closed trade in DB
                log_trade_close(ticket, current_price, pos.profit, exit_reason=f"{exit_reason_en} ({exit_reason_ar})")
                
                # Fetch trade info from DB for notifications
                try:
                    from db_manager import get_trade_history
                    history = get_trade_history(limit=50)
                    db_trade = next((t for t in history if t["ticket"] == ticket), None)
                    
                    # Notify via Telegram
                    result = "WIN" if pos.profit >= 0 else "LOSS"
                    close_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    from telegram_notifier import notify_trade_close
                    notify_trade_close(
                        ticket=ticket,
                        symbol=symbol,
                        action="BUY" if pos_type == 0 else "SELL",
                        volume=pos.volume,
                        entry_price=entry_price,
                        close_price=current_price,
                        profit=pos.profit,
                        result=result,
                        open_time=db_trade["open_time"] if db_trade else close_time_str,
                        close_time=close_time_str,
                        screenshot_url=db_trade.get("screenshot_url") if db_trade else None,
                        original_msg_id=db_trade.get("telegram_msg_id") if db_trade else None,
                        exit_reason=f"{exit_reason_en} ({exit_reason_ar})"
                    )
                except Exception as tg_ex:
                    print(f"[TELEGRAM] [ERROR] Failed to send M5 exit notification: {tg_ex}")
            continue # Go to next position

        pct_reached = current_dist / tp_dist

        # 2. Close early if 90% of target is reached (10% remaining)
        if pct_reached >= 0.90:
            print(f"[TP CLOSE EARLY] Position #{ticket} ({symbol}) reached {pct_reached*100:.1f}% of TP. Closing early.")
            
            # Close the trade on MT5
            close_res = close_position(ticket=ticket, comment="TP 90% Close Early")
            if close_res:
                # Log closed trade in DB
                log_trade_close(ticket, current_price, pos.profit, exit_reason="Take Profit 90% Close Early (إغلاق مبكر عند 90% من الهدف)")
                
                # Fetch trade info from DB for notifications
                try:
                    from db_manager import get_trade_history
                    history = get_trade_history(limit=50)
                    db_trade = next((t for t in history if t["ticket"] == ticket), None)
                    
                    # Notify via Telegram
                    result = "WIN" if pos.profit >= 0 else "LOSS"
                    close_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    from telegram_notifier import notify_trade_close
                    notify_trade_close(
                        ticket=ticket,
                        symbol=symbol,
                        action="BUY" if pos_type == 0 else "SELL",
                        volume=pos.volume,
                        entry_price=entry_price,
                        close_price=current_price,
                        profit=pos.profit,
                        result=result,
                        open_time=db_trade["open_time"] if db_trade else close_time_str,
                        close_time=close_time_str,
                        screenshot_url=db_trade.get("screenshot_url") if db_trade else None,
                        original_msg_id=db_trade.get("telegram_msg_id") if db_trade else None,
                        exit_reason="Take Profit 90% Close Early (إغلاق مبكر عند 90% من الهدف)"
                    )
                except Exception as tg_ex:
                    print(f"[TELEGRAM] [ERROR] Failed to send close early alert: {tg_ex}")
            continue # Go to next position

        # 2. Move SL to entry (breakeven) if 50% of target is reached (50% remaining)
        if pct_reached >= 0.50:
            # Check if SL is already at or better than entry
            is_sl_already_breakeven = False
            if pos_type == 0: # BUY
                if sl >= entry_price:
                    is_sl_already_breakeven = True
            else: # SELL
                if sl > 0 and sl <= entry_price:
                    is_sl_already_breakeven = True

            if not is_sl_already_breakeven:
                new_sl = round(entry_price, digits)
                new_tp = round(tp, digits)
                
                print(f"[BREAKEVEN] Position #{ticket} ({symbol}) reached {pct_reached*100:.1f}% of TP. Moving SL to Entry ({new_sl}).")
                
                # Modify SL on MT5
                mod_res = modify_position_sl_tp(ticket, new_sl, new_tp)
                if mod_res:
                    # Update DB
                    try:
                        from db_manager import update_trade_sl_tp
                        update_trade_sl_tp(ticket, new_sl, new_tp)
                    except Exception as db_err:
                        print(f"[ERROR] Failed to update SL to breakeven in DB: {db_err}")
                        
                    # Notify Telegram
                    try:
                        from telegram_notifier import get_telegram_config, send_telegram_message
                        enabled, token, chat_id = get_telegram_config()
                        if enabled and token and chat_id:
                            msg = (
                                f"🛡️ *BREAKEVEN ENFORCED (تحريك الاستوب لنقطة الدخول)*\n\n"
                                f"• *Symbol:* `{symbol}`\n"
                                f"• *Ticket:* #{ticket}\n"
                                f"• *Action:* `{'BUY' if pos_type == 0 else 'SELL'}`\n"
                                f"• *Entry Price:* `{entry_price:.5f}`\n"
                                f"• *Current Price:* `{current_price:.5f}`\n"
                                f"• *New SL:* `{new_sl:.5f}` (Breakeven)\n\n"
                                f"⚙️ *Reason:* تم الوصول إلى 50% من الهدف وحماية الصفقة عند نقطة الدخول."
                            )
                            send_telegram_message(token, chat_id, msg)
                    except Exception as tg_err:
                        print(f"[ERROR] Failed to send Telegram breakeven notification: {tg_err}")


def check_and_execute_trading_cycle():
    """
    Executes a single scanning cycle across all symbols stored in the SQLite settings database.
    """
    global last_scan_reports
    print("\n" + "=" * 60)
    print(f"🔄 STARTING SCANNING CYCLE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    last_scan_reports.clear()

    # 1. Load Settings from SQLite Database
    settings = get_settings()
    symbols_to_trade = settings.get("symbols", ["EURUSDm"])
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

    # Manage active positions (SL to breakeven at 50% TP, and Close early at 90% TP)
    try:
        manage_active_positions_tp_sl()
    except Exception as e:
        print(f"[ERROR] Active positions TP/SL management failed: {e}")

    # Get active positions to check limits
    active_positions = get_open_positions()
    print(f"[ACCOUNT] Active Positions: {len(active_positions)} / {max_positions}")

    # Check Max Positions Limit
    if len(active_positions) >= max_positions:
        print("[RISK] Maximum positions reached. Skipping analysis to protect margin.")
        for symbol in symbols_to_trade:
            last_scan_reports[symbol] = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "Skipped (Max Positions)",
                "details": f"Maximum active positions reached ({len(active_positions)} / {max_positions}).",
                "structure": None
            }
        return

    # Fetch active news events if filter is enabled
    active_news_events = []
    if news_filter_enabled:
        _, active_news_events = is_news_time(minutes_before=30, minutes_after=30)
        if active_news_events:
            print(f"\n[NEWS FILTER] Detected {len(active_news_events)} active high-impact events. Checking symbol exposure...")

    # Keep track of symbols we already have open positions for
    open_symbols = [pos.symbol for pos in active_positions]

    # Initialize AI Engine (it will dynamically load the strategy prompt from DB)
    try:
        engine = AITradingEngine(model_name="gemini-2.5-flash")
    except Exception as e:
        print(f"[ERROR] Failed to initialize AI Engine: {e}")
        return

    # 3. Analyze each symbol
    for symbol in symbols_to_trade:
        print(f"\n--------------------------------------------------")
        print(f"🔎 Scanning symbol: {symbol}")
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

        tf_constant = get_timeframe(DEFAULT_TIMEFRAME)
        if tf_constant is None:
            print(f"[ERROR] Invalid timeframe configuration: {DEFAULT_TIMEFRAME}")
            last_scan_reports[symbol] = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "Error",
                "details": f"Invalid timeframe configuration: {DEFAULT_TIMEFRAME}",
                "structure": None
            }
            continue

        # Fetch count
        count_to_fetch = max(50, gann_lookback + 10)
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
                "sl_price": dyn["sl_price"]
            }
            print(f"[GANN SETUP] Valid {setup['type']} structure: A={setup['A']}, B={setup['B']}, C={setup['C']} (Retracement: {retr_pct:.1f}%)")
            proposed_action = setup["type"]
            technical_reason = f"Gann breakout {proposed_action} pivot setup."
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

        # Run AI analysis (if enabled) or bypass
        decision = "HOLD"
        trade_params = None
        reasoning = ""

        if ai_evaluation_enabled:
            try:
                ai_result = engine.analyze_market(
                    symbol=symbol,
                    timeframe=DEFAULT_TIMEFRAME,
                    candles_data=candles_text,
                    account_info=account_info,
                    current_price=price_info,
                    gann_context=gann_context,
                    active_positions=active_positions,
                    proposed_action=proposed_action
                )
                decision = ai_result.get("decision", "HOLD")
                if decision == "ERROR":
                    raise Exception(ai_result.get("analysis", "Gemini API call failed"))
                if decision in ["APPROVE", "BUY", "SELL"]:
                    decision = proposed_action
                else:
                    decision = "HOLD"
                trade_params = ai_result.get("trade_params")
                reasoning = ai_result.get("reasoning", ai_result.get("reason", "HOLD - AI rejected proposed trade"))
            except Exception as e:
                print(f"[ERROR] AI analysis failed for {symbol}: {e}")
                if fallback_to_technical:
                    print(f"[FALLBACK] Gemini API call failed. Falling back to pure technical execution!")
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
                        "reason": f"Fallback: {technical_reason}"
                    }
                    reasoning = f"Gemini API error ({str(e)}). Executing technical fallback: {proposed_action}. Details: {technical_reason}"
                else:
                    last_scan_reports[symbol] = {
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "Error",
                        "details": f"Gemini AI API call failed: {str(e)} and fallback is disabled.",
                        "structure": gann_context
                    }
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
        last_scan_reports[symbol] = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "status": f"AI Approved ({decision})" if ai_evaluation_enabled else f"Technical ({decision})",
            "details": reasoning,
            "structure": gann_context
        }

        # Check active positions for this symbol to manage reversals and prevent double entries
        symbol_positions = [pos for pos in active_positions if pos.magic == MAGIC_NUMBER and pos.symbol == symbol]
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
            symbol_positions = [pos for pos in active_positions if pos.magic == MAGIC_NUMBER and pos.symbol == symbol]
            has_opposite_direction = False
            
        # If we have active positions in the same direction, skip opening a new trade (Prevent Double Entry)
        if decision in ["BUY", "SELL"] and has_same_direction:
            skip_msg = f"Already have an active {decision} position on {symbol}. Skipping new entry to prevent double entry."
            print(f"[SKIP] {skip_msg}")
            last_scan_reports[symbol]["status"] = "Already Open"
            last_scan_reports[symbol]["details"] = skip_msg
            continue

        # Check if the last closed trade was a Stop Loss and blocks re-entry
        if decision in ["BUY", "SELL"]:
            if is_setup_stopped_out(symbol, gann_context, decision):
                block_msg = f"Blocked re-entry on {symbol} in direction {decision}. The setup/symbol recently hit Stop Loss."
                print(f"[BLOCKED] {block_msg}")
                last_scan_reports[symbol]["status"] = "Blocked (Stop Loss)"
                last_scan_reports[symbol]["details"] = block_msg
                continue

        # 4. Handle decision
        if decision in ["BUY", "SELL"] and trade_params:
            print(f"\n🎯 [AI SIGNAL] Triggered {decision} on {symbol}")
            
            reason = trade_params.get("reason", "AI Strategy Entry")
            if gann_enabled and 'dyn' in locals() and dyn:
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

            # --- RISK SHIELD GUARD CHECK (MAX 3% LOSS) ---
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                point = symbol_info.point
                tick_value = symbol_info.trade_tick_value
                volume_step = symbol_info.volume_step
                min_volume = symbol_info.volume_min
                max_volume = symbol_info.volume_max
                
                sl_distance_points = abs(entry_price - sl) / point
                if sl_distance_points > 0:
                    estimated_loss = sl_distance_points * volume * tick_value
                    max_allowed_loss = balance * 0.03 # 3% of balance
                    
                    if estimated_loss > max_allowed_loss:
                        old_volume = volume
                        # Downscale volume
                        volume = max_allowed_loss / (sl_distance_points * tick_value)
                        volume = round(volume / volume_step) * volume_step
                        volume = max(min_volume, min(volume, max_volume))
                        volume = round(volume, 2)
                        
                        # Re-calculate loss with the new downscaled volume
                        new_estimated_loss = sl_distance_points * volume * tick_value
                        print(f"[RISK SHIELD] Estimated loss of ${estimated_loss:.2f} exceeded 3% cap (${max_allowed_loss:.2f}). Downscaling lot from {old_volume} to {volume}.")
                        
                        # If even at the absolute minimum volume the risk still exceeds 3%, reject the trade!
                        if new_estimated_loss > max_allowed_loss:
                            reject_msg = f"Trade rejected: Minimum volume {volume} lots still risks ${new_estimated_loss:.2f} which exceeds 3% of balance (${max_allowed_loss:.2f})."
                            print(f"[REJECT] {reject_msg}")
                            last_scan_reports[symbol]["status"] = "Risk Shield Rejection"
                            last_scan_reports[symbol]["details"] = reject_msg
                            continue

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
                        gann_context=gann_context
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

    print("\n" + "=" * 60)
    print("🔄 SCANNING CYCLE COMPLETED")
    print("=" * 60)


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

    try:
        if args.loop:
            while True:
                # Ensure connection is active
                if mt5.terminal_info() is None or mt5.account_info() is None:
                    print("[WARNING] Connection lost. Attempting to reconnect...")
                    if not connect_mt5():
                        print("[ERROR] Reconnection failed. Retrying next cycle.")
                        time.sleep(min(30, args.interval))
                        continue
                
                check_and_execute_trading_cycle()
                print(f"\n[LOOP] Sleeping for {args.interval} seconds...")
                time.sleep(args.interval)
        else:
            check_and_execute_trading_cycle()
        
    except KeyboardInterrupt:
        print("\n[INFO] Bot stopped by user.")
    except Exception as e:
        print(f"[ERROR] Unexpected error in main loop: {e}")
    finally:
        # Disconnect safely
        disconnect_mt5()

if __name__ == "__main__":
    main()
