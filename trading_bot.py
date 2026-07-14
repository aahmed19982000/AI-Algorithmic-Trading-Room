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
from gann_helper import detect_market_structure, calculate_gann_levels

# Load configuration
load_dotenv()
MAGIC_NUMBER = int(os.getenv("BOT_MAGIC", "20260604"))
DEFAULT_TIMEFRAME = os.getenv("TRADING_TIMEFRAME", "H4")

# Global dict to store the results of the last scanning cycle per symbol
last_scan_reports = {}
last_ai_scan_times = {}


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


def check_m5_exit_signal(symbol, pos_type):
    """
    Checks if there is an exit signal on M5 timeframe:
    - For BUY (0): returns True if a Lower Low (LL) is formed on M5,
      with a correction separating them, and a candle Close below the previous low.
    - For SELL (1): returns True if a Higher High (HH) is formed on M5,
      with a correction separating them, and a candle Close above the previous high.
    """
    df = get_candles(symbol=symbol, timeframe=mt5.TIMEFRAME_M5, count=60)
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

        # Check if we need to open the next leg (ONLY if Grid is explicitly enabled)
        if grid_enabled and drawdown_pips >= grid_step:
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
            
            if len(pos_list) == 1:
                grid_tp = float(settings.get("grid_tp", 20.0))
                tp_price = weighted_entry + (grid_tp * pip_size) if basket_type == 'BUY' else weighted_entry - (grid_tp * pip_size)
            else:
                tp_price = weighted_entry + (target_profit * pip_size) if basket_type == 'BUY' else weighted_entry - (target_profit * pip_size)
                
            last_entry_price = pos_list[-1].price_open
            is_gann_trade = any("gann" in (pos.comment or "").lower() for pos in pos_list)
            
            if is_gann_trade and pos_list[0].sl > 0:
                sl_price = pos_list[0].sl
                print(f"[GRID INFO] Gann trade detected. Preserving oldest SL price: {sl_price}")
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
                
            # 3. Check M5 Swing Exit
            if check_m5_exit_signal(symbol, 0 if basket_type == 'BUY' else 1):
                exit_reason_ar = "تكوين قمة أعلى (للبيع) أو قاع أدنى (للشراء) على M5"
                exit_reason_en = f"M5 {'Lower Low' if basket_type == 'BUY' else 'Higher High'} Exit"
                print(f"[M5 EXIT] Closing basket for {symbol} due to M5 exit signal.")
                for pos in pos_list:
                    close_res = close_position(ticket=pos.ticket, comment=exit_reason_en[:28])
                    if close_res:
                        log_trade_close(pos.ticket, current_price, pos.profit, exit_reason=f"{exit_reason_en} ({exit_reason_ar})")
                continue
                
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


def check_and_execute_trading_cycle():
    """
    Executes a single scanning cycle across all symbols stored in the SQLite settings database.
    """
    global last_scan_reports
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

    # 3. Analyze each symbol
    for symbol in ([] if max_positions_reached else symbols_to_trade):
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

        # --- PRE-AI VALIDATION CHECKS (Token Savers) ---
        # 1. Check active positions for this symbol to prevent double entries in same direction
        symbol_positions = [pos for pos in active_positions if pos.magic == MAGIC_NUMBER and pos.symbol == symbol]
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
            # Check if we already did an AI scan for this symbol on the current candle to save API usage
            current_candle_time = str(candles_df['Time'].iloc[-1])
            if last_ai_scan_times.get(symbol) == current_candle_time:
                print(f"[AI CACHE] Already evaluated {symbol} on the current candle ({current_candle_time}). Skipping AI call to save tokens.")
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
                
                # Record scan timestamp in cache
                last_ai_scan_times[symbol] = current_candle_time
            except Exception as e:
                print(f"[ERROR] AI analysis failed for {symbol}: {e}")
                ai_failure_count += 1
                
                # Temporarily disable auto_trade to stop further trades until user reviews
                try:
                    from db_manager import save_settings
                    save_settings({"auto_trade": "0"})
                    
                    from telegram_notifier import get_telegram_config, send_telegram_message
                    enabled, token, chat_id = get_telegram_config()
                    if enabled and token and chat_id:
                        err_alert = (
                            f"🚨 *فشل الاتصال بـ Gemini (Gemini Connection Failed)*\n\n"
                            f"• *الزوج:* `{symbol}`\n"
                            f"• *نوع الخطأ:* `{str(e)[:150]}`\n\n"
                            f"🛑 *قرار الحماية:* تم إيقاف التداول التلقائي تلقائياً لحماية حسابك من التداول الفني العشوائي.\n"
                            f"💬 لتفعيل التداول الفني البديل يدوياً أرسل `/start_trade`."
                        )
                        send_telegram_message(token, chat_id, err_alert)
                except Exception as db_err:
                    print(f"[ERROR] Failed to save disabled trade state or send alert: {db_err}")

                last_scan_reports[symbol] = {
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "status": "Error (Halted)",
                    "details": f"Gemini AI API call failed: {str(e)}. Trading halted.",
                    "structure": gann_context
                }
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

    # 3. Send Telegram Cycle Report
    try:
        from telegram_notifier import get_telegram_config, send_telegram_message
        enabled, token, chat_id = get_telegram_config()
        if enabled and token and chat_id:
            # Active positions summary
            active_count = len(latest_active)
            total_lots = sum(pos.volume for pos in latest_active)
            total_profit = sum(pos.profit for pos in latest_active)
            
            # Find signals triggered in this cycle
            triggered_signals = []
            for sym, report in last_scan_reports.items():
                status = report.get("status", "")
                if "Executed" in status or "AI Approved" in status or "Technical" in status or "Approved" in status:
                    triggered_signals.append(f"• `{sym}`: {status} ({report.get('details', '')[:100]})")
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
            
            ai_err_str = f"\n⚠️ *خطأ اتصال الذكاء الاصطناعي (Gemini):* فشل الاتصال لـ `{ai_failure_count}` أزواج (تم تفعيل البديل الفني)." if ai_failure_count > 0 else ""

            # Token usage details
            usage_str = ""
            if cycle_in_tokens > 0 or cycle_out_tokens > 0:
                usage_str = (
                    f"📊 *استهلاك الذكاء الاصطناعي (AI Token Usage):*\n"
                    f"• الـ Tokens المرسلة (Input): `{cycle_in_tokens}` Tokens\n"
                    f"• الـ Tokens المستلمة (Output): `{cycle_out_tokens}` Tokens\n"
                    f"• التكلفة التقديرية للدورة: `${cycle_cost:.5f} USD`\n\n"
                )

            report_msg = (
                f"🔄 *تقرير دورة الفحص (Scanning Cycle Report)*\n"
                f"⏰ *الوقت:* `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n\n"
                f"💳 *حالة الحساب (Account Details):*\n"
                f"• الرصيد (Balance): `{balance_val:.2f} {currency}`\n"
                f"• السيولة (Equity): `{equity_val:.2f} {currency}`\n"
                f"• الهامش الحر (Free Margin): `{free_margin:.2f} {currency}`\n\n"
                f"📈 *الصفقات المفتوحة (Portfolio Status):*\n"
                f"• إجمالي الصفقات: `{active_count}` صفقات\n"
                f"• إجمالي العقود: `{total_lots:.2f} lots`\n"
                f"• الأرباح العائمة: `{total_profit:+.2f} {currency}`\n\n"
                f"{usage_str}"
                f"🎯 *أحداث الدورة الحالية (Current Signals):*\n"
                f"{signals_str}\n{ai_err_str}\n"
                f"⚙️ تم فحص {len(symbols_to_trade)} أزواج بنجاح."
            )
            send_telegram_message(token, chat_id, report_msg)
            print("[TELEGRAM] Scanning cycle report sent successfully.")
    except Exception as tg_report_ex:
        print(f"[ERROR] Failed to send Telegram cycle report: {tg_report_ex}")

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
