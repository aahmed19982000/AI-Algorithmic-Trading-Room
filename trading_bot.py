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

# Load configuration
load_dotenv()
MAGIC_NUMBER = int(os.getenv("BOT_MAGIC", "20260604"))
DEFAULT_TIMEFRAME = os.getenv("TRADING_TIMEFRAME", "H4")


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
            history_deals = mt5.history_deals_get(ticket=ticket)
            if history_deals and len(history_deals) > 0:
                # Find the deal that closed the position (usually DEAL_ENTRY_OUT)
                for deal in history_deals:
                    if deal.entry == mt5.DEAL_ENTRY_OUT or deal.position_id == ticket:
                        # Use deal price and profit
                        close_price = deal.price
                        profit = deal.profit
            
            # Log as closed in database
            log_trade_close(ticket, close_price, profit)
            print(f"[SYNC] Closed position ticket {ticket} detected and logged in DB (Profit: ${profit})")


def manage_active_positions_grid():
    """
    Monitors active positions in MetaTrader 5.
    If a position goes into drawdown by grid_step, opens the next Martingale grid leg
    and modifies the TP/SL of all active legs of that symbol.
    """
    settings = get_settings()
    grid_enabled = int(settings.get("grid_enabled", 1)) == 1
    if not grid_enabled:
        return

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

        print(f"[GRID INFO] Symbol {symbol} has {len(pos_list)} active legs. Drawdown from last leg: {drawdown_pips:.1f} pips.")

        # Check if we need to open the next leg
        if drawdown_pips >= grid_step:
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
            sl_price = last_entry_price - (grid_sl * pip_size) if basket_type == 'BUY' else last_entry_price + (grid_sl * pip_size)
            
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
    print("\n" + "=" * 60)
    print(f"🔄 STARTING SCANNING CYCLE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Load Settings from SQLite Database
    settings = get_settings()
    symbols_to_trade = settings.get("symbols", ["EURUSDm"])
    risk_percent = float(settings.get("risk_percent", 1.0))
    max_positions = int(settings.get("max_positions", 2))
    auto_trade_enabled = int(settings.get("auto_trade", 1)) == 1
    news_filter_enabled = int(settings.get("news_filter", 1)) == 1

    print(f"[CONFIG] Symbols: {symbols_to_trade}")
    print(f"[CONFIG] Risk: {risk_percent}% per trade | Max Positions: {max_positions}")
    print(f"[CONFIG] Mode: {'AUTO-TRADE' if auto_trade_enabled else 'SIGNAL-ONLY (MANUAL)'}")
    print(f"[CONFIG] News Filter: {'ENABLED' if news_filter_enabled else 'DISABLED'}")

    # 2. Sync Positions & Log Balance Snapshot
    account_info = get_account_info()
    if not account_info:
        print("[ERROR] Could not fetch account info. Skipping cycle.")
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

    # Check Max Positions Limit
    if len(active_positions) >= max_positions:
        print("[RISK] Maximum positions reached. Skipping analysis to protect margin.")
        return

    # Check Economic News Filter
    if news_filter_enabled:
        should_freeze, active_news = is_news_time(minutes_before=60, minutes_after=30)
        if should_freeze:
            print("\n🚫 [NEWS FILTER] Trading is currently FROZEN due to upcoming economic news:")
            for event in active_news:
                print(f"  - [{event['importance']}] {event['currency']} - {event['title']} at {event['time']} (in {event['time_diff_minutes']} mins)")
            print("Skipping cycle scan for new trades.")
            return

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

        # Skip if we already have an open trade for this symbol to prevent over-exposure
        if symbol in open_symbols:
            print(f"[SKIP] Already have an open position on {symbol}. Skipping to avoid double entry.")
            continue

        # Get current price
        price_info = get_current_price(symbol)
        if not price_info:
            print(f"[ERROR] Could not fetch price for {symbol}. Skipping.")
            continue

        # Fetch recent 20 candles for technical context
        tf_constant = get_timeframe(DEFAULT_TIMEFRAME)
        if tf_constant is None:
            print(f"[ERROR] Invalid timeframe configuration: {DEFAULT_TIMEFRAME}")
            continue

        candles_df = get_candles(symbol=symbol, timeframe=tf_constant, count=20)
        if candles_df is None or len(candles_df) < 10:
            print(f"[ERROR] Not enough candle data for {symbol}. Skipping.")
            continue

        # Format candles for the AI model
        candles_text = format_candles_for_ai(candles_df, last_n=20)

        # Run AI analysis
        ai_result = engine.analyze_market(
            symbol=symbol,
            timeframe=DEFAULT_TIMEFRAME,
            candles_data=candles_text,
            account_info=account_info,
            current_price=price_info
        )

        decision = ai_result.get("decision", "HOLD")
        trade_params = ai_result.get("trade_params")

        # 4. Handle decision
        if decision in ["BUY", "SELL"] and trade_params:
            print(f"\n🎯 [AI SIGNAL] Triggered {decision} on {symbol}")
            
            # --- DYNAMIC GRID PARAMETERS OVERRIDE ---
            grid_enabled = int(settings.get("grid_enabled", 1)) == 1
            if grid_enabled:
                grid_sl = float(settings.get("grid_sl", 20.0))
                grid_tp = float(settings.get("grid_tp", 20.0))
                symbol_info = mt5.symbol_info(symbol)
                pip_size = 0.01 if (symbol.upper().endswith("JPY") or "JPY" in symbol.upper()) else 0.0001
                entry_price = price_info['ask'] if decision == "BUY" else price_info['bid']
                
                # Override SL and TP from database settings
                sl = entry_price - (grid_sl * pip_size) if decision == "BUY" else entry_price + (grid_sl * pip_size)
                tp = entry_price + (grid_tp * pip_size) if decision == "BUY" else entry_price - (grid_tp * pip_size)
                # Round
                sl = round(sl, symbol_info.digits)
                tp = round(tp, symbol_info.digits)
                print(f"[GRID OVERRIDE] Enforcing grid parameters. SL: {sl} ({grid_sl} pips), TP: {tp} ({grid_tp} pips)")
            else:
                sl = float(trade_params.get("sl", 0.0))
                tp = float(trade_params.get("tp", 0.0))
                
            reason = trade_params.get("reason", "AI Strategy Entry")
            entry_price = price_info['ask'] if decision == "BUY" else price_info['bid']
            
            if sl <= 0:
                print(f"[REJECT] Trade rejected: Stop Loss is mandatory but was not specified (SL: {sl})")
                continue

            # --- RISK/REWARD (R:R) RATIO CHECK ---
            sl_distance = abs(entry_price - sl)
            tp_distance = abs(tp - entry_price)
            
            if sl_distance <= 0:
                print("[REJECT] Trade rejected: Stop Loss is at or ahead of entry price.")
                continue
                
            rr_ratio = tp_distance / sl_distance
            min_rr_ratio = float(settings.get("min_rr_ratio", 1.5))
            if rr_ratio < min_rr_ratio:
                print(f"[REJECT] Trade rejected: Risk/Reward ratio is 1:{rr_ratio:.2f} (must be at least 1:{min_rr_ratio:.2f})")
                continue
                
            print(f"[OK] Risk validation passed: Risk/Reward ratio is 1:{rr_ratio:.2f}")

            # --- DYNAMIC LOT SIZE CALCULATION ---
            volume = calculate_lot_size(symbol, sl, entry_price, risk_percent=risk_percent)
            print(f"[LOTS] Calculated Dynamic Lot: {volume} lots (risking {risk_percent}% of balance)")

            if not auto_trade_enabled:
                print(f"[MANUAL] Signal-Only Mode: Trade was NOT executed. Signal: {decision} on {symbol}")
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
                log_trade_open(
                    ticket=trade_res["ticket"],
                    symbol=symbol,
                    action=decision,
                    volume=volume,
                    entry_price=trade_res["price"],
                    sl=sl,
                    tp=tp,
                    reason=reason
                )
                print(f"[DB] Trade ticket {trade_res['ticket']} logged successfully in SQLite.")
        else:
            print(f"\n➡️ [AI DECISION] {decision}")
            print(f"  Reasoning: {ai_result.get('analysis', '')[:300]}")

    print("\n" + "=" * 60)
    print("🔄 SCANNING CYCLE COMPLETED")
    print("=" * 60)


def main():
    print("=" * 60)
    print("          AI ALGORITHMIC TRADING BOT")
    print("============================================================\n")

    # Connect to MT5
    if not connect_mt5():
        print("[FATAL] Could not connect to MT5 Terminal. Exiting.")
        sys.exit(1)

    try:
        # Run a single scanning cycle on startup
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
