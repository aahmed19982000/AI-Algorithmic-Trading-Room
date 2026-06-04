"""
Flask Web Dashboard Server
==========================
Serves the web dashboard frontend and exposes APIs for trading status,
positions, history, and system settings.
Also runs the trading bot scanner in a background thread.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import time
import threading
from flask import Flask, jsonify, request, render_template
import MetaTrader5 as mt5

from mt5_connection import connect_mt5, disconnect_mt5
from mt5_data import get_account_info
from mt5_orders import get_open_positions, close_position
from db_manager import (
    get_settings, 
    save_settings, 
    get_trade_history, 
    get_balance_history,
    log_balance_snapshot
)
from news_filter import get_upcoming_impactful_events, is_news_time
from trading_bot import check_and_execute_trading_cycle

app = Flask(__name__)

# Global flag to control the background bot scanning thread
bot_running = True
last_scan_time = "Never"
scan_in_progress = False

def bot_background_loop():
    """
    Background thread loop that executes the trading bot scan cycle every 5 minutes.
    """
    global last_scan_time, scan_in_progress
    
    print("[BOT THREAD] Background scanning thread started.")
    
    # Wait 5 seconds on startup for server to initialize
    time.sleep(5)
    
    while bot_running:
        try:
            # 1. Initialize MT5 if not already connected
            # (Flask API requests might also connect/disconnect, so we connect/verify)
            if not mt5.initialize(path=os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe")):
                print("[BOT THREAD] [ERROR] Could not initialize MT5 Terminal.")
                time.sleep(30)
                continue
                
            # Log in
            MT5_LOGIN = os.getenv("MT5_LOGIN")
            MT5_PASSWORD = os.getenv("MT5_PASSWORD")
            MT5_SERVER = os.getenv("MT5_SERVER")
            if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
                mt5.login(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER)

            # 2. Run the scanning cycle
            scan_in_progress = True
            check_and_execute_trading_cycle()
            last_scan_time = time.strftime('%H:%M:%S')
            scan_in_progress = False
            
        except Exception as e:
            print(f"[BOT THREAD] [ERROR] Unexpected exception: {e}")
            scan_in_progress = False
            
        # Sleep for 5 minutes (300 seconds)
        for _ in range(300):
            if not bot_running:
                break
            time.sleep(1)

    print("[BOT THREAD] Background scanning thread stopped.")


# ============================================================
#  Web Page Routes
# ============================================================

@app.route('/')
def home():
    """Serve the main Dashboard HTML page."""
    return render_template("index.html")


# ============================================================
#  API Routes
# ============================================================

@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Get live status of MT5 connection, account metrics, open positions,
    recent trade history, and active economic news.
    """
    # 1. Verify MT5 status
    is_connected = False
    account_summary = {}
    positions = []
    
    # Try to initialize and fetch
    if mt5.initialize(path=os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe")):
        MT5_LOGIN = os.getenv("MT5_LOGIN")
        MT5_PASSWORD = os.getenv("MT5_PASSWORD")
        MT5_SERVER = os.getenv("MT5_SERVER")
        
        if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
            if mt5.login(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER):
                is_connected = True
                
                # Fetch account summary
                acc = get_account_info()
                if acc:
                    account_summary = acc
                    
                # Fetch positions
                pos_list = get_open_positions()
                for p in pos_list:
                    positions.append({
                        "ticket": p.ticket,
                        "symbol": p.symbol,
                        "action": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                        "volume": p.volume,
                        "price_open": p.price_open,
                        "price_current": p.price_current,
                        "sl": p.sl,
                        "tp": p.tp,
                        "profit": round(p.profit, 2),
                        "comment": p.comment
                    })

    # 2. Get history from SQLite
    history = get_trade_history(limit=20)
    
    # 3. Get upcoming news
    upcoming_news = get_upcoming_impactful_events(minutes_before=60, minutes_after=30)
    news_active, _ = is_news_time(minutes_before=60, minutes_after=30)

    # 4. Get last scan time
    settings = get_settings()

    return jsonify({
        "mt5_connected": is_connected,
        "last_scan_time": last_scan_time,
        "scan_in_progress": scan_in_progress,
        "account": account_summary,
        "positions": positions,
        "history": history,
        "news": upcoming_news,
        "news_freeze_active": news_active,
        "symbols_configured": settings.get("symbols", [])
    })


@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """
    GET: Retrieve system settings.
    POST: Update settings.
    """
    if request.method == 'GET':
        return jsonify(get_settings())
    
    elif request.method == 'POST':
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
            
        save_settings(data)
        
        # If active symbols changed, re-init symbols in MT5
        symbols = data.get("symbols")
        if symbols and mt5.initialize():
            for sym in symbols:
                mt5.symbol_select(sym, True)
                
        return jsonify({"success": True})


@app.route('/api/trade/close', methods=['POST'])
def close_trade():
    """
    Close an active position manually from the dashboard.
    """
    data = request.json
    if not data or "ticket" not in data:
        return jsonify({"success": False, "error": "Missing position ticket"}), 400
        
    ticket = int(data["ticket"])
    
    # Verify connected
    if not mt5.initialize(path=os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe")):
        return jsonify({"success": False, "error": "MT5 Terminal offline"}), 500
        
    MT5_LOGIN = os.getenv("MT5_LOGIN")
    MT5_PASSWORD = os.getenv("MT5_PASSWORD")
    MT5_SERVER = os.getenv("MT5_SERVER")
    if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
        mt5.login(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER)
        
    # Send close command
    success = close_position(ticket=ticket, comment="Manual Web Close")
    
    if success:
        # Sync positions to database immediately
        from trading_bot import sync_db_with_mt5_positions
        sync_db_with_mt5_positions()
        return jsonify({"success": True})
        
    return jsonify({"success": False, "error": "Failed to close trade in MT5"}), 500


@app.route('/api/scan', methods=['POST'])
def trigger_scan():
    """
    Manually triggers a scanning cycle in the background immediately.
    """
    global last_scan_time, scan_in_progress
    if scan_in_progress:
        return jsonify({"success": False, "error": "Scan already in progress"}), 400
        
    # Trigger in a separate temporary thread so Flask request finishes instantly
    def run_manual_scan():
        global last_scan_time, scan_in_progress
        scan_in_progress = True
        try:
            if mt5.initialize(path=os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe")):
                MT5_LOGIN = os.getenv("MT5_LOGIN")
                MT5_PASSWORD = os.getenv("MT5_PASSWORD")
                MT5_SERVER = os.getenv("MT5_SERVER")
                if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
                    mt5.login(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER)
            check_and_execute_trading_cycle()
            last_scan_time = time.strftime('%H:%M:%S')
        except Exception as e:
            print(f"[MANUAL SCAN] Error: {e}")
        scan_in_progress = False

    t = threading.Thread(target=run_manual_scan)
    t.start()
    return jsonify({"success": True})


@app.route('/api/chart-data', methods=['GET'])
def get_chart_data():
    """
    Get balance history for performance chart.
    """
    history = get_balance_history(limit=50)
    return jsonify(history)


# ============================================================
#  Server Startup & Tear-down
# ============================================================

if __name__ == '__main__':
    # Start the background scanning loop thread
    bot_thread = threading.Thread(target=bot_background_loop)
    bot_thread.daemon = True
    bot_thread.start()
    
    try:
        # Run Flask server
        print("\n" + "=" * 55)
        print("🚀 WEB DASHBOARD RUNNING: http://127.0.0.1:5000")
        print("=" * 55 + "\n")
        app.run(host='127.0.0.1', port=5000, debug=False)
    finally:
        # Stop background thread on server shutdown
        bot_running = False
        print("Stopping background bot thread...")
