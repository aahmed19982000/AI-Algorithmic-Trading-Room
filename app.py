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


@app.route('/backtest')
def backtest():
    """Serve the backtesting HTML page."""
    return render_template("backtest.html")


@app.route('/api/backtest', methods=['POST'])
def run_backtest_api():
    """
    Run strategy backtest and return metrics, trade logs, and recent price data.
    """
    from datetime import datetime
    data = request.json or {}
    symbol = data.get("symbol", "EURUSDm")
    timeframe = data.get("timeframe", "M30")
    geometry = data.get("geometry", "square")
    lookback = int(data.get("lookback", 100))
    years = float(data.get("years", 1.0))
    use_grid = bool(data.get("use_grid", True))
    
    # 1. Connect to MT5
    if not connect_mt5():
        return jsonify({"success": False, "error": "Could not connect to MT5 Terminal"}), 500
        
    try:
        # Import backtest functions
        from backtest_gann import fetch_backtest_data, run_backtest_gann
        
        # 2. Fetch rates
        df = fetch_backtest_data(symbol, timeframe, years=years)
        if df is None or len(df) == 0:
            return jsonify({"success": False, "error": f"Failed to download historical rates for {symbol}"}), 500
            
        # 3. Run simulation
        trades = run_backtest_gann(df, symbol, geometry=geometry, lookback=lookback, use_grid=use_grid)
        
        # 4. Calculate metrics
        total_trades = len(trades)
        wins = sum(1 for t in trades if t["result"] == "WIN")
        losses = sum(1 for t in trades if t["result"] == "LOSS")
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        
        gross_win_pips = sum(t["pips"] for t in trades if t["pips"] > 0)
        gross_loss_pips = abs(sum(t["pips"] for t in trades if t["pips"] < 0))
        profit_factor = (gross_win_pips / gross_loss_pips) if gross_loss_pips > 0 else (99.99 if gross_win_pips > 0 else 1.0)
            
        balance = 500.0
        peak_balance = 500.0
        max_drawdown = 0.0
        
        # Compute simulated account balance curve
        balance_curve = []
        for t in trades:
            profit = t["profit_usd_raw"]
            balance += profit
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance * 100.0
            if dd > max_drawdown:
                max_drawdown = dd
            
            t_exit_str = t["exit_time"].strftime('%Y-%m-%d %H:%M:%S') if isinstance(t["exit_time"], datetime) else str(t["exit_time"])
            balance_curve.append({
                "time": t_exit_str,
                "balance": round(balance, 2)
            })
            
        roi = ((balance - 500.0) / 500.0) * 100.0
        
        # Format trades list for frontend
        import pandas as pd
        formatted_trades = []
        df_indexed = df.set_index('time')
        for t in trades:
            t_entry_str = t["entry_time"].strftime('%Y-%m-%d %H:%M:%S') if isinstance(t["entry_time"], datetime) else str(t["entry_time"])
            t_exit_str = t["exit_time"].strftime('%Y-%m-%d %H:%M:%S') if isinstance(t["exit_time"], datetime) else str(t["exit_time"])
            
            # Lookup entry/exit prices from dataframe
            try:
                entry_price_row = df_indexed.loc[t["entry_time"]]
                entry_price = float(entry_price_row["Close"].iloc[0] if isinstance(entry_price_row, pd.DataFrame) else entry_price_row["Close"])
            except Exception:
                entry_price = 0.0
                
            try:
                exit_price_row = df_indexed.loc[t["exit_time"]]
                exit_price = float(exit_price_row["Close"].iloc[0] if isinstance(exit_price_row, pd.DataFrame) else exit_price_row["Close"])
            except Exception:
                exit_price = 0.0
            
            formatted_trades.append({
                "type": t["type"],
                "entry_time": t_entry_str,
                "exit_time": t_exit_str,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "legs": t["legs"],
                "volume": t["volume"],
                "pips": round(t["pips"], 1),
                "profit_usd": round(t["profit_usd_raw"], 2),
                "result": t["result"],
                "sl_price": t.get("sl_price", 0.0),
                "tp_price": t.get("tp_price", 0.0),
                "gann_data": t.get("gann_data")
            })
            
        # Get the last 1000 candles for price visualization
        chart_candles = []
        df_subset = df.tail(1000)
        for _, row in df_subset.iterrows():
            chart_candles.append({
                "time": row["time"].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row["time"], datetime) else str(row["time"]),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
            })
            
        return jsonify({
            "success": True,
            "metrics": {
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate": round(win_rate, 1),
                "profit_factor": round(profit_factor, 2),
                "final_balance": round(balance, 2),
                "roi": round(roi, 1),
                "max_drawdown": round(max_drawdown, 1)
            },
            "trades": formatted_trades,
            "balance_curve": balance_curve,
            "candles": chart_candles
        })
        
    except Exception as e:
        print(f"[API BACKTEST] Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        disconnect_mt5()


@app.route('/api/backtest/chart', methods=['POST'])
def generate_backtest_chart():
    """Dynamically generate and save an annotated chart screenshot for a backtest trade."""
    data = request.json or {}
    symbol = data.get("symbol", "EURUSDm")
    timeframe = data.get("timeframe", "M30")
    entry_time_str = data.get("entry_time")
    exit_time_str = data.get("exit_time")
    entry_price = float(data.get("entry_price", 0.0))
    sl_price = float(data.get("sl_price", 0.0))
    tp_price = float(data.get("tp_price", 0.0))
    gann_data = data.get("gann_data")
    
    if not connect_mt5():
        return jsonify({"success": False, "error": "MT5 Terminal offline"}), 500
        
    try:
        from datetime import datetime, timedelta
        from mt5_data import get_timeframe
        
        tf_constant = get_timeframe(timeframe)
        if tf_constant is None:
            tf_constant = mt5.TIMEFRAME_M30
            
        entry_dt = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
        exit_dt = datetime.strptime(exit_time_str, "%Y-%m-%d %H:%M:%S")
        
        # Determine timeframe minutes for padding
        def get_tf_mins(tf_s):
            tf_s = tf_s.upper()
            if tf_s.startswith("M"):
                return int(tf_s[1:]) if tf_s[1:].isdigit() else 30
            elif tf_s.startswith("H"):
                return int(tf_s[1:]) * 60 if tf_s[1:].isdigit() else 60
            elif tf_s == "D1":
                return 1440
            return 30
            
        tf_mins = get_tf_mins(timeframe)
        start_dt = entry_dt - timedelta(minutes=tf_mins * 30)
        end_dt = exit_dt + timedelta(minutes=tf_mins * 30)
        
        rates = mt5.copy_rates_range(symbol, tf_constant, start_dt, end_dt)
        if rates is None or len(rates) == 0:
            # Fallback: copy from entry time
            rates = mt5.copy_rates_from(symbol, tf_constant, entry_dt, 100)
            
        if rates is None or len(rates) == 0:
            return jsonify({"success": False, "error": "Failed to fetch chart data"}), 404
            
        import pandas as pd
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)
        
        # Generate chart filename dynamically
        import os
        import hashlib
        h = hashlib.md5(f"{symbol}_{entry_time_str}_{exit_time_str}".encode('utf-8')).hexdigest()
        filename = f"backtest_{h}.png"
        
        static_dir = os.path.join(os.getcwd(), 'static', 'screenshots')
        os.makedirs(static_dir, exist_ok=True)
        filepath = os.path.join(static_dir, filename)
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import mplfinance as mpf
        
        # Setup colors
        mc = mpf.make_marketcolors(
            up='#2ecc71', down='#e74c3c',
            edge='inherit', wick='inherit',
            volume='#3498db', inherit=True
        )
        s = mpf.make_mpf_style(
            base_mpf_style='charles',
            marketcolors=mc,
            gridcolor='#2d3748',
            facecolor='#1e272e',
            figcolor='#1e272e',
            gridstyle='dashed'
        )
        
        hlines_dict = dict(
            hlines=[sl_price, entry_price, tp_price],
            colors=['#e74c3c', '#f1c40f', '#2ecc71'],
            linestyle='dashed',
            linewidths=1.5
        )
        
        setup_title = f"{symbol} ({timeframe}) - Backtest Trade"
        if gann_data:
            setup_title += f"\nGann Setup: A={gann_data.get('A', 0):.5f}, B={gann_data.get('B', 0):.5f}, C={gann_data.get('C', 0):.5f} ({gann_data.get('type', '')})"
            
        fig, axlist = mpf.plot(
            df,
            type='candle',
            style=s,
            hlines=hlines_dict,
            title=setup_title,
            volume=False,
            savefig=dict(fname=filepath, dpi=150, bbox_inches='tight'),
            returnfig=True
        )
        
        ax = axlist[0]
        x_pos = len(df) - 5
        ax.text(x_pos, sl_price, '  SL (Point A)', color='#e74c3c', fontsize=8, fontweight='bold', va='center')
        ax.text(x_pos, entry_price, '  Entry', color='#f1c40f', fontsize=8, fontweight='bold', va='center')
        ax.text(x_pos, tp_price, '  TP (Target)', color='#2ecc71', fontsize=8, fontweight='bold', va='center')
        
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return jsonify({"success": True, "screenshot_url": f"/static/screenshots/{filename}"})
        
    except Exception as e:
        print(f"[API CHART] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        disconnect_mt5()


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
