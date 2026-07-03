"""
Flask Web Dashboard Server
==========================
Serves the web dashboard frontend and exposes APIs for trading status,
positions, history, and system settings.
Also runs the trading bot scanner in a background thread.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import collections

class LogCaptureStream:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.buffer = collections.deque(maxlen=1000)

    def write(self, message):
        self.original_stream.write(message)
        if message and message.strip():
            timestamp = time.strftime('%H:%M:%S')
            for line in message.splitlines():
                line_stripped = line.strip()
                if line_stripped:
                    self.buffer.append(f"[{timestamp}] {line_stripped}")

    def flush(self):
        self.original_stream.flush()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)

sys.stdout = LogCaptureStream(sys.stdout)
sys.stderr = LogCaptureStream(sys.stderr)

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
    log_balance_snapshot,
    get_db_connection
)
from news_filter import get_upcoming_impactful_events, is_news_time
from trading_bot import check_and_execute_trading_cycle, last_scan_reports

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


@app.route('/portfolio')
def portfolio():
    """Serve the live portfolio summary report page."""
    return render_template("portfolio.html")


@app.route('/bot-status')
def bot_status():
    """Serve the bot status and activity console page."""
    return render_template("bot_status.html")


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
        
        # Determine start date based on pivot A if available to show the complete pattern
        if gann_data and gann_data.get("time_A"):
            try:
                time_A_dt = datetime.strptime(gann_data["time_A"], "%Y-%m-%d %H:%M:%S")
                # Pad starting date 10 candles before Point A to show structure context
                start_dt = time_A_dt - timedelta(minutes=tf_mins * 10)
            except Exception:
                start_dt = entry_dt - timedelta(minutes=tf_mins * 300)
        else:
            start_dt = entry_dt - timedelta(minutes=tf_mins * 300)
            
        end_dt = exit_dt + timedelta(minutes=tf_mins * 20)
        
        rates = mt5.copy_rates_range(symbol, tf_constant, start_dt, end_dt)
        if rates is None or len(rates) == 0:
            # Fallback: copy from entry time (300 candles to ensure we capture the pivot pattern)
            rates = mt5.copy_rates_from(symbol, tf_constant, entry_dt, 300)
            
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
        
        # Find integer index of a timestamp
        def get_idx_of_time(df_in, dt_str):
            try:
                target_dt = pd.to_datetime(dt_str)
                idx = df_in.index.get_indexer([target_dt], method='nearest')[0]
                if idx is not None and idx >= 0:
                    return int(idx)
            except Exception:
                pass
            return None

        # Resolve entry, exit, and pivot indices on the uncropped dataframe
        idx_entry_uncropped = get_idx_of_time(df, entry_time_str)
        idx_exit_uncropped = get_idx_of_time(df, exit_time_str)
        
        idx_A_uncropped = None
        idx_B_uncropped = None
        idx_C_uncropped = None
        
        if gann_data:
            trade_type = gann_data.get("type", "BUY")
            val_A = gann_data.get("A")
            val_B = gann_data.get("B")
            val_C = gann_data.get("C")
            
            # Resolve Point A index
            if gann_data.get("time_A"):
                idx_A_uncropped = get_idx_of_time(df, gann_data["time_A"])
            if (idx_A_uncropped is None or idx_A_uncropped < 0) and val_A is not None:
                search_limit = idx_entry_uncropped if idx_entry_uncropped is not None else len(df)
                col = 'High' if trade_type == 'SELL' else 'Low'
                best_idx = None
                min_diff = float('inf')
                for i in range(0, search_limit):
                    diff = abs(df.iloc[i][col] - val_A)
                    if diff < min_diff:
                        min_diff = diff
                        best_idx = i
                if min_diff <= val_A * 0.0005:
                    idx_A_uncropped = best_idx
                    
            # Resolve Point B index
            if gann_data.get("time_B"):
                idx_B_uncropped = get_idx_of_time(df, gann_data["time_B"])
            if (idx_B_uncropped is None or idx_B_uncropped < 0) and val_B is not None:
                search_limit = idx_entry_uncropped if idx_entry_uncropped is not None else len(df)
                col = 'Low' if trade_type == 'SELL' else 'High'
                best_idx = None
                min_diff = float('inf')
                for i in range(0, search_limit):
                    diff = abs(df.iloc[i][col] - val_B)
                    if diff < min_diff:
                        min_diff = diff
                        best_idx = i
                if min_diff <= val_B * 0.0005:
                    idx_B_uncropped = best_idx
                    
            # Resolve Point C index
            if gann_data.get("time_C"):
                idx_C_uncropped = get_idx_of_time(df, gann_data["time_C"])
            if (idx_C_uncropped is None or idx_C_uncropped < 0) and val_C is not None:
                search_limit = idx_entry_uncropped if idx_entry_uncropped is not None else len(df)
                col = 'High' if trade_type == 'SELL' else 'Low'
                best_idx = None
                min_diff = float('inf')
                for i in range(0, search_limit):
                    diff = abs(df.iloc[i][col] - val_C)
                    if diff < min_diff:
                        min_diff = diff
                        best_idx = i
                if min_diff <= val_C * 0.0005:
                    idx_C_uncropped = best_idx
                    
        # Crop df to start 10 candles before Point A if found
        crop_offset = 0
        if idx_A_uncropped is not None and idx_A_uncropped >= 10:
            crop_offset = idx_A_uncropped - 10
            df = df.iloc[crop_offset:]
            
        # Calculate final cropped indices
        idx_A = idx_A_uncropped - crop_offset if idx_A_uncropped is not None else None
        idx_B = idx_B_uncropped - crop_offset if idx_B_uncropped is not None else None
        idx_C = idx_C_uncropped - crop_offset if idx_C_uncropped is not None else None
        idx_entry = idx_entry_uncropped - crop_offset if idx_entry_uncropped is not None else None
        idx_exit = idx_exit_uncropped - crop_offset if idx_exit_uncropped is not None else None
        
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
        
        # Build horizontal lines
        original_sl = gann_data.get("A") if gann_data else None
        hlines = [entry_price, tp_price]
        hl_colors = ['#f1c40f', '#2ecc71']
        
        if original_sl and abs(sl_price - original_sl) > 1e-5:
            hlines.extend([sl_price, original_sl])
            hl_colors.extend(['#e74c3c', '#d35400']) # red for trailed SL, orange for original SL (Point A)
        else:
            hlines.append(sl_price)
            hl_colors.append('#e74c3c')
            
        hlines_dict = dict(
            hlines=hlines,
            colors=hl_colors,
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
        x_pos = max(0, len(df) - 5)
        
        # Draw horizontal labels avoiding overlaps
        if abs(sl_price - entry_price) < 1e-5:
            ax.text(x_pos, entry_price, '  Entry / Trailed SL', color='#f1c40f', fontsize=8, fontweight='bold', va='center')
        else:
            ax.text(x_pos, entry_price, '  Entry', color='#f1c40f', fontsize=8, fontweight='bold', va='center')
            ax.text(x_pos, sl_price, '  SL (Trailed)', color='#e74c3c', fontsize=8, fontweight='bold', va='center')
            
        ax.text(x_pos, tp_price, '  TP (Target)', color='#2ecc71', fontsize=8, fontweight='bold', va='center')
        
        if original_sl and abs(sl_price - original_sl) > 1e-5:
            ax.text(x_pos, original_sl, '  Original SL (Point A)', color='#d35400', fontsize=8, fontweight='bold', va='center')
        
        # Swing pivot indices are already calculated and shifted above

        # Plot Connectors and circles
        pivots_x = []
        pivots_y = []
        
        if gann_data:
            trade_type = gann_data.get("type", "BUY")
            
            if idx_A is not None and idx_A >= 0:
                val_A = gann_data.get("A")
                if val_A:
                    pivots_x.append(idx_A)
                    pivots_y.append(val_A)
                    ax.plot(idx_A, val_A, marker='o', color='#f39c12', markersize=8, zorder=5)
                    va_A = 'top' if trade_type == 'BUY' else 'bottom'
                    ax.text(idx_A, val_A, '  Point A (Origin)', color='#f39c12', fontsize=9, fontweight='bold', va=va_A, ha='left')
                    
            if idx_B is not None and idx_B >= 0:
                val_B = gann_data.get("B")
                if val_B:
                    pivots_x.append(idx_B)
                    pivots_y.append(val_B)
                    ax.plot(idx_B, val_B, marker='o', color='#e67e22', markersize=8, zorder=5)
                    va_B = 'bottom' if trade_type == 'BUY' else 'top'
                    ax.text(idx_B, val_B, '  Point B (Breakout)', color='#e67e22', fontsize=9, fontweight='bold', va=va_B)
                    
            if idx_C is not None and idx_C >= 0:
                val_C = gann_data.get("C")
                if val_C:
                    pivots_x.append(idx_C)
                    pivots_y.append(val_C)
                    ax.plot(idx_C, val_C, marker='o', color='#3498db', markersize=8, zorder=5)
                    va_C = 'top' if trade_type == 'BUY' else 'bottom'
                    ax.text(idx_C, val_C, '  Point C (Correction)', color='#3498db', fontsize=9, fontweight='bold', va=va_C)
                    
            if len(pivots_x) > 1:
                # Sort by x coordinate to draw line correctly in chronological order
                pivot_pts = sorted(zip(pivots_x, pivots_y))
                px, py = zip(*pivot_pts)
                ax.plot(px, py, color='#9b59b6', linestyle='dotted', linewidth=1.5, zorder=4)

        # 2. Plot Entry and Exit vertical lines and markers
        idx_exit = get_idx_of_time(df, exit_time_str)
        
        if idx_entry is not None and idx_entry >= 0:
            ax.axvline(x=idx_entry, color='#f1c40f', linestyle='--', linewidth=0.8, alpha=0.7)
            trade_type = gann_data.get('type') if gann_data else 'BUY'
            ax.plot(idx_entry, entry_price, marker='^' if trade_type == 'BUY' else 'v', 
                    color='#2ecc71' if trade_type == 'BUY' else '#e74c3c', markersize=8, zorder=6)
            
        if idx_exit is not None and idx_exit >= 0:
            ax.axvline(x=idx_exit, color='#95a5a6', linestyle='--', linewidth=0.8, alpha=0.7)
            result = data.get("result")
            ax.plot(idx_exit, df.iloc[idx_exit]['Close'], marker='o', 
                    color='#e74c3c' if result == 'LOSS' else '#2ecc71', markersize=6, zorder=6)

        # 3. Draw outcome banner
        result = data.get("result")
        profit_usd = data.get("profit_usd")
        if result:
            result_color = '#2ecc71' if result == 'WIN' else '#e74c3c'
            try:
                prof_val = float(profit_usd)
                result_text = f"Result: {result} (${prof_val:+.2f})"
            except Exception:
                result_text = f"Result: {result}"
                
            ax.text(0.02, 0.93, result_text, transform=ax.transAxes, color=result_color, 
                    fontsize=10, fontweight='bold', bbox=dict(facecolor='#1e272e', alpha=0.85, edgecolor=result_color, boxstyle='round,pad=0.4'))

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


@app.route('/api/mt5-symbols', methods=['GET'])
def get_mt5_symbols():
    """Fetch all available symbols from MT5 ending with 'm'."""
    if not mt5.initialize(path=os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe")):
        return jsonify({"success": False, "error": "Could not connect to MT5"}), 500
    try:
        MT5_LOGIN = os.getenv("MT5_LOGIN")
        MT5_PASSWORD = os.getenv("MT5_PASSWORD")
        MT5_SERVER = os.getenv("MT5_SERVER")
        if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
            mt5.login(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER)
            
        # Curated list of desired symbols (Majors, Famous Crosses, Gold, Silver, Oil, Cryptos)
        CURATED_SYMBOLS = [
            # Majors
            "EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm", "USDCADm", "USDCHFm", "NZDUSDm",
            # Famous Crosses
            "EURGBPm", "EURJPYm", "GBPJPYm", "EURAUDm", "EURCADm", "EURCHFm", "EURNZDm", "GBPAUDm", "GBPCADm", "AUDJPYm",
            # Metals
            "XAUUSDm", "XAGUSDm",
            # Oil
            "USOILm", "UKOILm",
            # Cryptos
            "BTCUSDm", "ETHUSDm", "SOLUSDm", "BNBUSDm", "XRPUSDm", "LTCUSDm", "DOGEUSDm"
        ]
        
        all_symbols = mt5.symbols_get()
        if not all_symbols:
            return jsonify({"success": True, "symbols": ["EURUSDm", "GBPUSDm", "USDJPYm"]})
            
        supported_names = {s.name for s in all_symbols}
        curated_filtered = [sym for sym in CURATED_SYMBOLS if sym in supported_names]
        
        return jsonify({"success": True, "symbols": curated_filtered})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/portfolio-report', methods=['GET'])
def get_portfolio_report():
    """
    Generate live portfolio statistics and historical trade summary by symbol.
    """
    is_connected = False
    account_summary = {}
    positions = []
    
    # 1. Connect and fetch live account details and active positions from MT5
    if mt5.initialize(path=os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe")):
        MT5_LOGIN = os.getenv("MT5_LOGIN")
        MT5_PASSWORD = os.getenv("MT5_PASSWORD")
        MT5_SERVER = os.getenv("MT5_SERVER")
        
        if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
            if mt5.login(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER):
                is_connected = True
                acc = get_account_info()
                if acc:
                    account_summary = acc
                pos_list = get_open_positions()
                for p in pos_list:
                    positions.append({
                        "ticket": p.ticket,
                        "symbol": p.symbol,
                        "action": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                        "volume": p.volume,
                        "profit": p.profit
                    })
                    
    # 2. Get closed trades history from SQLite database
    closed_trades = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, profit FROM trades WHERE status = 'CLOSED'")
        closed_trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        print(f"[PORTFOLIO API] Error reading closed trade history: {e}")
        
    # Get configured symbols from settings
    settings = get_settings()
    configured_symbols = settings.get("symbols", [])
    
    # Extract unique symbols from all sources
    all_symbols = set(configured_symbols)
    for p in positions:
        all_symbols.add(p["symbol"])
    for t in closed_trades:
        all_symbols.add(t["symbol"])
        
    rows = []
    total_active_trades = len(positions)
    total_active_lots = sum(p["volume"] for p in positions)
    total_active_profit = sum(p["profit"] for p in positions)
    
    total_closed_profit = sum(t["profit"] for t in closed_trades)
    closed_wins = sum(1 for t in closed_trades if t["profit"] > 0)
    closed_losses = sum(1 for t in closed_trades if t["profit"] <= 0)
    overall_win_rate = (closed_wins / len(closed_trades) * 100.0) if len(closed_trades) > 0 else 0.0
    
    for symbol in sorted(all_symbols):
        # Filter active positions for this symbol
        sym_active = [p for p in positions if p["symbol"] == symbol]
        sym_active_count = len(sym_active)
        sym_active_lots = sum(p["volume"] for p in sym_active)
        sym_active_profit = sum(p["profit"] for p in sym_active)
        
        # Filter closed trades for this symbol
        sym_closed = [t for t in closed_trades if t["symbol"] == symbol]
        sym_closed_wins = sum(1 for t in sym_closed if t["profit"] > 0)
        sym_closed_losses = sum(1 for t in sym_closed if t["profit"] <= 0)
        sym_closed_profit = sum(t["profit"] for t in sym_closed)
        
        rows.append({
            "symbol": symbol,
            "active_trades": sym_active_count,
            "total_lots": sym_active_lots,
            "floating_profit": round(sym_active_profit, 2),
            "closed_wins": sym_closed_wins,
            "closed_losses": sym_closed_losses,
            "total_closed_profit": round(sym_closed_profit, 2)
        })
        
    return jsonify({
        "success": True,
        "connected": is_connected,
        "account": account_summary,
        "summary": {
            "total_active_trades": total_active_trades,
            "total_active_lots": total_active_lots,
            "total_active_profit": round(total_active_profit, 2),
            "total_closed_profit": round(total_closed_profit, 2),
            "overall_win_rate": round(overall_win_rate, 1)
        },
        "rows": rows
    })


@app.route('/api/bot-activity', methods=['GET'])
def get_bot_activity():
    """
    Get live bot logs and the last scan report per symbol.
    """
    logs = []
    if hasattr(sys.stdout, 'buffer'):
        logs = list(sys.stdout.buffer)
        
    is_connected = False
    if mt5.initialize(path=os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe")):
        MT5_LOGIN = os.getenv("MT5_LOGIN")
        MT5_PASSWORD = os.getenv("MT5_PASSWORD")
        MT5_SERVER = os.getenv("MT5_SERVER")
        if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
            is_connected = bool(mt5.login(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER))
            
    settings = get_settings()
    auto_trade = int(settings.get("auto_trade", 1)) == 1
    
    return jsonify({
        "success": True,
        "logs": logs,
        "scan_reports": last_scan_reports,
        "last_scan_time": last_scan_time,
        "scan_in_progress": scan_in_progress,
        "mt5_connected": is_connected,
        "auto_trade": auto_trade
    })


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
    upcoming_news = get_upcoming_impactful_events(minutes_before=30, minutes_after=30)
    news_active, _ = is_news_time(minutes_before=30, minutes_after=30)

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


@app.route('/api/icons-config', methods=['GET'])
def get_icons_config():
    """
    Return the full icons catalogue with enabled/disabled status
    for Trading Platforms, Financial Instruments, and Regulations.
    """
    # Master catalogue — all possible items
    ALL_PLATFORMS = [
        {"id": "mt5",         "name": "MetaTrader 5",  "icon": "fa-brands fa-windows",     "color": "#6c5ce7"},
        {"id": "mt4",         "name": "MetaTrader 4",  "icon": "fa-brands fa-windows",     "color": "#a29bfe"},
        {"id": "ctrader",     "name": "cTrader",        "icon": "fa-solid fa-c",             "color": "#00cec9"},
        {"id": "tradingview", "name": "TradingView",    "icon": "fa-solid fa-chart-candlestick","color": "#2ecc71"},
        {"id": "dxtrade",     "name": "DXtrade",        "icon": "fa-solid fa-d",             "color": "#f39c12"},
        {"id": "matchtrader", "name": "Match-Trader",   "icon": "fa-solid fa-m",             "color": "#e17055"},
    ]
    ALL_INSTRUMENTS = [
        {"id": "forex",       "name": "Forex",          "icon": "fa-solid fa-money-bill-transfer", "color": "#6c5ce7"},
        {"id": "gold",        "name": "Gold (XAU)",     "icon": "fa-solid fa-coins",         "color": "#f1c40f"},
        {"id": "crypto",      "name": "Crypto",         "icon": "fa-brands fa-bitcoin",      "color": "#f39c12"},
        {"id": "oil",         "name": "Oil",            "icon": "fa-solid fa-oil-well",      "color": "#636e72"},
        {"id": "silver",      "name": "Silver (XAG)",   "icon": "fa-solid fa-gem",           "color": "#b2bec3"},
        {"id": "indices",     "name": "Stock Indices",  "icon": "fa-solid fa-chart-line",    "color": "#00cec9"},
        {"id": "stocks",      "name": "Stocks",         "icon": "fa-solid fa-building-columns","color": "#55efc4"},
        {"id": "commodities", "name": "Commodities",    "icon": "fa-solid fa-wheat-awn",     "color": "#fdcb6e"},
    ]
    ALL_REGULATIONS = [
        {"id": "fca",   "name": "FCA",   "subtitle": "United Kingdom",    "icon": "fa-solid fa-shield-halved", "color": "#6c5ce7"},
        {"id": "cysec", "name": "CySEC", "subtitle": "Cyprus",            "icon": "fa-solid fa-shield-halved", "color": "#0984e3"},
        {"id": "asic",  "name": "ASIC",  "subtitle": "Australia",         "icon": "fa-solid fa-shield-halved", "color": "#00b894"},
        {"id": "fsca",  "name": "FSCA",  "subtitle": "South Africa",      "icon": "fa-solid fa-shield-halved", "color": "#e17055"},
        {"id": "cma",   "name": "CMA",   "subtitle": "Kenya",             "icon": "fa-solid fa-shield-halved", "color": "#fdcb6e"},
        {"id": "fsa",   "name": "FSA",   "subtitle": "Seychelles",        "icon": "fa-solid fa-shield-halved", "color": "#a29bfe"},
        {"id": "sfsa",  "name": "SFSA",  "subtitle": "St. Vincent",       "icon": "fa-solid fa-shield-halved", "color": "#55efc4"},
        {"id": "vfsc",  "name": "VFSC",  "subtitle": "Vanuatu",           "icon": "fa-solid fa-shield-halved", "color": "#fd79a8"},
    ]

    settings = get_settings()
    enabled_platforms    = settings.get("trading_platforms", [p["id"] for p in ALL_PLATFORMS])
    enabled_instruments  = settings.get("financial_instruments", [i["id"] for i in ALL_INSTRUMENTS])
    enabled_regulations  = settings.get("regulations", [r["id"] for r in ALL_REGULATIONS])

    # Attach enabled flag to each item
    for item in ALL_PLATFORMS:
        item["enabled"] = item["id"] in enabled_platforms
    for item in ALL_INSTRUMENTS:
        item["enabled"] = item["id"] in enabled_instruments
    for item in ALL_REGULATIONS:
        item["enabled"] = item["id"] in enabled_regulations

    return jsonify({
        "success": True,
        "trading_platforms":    ALL_PLATFORMS,
        "financial_instruments": ALL_INSTRUMENTS,
        "regulations":          ALL_REGULATIONS,
    })


@app.route('/chat')
def chat_page():
    """
    Render the AI Analyst Chat interface.
    """

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY open_time DESC LIMIT 15")
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        print(f"[CHAT PAGE] Error reading trade history: {e}")
        trades = []
    return render_template('chat.html', trades=trades)



@app.route('/api/chat', methods=['POST'])
def api_chat():
    """
    Handle chat messages with Gemini, injecting recent trades context.
    """
    try:
        import google.generativeai as genai
        import json
        
        data = request.get_json() or {}
        user_message = data.get("message", "")
        history = data.get("history", [])
        
        # 1. Fetch recent trades from DB for context
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY open_time DESC LIMIT 20")
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        # 2. Build trades context
        trades_context = "RECENT TRADES LIST:\n"
        for t in trades:
            gann_data_str = ""
            if t.get("gann_data"):
                try:
                    gd = json.loads(t["gann_data"])
                    gann_data_str = f", Gann: Type={gd.get('type')}, Pivots (A={gd.get('A')}, B={gd.get('B')}, C={gd.get('C')}), Pullback={gd.get('retracement_pct', 0):.1f}%, TargetAngle={gd.get('target_angle')}°"
                except Exception:
                    pass
            trades_context += (
                f"- Ticket #{t['ticket']}, Symbol: {t['symbol']}, Action: {t['action']}, "
                f"Volume: {t['volume']} lots, Entry Price: {t['entry_price']}, "
                f"Close Price: {t['close_price'] or 'N/A'}, TP: {t['tp']}, SL: {t['sl']}, "
                f"Profit: {t['profit']} USD, Open: {t['open_time']}, Close: {t['close_time'] or 'OPEN'}, "
                f"Reason: {t['reason']}{gann_data_str}\n"
            )
            
        # 3. System Instruction for Analyst
        system_instruction = (
            "You are a professional AI Trading Analyst Coach. The user is trading Forex and other assets using an MT5 bot powered by a Gann geometry strategy combined with a Gemini analysis filter.\n"
            "Below is the history of the 20 most recent trades executed by the bot. Use this history to answer all user queries, analyze specific trades, and explain why trades were opened/closed:\n\n"
            f"{trades_context}\n\n"
            "Your instructions:\n"
            "1. Answer the user's questions about these trades, active positions, and market operations.\n"
            "2. Explain why trades were opened (referencing their reasons, Gann pivots A/B/C, Fibonacci retracement %, and angles) and why they were closed (e.g. hitting TP, SL, or Martingale grid averaging).\n"
            "3. If a user asks about a specific trade, locate it in the list above using its ticket number, symbol, or open time.\n"
            "4. Provide constructive feedback, stats (e.g. win rate, total profit/loss), and trade breakdown.\n"
            "5. ALWAYS respond in the user's language (if they ask in Arabic, respond in Arabic; if English, respond in English). Keep your answers concise, clear, and professional."
        )
        
        # 4. Initialize Gemini
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            return jsonify({"success": False, "error": "GEMINI_API_KEY missing in server env"}), 500
            
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_instruction
        )
        
        # 5. Format history for Gemini chat API
        gemini_history = []
        for h in history:
            role = h.get("role", "user")
            text = h.get("text", "")
            gemini_history.append({
                "role": "user" if role == "user" else "model",
                "parts": [text]
            })
            
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(user_message)
        
        return jsonify({
            "success": True,
            "response": response.text
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/evaluate-performance', methods=['GET', 'POST'])
def api_evaluate_performance():
    """
    Fetch closed trade history, calculate statistics, find losing patterns,
    and prompt Gemini to generate a read-only performance audit with solutions.
    """
    try:
        import google.generativeai as genai
        import json
        
        # 1. Fetch closed trades from database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY close_time DESC")
        closed_trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        if not closed_trades:
            return jsonify({
                "success": True,
                "response": "⚠️ لم يتم العثور على أي صفقات مغلقة في قاعدة البيانات حتى الآن لإجراء التقييم عليها."
            })
            
        # 2. Calculate summary statistics
        total_trades = len(closed_trades)
        wins = [t for t in closed_trades if t['profit'] > 0]
        losses = [t for t in closed_trades if t['profit'] <= 0]
        
        total_wins = len(wins)
        total_losses = len(losses)
        win_rate = (total_wins / total_trades) * 100.0 if total_trades > 0 else 0.0
        
        net_profit = sum(t['profit'] for t in closed_trades)
        gross_profit = sum(t['profit'] for t in wins)
        gross_loss = abs(sum(t['profit'] for t in losses))
        
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.99 if gross_profit > 0 else 1.0)
        
        # Symbol analysis
        symbol_stats = {}
        for t in closed_trades:
            sym = t['symbol']
            if sym not in symbol_stats:
                symbol_stats[sym] = {"trades": 0, "wins": 0, "losses": 0, "profit": 0.0}
            symbol_stats[sym]["trades"] += 1
            symbol_stats[sym]["profit"] += t['profit']
            if t['profit'] > 0:
                symbol_stats[sym]["wins"] += 1
            else:
                symbol_stats[sym]["losses"] += 1
                
        # Format stats for AI context
        stats_context = (
            f"SUMMARY STATISTICS:\n"
            f"- Total Closed Trades: {total_trades}\n"
            f"- Wins: {total_wins} ({win_rate:.1f}% Win Rate)\n"
            f"- Losses: {total_losses}\n"
            f"- Net Profit: {net_profit:.2f} USD\n"
            f"- Profit Factor: {profit_factor:.2f}\n"
            f"PER-SYMBOL BREAKDOWN:\n"
        )
        for sym, s in symbol_stats.items():
            sym_win_rate = (s["wins"] / s["trades"] * 100) if s["trades"] > 0 else 0.0
            stats_context += f"  * {sym}: {s['trades']} trades, {s['wins']}W/{s['losses']}L ({sym_win_rate:.1f}% WR), Net Profit: {s['profit']:.2f} USD\n"
            
        # 3. Format detailed losing trades (up to 50 for depth without overflowing token context)
        losing_trades = losses[:50]
        losses_context = "RECENT LOSING TRADES DETAILS:\n"
        for t in losing_trades:
            gann_data_str = ""
            if t.get("gann_data"):
                try:
                    gd = json.loads(t["gann_data"])
                    gann_data_str = f", Gann: Type={gd.get('type')}, Pivots (A={gd.get('A')}, B={gd.get('B')}, C={gd.get('C')}), Retracement={gd.get('retracement_pct', 0):.1f}%, TargetAngle={gd.get('target_angle')}°"
                except Exception:
                    pass
            losses_context += (
                f"- Ticket #{t['ticket']}, Symbol: {t['symbol']}, Action: {t['action']}, "
                f"Volume: {t['volume']} lots, Entry Price: {t['entry_price']}, Close Price: {t['close_price']}, "
                f"Profit: {t['profit']} USD, Open Time: {t['open_time']}, Close Time: {t['close_time']}, "
                f"Reason: {t['reason']}{gann_data_str}\n"
            )
            
        # Get active settings for context
        settings = get_settings()
        settings_context = "CURRENT BOT CONFIGURATIONS:\n"
        for k, v in settings.items():
            if k not in ["telegram_token", "telegram_chat_id", "strategy_prompt"]: # hide secrets and long prompts
                settings_context += f"- {k}: {v}\n"
                
        # 4. Prompt Gemini
        system_instruction = (
            "You are a professional Senior Quantitative Trading Auditor and Risk Coach. The user runs an MT5 trading bot utilizing a Gann pivot geometry breakout strategy combined with a Martingale grid scaling system.\n"
            "Your task is to analyze the bot's overall trading performance, specifically focusing on its losing trades, to identify common patterns, operational mistakes, and structural errors that lead to losses.\n"
            "Then, provide practical solutions and recommendations to optimize settings (e.g. adjusting grid sizes, disabling poorly performing currency pairs, refining Gann entries, avoiding high-impact news times).\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. You must NOT attempt to change any code, files, or settings. You are a READ-ONLY analyst. Keep all suggestions strictly as recommendations for the user to configure manually.\n"
            "2. Focus heavily on identifying the specific errors in the provided list of losing trades. Categorize these errors (e.g., 'Martingale grid too aggressive', 'Trading during news', 'Bad Gann breakouts on specific pairs').\n"
            "3. Present your report in beautiful, professional Arabic markdown. Use sections, bullet points, and highlight key terms (using bold/italics or tables).\n"
            "4. Keep the tone constructive, analytical, and professional. Explain the math or logic behind your recommendations."
        )
        
        prompt = (
            f"Here is the bot's performance data:\n\n"
            f"{stats_context}\n"
            f"{settings_context}\n"
            f"{losses_context}\n\n"
            "Please perform the performance audit and provide the Arabic report now."
        )
        
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            return jsonify({"success": False, "error": "GEMINI_API_KEY missing in server env"}), 500
            
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_instruction
        )
        
        response = model.generate_content(prompt)
        
        return jsonify({
            "success": True,
            "response": response.text
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
#  Server Startup & Tear-down
# ============================================================

if __name__ == '__main__':
    # Start the background scanning loop thread
    bot_thread = threading.Thread(target=bot_background_loop)
    bot_thread.daemon = True
    bot_thread.start()
    
    # Start the Telegram listener thread
    try:
        from telegram_listener import start_telegram_listener
        start_telegram_listener()
    except Exception as tg_err:
        print(f"[SERVER STARTUP] [ERROR] Failed to start Telegram listener: {tg_err}")
    
    try:
        # Run Flask server
        import socket
        local_ip = "127.0.0.1"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass
            
        print("\n" + "=" * 55)
        print("🚀 WEB DASHBOARD RUNNING:")
        print(f"   - Local:  http://127.0.0.1:5000")
        print(f"   - Wi-Fi:  http://{local_ip}:5000 (Use on Mobile)")
        print("=" * 55 + "\n")
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        # Stop background threads on server shutdown
        bot_running = False
        print("Stopping background bot thread...")
        try:
            from telegram_listener import stop_telegram_listener
            stop_telegram_listener()
        except Exception:
            pass
