import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to run headlessly
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta

def generate_chart_screenshot(symbol, ticket, entry_price, sl_price, tp_price, gann_data=None, timeframe_str="M30", entry_time_str=None, exit_time_str=None, exit_price=None, result=None, profit_usd=None, strategy_label="Gann"):
    """
    Fetches historical candle data from MT5 and plots the candlesticks along with
    Entry, TP, and SL horizontal lines, and highlights the Gann A-B-C pivot structure.
    Saves the resulting image to the static screenshots folder.
    """
    # 1. Fetch candles from MT5
    if not mt5.initialize():
        raise RuntimeError("MT5 terminal offline")
        
    from mt5_data import get_timeframe
    tf = get_timeframe(timeframe_str)
    if tf is None:
        tf = mt5.TIMEFRAME_M30
        
    # Get tf minutes for padding
    def get_tf_mins(tf_s):
        tf_s = tf_s.upper()
        if tf_s.startswith("M"):
            return int(tf_s[1:]) if tf_s[1:].isdigit() else 30
        elif tf_s.startswith("H"):
            return int(tf_s[1:]) * 60 if tf_s[1:].isdigit() else 60
        elif tf_s == "D1":
            return 1440
        return 30
        
    tf_mins = get_tf_mins(timeframe_str)
    rates = None
    
    # Try to fetch range starting from Point A to now (or exit time if closed)
    if gann_data and gann_data.get("time_A"):
        try:
            time_A_dt = datetime.strptime(gann_data["time_A"], "%Y-%m-%d %H:%M:%S")
            start_date = time_A_dt - timedelta(minutes=tf_mins * 10)
        except Exception as ex:
            print(f"[SCREENSHOT] Error calculating start_dt from time_A: {ex}")
            start_date = datetime.now() - timedelta(minutes=tf_mins * 300)
    else:
        start_date = datetime.now() - timedelta(minutes=tf_mins * 300)
        
    if exit_time_str:
        try:
            exit_dt = datetime.strptime(exit_time_str, "%Y-%m-%d %H:%M:%S")
            end_date = exit_dt + timedelta(minutes=tf_mins * 20)
        except Exception:
            end_date = datetime.now() + timedelta(minutes=tf_mins * 20)
    else:
        end_date = datetime.now() + timedelta(minutes=tf_mins * 20)
        
    rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
            
    if rates is None or len(rates) == 0:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, 300)
        
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to fetch rates for {symbol}")
        
    # Convert to DataFrame
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

    # Resolve entry and swing pivot indices on the uncropped dataframe
    idx_entry_uncropped = get_idx_of_time(df, entry_time_str) if entry_time_str else len(df) - 1
    if idx_entry_uncropped is None or idx_entry_uncropped < 0:
        idx_entry_uncropped = len(df) - 1
        
    idx_exit_uncropped = get_idx_of_time(df, exit_time_str) if exit_time_str else None
    
    idx_X_uncropped = None
    idx_A_uncropped = None
    idx_B_uncropped = None
    idx_C_uncropped = None
    
    val_X = None
    val_A = None
    val_B = None
    val_C = None
    
    # Check if this is a reversal classical pattern
    is_reversal_classical = False
    if gann_data and strategy_label == "Classical Patterns" and not gann_data.get("is_continuation", False):
        pattern_info = gann_data.get("pattern_info")
        if pattern_info:
            is_reversal_classical = True
            val_A = pattern_info.get("shoulder_a")
            val_B = pattern_info.get("head")
            val_C = pattern_info.get("shoulder_b")
            if pattern_info.get("time_sa"):
                idx_A_uncropped = get_idx_of_time(df, pattern_info["time_sa"])
            if pattern_info.get("time_head"):
                idx_B_uncropped = get_idx_of_time(df, pattern_info["time_head"])
            if pattern_info.get("time_sb"):
                idx_C_uncropped = get_idx_of_time(df, pattern_info["time_sb"])
    
    elif gann_data:
        val_X = gann_data.get("X")
        val_A = gann_data.get("A")
        val_B = gann_data.get("B")
        val_C = gann_data.get("C")
        trade_type = gann_data.get("type", "BUY")
        
        # Resolve Point X
        if gann_data.get("time_X"):
            idx_X_uncropped = get_idx_of_time(df, gann_data["time_X"])
        if (idx_X_uncropped is None or idx_X_uncropped < 0) and val_X is not None:
            search_limit = idx_entry_uncropped
            col = 'Low' if trade_type == 'SELL' else 'High'
            best_idx = None
            min_diff = float('inf')
            for i in range(0, search_limit):
                diff = abs(df.iloc[i][col] - val_X)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i
            if min_diff <= val_X * 0.0005:
                idx_X_uncropped = best_idx

        # Resolve Point A index
        if gann_data.get("time_A"):
            idx_A_uncropped = get_idx_of_time(df, gann_data["time_A"])
        if (idx_A_uncropped is None or idx_A_uncropped < 0) and val_A is not None:
            search_limit = idx_entry_uncropped
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
            search_limit = idx_entry_uncropped
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
            search_limit = idx_entry_uncropped
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
                
    # Crop df to start 10 candles before the pattern start if found
    crop_offset = 0
    start_pivot_idx = idx_A_uncropped
    if idx_X_uncropped is not None:
        start_pivot_idx = idx_X_uncropped
    elif gann_data and strategy_label == "Classical Patterns" and gann_data.get("is_continuation", False):
        consol_start_time = gann_data.get("consol_start_time")
        if consol_start_time:
            start_pivot_idx = get_idx_of_time(df, consol_start_time)
            
    if start_pivot_idx is not None and start_pivot_idx >= 10:
        crop_offset = start_pivot_idx - 10
        df = df.iloc[crop_offset:]
        
    # Calculate final cropped indices
    idx_X = idx_X_uncropped - crop_offset if idx_X_uncropped is not None else None
    idx_A = idx_A_uncropped - crop_offset if idx_A_uncropped is not None else None
    idx_B = idx_B_uncropped - crop_offset if idx_B_uncropped is not None else None
    idx_C = idx_C_uncropped - crop_offset if idx_C_uncropped is not None else None
    idx_entry = idx_entry_uncropped - crop_offset if idx_entry_uncropped is not None else None
    idx_exit = idx_exit_uncropped - crop_offset if idx_exit_uncropped is not None else None
    
    # 2. Setup directory
    static_dir = os.path.join(os.getcwd(), 'static', 'screenshots')
    os.makedirs(static_dir, exist_ok=True)
    filename = f"{ticket}.png"
    filepath = os.path.join(static_dir, filename)
    
    # 3. Create chart style and plot
    # Setup custom colors
    mc = mpf.make_marketcolors(
        up='#2ecc71', down='#e74c3c',
        edge='inherit', wick='inherit',
        volume='#3498db', inherit=True
    )
    s = mpf.make_mpf_style(
        base_mpf_style='charles',
        marketcolors=mc,
        gridcolor='#2d3748',
        facecolor='#1e272e',  # Dark slate background
        figcolor='#1e272e',
        gridstyle='dashed'
    )
    
    # Build horizontal lines
    original_sl = val_A
    hlines = [entry_price, tp_price]
    hl_colors = ['#f1c40f', '#2ecc71']
    
    if original_sl and abs(sl_price - original_sl) > 1e-5:
        hlines.extend([sl_price, original_sl])
        hl_colors.extend(['#e74c3c', '#d35400']) # red for trailed SL, orange for original SL
    else:
        hlines.append(sl_price)
        hl_colors.append('#e74c3c')
        
    hlines_dict = dict(
        hlines=hlines,
        colors=hl_colors,
        linestyle='dashed',
        linewidths=1.5
    )
    
    # Prepare title text
    setup_title = f"{symbol} ({timeframe_str}) - Trade #{ticket}"
    if gann_data:
        p_name = gann_data.get("pattern", strategy_label)
        if strategy_label == "Classical Patterns" and gann_data.get("is_continuation", False):
            setup_title += f"\nContinuation Pattern: {p_name} ({gann_data.get('type', '')})"
        elif strategy_label == "Classical Patterns":
            setup_title += f"\nReversal Pattern: {p_name} ({gann_data.get('type', '')})"
        elif strategy_label == "Harmonic Patterns":
            setup_title += f"\nHarmonic Pattern: {p_name} ({gann_data.get('type', '')})"
        else:
            setup_title += f"\n{strategy_label} Setup: A={val_A:.5f}, B={val_B:.5f}, C={val_C:.5f} ({gann_data.get('type', '')})"

    point_labels = {
        "Elliott Wave": ("Wave 0 (Start)", "Wave 1 (Peak)", "Wave 2 (Correction)"),
        "Harmonic Patterns": ("X", "A", "B", "C"),
    }.get(strategy_label, ("Point A (Origin)", "Point B (Breakout)", "Point C (Correction)"))
    
    if is_reversal_classical:
        point_labels = ("Left Shoulder", "Head", "Right Shoulder")
        
    # Plot using mplfinance
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
    
    # Annotate lines on chart
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
        ax.text(x_pos, original_sl, '  Original SL', color='#d35400', fontsize=8, fontweight='bold', va='center')
    
    # Plot Connectors and circles
    pivots_x = []
    pivots_y = []
    
    if gann_data:
        trade_type = gann_data.get("type", "BUY")
        
        # 1. Harmonic Patterns Drawing
        if strategy_label == "Harmonic Patterns":
            if all(v is not None for v in [idx_X, idx_A, idx_B, idx_C, idx_entry]) and all(val is not None for val in [val_X, val_A, val_B, val_C]):
                # Connect standard X-A-B-C-D leg
                ax.plot([idx_X, idx_A, idx_B, idx_C, idx_entry], 
                        [val_X, val_A, val_B, val_C, entry_price], 
                        color='#9b59b6', linestyle='-', linewidth=2, zorder=4)
                # Triangle X-A-B
                ax.plot([idx_X, idx_B], [val_X, val_B], color='#9b59b6', linestyle='--', linewidth=1, zorder=3)
                # Triangle B-C-D
                ax.plot([idx_B, idx_entry], [val_B, entry_price], color='#9b59b6', linestyle='--', linewidth=1, zorder=3)
                # Connect A-C
                ax.plot([idx_A, idx_C], [val_A, val_C], color='#9b59b6', linestyle='--', linewidth=1, zorder=3)
                
                # Markers and text
                ax.plot(idx_X, val_X, marker='o', color='#e74c3c', markersize=8, zorder=5)
                ax.text(idx_X, val_X, '  X', color='#e74c3c', fontsize=10, fontweight='bold', va='center')
                ax.plot(idx_A, val_A, marker='o', color='#2ecc71', markersize=8, zorder=5)
                ax.text(idx_A, val_A, '  A', color='#2ecc71', fontsize=10, fontweight='bold', va='center')
                ax.plot(idx_B, val_B, marker='o', color='#f1c40f', markersize=8, zorder=5)
                ax.text(idx_B, val_B, '  B', color='#f1c40f', fontsize=10, fontweight='bold', va='center')
                ax.plot(idx_C, val_C, marker='o', color='#3498db', markersize=8, zorder=5)
                ax.text(idx_C, val_C, '  C', color='#3498db', fontsize=10, fontweight='bold', va='center')
                ax.plot(idx_entry, entry_price, marker='o', color='#1abc9c', markersize=8, zorder=5)
                ax.text(idx_entry, entry_price, '  D', color='#1abc9c', fontsize=10, fontweight='bold', va='center')
                
        # 2. Classical Continuation Patterns (Trendlines)
        elif strategy_label == "Classical Patterns" and gann_data.get("is_continuation", False):
            pattern_info = gann_data.get("pattern_info")
            consol_start_time = gann_data.get("consol_start_time")
            if pattern_info and consol_start_time:
                idx_consol_start = get_idx_of_time(df, consol_start_time)
                if idx_consol_start is not None and idx_consol_start >= 0:
                    idxs = []
                    y_uppers = []
                    y_lowers = []
                    for idx_val in range(idx_consol_start, len(df)):
                        x = idx_val - idx_consol_start
                        y_up = pattern_info["upper_slope"] * x + pattern_info["upper_intercept"]
                        y_lo = pattern_info["lower_slope"] * x + pattern_info["lower_intercept"]
                        idxs.append(idx_val)
                        y_uppers.append(y_up)
                        y_lowers.append(y_lo)
                    ax.plot(idxs, y_uppers, color='#3498db', linestyle='-', linewidth=2, zorder=4)
                    ax.plot(idxs, y_lowers, color='#e74c3c', linestyle='-', linewidth=2, zorder=4)
                    
        # 3. Gann / Elliott Wave / Reversal Classical Patterns (A-B-C pivot structure)
        else:
            if idx_A is not None and idx_A >= 0 and val_A is not None:
                pivots_x.append(idx_A)
                pivots_y.append(val_A)
                ax.plot(idx_A, val_A, marker='o', color='#f39c12', markersize=8, zorder=5)
                va_A = 'top' if trade_type == 'BUY' else 'bottom'
                ax.text(idx_A, val_A, f'  {point_labels[0]}', color='#f39c12', fontsize=9, fontweight='bold', va=va_A, ha='left')
                
            if idx_B is not None and idx_B >= 0 and val_B is not None:
                pivots_x.append(idx_B)
                pivots_y.append(val_B)
                ax.plot(idx_B, val_B, marker='o', color='#e67e22', markersize=8, zorder=5)
                va_B = 'bottom' if trade_type == 'BUY' else 'top'
                ax.text(idx_B, val_B, f'  {point_labels[1]}', color='#e67e22', fontsize=9, fontweight='bold', va=va_B)
                
            if idx_C is not None and idx_C >= 0 and val_C is not None:
                pivots_x.append(idx_C)
                pivots_y.append(val_C)
                ax.plot(idx_C, val_C, marker='o', color='#3498db', markersize=8, zorder=5)
                va_C = 'top' if trade_type == 'BUY' else 'bottom'
                ax.text(idx_C, val_C, f'  {point_labels[2]}', color='#3498db', fontsize=9, fontweight='bold', va=va_C)
                
            if len(pivots_x) > 1:
                pivot_pts = sorted(zip(pivots_x, pivots_y))
                px, py = zip(*pivot_pts)
                ax.plot(px, py, color='#9b59b6', linestyle='dotted', linewidth=1.5, zorder=4)

    # 4. Plot Entry and Exit vertical lines and markers
    if idx_entry is not None and idx_entry >= 0:
        ax.axvline(x=idx_entry, color='#f1c40f', linestyle='--', linewidth=0.8, alpha=0.7)
        trade_type = gann_data.get('type') if gann_data else 'BUY'
        ax.plot(idx_entry, entry_price, marker='^' if trade_type == 'BUY' else 'v', 
                color='#2ecc71' if trade_type == 'BUY' else '#e74c3c', markersize=8, zorder=6)
        
    if idx_exit is not None and idx_exit >= 0:
        ax.axvline(x=idx_exit, color='#95a5a6', linestyle='--', linewidth=0.8, alpha=0.7)
        exit_val = exit_price if exit_price else df.iloc[idx_exit]['Close']
        ax.plot(idx_exit, exit_val, marker='o', 
                color='#e74c3c' if result == 'LOSS' else '#2ecc71', markersize=6, zorder=6)

    # 5. Draw outcome banner
    if result:
        result_color = '#2ecc71' if result == 'WIN' else '#e74c3c'
        try:
            prof_val = float(profit_usd)
            result_text = f"Result: {result} (${prof_val:+.2f})"
        except Exception:
            result_text = f"Result: {result}"
            
        ax.text(0.02, 0.93, result_text, transform=ax.transAxes, color=result_color, 
                fontsize=10, fontweight='bold', bbox=dict(facecolor='#1e272e', alpha=0.85, edgecolor=result_color, boxstyle='round,pad=0.4'))

    # Save again with annotations
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return f"/static/screenshots/{filename}"
