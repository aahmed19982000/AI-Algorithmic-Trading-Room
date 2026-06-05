import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to run headlessly
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime

def generate_chart_screenshot(symbol, ticket, entry_price, sl_price, tp_price, gann_data=None, timeframe_str="M30"):
    """
    Fetches historical candle data from MT5 and plots the candlesticks along with
    Entry, TP, and SL horizontal lines. Saves the resulting image to the static screenshots folder.
    """
    # 1. Fetch candles from MT5
    if not mt5.initialize():
        raise RuntimeError("MT5 terminal offline")
        
    from mt5_data import get_timeframe
    tf = get_timeframe(timeframe_str)
    if tf is None:
        tf = mt5.TIMEFRAME_M30
        
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 100)
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
    
    # Add horizontal lines for SL, Entry, TP
    hlines_dict = dict(
        hlines=[sl_price, entry_price, tp_price],
        colors=['#e74c3c', '#f1c40f', '#2ecc71'],
        linestyle='dashed',
        linewidths=1.5
    )
    
    # Prepare title text
    setup_title = f"{symbol} ({timeframe_str}) - Trade #{ticket}"
    if gann_data:
        setup_title += f"\nGann Setup: A={gann_data.get('A', 0):.5f}, B={gann_data.get('B', 0):.5f}, C={gann_data.get('C', 0):.5f} ({gann_data.get('type', '')})"
        
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
    # Draw text annotations for SL, Entry, TP
    # Get x coordinate for label placement (right side of chart)
    x_pos = len(df) - 5
    # Configure text color matching the lines
    ax.text(x_pos, sl_price, '  SL (Point A)', color='#e74c3c', fontsize=8, fontweight='bold', va='center')
    ax.text(x_pos, entry_price, '  Entry', color='#f1c40f', fontsize=8, fontweight='bold', va='center')
    ax.text(x_pos, tp_price, '  TP (Target)', color='#2ecc71', fontsize=8, fontweight='bold', va='center')
    
    # Save again with annotations
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return f"/static/screenshots/{filename}"
