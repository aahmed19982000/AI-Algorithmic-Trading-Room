"""
Live Elliott Wave (Wave 0->1->2) Chart Analyzer
================================================
Standalone version of the project's Elliott Wave strategy
(see backtest_elliott_wave.py / trading_bot.py's manage_elliott_wave_strategy)
that does NOT require MetaTrader5 or a running MT5 terminal.

Uses Yahoo Finance (yfinance) for OHLC data instead of MT5, so it can run
on any OS. The pivot-detection and wave-validation math is imported
directly from gann_helper.py (pure functions, no MT5 dependency) so the
result matches the live bot's logic.

Usage:
    python3 analyze_elliott_wave_live.py --symbol EURUSD=X --interval 1h --period 90d
"""
import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf

from gann_helper import calculate_fibonacci_extension, _check_wave1_impulse_quality

WINDOW = 5
MIN_RETRACEMENT = 0.382
MAX_RETRACEMENT = 0.786
MAX_WAVE1_INTERNAL_RETRACEMENT = 0.618
EXTENSION_RATIO = 1.618
FRESHNESS_BARS = 5


def fetch_data(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close"})
    df = df.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={time_col: "time"})
    df = df[["time", "Open", "High", "Low", "Close"]].dropna().reset_index(drop=True)
    return df


def find_pivots(df):
    pivots_high, pivots_low = [], []
    for i in range(WINDOW, len(df) - WINDOW):
        window_slice = range(i - WINDOW, i + WINDOW + 1)
        if all(df.loc[j, 'High'] <= df.loc[i, 'High'] for j in window_slice):
            pivots_high.append((i, df.loc[i, 'High']))
        if all(df.loc[j, 'Low'] >= df.loc[i, 'Low'] for j in window_slice):
            pivots_low.append((i, df.loc[i, 'Low']))
    return pivots_high, pivots_low


def find_latest_setup(df, pivots_high, pivots_low, lookback=150):
    """Same A(Wave0)->B(Wave1)->C(Wave2) scan as run_backtest_elliott_wave(),
    but run once against the most recent bar instead of bar-by-bar."""
    i = len(df) - 1
    lookback_start = max(0, i - lookback)
    window_highs = [p for p in pivots_high if lookback_start <= p[0] and p[0] + WINDOW <= i]
    window_lows = [p for p in pivots_low if lookback_start <= p[0] and p[0] + WINDOW <= i]

    setup = None

    # Bullish: Low A -> High B -> Low C
    if len(window_lows) >= 2 and len(window_highs) >= 1:
        for idx_C, val_C in reversed(window_lows):
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
                if MIN_RETRACEMENT <= ratio <= MAX_RETRACEMENT:
                    if _check_wave1_impulse_quality(window_highs, window_lows, idx_A, val_A, idx_B, val_B,
                                                     MAX_WAVE1_INTERNAL_RETRACEMENT, is_buy=True):
                        setup = {"type": "BUY", "idx_A": idx_A, "val_A": val_A, "idx_B": idx_B, "val_B": val_B,
                                 "idx_C": idx_C, "val_C": val_C, "ratio": ratio}
                        break

    # Bearish: High A -> Low B -> High C
    if setup is None and len(window_highs) >= 2 and len(window_lows) >= 1:
        for idx_C, val_C in reversed(window_highs):
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
                if MIN_RETRACEMENT <= ratio <= MAX_RETRACEMENT:
                    if _check_wave1_impulse_quality(window_highs, window_lows, idx_A, val_A, idx_B, val_B,
                                                     MAX_WAVE1_INTERNAL_RETRACEMENT, is_buy=False):
                        setup = {"type": "SELL", "idx_A": idx_A, "val_A": val_A, "idx_B": idx_B, "val_B": val_B,
                                 "idx_C": idx_C, "val_C": val_C, "ratio": ratio}
                        break

    return setup


def plot_chart(df, symbol, interval, setup, recent_pivots, out_path):
    plot_df = df.set_index('time')

    mc = mpf.make_marketcolors(up='#2ecc71', down='#e74c3c', edge='inherit', wick='inherit', inherit=True)
    style = mpf.make_mpf_style(base_mpf_style='charles', marketcolors=mc, gridcolor='#2d3748',
                                facecolor='#1e272e', figcolor='#1e272e', gridstyle='dashed')

    plot_window = min(len(df), 250)
    plot_df_tail = plot_df.tail(plot_window)
    offset = len(df) - plot_window

    title = f"{symbol} ({interval}) - Elliott Wave Analysis"
    if setup:
        title += f"\n{setup['type']} setup: Wave2 retraced {setup['ratio']*100:.1f}% of Wave1"

    fig, axlist = mpf.plot(plot_df_tail, type='candle', style=style, title=title,
                            volume=False, returnfig=True, figsize=(14, 8))
    ax = axlist[0]

    def x(idx):
        return idx - offset

    # Draw the recent swing structure (zigzag of confirmed pivots) so the
    # overall wave pattern is visible even outside the highlighted setup.
    zz_x, zz_y = [], []
    for idx, val in recent_pivots:
        if idx >= offset:
            zz_x.append(x(idx))
            zz_y.append(val)
    if len(zz_x) > 1:
        ax.plot(zz_x, zz_y, color='#7f8c8d', linestyle='-', linewidth=1, alpha=0.6, zorder=2)

    if setup:
        trade_type = setup["type"]
        pts = [("Wave 0 (Start)", setup["idx_A"], setup["val_A"], '#f39c12'),
               ("Wave 1 (Peak)", setup["idx_B"], setup["val_B"], '#e67e22'),
               ("Wave 2 (Correction)", setup["idx_C"], setup["val_C"], '#3498db')]
        px, py = [], []
        for label, idx, val, color in pts:
            xi = x(idx)
            px.append(xi)
            py.append(val)
            ax.plot(xi, val, marker='o', color=color, markersize=9, zorder=5)
            va = ('top' if trade_type == 'BUY' else 'bottom') if 'Start' in label else \
                 ('bottom' if trade_type == 'BUY' else 'top') if 'Peak' in label else \
                 ('top' if trade_type == 'BUY' else 'bottom')
            ax.text(xi, val, f"  {label}", color=color, fontsize=9, fontweight='bold', va=va)
        ax.plot(px, py, color='#9b59b6', linestyle='dotted', linewidth=2, zorder=4)

        wave0, wave1 = setup["val_A"], setup["val_B"]
        tp_target = calculate_fibonacci_extension(wave0, wave1, EXTENSION_RATIO)
        last_close = float(df['Close'].iloc[-1])

        hlines = [wave0, tp_target, last_close]
        hl_colors = ['#e74c3c', '#2ecc71', '#f1c40f']
        ax.hlines(hlines, xmin=0, xmax=plot_window - 1, colors=hl_colors, linestyles='dashed', linewidth=1.2, alpha=0.8)
        ax.text(plot_window - 1, wave0, '  Invalidation (SL)', color='#e74c3c', fontsize=8, fontweight='bold', va='center')
        ax.text(plot_window - 1, tp_target, f'  Wave 3 Target ({EXTENSION_RATIO}x ext)', color='#2ecc71', fontsize=8, fontweight='bold', va='center')
        ax.text(plot_window - 1, last_close, '  Current Price', color='#f1c40f', fontsize=8, fontweight='bold', va='center')

        i_last = len(df) - 1
        is_fresh = (i_last - setup["idx_C"]) <= FRESHNESS_BARS
        confirmed = (trade_type == "BUY" and last_close > setup["val_C"]) or \
                    (trade_type == "SELL" and last_close < setup["val_C"])
        if is_fresh and confirmed:
            status = "ACTIVE SETUP - entry conditions met"
            status_color = '#2ecc71'
        elif is_fresh:
            status = "Setup fresh - awaiting reversal confirmation off Wave 2"
            status_color = '#f1c40f'
        else:
            status = "Most recent qualifying setup (not fresh - informational only)"
            status_color = '#95a5a6'
        ax.text(0.02, 0.95, status, transform=ax.transAxes, color=status_color, fontsize=10, fontweight='bold',
                bbox=dict(facecolor='#1e272e', alpha=0.85, edgecolor=status_color, boxstyle='round,pad=0.4'))
    else:
        ax.text(0.02, 0.95, "No qualifying Wave 0-1-2 setup found in the current lookback window",
                transform=ax.transAxes, color='#e74c3c', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='#1e272e', alpha=0.85, edgecolor='#e74c3c', boxstyle='round,pad=0.4'))

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Analyze a symbol's current Elliott Wave (0->1->2) structure and draw it.")
    parser.add_argument("--symbol", type=str, default="EURUSD=X", help="Yahoo Finance ticker, e.g. EURUSD=X")
    parser.add_argument("--interval", type=str, default="1h", help="Candle interval, e.g. 1h, 4h, 1d")
    parser.add_argument("--period", type=str, default="90d", help="History period, e.g. 90d, 6mo, 1y")
    parser.add_argument("--lookback", type=int, default=150, help="Pivot-scan lookback window (bars)")
    parser.add_argument("--out", type=str, default="elliott_wave_analysis.png", help="Output PNG path")
    args = parser.parse_args()

    df = fetch_data(args.symbol, args.interval, args.period)
    if df is None or len(df) < 2 * WINDOW + 10:
        print(f"[ERROR] Not enough data returned for {args.symbol} ({args.interval}, {args.period}).")
        sys.exit(1)

    pivots_high, pivots_low = find_pivots(df)
    setup = find_latest_setup(df, pivots_high, pivots_low, lookback=args.lookback)

    all_pivots = sorted(pivots_high + pivots_low, key=lambda p: p[0])
    recent_pivots = all_pivots[-10:]

    out_path = plot_chart(df, args.symbol, args.interval, setup, recent_pivots, args.out)
    print(f"[OK] Chart saved to {out_path}")

    if setup:
        print(f"Setup: {setup['type']}")
        print(f"  Wave 0 (A): {setup['val_A']:.5f}")
        print(f"  Wave 1 (B): {setup['val_B']:.5f}")
        print(f"  Wave 2 (C): {setup['val_C']:.5f}  (retraced {setup['ratio']*100:.1f}% of Wave 1)")
        tp_target = calculate_fibonacci_extension(setup['val_A'], setup['val_B'], EXTENSION_RATIO)
        print(f"  Wave 3 target ({EXTENSION_RATIO}x ext): {tp_target:.5f}")
        print(f"  Current price: {df['Close'].iloc[-1]:.5f}")
    else:
        print("No qualifying Wave 0-1-2 setup found in the current lookback window.")


if __name__ == "__main__":
    main()
