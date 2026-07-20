"""
Harmonic Patterns Backtester (Gartley / Bat / Butterfly / Crab)
=================================================================
Downloads historical candles for a symbol/timeframe and replays the same
Harmonic Patterns logic used live in trading_bot.py
(manage_harmonic_patterns_strategy / check_harmonic_pattern_signal), bar by
bar, then reports win rate, profit factor, ROI, and max drawdown, broken
down by which pattern (Gartley/Bat/Butterfly/Crab) triggered each trade.

Reuses fetch_backtest_data()/get_currency_conversion_rate() from
backtest_gann.py and detect_harmonic_pattern()/HARMONIC_PATTERNS from
gann_helper.py, so the pattern-detection math matches the live strategy
exactly. The entry-decision logic (PRZ proximity + RSI confirmation) is
replicated inline, matching backtest_range_trading.py's approach, to keep
this script's dependencies limited to MT5/pandas/gann_helper.

detect_harmonic_pattern() is called on a bounded trailing slice of the
dataframe ending at each bar (lookback+100 candles, matching the live
strategy's fetch size) rather than the full growing history — same
lookahead-avoidance reasoning as backtest_range_trading.py's use of
detect_range_zone(): a pivot near the end of a precomputed-on-everything
scan could be "confirmed" using bars that wouldn't exist yet in a real-time
replay. .reset_index(drop=True) is required after slicing since the pivot
detector uses positional .loc[j, ...] indexing.

RSI and ATR, by contrast, are precomputed ONCE on the full dataframe (like
backtest_range_trading.py's ADX/ATR treatment): both are causal rolling
computations with no lookahead risk, so there's no need to recompute them
per bar on a bounded window.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import MetaTrader5 as mt5
from dotenv import load_dotenv

from mt5_connection import connect_mt5, disconnect_mt5
from backtest_gann import fetch_backtest_data, get_currency_conversion_rate
from gann_helper import detect_harmonic_pattern, HARMONIC_PATTERNS

load_dotenv()


def _compute_atr(df, period=14):
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def run_backtest_harmonic_patterns(df, symbol, lookback=100, swing_window=5, patterns=None,
                                    ratio_tolerance_pct=5.0, prz_confluence_pct=10.0,
                                    entry_zone_pct=0.15, atr_period=14, atr_sl_multiplier=1.0,
                                    tp_retracement_ratio=0.618, rsi_overbought=65.0, rsi_oversold=35.0,
                                    min_rr_ratio=1.0, min_tp_spread_multiple=2.0, freshness_bars=10):
    if patterns is None:
        patterns = list(HARMONIC_PATTERNS.keys())

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[ERROR] Symbol {symbol} not found in MT5.")
        return []

    contract_size = symbol_info.trade_contract_size
    currency_conversion_rate = get_currency_conversion_rate(symbol_info)
    contract_size_usd = contract_size * currency_conversion_rate
    if currency_conversion_rate != 1.0:
        print(f"[INFO] {symbol}: profit currency is {symbol_info.currency_profit}, applying conversion rate {currency_conversion_rate:.5f}.")

    # Historical spread isn't available via MT5's API — approximate with the current
    # live spread, applied uniformly across the whole backtest. Noted limitation.
    tick = mt5.symbol_info_tick(symbol)
    spread = (tick.ask - tick.bid) if tick else symbol_info.spread * symbol_info.point
    print(f"[INFO] {symbol}: using current live spread ({spread:.5f}) as a static approximation for the whole backtest period.")

    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi14'] = 100 - (100 / (1 + gain / loss))
    df['atr'] = _compute_atr(df, period=atr_period)

    sim_balance = 500.0
    risk_percent = 3.0
    window_span = lookback + 100  # matches the live strategy's get_candles(count=lookback+100) fetch size
    min_bars_needed = max(atr_period + 5, 20)

    basket = None  # {"type":, "pattern":, "entry":, "vol":, "sl":, "tp":, "open_time":}
    trades = []

    for i in range(min_bars_needed, len(df)):
        if basket is None and sim_balance <= 0:
            break  # account blown — see backtest_sma5_reversion.py for why this guard exists

        current_time = df.loc[i, 'time']
        close_curr = df.loc[i, 'Close']
        high_curr = df.loc[i, 'High']
        low_curr = df.loc[i, 'Low']

        # --- Manage an active position ---
        if basket is not None:
            # SL sits beyond point X — the standard harmonic invalidation level.
            # No separate invalidation check is needed here (unlike Elliott Wave):
            # entry happens near D, which sits on the opposite side of the whole
            # pattern from X, so the SL alone already captures the real
            # invalidation rule (see manage_harmonic_patterns_strategy's docstring).
            hit_sl = (basket["type"] == "BUY" and low_curr <= basket["sl"]) or \
                     (basket["type"] == "SELL" and high_curr >= basket["sl"])
            hit_tp = (basket["type"] == "BUY" and high_curr >= basket["tp"]) or \
                     (basket["type"] == "SELL" and low_curr <= basket["tp"])

            if hit_sl:
                profit_usd = ((basket["sl"] - basket["entry"]) if basket["type"] == "BUY" else (basket["entry"] - basket["sl"])) * basket["vol"] * contract_size_usd
                trades.append({"type": basket["type"], "pattern": basket["pattern"], "entry_time": basket["open_time"], "exit_time": current_time,
                               "volume": basket["vol"], "profit_usd_raw": profit_usd, "result": "LOSS" if profit_usd < 0 else "WIN"})
                sim_balance += profit_usd
                basket = None
            elif hit_tp:
                profit_usd = ((basket["tp"] - basket["entry"]) if basket["type"] == "BUY" else (basket["entry"] - basket["tp"])) * basket["vol"] * contract_size_usd
                trades.append({"type": basket["type"], "pattern": basket["pattern"], "entry_time": basket["open_time"], "exit_time": current_time,
                               "volume": basket["vol"], "profit_usd_raw": profit_usd, "result": "WIN"})
                sim_balance += profit_usd
                basket = None
            continue  # either still open, or just closed — don't open a new one on the same bar

        # --- No active position: scan for a signal ---
        # detect_harmonic_pattern needs a bounded trailing slice (not the full
        # growing history) both for speed and to avoid the lookahead bias fixed
        # in backtest_gann.py: a pivot near the end of a precomputed-on-everything
        # scan could be "confirmed" using bars that wouldn't exist yet in a
        # real-time replay.
        slice_start = max(0, i + 1 - window_span)
        window_df = df.iloc[slice_start:i + 1].reset_index(drop=True)
        if len(window_df) < min_bars_needed:
            continue

        pattern_info = detect_harmonic_pattern(
            window_df, lookback=lookback, window=swing_window, patterns=patterns,
            ratio_tolerance_pct=ratio_tolerance_pct, prz_confluence_pct=prz_confluence_pct
        )
        if not pattern_info:
            continue

        # Freshness: C must have completed recently, not ancient history
        last_idx_in_window = len(window_df) - 1
        if last_idx_in_window - pattern_info["idx_C"] > freshness_bars:
            continue

        prz_price = pattern_info["prz_price"]
        if prz_price == 0:
            continue

        current_price = close_curr  # backtest stand-in for "live current price"
        dist_from_prz_pct = abs(current_price - prz_price) / prz_price * 100.0
        if dist_from_prz_pct > entry_zone_pct:
            continue

        rsi14 = df['rsi14'].iloc[i]
        if pd.isna(rsi14):
            continue

        pattern_type = pattern_info["type"]
        if pattern_type == "BUY":
            if rsi14 > rsi_oversold:
                continue
            decision = "BUY"
        else:
            if rsi14 < rsi_overbought:
                continue
            decision = "SELL"

        entry_price = close_curr
        atr = df['atr'].iloc[i]
        if pd.isna(atr):
            atr = 0.0

        point_X = pattern_info["X"]
        point_C = pattern_info["C"]

        if decision == "BUY":
            sl = point_X - atr * atr_sl_multiplier
            tp = prz_price + tp_retracement_ratio * (point_C - prz_price)
        else:
            sl = point_X + atr * atr_sl_multiplier
            tp = prz_price - tp_retracement_ratio * (prz_price - point_C)

        # TP must actually sit on the profitable side of entry (defense in
        # depth — same class of bug found and fixed in Range Trading/Classical
        # Patterns; lower risk here since entry is anchored close to the PRZ).
        if decision == "BUY" and tp <= entry_price:
            continue
        if decision == "SELL" and tp >= entry_price:
            continue

        sl_dist = abs(entry_price - sl)
        tp_dist = abs(tp - entry_price)
        if sl_dist <= 0:
            continue

        # Spread guard — same rationale as the other strategies: skip targets too
        # small relative to the (approximated) spread to be realistically closable.
        if tp_dist < spread * min_tp_spread_multiple:
            continue

        rr_ratio = tp_dist / sl_dist
        if rr_ratio < min_rr_ratio:
            continue

        # Dynamic lot sizing: risk 3% of current simulated balance (same convention as backtest_gann.py)
        risk_amount = sim_balance * (risk_percent / 100.0)
        vol = risk_amount / (sl_dist * contract_size_usd)
        vol_step = symbol_info.volume_step if symbol_info.volume_step > 0 else 0.01
        vol = round(vol / vol_step) * vol_step
        vol = max(symbol_info.volume_min, min(vol, symbol_info.volume_max))
        vol = round(vol, 2)

        # If the SL is so wide (relative to account size) that even the broker's
        # minimum lot forces more risk than intended, skip rather than silently
        # accept a multiple of the intended risk (seen producing impossible
        # >100%-loss single trades on high-contract-size symbols like XAGUSDm).
        actual_risk = vol * sl_dist * contract_size_usd
        if actual_risk > risk_amount * 2.0:
            continue

        basket = {"type": decision, "pattern": pattern_info["pattern"], "entry": entry_price, "vol": vol,
                  "sl": sl, "tp": tp, "open_time": current_time}

    return trades


def print_harmonic_report(trades, symbol, timeframe):
    print("=" * 65)
    print(f"   HARMONIC PATTERNS BACKTEST REPORT: {symbol} ({timeframe})")
    print("=" * 65)

    total_trades = len(trades)
    if total_trades == 0:
        print("  No trades were triggered for this symbol/period.")
        print("=" * 65)
        return

    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = sum(1 for t in trades if t["result"] == "LOSS")
    win_rate = (wins / total_trades) * 100.0

    gross_win = sum(t["profit_usd_raw"] for t in trades if t["profit_usd_raw"] > 0)
    gross_loss = abs(sum(t["profit_usd_raw"] for t in trades if t["profit_usd_raw"] < 0))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')

    balance = 500.0
    peak_balance = 500.0
    max_drawdown = 0.0
    for t in trades:
        balance += t["profit_usd_raw"]
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100.0
        if dd > max_drawdown:
            max_drawdown = dd

    print(f"  Total Trades:           {total_trades}")
    print(f"  Wins:                   {wins} (Win Rate: {win_rate:.1f}%)")
    print(f"  Losses:                 {losses}")
    print(f"  Profit Factor:          {profit_factor:.2f}")
    print("-" * 65)
    print("  BY PATTERN:")
    for name in sorted(set(t["pattern"] for t in trades)):
        p_trades = [t for t in trades if t["pattern"] == name]
        p_wins = sum(1 for t in p_trades if t["result"] == "WIN")
        p_profit = sum(t["profit_usd_raw"] for t in p_trades)
        print(f"    {name:<12} {len(p_trades):>4} trades, {p_wins}/{len(p_trades)} wins, ${p_profit:+.2f}")
    print("-" * 65)
    print("  SIMULATED ACCOUNT METRICS (Starting $500, Risking 3%):")
    print(f"  Final Account Balance:  ${balance:.2f} USD")
    print(f"  Net ROI:                {((balance - 500.0) / 500.0 * 100):+.1f}%")
    print(f"  Max Drawdown:           {max_drawdown:.1f}%")
    print("=" * 65)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest the Harmonic Patterns strategy on historical MT5 data.")
    parser.add_argument("--symbol", type=str, default="EURUSDm", help="Trading symbol, or comma-separated list e.g. EURUSDm,GBPUSDm")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe (e.g. H1, M30)")
    parser.add_argument("--years", type=int, default=1, help="Years of data to backtest")
    parser.add_argument("--lookback", type=int, default=100, help="Pivot-scan lookback window for X-A-B-C detection")
    parser.add_argument("--swing-window", type=int, default=5, help="Swing-pivot detection window (candles on each side)")
    parser.add_argument("--patterns", type=str, default="Gartley,Bat,Butterfly,Crab", help="Comma-separated pattern names to check")
    parser.add_argument("--ratio-tolerance-pct", type=float, default=5.0, help="Tolerance applied to each Fibonacci ratio band")
    parser.add_argument("--prz-confluence-pct", type=float, default=10.0, help="Max %% difference between the two D projections")
    parser.add_argument("--entry-zone-pct", type=float, default=0.15, help="%% of price from the PRZ required to trigger entry")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period (SL noise buffer)")
    parser.add_argument("--atr-sl-multiplier", type=float, default=1.0, help="SL distance beyond point X, as a multiple of ATR")
    parser.add_argument("--tp-retracement-ratio", type=float, default=0.618, help="TP retracement of the C->D leg back toward C")
    parser.add_argument("--rsi-overbought", type=float, default=65.0, help="RSI14 level required to confirm a SELL")
    parser.add_argument("--rsi-oversold", type=float, default=35.0, help="RSI14 level required to confirm a BUY")
    parser.add_argument("--min-rr", type=float, default=1.0, help="Minimum R:R ratio required to take the trade")
    parser.add_argument("--min-tp-spread-multiple", type=float, default=2.0, help="Require TP distance to be at least this many multiples of the spread")
    parser.add_argument("--freshness-bars", type=int, default=10, help="Max bars since C for the pattern to still be considered fresh")

    args = parser.parse_args()

    if not connect_mt5():
        print("[FATAL] Could not connect to MT5.")
        sys.exit(1)

    try:
        symbols = [s.strip() for s in args.symbol.split(",") if s.strip()]
        pattern_list = [p.strip() for p in args.patterns.split(",") if p.strip()]
        results = {}

        for sym in symbols:
            df = fetch_backtest_data(sym, args.timeframe, args.years)
            if df is None or len(df) == 0:
                print(f"[WARNING] Skipping symbol {sym} due to lack of historical data.")
                continue

            trades = run_backtest_harmonic_patterns(
                df, sym,
                lookback=args.lookback,
                swing_window=args.swing_window,
                patterns=pattern_list,
                ratio_tolerance_pct=args.ratio_tolerance_pct,
                prz_confluence_pct=args.prz_confluence_pct,
                entry_zone_pct=args.entry_zone_pct,
                atr_period=args.atr_period,
                atr_sl_multiplier=args.atr_sl_multiplier,
                tp_retracement_ratio=args.tp_retracement_ratio,
                rsi_overbought=args.rsi_overbought,
                rsi_oversold=args.rsi_oversold,
                min_rr_ratio=args.min_rr,
                min_tp_spread_multiple=args.min_tp_spread_multiple,
                freshness_bars=args.freshness_bars
            )
            results[sym] = trades
            print_harmonic_report(trades, sym, args.timeframe)

        if len(results) > 1:
            print("\n" + "=" * 95)
            print("                  COMPARATIVE HARMONIC PATTERNS BACKTEST SUMMARY")
            print("=" * 95)
            print(f"{'Symbol':<12} | {'Trades':<7} | {'Win Rate':<9} | {'Net ROI (%)':<12} | {'Max DD (%)':<11} | {'Profit Factor':<13}")
            print("-" * 95)
            for sym, trades in results.items():
                total = len(trades)
                if total == 0:
                    print(f"{sym:<12} | {'0':<7} | {'0.0%':<9} | {'+0.0%':<12} | {'0.0%':<11} | {'0.00':<13}")
                    continue
                wins = sum(1 for t in trades if t["result"] == "WIN")
                win_rate = (wins / total) * 100.0
                gross_win = sum(t["profit_usd_raw"] for t in trades if t["profit_usd_raw"] > 0)
                gross_loss = abs(sum(t["profit_usd_raw"] for t in trades if t["profit_usd_raw"] < 0))
                pf = gross_win / gross_loss if gross_loss > 0 else (99.99 if gross_win > 0 else 1.0)

                balance = 500.0
                peak = 500.0
                max_dd = 0.0
                for t in trades:
                    balance += t["profit_usd_raw"]
                    if balance > peak:
                        peak = balance
                    dd = (peak - balance) / peak * 100.0
                    if dd > max_dd:
                        max_dd = dd
                roi = ((balance - 500.0) / 500.0) * 100.0
                pf_str = f"{pf:.2f}" if pf < 99.99 else "99.99"
                print(f"{sym:<12} | {total:<7} | {f'{win_rate:.1f}%':<9} | {f'{roi:+.1f}%':<12} | {f'{max_dd:.1f}%':<11} | {pf_str:<13}")
            print("=" * 95)

    finally:
        disconnect_mt5()


if __name__ == "__main__":
    main()
