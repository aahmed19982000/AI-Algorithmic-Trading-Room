"""
Classical Chart Patterns Backtester
=====================================
Downloads historical candles for a symbol/timeframe and replays the same
Classical Chart Patterns logic used live in trading_bot.py
(manage_classical_patterns_strategy / check_classical_pattern_signal), bar
by bar, then reports win rate, profit factor, ROI, and max drawdown, broken
down by which pattern triggered each trade.

Reuses fetch_backtest_data()/get_currency_conversion_rate() from
backtest_gann.py and detect_continuation_pattern()/detect_reversal_pattern()
from gann_helper.py, so the pattern-detection math matches the live
strategy exactly. The entry-decision logic (pure breakout, no RSI gate) is
replicated inline, matching backtest_range_trading.py/
backtest_harmonic_patterns.py's approach.

Both detectors are called on a bounded trailing slice of the dataframe
ending at each bar (pole_lookback + consolidation_max_bars + 50 candles,
matching the live strategy's fetch size) rather than the full growing
history — same lookahead-avoidance reasoning as the other backtests this
session: a pivot near the end of a precomputed-on-everything scan could be
"confirmed" using bars that wouldn't exist yet in a real-time replay.
.reset_index(drop=True) is required after slicing since the pivot detector
uses positional .loc[j, ...] indexing.

ATR, by contrast, is precomputed ONCE on the full dataframe (like the other
backtests' ADX/ATR treatment): it's a causal rolling computation with no
lookahead risk.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import MetaTrader5 as mt5
from dotenv import load_dotenv

from mt5_connection import connect_mt5, disconnect_mt5
from backtest_gann import fetch_backtest_data, get_currency_conversion_rate
from gann_helper import detect_continuation_pattern, detect_reversal_pattern

load_dotenv()


def _compute_atr(df, period=14):
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def run_backtest_classical_patterns(df, symbol, pole_lookback=20, pole_min_move_pct=1.5, pole_min_efficiency=0.6,
                                     consolidation_max_bars=40, swing_window=3, flat_slope_threshold_pct=0.02,
                                     shoulder_tolerance_pct=2.0, neckline_tolerance_pct=1.0, patterns=None,
                                     atr_period=14, atr_sl_multiplier=1.0, tp_multiplier=1.0,
                                     min_rr_ratio=1.0, min_tp_spread_multiple=2.0):
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
    df['atr'] = _compute_atr(df, period=atr_period)

    sim_balance = 500.0
    risk_percent = 3.0
    window_span = pole_lookback + consolidation_max_bars + 50
    reversal_lookback = pole_lookback + consolidation_max_bars
    min_bars_needed = max(window_span, atr_period + 5)

    basket = None  # {"type":, "pattern":, "is_continuation":, "level":, "entry":, "vol":, "sl":, "tp":, "open_time":}
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
            hit_sl = (basket["type"] == "BUY" and low_curr <= basket["sl"]) or \
                     (basket["type"] == "SELL" and high_curr >= basket["sl"])
            hit_tp = (basket["type"] == "BUY" and high_curr >= basket["tp"]) or \
                     (basket["type"] == "SELL" and low_curr <= basket["tp"])

            # Invalidation: a completed candle closes back on the wrong side of
            # the breakout/neckline level — the breakout failed, exit now rather
            # than wait for the far SL (mirrors check_classical_pattern_invalidation).
            invalidated = False
            if not hit_sl and not hit_tp:
                if basket["type"] == "BUY" and close_curr < basket["level"]:
                    invalidated = True
                elif basket["type"] == "SELL" and close_curr > basket["level"]:
                    invalidated = True

            if hit_sl or invalidated:
                exit_price = basket["sl"] if hit_sl else close_curr
                profit_usd = ((exit_price - basket["entry"]) if basket["type"] == "BUY" else (basket["entry"] - exit_price)) * basket["vol"] * contract_size_usd
                trades.append({"type": basket["type"], "pattern": basket["pattern"], "entry_time": basket["open_time"], "exit_time": current_time,
                               "volume": basket["vol"], "profit_usd_raw": profit_usd, "result": "WIN" if profit_usd >= 0 else "LOSS"})
                sim_balance += profit_usd
                basket = None
            elif hit_tp:
                profit_usd = ((basket["tp"] - basket["entry"]) if basket["type"] == "BUY" else (basket["entry"] - basket["tp"])) * basket["vol"] * contract_size_usd
                trades.append({"type": basket["type"], "pattern": basket["pattern"], "entry_time": basket["open_time"], "exit_time": current_time,
                               "volume": basket["vol"], "profit_usd_raw": profit_usd, "result": "WIN"})
                sim_balance += profit_usd
                basket = None
            continue  # either still open, or just closed — don't open a new one on the same bar

        # --- No active position: scan for a signal on a bounded trailing slice ---
        slice_start = max(0, i + 1 - window_span)
        window_df = df.iloc[slice_start:i + 1].reset_index(drop=True)
        if len(window_df) < min_bars_needed:
            continue

        pattern_info = detect_continuation_pattern(
            window_df, pole_lookback=pole_lookback, pole_min_move_pct=pole_min_move_pct,
            pole_min_efficiency=pole_min_efficiency, consolidation_max_bars=consolidation_max_bars,
            swing_window=swing_window, flat_slope_threshold_pct=flat_slope_threshold_pct
        )
        is_continuation = pattern_info is not None
        if not pattern_info:
            pattern_info = detect_reversal_pattern(
                window_df, lookback=reversal_lookback, window=swing_window,
                shoulder_tolerance_pct=shoulder_tolerance_pct, neckline_tolerance_pct=neckline_tolerance_pct,
                patterns=patterns
            )
        if not pattern_info:
            continue

        decision = pattern_info["type"]
        level = pattern_info["breakout_level"] if is_continuation else pattern_info["neckline_level"]

        last_close = window_df['Close'].iloc[-1]  # == close_curr, the last row of the slice is bar i
        if decision == "BUY" and last_close <= level:
            continue
        if decision == "SELL" and last_close >= level:
            continue

        entry_price = close_curr
        atr = df['atr'].iloc[i]
        if pd.isna(atr):
            atr = 0.0

        if is_continuation:
            x_last = consolidation_max_bars - 1
            if decision == "BUY":
                far_level = pattern_info["lower_slope"] * x_last + pattern_info["lower_intercept"]
                sl = far_level - atr * atr_sl_multiplier
                tp = level + pattern_info["pole_height"] * tp_multiplier
            else:
                far_level = pattern_info["upper_slope"] * x_last + pattern_info["upper_intercept"]
                sl = far_level + atr * atr_sl_multiplier
                tp = level - pattern_info["pole_height"] * tp_multiplier
        else:
            head = pattern_info["head"]
            if decision == "BUY":
                sl = head - atr * atr_sl_multiplier
                tp = level + (level - head) * tp_multiplier
            else:
                sl = head + atr * atr_sl_multiplier
                tp = level - (head - level) * tp_multiplier

        # TP must actually sit on the profitable side of entry. Entry can occur
        # well beyond the breakout level/neckline on a strong move, so a target
        # computed purely from level +/- a fixed offset can land behind entry —
        # abs(tp - entry) would still look like a positive R:R, silently turning
        # a loss into a trade that reads as a "WIN" the moment tp is touched.
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
        # accept a multiple of the intended risk — this produced impossible
        # >100%-loss single trades on high-contract-size symbols (XAGUSDm) where
        # a trendline-projected SL landed far from entry.
        actual_risk = vol * sl_dist * contract_size_usd
        if actual_risk > risk_amount * 2.0:
            continue

        basket = {"type": decision, "pattern": pattern_info["pattern"], "level": level,
                  "entry": entry_price, "vol": vol, "sl": sl, "tp": tp, "open_time": current_time}

    return trades


def print_classical_report(trades, symbol, timeframe):
    print("=" * 65)
    print(f"   CLASSICAL CHART PATTERNS BACKTEST REPORT: {symbol} ({timeframe})")
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
        print(f"    {name:<28} {len(p_trades):>4} trades, {p_wins}/{len(p_trades)} wins, ${p_profit:+.2f}")
    print("-" * 65)
    print("  SIMULATED ACCOUNT METRICS (Starting $500, Risking 3%):")
    print(f"  Final Account Balance:  ${balance:.2f} USD")
    print(f"  Net ROI:                {((balance - 500.0) / 500.0 * 100):+.1f}%")
    print(f"  Max Drawdown:           {max_drawdown:.1f}%")
    print("=" * 65)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest the Classical Chart Patterns strategy on historical MT5 data.")
    parser.add_argument("--symbol", type=str, default="EURUSDm", help="Trading symbol, or comma-separated list e.g. EURUSDm,GBPUSDm")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe (e.g. H1, M30)")
    parser.add_argument("--years", type=int, default=1, help="Years of data to backtest")
    parser.add_argument("--pole-lookback", type=int, default=20, help="Bars used to evaluate the pole preceding a continuation pattern")
    parser.add_argument("--pole-min-move-pct", type=float, default=0.5, help="Minimum net %% move required for a valid pole")
    parser.add_argument("--pole-min-efficiency", type=float, default=0.4, help="Minimum net-move/total-move ratio required for a clean pole")
    parser.add_argument("--consolidation-max-bars", type=int, default=40, help="Bars in the consolidation window following the pole")
    parser.add_argument("--swing-window", type=int, default=3, help="Swing-pivot detection window (candles on each side)")
    parser.add_argument("--flat-slope-threshold-pct", type=float, default=0.02, help="Slope magnitude (%%/bar) below which a trendline counts as flat")
    parser.add_argument("--shoulder-tolerance-pct", type=float, default=3.0, help="Max %% difference between the two shoulders/peaks/troughs")
    parser.add_argument("--neckline-tolerance-pct", type=float, default=2.0, help="Max %% difference between the two neckline-forming points")
    parser.add_argument("--patterns", type=str, default="", help="Comma-separated reversal pattern names to check (default: all four)")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period (SL noise buffer)")
    parser.add_argument("--atr-sl-multiplier", type=float, default=1.0, help="SL distance beyond the far level, as a multiple of ATR")
    parser.add_argument("--tp-multiplier", type=float, default=1.0, help="Measured-move TP multiplier (pole height or head-to-neckline distance)")
    parser.add_argument("--min-rr", type=float, default=1.0, help="Minimum R:R ratio required to take the trade")
    parser.add_argument("--min-tp-spread-multiple", type=float, default=2.0, help="Require TP distance to be at least this many multiples of the spread")

    args = parser.parse_args()

    if not connect_mt5():
        print("[FATAL] Could not connect to MT5.")
        sys.exit(1)

    try:
        symbols = [s.strip() for s in args.symbol.split(",") if s.strip()]
        reversal_patterns = [p.strip() for p in args.patterns.split(",") if p.strip()] or None
        results = {}

        for sym in symbols:
            df = fetch_backtest_data(sym, args.timeframe, args.years)
            if df is None or len(df) == 0:
                print(f"[WARNING] Skipping symbol {sym} due to lack of historical data.")
                continue

            trades = run_backtest_classical_patterns(
                df, sym,
                pole_lookback=args.pole_lookback,
                pole_min_move_pct=args.pole_min_move_pct,
                pole_min_efficiency=args.pole_min_efficiency,
                consolidation_max_bars=args.consolidation_max_bars,
                swing_window=args.swing_window,
                flat_slope_threshold_pct=args.flat_slope_threshold_pct,
                shoulder_tolerance_pct=args.shoulder_tolerance_pct,
                neckline_tolerance_pct=args.neckline_tolerance_pct,
                patterns=reversal_patterns,
                atr_period=args.atr_period,
                atr_sl_multiplier=args.atr_sl_multiplier,
                tp_multiplier=args.tp_multiplier,
                min_rr_ratio=args.min_rr,
                min_tp_spread_multiple=args.min_tp_spread_multiple
            )
            results[sym] = trades
            print_classical_report(trades, sym, args.timeframe)

        if len(results) > 1:
            print("\n" + "=" * 95)
            print("                 COMPARATIVE CLASSICAL CHART PATTERNS BACKTEST SUMMARY")
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
