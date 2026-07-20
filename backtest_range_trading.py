"""
Range Trading Backtester
=========================
Downloads historical candles for a symbol/timeframe and replays the same
Range Trading logic used live in trading_bot.py (manage_range_trading_strategy /
check_range_trading_signal / check_range_trading_invalidation), bar by bar,
then reports win rate, profit factor, ROI, and max drawdown.

Reuses fetch_backtest_data()/get_currency_conversion_rate() from
backtest_gann.py and detect_range_zone()/calculate_adx() from gann_helper.py,
so the range-detection and trend-regime math matches the live strategy
exactly. The entry-decision and invalidation logic itself is replicated
inline (matching backtest_sma5_reversion.py's approach) rather than imported
from trading_bot.py, to keep this script's dependencies limited to
MT5/pandas/gann_helper instead of pulling in trading_bot.py's full
dependency chain (ai_engine, news_filter, etc.) just to run a backtest.

detect_range_zone() is called on a bounded trailing slice of the dataframe
ending at each bar (matching the live strategy's fetch size, lookback+100
candles) rather than the full growing history — pivot detection needs a
bound to avoid both O(n^2) cost and the lookahead bias fixed earlier in
backtest_gann.py/backtest_elliott_wave.py (a pivot near the end of a
precomputed-on-everything scan could be "confirmed" using future bars).
.reset_index(drop=True) is required after slicing since the pivot detector
uses positional .loc[j, ...] indexing.

ADX, by contrast, is precomputed ONCE on the full dataframe up front (like
RSI/ATR below) rather than recomputed per-bar on a bounded window: Wilder's
smoothing recursion is strictly causal (each value only depends on earlier
values), so there's no lookahead risk, and re-seeding it from scratch on a
truncated window every bar would actually be wrong — it would produce a
different, artificially choppy series than a continuously-smoothed
indicator, not just a slower one.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import MetaTrader5 as mt5
from dotenv import load_dotenv

from mt5_connection import connect_mt5, disconnect_mt5
from backtest_gann import fetch_backtest_data, get_currency_conversion_rate
from gann_helper import detect_range_zone, calculate_adx

load_dotenv()


def _compute_atr(df, period=14):
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def run_backtest_range_trading(df, symbol, lookback=100, swing_window=5, peak_tolerance_pct=0.15,
                                trough_tolerance_pct=0.15, adx_period=14, adx_threshold=25.0,
                                adx_exit_threshold=30.0, entry_zone_pct=0.15, min_range_pct=0.3,
                                max_range_pct=3.0, atr_period=14, atr_sl_multiplier=1.0,
                                tp_buffer_pct=10.0, breakout_tp_multiplier=1.0,
                                rsi_overbought=65.0, rsi_oversold=35.0, min_rr_ratio=1.0,
                                min_tp_spread_multiple=2.0):
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
    df['adx'] = calculate_adx(df, period=adx_period)

    sim_balance = 500.0
    risk_percent = 3.0
    window_span = lookback + 100  # matches the live strategy's get_candles(count=lookback+100) fetch size
    min_bars_needed = max(2 * adx_period + 5, atr_period + 5, 20)

    basket = None  # {"type":, "mode":, "entry":, "vol":, "sl":, "tp":, "range_top":, "range_bottom":, "open_time":}
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

            # Invalidation checks — mirrors check_range_trading_invalidation():
            # fade-mode ADX regime-break, breakout-mode failed-breakout (close
            # back inside the range). Evaluated once per bar here, which is
            # already the per-completed-candle cadence the live cache idiom
            # exists to approximate.
            invalidated = False
            if not hit_sl and not hit_tp:
                if basket["mode"] == "fade":
                    adx_now = df['adx'].iloc[i]
                    if pd.notna(adx_now) and adx_now > adx_exit_threshold:
                        invalidated = True
                elif basket["mode"] == "breakout":
                    if basket["range_bottom"] < close_curr < basket["range_top"]:
                        invalidated = True

            if hit_sl or invalidated:
                exit_price = basket["sl"] if hit_sl else close_curr
                profit_usd = ((exit_price - basket["entry"]) if basket["type"] == "BUY" else (basket["entry"] - exit_price)) * basket["vol"] * contract_size_usd
                trades.append({"type": basket["type"], "mode": basket["mode"], "entry_time": basket["open_time"], "exit_time": current_time,
                               "volume": basket["vol"], "profit_usd_raw": profit_usd, "result": "WIN" if profit_usd >= 0 else "LOSS"})
                sim_balance += profit_usd
                basket = None
            elif hit_tp:
                profit_usd = ((basket["tp"] - basket["entry"]) if basket["type"] == "BUY" else (basket["entry"] - basket["tp"])) * basket["vol"] * contract_size_usd
                trades.append({"type": basket["type"], "mode": basket["mode"], "entry_time": basket["open_time"], "exit_time": current_time,
                               "volume": basket["vol"], "profit_usd_raw": profit_usd, "result": "WIN"})
                sim_balance += profit_usd
                basket = None
            continue  # either still open, or just closed — don't open a new one on the same bar

        # --- No active position: scan for a signal ---
        # detect_range_zone needs a bounded trailing slice (not the full
        # growing history) both for speed and to avoid the same lookahead
        # bias fixed in backtest_gann.py: a pivot near the end of a
        # precomputed-on-everything scan could be "confirmed" using bars
        # that wouldn't exist yet in a real-time replay.
        slice_start = max(0, i + 1 - window_span)
        window_df = df.iloc[slice_start:i + 1].reset_index(drop=True)
        if len(window_df) < min_bars_needed:
            continue

        range_info = detect_range_zone(
            window_df, lookback=lookback, window=swing_window,
            peak_tolerance_pct=peak_tolerance_pct, trough_tolerance_pct=trough_tolerance_pct,
            min_range_pct=min_range_pct, max_range_pct=max_range_pct
        )
        if not range_info:
            continue

        range_top = range_info["range_top"]
        range_bottom = range_info["range_bottom"]
        range_width = range_top - range_bottom

        last_close = close_curr  # last completed bar in this replay IS bar i
        current_price = close_curr  # backtest stand-in for "live current price"

        decision = None
        mode = None

        # 1. Breakout mode — no ADX/RSI gating, a breakout is the market
        # starting to trend, so "not trending"/"exhausted" filters would be
        # self-contradictory here.
        if last_close > range_top:
            decision, mode = "BUY", "breakout"
        elif last_close < range_bottom:
            decision, mode = "SELL", "breakout"
        else:
            # 2. Fade mode — requires ADX < threshold (still ranging), price
            # within entry_zone_pct% of a boundary, RSI confirms exhaustion.
            adx_now = df['adx'].iloc[i]
            if pd.notna(adx_now) and adx_now < adx_threshold and range_width > 0:
                rsi14 = df['rsi14'].iloc[i]
                if pd.notna(rsi14):
                    dist_from_bottom_pct = abs(current_price - range_bottom) / range_bottom * 100.0
                    dist_from_top_pct = abs(current_price - range_top) / range_top * 100.0
                    if dist_from_bottom_pct <= entry_zone_pct and rsi14 <= rsi_oversold:
                        decision, mode = "BUY", "fade"
                    elif dist_from_top_pct <= entry_zone_pct and rsi14 >= rsi_overbought:
                        decision, mode = "SELL", "fade"

        if decision is None:
            continue

        entry_price = close_curr
        atr = df['atr'].iloc[i]
        if pd.isna(atr):
            atr = 0.0

        # SL unified by direction regardless of mode (deliberately wide for
        # a breakout — the far/opposite side of the range).
        if decision == "BUY":
            sl = range_bottom - atr * atr_sl_multiplier
        else:
            sl = range_top + atr * atr_sl_multiplier

        # TP is mode-dependent: fade targets the opposite boundary pulled in
        # by a fill-probability buffer; breakout targets a measured-move
        # projection of the range width.
        if mode == "fade":
            tp = (range_top - range_width * (tp_buffer_pct / 100.0)) if decision == "BUY" else (range_bottom + range_width * (tp_buffer_pct / 100.0))
        else:
            tp = (range_top + range_width * breakout_tp_multiplier) if decision == "BUY" else (range_bottom - range_width * breakout_tp_multiplier)

        # TP must actually sit on the profitable side of entry — a breakout
        # entry can occur well beyond range_top/range_bottom on a strong move,
        # so a measured-move TP computed purely from the boundary can land
        # behind entry (abs(tp-entry) would still look like a positive R:R).
        if decision == "BUY" and tp <= entry_price:
            continue
        if decision == "SELL" and tp >= entry_price:
            continue

        sl_dist = abs(entry_price - sl)
        tp_dist = abs(tp - entry_price)
        if sl_dist <= 0:
            continue

        # Spread guard — same rationale as the live strategy: skip targets too
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

        basket = {"type": decision, "mode": mode, "entry": entry_price, "vol": vol, "sl": sl, "tp": tp,
                  "range_top": range_top, "range_bottom": range_bottom, "open_time": current_time}

    return trades


def print_range_trading_report(trades, symbol, timeframe):
    print("=" * 65)
    print(f"   RANGE TRADING BACKTEST REPORT: {symbol} ({timeframe})")
    print("=" * 65)

    total_trades = len(trades)
    if total_trades == 0:
        print("  No trades were triggered for this symbol/period.")
        print("=" * 65)
        return

    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = sum(1 for t in trades if t["result"] == "LOSS")
    win_rate = (wins / total_trades) * 100.0

    breakout_trades = sum(1 for t in trades if t["mode"] == "breakout")
    fade_trades = sum(1 for t in trades if t["mode"] == "fade")

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

    print(f"  Total Trades:           {total_trades} ({breakout_trades} breakout, {fade_trades} fade)")
    print(f"  Wins:                   {wins} (Win Rate: {win_rate:.1f}%)")
    print(f"  Losses:                 {losses}")
    print(f"  Profit Factor:          {profit_factor:.2f}")
    print("-" * 65)
    print("  SIMULATED ACCOUNT METRICS (Starting $500, Risking 3%):")
    print(f"  Final Account Balance:  ${balance:.2f} USD")
    print(f"  Net ROI:                {((balance - 500.0) / 500.0 * 100):+.1f}%")
    print(f"  Max Drawdown:           {max_drawdown:.1f}%")
    print("=" * 65)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest the Range Trading strategy on historical MT5 data.")
    parser.add_argument("--symbol", type=str, default="EURUSDm", help="Trading symbol, or comma-separated list e.g. EURUSDm,GBPUSDm")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe (e.g. H1, M30)")
    parser.add_argument("--years", type=int, default=1, help="Years of data to backtest")
    parser.add_argument("--lookback", type=int, default=100, help="Pivot-scan lookback window for range detection")
    parser.add_argument("--swing-window", type=int, default=5, help="Swing-pivot detection window (candles on each side)")
    parser.add_argument("--peak-tolerance-pct", type=float, default=0.15, help="Max %% difference between the two most recent swing highs")
    parser.add_argument("--trough-tolerance-pct", type=float, default=0.15, help="Max %% difference between the two most recent swing lows")
    parser.add_argument("--adx-period", type=int, default=14, help="ADX smoothing period")
    parser.add_argument("--adx-threshold", type=float, default=25.0, help="Max ADX to still consider the market ranging (fade entry gate)")
    parser.add_argument("--adx-exit-threshold", type=float, default=30.0, help="ADX level that invalidates an open fade position")
    parser.add_argument("--entry-zone-pct", type=float, default=0.15, help="%% of price from a boundary required to trigger a fade")
    parser.add_argument("--min-range-pct", type=float, default=0.3, help="Minimum range width as %% of price")
    parser.add_argument("--max-range-pct", type=float, default=3.0, help="Maximum range width as %% of price")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period (SL noise buffer)")
    parser.add_argument("--atr-sl-multiplier", type=float, default=1.0, help="SL distance beyond the far boundary, as a multiple of ATR")
    parser.add_argument("--tp-buffer-pct", type=float, default=10.0, help="Fade TP pull-in from the opposite boundary, as %% of range width")
    parser.add_argument("--breakout-tp-multiplier", type=float, default=1.0, help="Breakout TP measured-move projection, as a multiple of range width")
    parser.add_argument("--rsi-overbought", type=float, default=65.0, help="RSI14 level required to confirm a fade SELL")
    parser.add_argument("--rsi-oversold", type=float, default=35.0, help="RSI14 level required to confirm a fade BUY")
    parser.add_argument("--min-rr", type=float, default=1.0, help="Minimum R:R ratio required to take the trade")
    parser.add_argument("--min-tp-spread-multiple", type=float, default=2.0, help="Require TP distance to be at least this many multiples of the spread")

    args = parser.parse_args()

    if not connect_mt5():
        print("[FATAL] Could not connect to MT5.")
        sys.exit(1)

    try:
        symbols = [s.strip() for s in args.symbol.split(",") if s.strip()]
        results = {}

        for sym in symbols:
            df = fetch_backtest_data(sym, args.timeframe, args.years)
            if df is None or len(df) == 0:
                print(f"[WARNING] Skipping symbol {sym} due to lack of historical data.")
                continue

            trades = run_backtest_range_trading(
                df, sym,
                lookback=args.lookback,
                swing_window=args.swing_window,
                peak_tolerance_pct=args.peak_tolerance_pct,
                trough_tolerance_pct=args.trough_tolerance_pct,
                adx_period=args.adx_period,
                adx_threshold=args.adx_threshold,
                adx_exit_threshold=args.adx_exit_threshold,
                entry_zone_pct=args.entry_zone_pct,
                min_range_pct=args.min_range_pct,
                max_range_pct=args.max_range_pct,
                atr_period=args.atr_period,
                atr_sl_multiplier=args.atr_sl_multiplier,
                tp_buffer_pct=args.tp_buffer_pct,
                breakout_tp_multiplier=args.breakout_tp_multiplier,
                rsi_overbought=args.rsi_overbought,
                rsi_oversold=args.rsi_oversold,
                min_rr_ratio=args.min_rr,
                min_tp_spread_multiple=args.min_tp_spread_multiple
            )
            results[sym] = trades
            print_range_trading_report(trades, sym, args.timeframe)

        if len(results) > 1:
            print("\n" + "=" * 95)
            print("                    COMPARATIVE RANGE TRADING BACKTEST SUMMARY")
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
