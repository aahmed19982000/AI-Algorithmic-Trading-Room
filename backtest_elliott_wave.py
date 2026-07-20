"""
Elliott Wave (Wave 2 -> Wave 3) Backtester
==========================================
Downloads historical candles for a symbol/timeframe and replays the same
Elliott Wave logic used live in trading_bot.py (manage_elliott_wave_strategy /
gann_helper.detect_elliott_wave2_setup), bar by bar, then reports win rate,
profit factor, ROI, and max drawdown.

Pivots are precomputed once (matching backtest_gann.py's approach) rather
than calling detect_elliott_wave2_setup() per bar, since that function
rescans the whole lookback window from scratch on every call — fine for a
single live check per cycle, too slow for a multi-thousand-bar backtest loop.

Reuses fetch_backtest_data() and get_currency_conversion_rate() from
backtest_gann.py, and calculate_fibonacci_extension() / _check_wave1_impulse_quality()
from gann_helper.py, so the backtest's math matches the live strategy exactly.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import MetaTrader5 as mt5
from dotenv import load_dotenv

from mt5_connection import connect_mt5, disconnect_mt5
from backtest_gann import fetch_backtest_data, get_currency_conversion_rate
from gann_helper import calculate_fibonacci_extension, _check_wave1_impulse_quality

load_dotenv()


def run_backtest_elliott_wave(df, symbol, lookback=100, min_retracement=0.382, max_retracement=0.786,
                               max_wave1_internal_retracement=0.618, extension_ratio=1.618, min_rr_ratio=1.0):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[ERROR] Symbol {symbol} not found in MT5.")
        return []

    contract_size = symbol_info.trade_contract_size
    currency_conversion_rate = get_currency_conversion_rate(symbol_info)
    contract_size_usd = contract_size * currency_conversion_rate
    if currency_conversion_rate != 1.0:
        print(f"[INFO] {symbol}: profit currency is {symbol_info.currency_profit}, applying conversion rate {currency_conversion_rate:.5f}.")

    # 1. Precompute all swing pivots once (window=5, same as gann_helper.py)
    window = 5
    pivots_high = []
    pivots_low = []
    for i in range(window, len(df) - window):
        is_high = all(df.loc[j, 'High'] <= df.loc[i, 'High'] for j in range(i - window, i + window + 1))
        if is_high:
            pivots_high.append((i, df.loc[i, 'High']))
        is_low = all(df.loc[j, 'Low'] >= df.loc[i, 'Low'] for j in range(i - window, i + window + 1))
        if is_low:
            pivots_low.append((i, df.loc[i, 'Low']))

    sim_balance = 500.0
    risk_percent = 3.0

    basket = None  # {"type":, "entry":, "vol":, "sl":, "tp":, "wave0":, "wave1":, "open_time":}
    trades = []

    for i in range(window, len(df)):
        if basket is None and sim_balance <= 0:
            break  # account blown — see backtest_sma5_reversion.py for why this guard exists

        current_time = df.loc[i, 'time']
        close_curr = df.loc[i, 'Close']
        high_curr = df.loc[i, 'High']
        low_curr = df.loc[i, 'Low']

        # --- Manage an active position ---
        if basket is not None:
            # Elliott invalidation: Wave 4 must never overlap Wave 1 (point B)'s
            # territory. This only becomes a meaningful check AFTER price has
            # actually confirmed Wave 3 by moving beyond Wave 1 at least once
            # since entry — entry happens right as price crosses Wave 2 (C),
            # which sits between A and B by construction, so checking overlap
            # from the very first bar would flag almost every fresh trade
            # immediately (Wave 3 hasn't started yet, that's not an overlap).
            if basket["type"] == "BUY" and high_curr > basket["wave1"]:
                basket["wave3_confirmed"] = True
            elif basket["type"] == "SELL" and low_curr < basket["wave1"]:
                basket["wave3_confirmed"] = True

            invalidated = basket.get("wave3_confirmed", False) and (
                (basket["type"] == "BUY" and low_curr < basket["wave1"]) or
                (basket["type"] == "SELL" and high_curr > basket["wave1"])
            )

            hit_sl = (basket["type"] == "BUY" and low_curr <= basket["sl"]) or \
                     (basket["type"] == "SELL" and high_curr >= basket["sl"])
            hit_tp = (basket["type"] == "BUY" and high_curr >= basket["tp"]) or \
                     (basket["type"] == "SELL" and low_curr <= basket["tp"])

            if hit_sl or invalidated:
                exit_price = basket["sl"] if hit_sl else close_curr
                profit_usd = ((exit_price - basket["entry"]) if basket["type"] == "BUY" else (basket["entry"] - exit_price)) * basket["vol"] * contract_size_usd
                trades.append({"type": basket["type"], "entry_time": basket["open_time"], "exit_time": current_time,
                               "volume": basket["vol"], "profit_usd_raw": profit_usd,
                               "result": "WIN" if profit_usd >= 0 else "LOSS"})
                sim_balance += profit_usd
                basket = None
            elif hit_tp:
                profit_usd = ((basket["tp"] - basket["entry"]) if basket["type"] == "BUY" else (basket["entry"] - basket["tp"])) * basket["vol"] * contract_size_usd
                trades.append({"type": basket["type"], "entry_time": basket["open_time"], "exit_time": current_time,
                               "volume": basket["vol"], "profit_usd_raw": profit_usd, "result": "WIN"})
                sim_balance += profit_usd
                basket = None
            continue

        # --- No active position: scan for a fresh Wave 0-1-2 setup ---
        # A pivot at index p needs `window` bars AFTER it to be confirmed (see the
        # is_high/is_low scan above), so it isn't actually knowable until bar p+window.
        # Filtering by p[0] < i alone would let a "fresh" pivot near i be confirmed
        # using bars that don't exist yet at bar i in a real-time replay — lookahead bias.
        lookback_start = max(0, i - lookback)
        window_highs = [p for p in pivots_high if lookback_start <= p[0] and p[0] + window <= i]
        window_lows = [p for p in pivots_low if lookback_start <= p[0] and p[0] + window <= i]

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
                    if min_retracement <= ratio <= max_retracement:
                        if _check_wave1_impulse_quality(window_highs, window_lows, idx_A, val_A, idx_B, val_B,
                                                         max_wave1_internal_retracement, is_buy=True):
                            setup = {"type": "BUY", "A": val_A, "B": val_B, "C": val_C, "idx_C": idx_C}
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
                    if min_retracement <= ratio <= max_retracement:
                        if _check_wave1_impulse_quality(window_highs, window_lows, idx_A, val_A, idx_B, val_B,
                                                         max_wave1_internal_retracement, is_buy=False):
                            setup = {"type": "SELL", "A": val_A, "B": val_B, "C": val_C, "idx_C": idx_C}
                            break

        if setup is None:
            continue

        # Freshness: Wave 2 (C) must have completed recently
        if i - setup["idx_C"] > 5:
            continue

        decision = setup["type"]
        wave0, wave1, wave2 = setup["A"], setup["B"], setup["C"]

        # Entry confirmation: current close already reversing away from C in the Wave 1 direction
        if decision == "BUY" and close_curr <= wave2:
            continue
        if decision == "SELL" and close_curr >= wave2:
            continue

        entry_price = close_curr
        sl = wave0
        tp = calculate_fibonacci_extension(wave0, wave1, extension_ratio)

        sl_dist = abs(entry_price - sl)
        tp_dist = abs(tp - entry_price)
        if sl_dist <= 0:
            continue
        rr_ratio = tp_dist / sl_dist
        if rr_ratio < min_rr_ratio:
            continue

        risk_amount = sim_balance * (risk_percent / 100.0)
        vol = risk_amount / (sl_dist * contract_size_usd)
        vol_step = symbol_info.volume_step if symbol_info.volume_step > 0 else 0.01
        vol = round(vol / vol_step) * vol_step
        vol = max(symbol_info.volume_min, min(vol, symbol_info.volume_max))
        vol = round(vol, 2)

        basket = {"type": decision, "entry": entry_price, "vol": vol, "sl": sl, "tp": tp,
                  "wave0": wave0, "wave1": wave1, "open_time": current_time}

    return trades


def print_elliott_report(trades, symbol, timeframe):
    print("=" * 65)
    print(f"   ELLIOTT WAVE (2->3) BACKTEST REPORT: {symbol} ({timeframe})")
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
    print("  SIMULATED ACCOUNT METRICS (Starting $500, Risking 3%):")
    print(f"  Final Account Balance:  ${balance:.2f} USD")
    print(f"  Net ROI:                {((balance - 500.0) / 500.0 * 100):+.1f}%")
    print(f"  Max Drawdown:           {max_drawdown:.1f}%")
    print("=" * 65)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest the Elliott Wave (Wave 2->3) strategy on historical MT5 data.")
    parser.add_argument("--symbol", type=str, default="EURUSDm", help="Trading symbol, or comma-separated list e.g. EURUSDm,GBPUSDm")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe (e.g. H1, M30)")
    parser.add_argument("--years", type=int, default=1, help="Years of data to backtest")
    parser.add_argument("--lookback", type=int, default=100, help="Pivot-scan lookback window")
    parser.add_argument("--min-retracement", type=float, default=0.382, help="Wave 2 minimum retracement of Wave 1")
    parser.add_argument("--max-retracement", type=float, default=0.786, help="Wave 2 maximum retracement of Wave 1")
    parser.add_argument("--max-wave1-internal-retracement", type=float, default=0.618, help="Wave 1 impulse-quality threshold")
    parser.add_argument("--extension-ratio", type=float, default=1.618, help="Wave 3 Fibonacci extension multiple")
    parser.add_argument("--min-rr", type=float, default=1.0, help="Minimum R:R ratio required to take the trade")

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

            trades = run_backtest_elliott_wave(
                df, sym,
                lookback=args.lookback,
                min_retracement=args.min_retracement,
                max_retracement=args.max_retracement,
                max_wave1_internal_retracement=args.max_wave1_internal_retracement,
                extension_ratio=args.extension_ratio,
                min_rr_ratio=args.min_rr
            )
            results[sym] = trades
            print_elliott_report(trades, sym, args.timeframe)

        if len(results) > 1:
            print("\n" + "=" * 95)
            print("                    COMPARATIVE ELLIOTT WAVE BACKTEST SUMMARY")
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
