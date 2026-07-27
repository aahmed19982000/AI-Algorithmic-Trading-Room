"""
SMA(5) Mean-Reversion Backtester
================================
Downloads historical candles for a symbol/timeframe and replays the same
SMA(5) mean-reversion logic used live in trading_bot.py
(manage_sma5_reversion_strategy / check_sma5_reversion_signal), bar by bar,
then reports win rate, profit factor, ROI, and max drawdown.

Reuses fetch_backtest_data() and get_currency_conversion_rate() from
backtest_gann.py so JPY/cross-pair profit figures are correctly converted
to the account's deposit currency (the same bug fixed there earlier).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import MetaTrader5 as mt5
from dotenv import load_dotenv

from mt5_connection import connect_mt5, disconnect_mt5
from backtest_gann import fetch_backtest_data, get_currency_conversion_rate
from gann_helper import calculate_adx

load_dotenv()


def run_backtest_sma5_reversion(df, symbol, threshold_pct=0.3, sl_multiplier=1.5, min_rr_ratio=0.5,
                                 rsi_overbought=65.0, rsi_oversold=35.0, min_tp_spread_multiple=2.0,
                                 adx_period=14, adx_threshold=25.0, momentum_tp_multiplier=2.0,
                                 momentum_enabled=True):
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

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df = df.copy()
    df['rsi14'] = 100 - (100 / (1 + gain / loss))
    df['adx'] = calculate_adx(df, period=adx_period)

    sim_balance = 500.0
    risk_percent = 3.0

    basket = None  # {"type": "BUY"/"SELL", "mode": "reversion"/"momentum", "entry":, "vol":, "sl":, "tp":, "open_time":}
    trades = []

    for i in range(20, len(df)):  # 20 = enough warmup for a non-NaN RSI(14)
        # No margin modeling here — a real broker would reject an order the account
        # can't support long before this point. Once simulated balance is wiped out,
        # stop trading this symbol rather than let it spiral to an impossible negative
        # ROI (this bit us on XAUUSDm/XAGUSDm: a clamped-to-volume_max lot size against
        # a $500 sim balance produced a single loss large enough to go deeply negative,
        # and the loop kept sizing new "risk 3% of balance" trades off that negative number).
        if basket is None and sim_balance <= 0:
            break

        current_time = df.loc[i, 'time']
        close_curr = df.loc[i, 'Close']
        high_curr = df.loc[i, 'High']
        low_curr = df.loc[i, 'Low']

        # --- Manage an active position ---
        if basket is not None:
            # Momentum-mode invalidation: if ADX (as of the last completed candle)
            # has fallen back below the entry threshold, the trend that justified
            # trading with the deviation has exhausted — close now rather than
            # wait for the far measured-move TP or the SMA5-anchored SL. Mirrors
            # check_sma5_momentum_invalidation() in trading_bot.py.
            if basket.get("mode") == "momentum":
                adx_mgmt = df['adx'].iloc[i - 1]
                if pd.notna(adx_mgmt) and adx_mgmt < adx_threshold:
                    if basket["type"] == "BUY":
                        profit_usd = (close_curr - basket["entry"]) * basket["vol"] * contract_size_usd
                    else:
                        profit_usd = (basket["entry"] - close_curr) * basket["vol"] * contract_size_usd
                    trades.append({"type": basket["type"], "entry_time": basket["open_time"], "exit_time": current_time,
                                   "volume": basket["vol"], "profit_usd_raw": profit_usd,
                                   "result": "WIN" if profit_usd > 0 else "LOSS"})
                    sim_balance += profit_usd
                    basket = None
                    continue

            if basket["type"] == "BUY":
                if low_curr <= basket["sl"]:
                    profit_usd = (basket["sl"] - basket["entry"]) * basket["vol"] * contract_size_usd
                    trades.append({"type": "BUY", "entry_time": basket["open_time"], "exit_time": current_time,
                                   "volume": basket["vol"], "profit_usd_raw": profit_usd, "result": "LOSS"})
                    sim_balance += profit_usd
                    basket = None
                elif high_curr >= basket["tp"]:
                    profit_usd = (basket["tp"] - basket["entry"]) * basket["vol"] * contract_size_usd
                    trades.append({"type": "BUY", "entry_time": basket["open_time"], "exit_time": current_time,
                                   "volume": basket["vol"], "profit_usd_raw": profit_usd, "result": "WIN"})
                    sim_balance += profit_usd
                    basket = None
            else:  # SELL
                if high_curr >= basket["sl"]:
                    profit_usd = (basket["entry"] - basket["sl"]) * basket["vol"] * contract_size_usd
                    trades.append({"type": "SELL", "entry_time": basket["open_time"], "exit_time": current_time,
                                   "volume": basket["vol"], "profit_usd_raw": profit_usd, "result": "LOSS"})
                    sim_balance += profit_usd
                    basket = None
                elif low_curr <= basket["tp"]:
                    profit_usd = (basket["entry"] - basket["tp"]) * basket["vol"] * contract_size_usd
                    trades.append({"type": "SELL", "entry_time": basket["open_time"], "exit_time": current_time,
                                   "volume": basket["vol"], "profit_usd_raw": profit_usd, "result": "WIN"})
                    sim_balance += profit_usd
                    basket = None
            continue  # either still open, or just closed — don't open a new one on the same bar

        # --- No active position: check for a new signal ---
        # Mirrors check_sma5_reversion_signal(): SMA5 of the 5 bars strictly before
        # the current one, current bar's close stands in for "live current price".
        sma5 = df['Close'].iloc[i - 5:i].mean()
        current_price = close_curr
        deviation_pct = (current_price - sma5) / sma5 * 100.0
        rsi14 = df['rsi14'].iloc[i - 1]  # last completed bar, matches check_sma5_reversion_signal's iloc[-2]
        adx_now = df['adx'].iloc[i - 1]  # same completed-bar convention as RSI

        if pd.isna(rsi14):
            continue

        # ADX-regime gate, mirrors check_sma5_reversion_signal(): ranging (ADX <
        # threshold, or NaN during warmup) keeps the original fade behavior;
        # trending (ADX >= threshold) flips to trading with the deviation instead
        # of fading it. momentum_enabled=False forces reversion-only regardless of
        # ADX — mirrors trading_bot.py gating momentum to sma5_reversion_momentum_symbols.
        trending = momentum_enabled and pd.notna(adx_now) and adx_now >= adx_threshold
        mode = "momentum" if trending else "reversion"

        if deviation_pct >= threshold_pct:
            if rsi14 < rsi_overbought:
                continue  # momentum not exhausted yet
            decision = "BUY" if trending else "SELL"
        elif deviation_pct <= -threshold_pct:
            if rsi14 > rsi_oversold:
                continue
            decision = "SELL" if trending else "BUY"
        else:
            continue

        entry_price = current_price
        deviation_distance = abs(entry_price - sma5)

        if mode == "momentum":
            # SL anchored at SMA5 itself (the invalidation level), TP a measured-move
            # projection away from it. Mirrors manage_sma5_reversion_strategy()'s
            # momentum branch in trading_bot.py.
            sl = sma5 if decision == "BUY" else (sma5 + spread)
            tp = (entry_price + momentum_tp_multiplier * deviation_distance if decision == "BUY"
                  else entry_price - momentum_tp_multiplier * deviation_distance)
        else:
            sl_distance = sl_multiplier * deviation_distance
            sl = entry_price - sl_distance if decision == "BUY" else entry_price + sl_distance
            tp = sma5  # unchanged from original reversion-only backtest (no spread adjustment here)

        sl_dist = abs(entry_price - sl)
        tp_dist = abs(tp - entry_price)
        if sl_dist <= 0:
            continue

        # Spread guard — same rationale as the live strategy: skip targets too small
        # relative to the (approximated) spread to be realistically closable.
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

        basket = {"type": decision, "mode": mode, "entry": entry_price, "vol": vol, "sl": sl, "tp": tp, "open_time": current_time}

    return trades


def print_sma5_report(trades, symbol, timeframe):
    print("=" * 65)
    print(f"   SMA(5) MEAN-REVERSION BACKTEST REPORT: {symbol} ({timeframe})")
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
    parser = argparse.ArgumentParser(description="Backtest the SMA(5) mean-reversion strategy on historical MT5 data.")
    parser.add_argument("--symbol", type=str, default="EURUSDm", help="Trading symbol, or comma-separated list e.g. EURUSDm,GBPUSDm")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe (e.g. H1, M30)")
    parser.add_argument("--years", type=int, default=1, help="Years of data to backtest")
    parser.add_argument("--threshold", type=float, default=0.3, help="Deviation %% from SMA5 required to trigger a signal")
    parser.add_argument("--sl-multiplier", type=float, default=1.5, help="SL distance as a multiple of the entry deviation distance")
    parser.add_argument("--min-rr", type=float, default=0.5, help="Minimum R:R ratio required to take the trade")
    parser.add_argument("--rsi-overbought", type=float, default=65.0, help="RSI14 level required to confirm a SELL")
    parser.add_argument("--rsi-oversold", type=float, default=35.0, help="RSI14 level required to confirm a BUY")
    parser.add_argument("--min-tp-spread-multiple", type=float, default=2.0, help="Require TP distance to be at least this many multiples of the spread")
    parser.add_argument("--adx-period", type=int, default=14, help="ADX smoothing period")
    parser.add_argument("--adx-threshold", type=float, default=45.0, help="ADX level at/above which the strategy switches from fading the deviation (reversion) to trading with it (momentum) — 45 validated better than 25 in backtesting; a low threshold misclassifies too many normal reversion setups as momentum")
    parser.add_argument("--momentum-tp-multiplier", type=float, default=2.0, help="Momentum-mode TP distance as a multiple of the entry deviation-from-SMA5 distance")
    parser.add_argument("--momentum-symbols", type=str, default="BTCUSDm,ETHUSDm,SOLUSDm,XRPUSDm",
                         help="Comma-separated symbols allowed to use momentum mode (matches sma5_reversion_momentum_symbols live default); every other symbol stays reversion-only regardless of ADX. Pass an empty string to disable momentum entirely.")

    args = parser.parse_args()
    momentum_symbols = {s.strip() for s in args.momentum_symbols.split(",") if s.strip()}

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

            trades = run_backtest_sma5_reversion(
                df, sym,
                threshold_pct=args.threshold,
                sl_multiplier=args.sl_multiplier,
                min_rr_ratio=args.min_rr,
                rsi_overbought=args.rsi_overbought,
                rsi_oversold=args.rsi_oversold,
                min_tp_spread_multiple=args.min_tp_spread_multiple,
                adx_period=args.adx_period,
                adx_threshold=args.adx_threshold,
                momentum_tp_multiplier=args.momentum_tp_multiplier,
                momentum_enabled=sym in momentum_symbols
            )
            results[sym] = trades
            print_sma5_report(trades, sym, args.timeframe)

        if len(results) > 1:
            print("\n" + "=" * 95)
            print("                  COMPARATIVE SMA(5) REVERSION BACKTEST SUMMARY")
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
