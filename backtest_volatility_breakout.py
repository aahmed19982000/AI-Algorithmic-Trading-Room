"""
Volatility Breakout (ATR + Bollinger Squeeze) Backtester
==========================================================
Downloads historical candles for a symbol/timeframe and replays the same
volatility breakout logic used live in trading_bot.py
(manage_volatility_breakout_strategy / check_volatility_breakout_signal), bar
by bar, then reports win rate, profit factor, ROI, and max drawdown.

Reuses fetch_backtest_data() and get_currency_conversion_rate() from
backtest_gann.py so JPY/cross-pair profit figures are correctly converted
to the account's deposit currency.

Limitation: this backtest does NOT replay the live strategy's higher-timeframe
(default D1) trend confirmation filter — doing so correctly requires fetching
a second timeframe per symbol and reindexing it (forward-filled) onto the
trading timeframe, which is out of scope here. Results from this script are
therefore an upper bound on trade frequency versus the live strategy with
volatility_breakout_mtf_enabled=1; expect fewer, more selective live signals.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import MetaTrader5 as mt5
from dotenv import load_dotenv

from mt5_connection import connect_mt5, disconnect_mt5
from backtest_gann import fetch_backtest_data, get_currency_conversion_rate

load_dotenv()


def run_backtest_volatility_breakout(df, symbol, bb_period=20, bb_stddev=2.0, squeeze_lookback=50,
                                      squeeze_percentile=20.0, atr_period=14, atr_sl_multiplier=1.5,
                                      rr_ratio_target=1.5, min_rr_ratio=1.2, min_tp_spread_multiple=3.0):
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
    df['bb_mid'] = df['Close'].rolling(window=bb_period).mean()
    bb_std = df['Close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_mid'] + bb_stddev * bb_std
    df['bb_lower'] = df['bb_mid'] - bb_stddev * bb_std
    df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=atr_period).mean()

    sim_balance = 500.0
    risk_percent = 3.0

    basket = None  # {"type": "BUY"/"SELL", "entry":, "vol":, "sl":, "tp":, "open_time":}
    trades = []

    warmup = bb_period + squeeze_lookback + atr_period + 5

    for i in range(warmup, len(df)):
        # No margin modeling here — same rationale as backtest_sma5_reversion.py:
        # once the simulated balance is wiped out, stop trading this symbol.
        if basket is None and sim_balance <= 0:
            break

        current_time = df.loc[i, 'time']
        close_curr = df.loc[i, 'Close']
        high_curr = df.loc[i, 'High']
        low_curr = df.loc[i, 'Low']

        # --- Manage an active position ---
        if basket is not None:
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
        # Mirrors check_volatility_breakout_signal(): the "breakout candle" is the
        # last completed bar (i-1), the "squeeze candle" is the one before it (i-2),
        # and bar i's close stands in for the live current price (same convention
        # as backtest_sma5_reversion.py).
        breakout_close = df['Close'].iloc[i - 1]
        upper = df['bb_upper'].iloc[i - 1]
        lower = df['bb_lower'].iloc[i - 1]
        atr_value = df['atr'].iloc[i - 1]

        squeeze_bandwidth = df['bb_bandwidth'].iloc[i - 2]
        bandwidth_window = df['bb_bandwidth'].iloc[i - 2 - squeeze_lookback:i - 2]

        if bandwidth_window.isna().any() or pd.isna(squeeze_bandwidth) or pd.isna(atr_value):
            continue

        squeeze_threshold = bandwidth_window.quantile(squeeze_percentile / 100.0)
        if squeeze_bandwidth > squeeze_threshold:
            continue  # no squeeze precedes this bar

        if breakout_close > upper:
            decision = "BUY"
        elif breakout_close < lower:
            decision = "SELL"
        else:
            continue

        entry_price = close_curr
        sl_distance = atr_value * atr_sl_multiplier
        tp_distance = sl_distance * rr_ratio_target
        sl = entry_price - sl_distance if decision == "BUY" else entry_price + sl_distance
        tp = entry_price + tp_distance if decision == "BUY" else entry_price - tp_distance

        sl_dist = abs(entry_price - sl)
        tp_dist = abs(tp - entry_price)
        if sl_dist <= 0:
            continue

        # Spread guard — same rationale as the live strategy.
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

        basket = {"type": decision, "entry": entry_price, "vol": vol, "sl": sl, "tp": tp, "open_time": current_time}

    return trades


def print_volatility_breakout_report(trades, symbol, timeframe):
    print("=" * 65)
    print(f"   VOLATILITY BREAKOUT BACKTEST REPORT: {symbol} ({timeframe})")
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
    parser = argparse.ArgumentParser(description="Backtest the volatility breakout (ATR + Bollinger squeeze) strategy on historical MT5 data.")
    parser.add_argument("--symbol", type=str, default="EURUSDm", help="Trading symbol, or comma-separated list e.g. EURUSDm,GBPUSDm")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe (e.g. H1, M30)")
    parser.add_argument("--years", type=int, default=1, help="Years of data to backtest")
    parser.add_argument("--bb-period", type=int, default=20, help="Bollinger Band period")
    parser.add_argument("--bb-stddev", type=float, default=2.0, help="Bollinger Band standard deviation multiplier")
    parser.add_argument("--squeeze-lookback", type=int, default=50, help="Number of prior bars used to judge whether bandwidth is a squeeze")
    parser.add_argument("--squeeze-percentile", type=float, default=20.0, help="Bandwidth percentile threshold defining a squeeze")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period")
    parser.add_argument("--atr-sl-multiplier", type=float, default=1.5, help="SL distance as a multiple of ATR")
    parser.add_argument("--rr-ratio", type=float, default=1.5, help="TP distance as a multiple of the SL distance")
    parser.add_argument("--min-rr", type=float, default=1.2, help="Minimum R:R ratio required to take the trade")
    parser.add_argument("--min-tp-spread-multiple", type=float, default=3.0, help="Require TP distance to be at least this many multiples of the spread")

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

            trades = run_backtest_volatility_breakout(
                df, sym,
                bb_period=args.bb_period,
                bb_stddev=args.bb_stddev,
                squeeze_lookback=args.squeeze_lookback,
                squeeze_percentile=args.squeeze_percentile,
                atr_period=args.atr_period,
                atr_sl_multiplier=args.atr_sl_multiplier,
                rr_ratio_target=args.rr_ratio,
                min_rr_ratio=args.min_rr,
                min_tp_spread_multiple=args.min_tp_spread_multiple
            )
            results[sym] = trades
            print_volatility_breakout_report(trades, sym, args.timeframe)

        if len(results) > 1:
            print("\n" + "=" * 95)
            print("              COMPARATIVE VOLATILITY BREAKOUT BACKTEST SUMMARY")
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
