"""
MT5 Full Connection Test
========================
Test script: Connect, fetch account info, get 100 H4 EURUSD candles.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from mt5_connection import connect_mt5, disconnect_mt5
from mt5_data import get_candles, get_current_price, print_account_summary
import MetaTrader5 as mt5


def run_test():
    print("=" * 55)
    print("       MT5 FULL CONNECTION TEST")
    print("=" * 55)

    # ── Step 1: Connect ──
    print("\n[TEST 1] Connecting to MT5...")
    if not connect_mt5():
        print("[FAIL] Could not connect to MT5")
        return False

    # ── Step 2: Account Info ──
    print("\n[TEST 2] Fetching account info...")
    print_account_summary()

    # ── Step 3: Current Price ──
    print("\n[TEST 3] Fetching current EURUSDm price...")
    price = get_current_price("EURUSDm")
    if price:
        print(f"  Bid:    {price['bid']}")
        print(f"  Ask:    {price['ask']}")
        print(f"  Spread: {price['spread']} pips")
        print(f"  Time:   {price['time']}")
    else:
        print("  [FAIL] Could not fetch price")

    # ── Step 4: Fetch 100 H4 Candles ──
    print("\n[TEST 4] Fetching 100 H4 candles for EURUSDm...")
    df = get_candles(symbol="EURUSDm", timeframe=mt5.TIMEFRAME_H4, count=100)

    if df is not None:
        print(f"  Candles fetched: {len(df)}")
        print(f"  Date range: {df['Time'].iloc[0]} -> {df['Time'].iloc[-1]}")

        print("\n  --- First 5 candles ---")
        print(df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']].head().to_string(index=False))

        print("\n  --- Last 5 candles ---")
        print(df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']].tail().to_string(index=False))

        print(f"\n  --- Statistics ---")
        print(f"  Highest High:  {df['High'].max()}")
        print(f"  Lowest Low:    {df['Low'].min()}")
        print(f"  Avg Close:     {df['Close'].mean():.5f}")
        print(f"  Avg Volume:    {df['Volume'].mean():.0f}")
    else:
        print("  [FAIL] Could not fetch candles")

    # ── Step 5: Disconnect ──
    disconnect_mt5()

    print("\n" + "=" * 55)
    print("       ALL TESTS PASSED!")
    print("=" * 55)
    return True


if __name__ == "__main__":
    run_test()
