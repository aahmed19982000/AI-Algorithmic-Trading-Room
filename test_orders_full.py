"""
MT5 Order Execution Integration Test
=====================================
Connects to MT5, opens a 0.01 lot BUY trade on EURUSDm, and closes it immediately.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import MetaTrader5 as mt5
from mt5_connection import connect_mt5, disconnect_mt5
from mt5_orders import open_trade, close_position, get_open_positions
from mt5_data import get_current_price

def run_order_test():
    print("=" * 60)
    print("       MT5 ORDER EXECUTION TEST")
    print("=" * 60)

    # 1. Connect to MT5
    if not connect_mt5():
        print("[FAIL] Connection to MT5 terminal failed")
        return False

    symbol = "EURUSDm"

    try:
        # 2. Get current price
        price_info = get_current_price(symbol)
        if not price_info:
            print(f"[FAIL] Could not fetch price for {symbol}")
            return False

        ask = price_info['ask']
        bid = price_info['bid']
        print(f"\n[INFO] Current price for {symbol}: Bid={bid}, Ask={ask}")

        # Set conservative SL/TP
        # 1 pip = 0.0001 for EURUSD
        sl_price = ask - 0.0020  # 20 pips below entry
        tp_price = ask + 0.0030  # 30 pips above entry

        # 3. Open a BUY position
        trade_result = open_trade(
            action="BUY",
            symbol=symbol,
            volume=0.01,
            sl=sl_price,
            tp=tp_price,
            magic=999111,
            comment="Automated Test Order"
        )

        if not trade_result or not trade_result["success"]:
            print("[FAIL] Trade execution failed")
            return False

        ticket = trade_result["ticket"]
        print(f"\n[INFO] Open positions count: {len(get_open_positions())}")

        # Wait 2 seconds so user can see it in terminal
        print("\n⏳ Waiting 2 seconds before closing trade...")
        time.sleep(2)

        # 4. Close the position
        close_result = close_position(ticket=ticket, magic=999111, comment="Automated Test Close")
        
        if close_result:
            print("[OK] Test passed! Position opened and closed successfully.")
        else:
            print("[FAIL] Failed to close position")
            return False

    except Exception as e:
        print(f"[ERROR] Exception during test: {e}")
        return False

    finally:
        # 5. Disconnect
        disconnect_mt5()
        print("=" * 60)

    return True

if __name__ == "__main__":
    run_order_test()
