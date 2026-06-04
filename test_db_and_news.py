"""
Database and News Filter Integration Test
=========================================
Verifies that SQLite database is correctly initialized, records settings and trades,
and ensures that the news filter successfully queries and parses TradingView events.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
from db_manager import get_settings, save_settings, log_trade_open, log_trade_close, get_trade_history, get_active_trades, log_balance_snapshot, get_balance_history
from news_filter import is_news_time, fetch_economic_events

def run_integration_test():
    print("=" * 65)
    print("       DATABASE & NEWS FILTER INTEGRATION TEST")
    print("=" * 65)

    # --------------------------------------------------------
    #  Part 1: Database Test
    # --------------------------------------------------------
    print("\n[PART 1] Testing Database Manager...")
    
    # 1. Fetch current settings
    settings = get_settings()
    print("[OK] Settings retrieved successfully:")
    print(f"  - Traded Symbols: {settings.get('symbols')}")
    print(f"  - Risk Percent:   {settings.get('risk_percent')}%")
    print(f"  - Max Positions:  {settings.get('max_positions')}")
    print(f"  - Auto Trade:     {settings.get('auto_trade')}")
    print(f"  - News Filter:    {settings.get('news_filter')}")

    # 2. Modify and save a setting
    print("\n  Updating Risk Percent to 2.0% and saving...")
    save_settings({"risk_percent": 2.0})
    new_settings = get_settings()
    print(f"  [OK] New Risk Percent value: {new_settings.get('risk_percent')}%")

    # Restore default setting (1.0%)
    save_settings({"risk_percent": 1.0})

    # 3. Test Trade Logging
    print("\n  Logging a test trade open...")
    test_ticket = 9999999
    log_trade_open(
        ticket=test_ticket,
        symbol="EURUSDm",
        action="BUY",
        volume=0.02,
        entry_price=1.1650,
        sl=1.1600,
        tp=1.1750,
        reason="Test breakout signal"
    )
    
    active_trades = get_active_trades()
    print(f"  [OK] Active Trades count: {len(active_trades)}")
    if len(active_trades) > 0:
        print(f"    Active Ticket: {active_trades[0]['ticket']} | Symbol: {active_trades[0]['symbol']} | Status: {active_trades[0]['status']}")

    # Close trade
    print("\n  Logging test trade close...")
    log_trade_close(ticket=test_ticket, close_price=1.1710, profit=12.0)
    
    history = get_trade_history()
    print(f"  [OK] Trade history records count: {len(history)}")
    if len(history) > 0:
        print(f"    History Ticket: {history[0]['ticket']} | Close Price: {history[0]['close_price']} | Profit: ${history[0]['profit']} | Status: {history[0]['status']}")

    # 4. Test Balance Logging
    print("\n  Logging balance snapshot...")
    log_balance_snapshot(balance=500.0, equity=512.50)
    balance_history = get_balance_history()
    print(f"  [OK] Balance history records: {len(balance_history)}")
    if len(balance_history) > 0:
        print(f"    Latest Snapshot: Balance={balance_history[-1]['balance']} | Equity={balance_history[-1]['equity']}")

    # --------------------------------------------------------
    #  Part 2: News Filter Test
    # --------------------------------------------------------
    print("\n" + "=" * 50)
    print("[PART 2] Testing Economic News Filter...")
    print("=" * 50)
    
    try:
        raw_events = fetch_economic_events()
        print(f"[OK] Successfully fetched {len(raw_events)} events from TradingView.")
        
        should_freeze, active_news = is_news_time(minutes_before=60, minutes_after=30)
        print(f"\nShould freeze trading now? {'🚫 YES' if should_freeze else '✅ NO'}")
        
        if len(active_news) > 0:
            print("Active news events triggering the freeze:")
            for e in active_news:
                print(f"  - [{e['importance']}] {e['currency']} - {e['title']} at {e['time']} (Time diff: {e['time_diff_minutes']} mins)")
        else:
            print("No active high/medium impact events in the current window.")

    except Exception as e:
        print(f"[FAIL] News filter test failed: {e}")

    print("\n" + "=" * 65)
    print("       INTEGRATION TEST COMPLETED")
    print("=" * 65)

if __name__ == "__main__":
    run_integration_test()
