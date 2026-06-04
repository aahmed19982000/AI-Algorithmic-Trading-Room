"""
AI Engine Standalone Test
=========================
Test the AI trading engine WITHOUT MetaTrader 5 connection.
Uses simulated market data to verify AI decision-making.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
from ai_engine import AITradingEngine, format_candles_for_ai
import pandas as pd


# ============================================================
#  Simulated Market Data (No MT5 needed)
# ============================================================

def create_sample_uptrend():
    """Create simulated H4 candles showing an uptrend."""
    data = {
        'Time': pd.date_range('2026-05-28', periods=20, freq='4h'),
        'Open':  [1.1500, 1.1510, 1.1520, 1.1530, 1.1525, 1.1540, 1.1555, 1.1560, 1.1570, 1.1580,
                   1.1575, 1.1590, 1.1600, 1.1610, 1.1605, 1.1620, 1.1635, 1.1640, 1.1650, 1.1655],
        'High':  [1.1520, 1.1530, 1.1540, 1.1545, 1.1550, 1.1560, 1.1570, 1.1580, 1.1590, 1.1595,
                   1.1600, 1.1610, 1.1620, 1.1625, 1.1630, 1.1640, 1.1650, 1.1660, 1.1665, 1.1670],
        'Low':   [1.1495, 1.1505, 1.1515, 1.1520, 1.1515, 1.1530, 1.1545, 1.1555, 1.1565, 1.1570,
                   1.1570, 1.1585, 1.1595, 1.1600, 1.1600, 1.1615, 1.1630, 1.1635, 1.1645, 1.1650],
        'Close': [1.1515, 1.1525, 1.1535, 1.1530, 1.1545, 1.1558, 1.1565, 1.1575, 1.1585, 1.1580,
                   1.1595, 1.1605, 1.1615, 1.1610, 1.1625, 1.1638, 1.1645, 1.1655, 1.1660, 1.1665],
        'Volume': [5000]*20
    }
    return pd.DataFrame(data)


def create_sample_downtrend():
    """Create simulated H4 candles showing a downtrend."""
    data = {
        'Time': pd.date_range('2026-05-28', periods=20, freq='4h'),
        'Open':  [1.1700, 1.1690, 1.1680, 1.1670, 1.1675, 1.1660, 1.1645, 1.1640, 1.1630, 1.1620,
                   1.1625, 1.1610, 1.1600, 1.1590, 1.1595, 1.1580, 1.1565, 1.1560, 1.1550, 1.1545],
        'High':  [1.1710, 1.1695, 1.1685, 1.1680, 1.1680, 1.1665, 1.1650, 1.1645, 1.1635, 1.1630,
                   1.1630, 1.1615, 1.1605, 1.1600, 1.1600, 1.1585, 1.1570, 1.1565, 1.1555, 1.1550],
        'Low':   [1.1685, 1.1675, 1.1665, 1.1660, 1.1655, 1.1640, 1.1630, 1.1625, 1.1615, 1.1610,
                   1.1605, 1.1595, 1.1585, 1.1580, 1.1575, 1.1560, 1.1550, 1.1545, 1.1535, 1.1530],
        'Close': [1.1690, 1.1678, 1.1668, 1.1665, 1.1658, 1.1642, 1.1635, 1.1628, 1.1618, 1.1622,
                   1.1608, 1.1598, 1.1588, 1.1592, 1.1578, 1.1562, 1.1555, 1.1548, 1.1542, 1.1535],
        'Volume': [6000]*20
    }
    return pd.DataFrame(data)


def create_sample_ranging():
    """Create simulated H4 candles showing a ranging/sideways market."""
    data = {
        'Time': pd.date_range('2026-05-28', periods=20, freq='4h'),
        'Open':  [1.1600, 1.1605, 1.1595, 1.1600, 1.1610, 1.1605, 1.1595, 1.1600, 1.1608, 1.1602,
                   1.1598, 1.1605, 1.1610, 1.1600, 1.1595, 1.1602, 1.1608, 1.1600, 1.1605, 1.1598],
        'High':  [1.1615, 1.1615, 1.1610, 1.1615, 1.1620, 1.1615, 1.1610, 1.1615, 1.1618, 1.1612,
                   1.1612, 1.1615, 1.1618, 1.1612, 1.1610, 1.1615, 1.1618, 1.1612, 1.1615, 1.1610],
        'Low':   [1.1590, 1.1590, 1.1585, 1.1590, 1.1595, 1.1590, 1.1585, 1.1590, 1.1595, 1.1590,
                   1.1588, 1.1592, 1.1595, 1.1588, 1.1585, 1.1590, 1.1595, 1.1590, 1.1592, 1.1588],
        'Close': [1.1605, 1.1598, 1.1602, 1.1608, 1.1605, 1.1598, 1.1600, 1.1608, 1.1602, 1.1598,
                   1.1605, 1.1610, 1.1600, 1.1595, 1.1602, 1.1608, 1.1600, 1.1605, 1.1598, 1.1602],
        'Volume': [3000]*20
    }
    return pd.DataFrame(data)


# ============================================================
#  Test Runner
# ============================================================

def run_test(scenario_name, df, current_bid, current_ask):
    """Run a single test scenario."""
    print("\n" + "=" * 60)
    print(f"  SCENARIO: {scenario_name}")
    print("=" * 60)

    # Simulated account info
    account_info = {
        "balance": 500.0,
        "equity": 500.0,
        "margin_free": 500.0,
        "currency": "USD",
        "leverage": 200
    }

    # Simulated current price
    current_price = {
        "bid": current_bid,
        "ask": current_ask,
        "spread": round((current_ask - current_bid) * 10000, 1)
    }

    # Format candles for AI
    candles_text = format_candles_for_ai(df, last_n=20)

    # Initialize AI Engine
    engine = AITradingEngine(model_name="gemini-2.5-flash")

    # Get AI decision
    result = engine.analyze_market(
        symbol="EURUSDm",
        timeframe="H4",
        candles_data=candles_text,
        account_info=account_info,
        current_price=current_price
    )

    # Display result
    print("\n" + "-" * 60)
    print(f"  RESULT:")
    print(f"  Decision:  {result['decision']}")

    if result['trade_params']:
        tp = result['trade_params']
        print(f"  Action:    {tp['action']}")
        print(f"  Symbol:    {tp['symbol']}")
        print(f"  Volume:    {tp['volume']}")
        print(f"  SL:        {tp['sl']}")
        print(f"  TP:        {tp['tp']}")
        print(f"  Reason:    {tp['reason']}")

        # Validate risk rules
        print("\n  --- RISK VALIDATION ---")
        sl_distance = abs(current_bid - tp['sl']) * 10000
        tp_distance = abs(tp['tp'] - current_bid) * 10000
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
        print(f"  SL Distance: {sl_distance:.1f} pips")
        print(f"  TP Distance: {tp_distance:.1f} pips")
        print(f"  Risk/Reward: 1:{rr_ratio:.2f} {'[OK]' if rr_ratio >= 1.5 else '[WARNING: < 1:1.5]'}")
        print(f"  Volume:      {tp['volume']} {'[OK]' if tp['volume'] <= 0.1 else '[WARNING: too high]'}")
    else:
        print(f"  Analysis:  {result['analysis'][:300]}")

    print("-" * 60)
    return result


def main():
    print("=" * 60)
    print("       AI TRADING ENGINE - STANDALONE TEST")
    print("       (No MT5 connection required)")
    print("=" * 60)

    # Test 1: Uptrend
    df_up = create_sample_uptrend()
    run_test(
        "UPTREND - Should suggest BUY",
        df_up,
        current_bid=1.1665,
        current_ask=1.1666
    )

    # Test 2: Downtrend
    df_down = create_sample_downtrend()
    run_test(
        "DOWNTREND - Should suggest SELL",
        df_down,
        current_bid=1.1535,
        current_ask=1.1536
    )

    # Test 3: Ranging
    df_range = create_sample_ranging()
    run_test(
        "RANGING MARKET - Should suggest HOLD",
        df_range,
        current_bid=1.1602,
        current_ask=1.1603
    )

    print("\n" + "=" * 60)
    print("       ALL TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
