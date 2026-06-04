"""
MT5 Data Module
================
Fetch live price data (candlesticks) and account info from MetaTrader 5.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime


# ============================================================
#  Candlestick / Price Data
# ============================================================

def get_candles(symbol="EURUSD", timeframe=mt5.TIMEFRAME_H4, count=100):
    """
    Fetch candlestick data for a given symbol.

    Args:
        symbol:    Trading pair (e.g. "EURUSD", "GBPUSD", "XAUUSD")
        timeframe: MT5 timeframe constant (e.g. mt5.TIMEFRAME_H4)
        count:     Number of candles to fetch

    Returns:
        pandas.DataFrame or None if failed
    """
    # Verify symbol is available
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[ERROR] Symbol '{symbol}' not found")
        return None

    # Enable symbol if not visible in MarketWatch
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"[ERROR] Failed to enable symbol '{symbol}'")
            return None

    # Fetch candles
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        print(f"[ERROR] Failed to fetch candles for {symbol}")
        print(f"  Error: {error}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)

    # Convert timestamp to readable datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Rename columns for clarity
    df.rename(columns={
        'time': 'Time',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume',
        'spread': 'Spread',
        'real_volume': 'Real_Volume'
    }, inplace=True)

    return df


def get_current_price(symbol="EURUSD"):
    """
    Get the current Bid/Ask price for a symbol.

    Args:
        symbol: Trading pair

    Returns:
        dict with bid, ask, spread or None if failed
    """
    # Verify symbol is available
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[ERROR] Symbol '{symbol}' not found")
        return None

    # Enable symbol if not visible in MarketWatch
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"[ERROR] Failed to enable symbol '{symbol}'")
            return None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] Failed to get tick price for {symbol}")
        return None

    return {
        "symbol": symbol,
        "bid": tick.bid,
        "ask": tick.ask,
        "spread": round((tick.ask - tick.bid) * 10000, 1),  # in pips (for forex)
        "time": datetime.fromtimestamp(tick.time)
    }


# ============================================================
#  Account Info
# ============================================================

def get_account_info():
    """
    Get full account information.

    Returns:
        dict with account details or None if failed
    """
    info = mt5.account_info()
    if info is None:
        print("[ERROR] Failed to get account info")
        return None

    return {
        "name": info.name,
        "login": info.login,
        "server": info.server,
        "currency": info.currency,
        "balance": info.balance,
        "equity": info.equity,
        "profit": info.profit,
        "margin": info.margin,
        "margin_free": info.margin_free,
        "margin_level": info.margin_level,
        "leverage": info.leverage,
        "trade_mode": "Real" if info.trade_mode == 0 else "Demo"
    }


def print_account_summary():
    """Print a formatted account summary."""
    info = get_account_info()
    if info is None:
        return

    print("\n" + "=" * 50)
    print("             ACCOUNT SUMMARY")
    print("=" * 50)
    print(f"  Account:      {info['login']} ({info['trade_mode']})")
    print(f"  Server:       {info['server']}")
    print(f"  Balance:      {info['balance']:.2f} {info['currency']}")
    print(f"  Equity:       {info['equity']:.2f} {info['currency']}")
    print(f"  Profit:       {info['profit']:.2f} {info['currency']}")
    print(f"  Free Margin:  {info['margin_free']:.2f} {info['currency']}")
    print(f"  Leverage:     1:{info['leverage']}")
    print("=" * 50)


# ============================================================
#  Timeframe Helper
# ============================================================

TIMEFRAMES = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


def get_timeframe(name):
    """Convert timeframe string to MT5 constant. e.g. 'H4' -> mt5.TIMEFRAME_H4"""
    tf = TIMEFRAMES.get(name.upper())
    if tf is None:
        print(f"[ERROR] Invalid timeframe '{name}'. Valid: {list(TIMEFRAMES.keys())}")
    return tf
