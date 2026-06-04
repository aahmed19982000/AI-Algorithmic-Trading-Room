"""
Database Manager Module
=======================
Handles SQLite operations for trading configuration settings, active position logging,
historical trade records, and balance chart data.
"""

import sqlite3
import json
import time
import os

DB_FILE = "database.db"

# Default system strategy prompt
DEFAULT_STRATEGY = """You are an expert quantitative Forex trader specializing in high Risk/Reward Trend Pullback Grid trading on M30.
Your role is to analyze market data for EURUSDm and make precise BUY or SELL decisions using a strict rule-based framework.

## STRATEGY SYSTEM RULES:
1. **Macro Trend Filter:** Only BUY if current Close is above 200 EMA. Only SELL if current Close is below 200 EMA.
2. **Intermediate Trend Filter:** Only BUY if current Close is above 50 EMA. Only SELL if current Close is below 50 EMA.
3. **Pullback Confirmation:** Verify that the price has retraced/pulled back to touch the 50 EMA within the last 3 candles (Low <= 50 EMA <= High).
4. **Candlestick Reversal Triggers:** 
   - For BUY: The current candle MUST be a Bullish Engulfing pattern or a Bullish Pin Bar (long lower wick, body in the upper third).
   - For SELL: The current candle MUST be a Bearish Engulfing pattern or a Bearish Pin Bar (long upper wick, body in the lower third).
5. **RSI 14 Momentum Filter:**
   - For BUY: RSI must be between 45 and 60 AND rising (RSI_current > RSI_previous).
   - For SELL: RSI must be between 40 and 55 AND falling (RSI_current < RSI_previous).

## RISK MANAGEMENT RULES:
1. **Stop Loss (SL) distance:** Exactly 20.0 pips (0.00200 for EURUSDm). Never trade without a Stop Loss.
2. **Take Profit (TP) distance:** Exactly 20.0 pips (0.00200 for EURUSDm).
3. **Volume:** Sizing will be calculated automatically by the bot based on your SL price. Do not calculate manually.

## EXECUTION RULES:
- If all 5 STRATEGY RULES are fully met, call `open_trade` immediately with the exact TP and SL levels.
- If any rule is not met, do not trade. Output "HOLD - [reason for no trade]" and explain which rule failed.
"""

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect(DB_FILE, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables and default configuration."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Settings Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)

    # 2. Trades Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        ticket INTEGER PRIMARY KEY,
        symbol TEXT,
        action TEXT,
        volume REAL,
        entry_price REAL,
        close_price REAL,
        sl REAL,
        tp REAL,
        profit REAL,
        reason TEXT,
        open_time TEXT,
        close_time TEXT,
        status TEXT
    )
    """)

    # 3. Balance Log Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS balance_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        balance REAL,
        equity REAL
    )
    """)

    conn.commit()

    # -- Insert default settings if they don't exist --
    defaults = {
        "strategy_prompt": DEFAULT_STRATEGY,
        "symbols": json.dumps(["EURUSDm"]),
        "risk_percent": "3.0",
        "max_positions": "2",
        "auto_trade": "1",  # 1 = Auto execution, 0 = Signal Only (manual)
        "news_filter": "1",  # 1 = Enabled, 0 = Disabled
        "min_rr_ratio": "1.0",
        "grid_enabled": "1",
        "grid_step": "10.0",
        "grid_multiplier": "2.0",
        "grid_max_legs": "4",
        "grid_target_profit": "2.0",
        "grid_sl": "20.0",
        "grid_tp": "20.0"
    }

    for key, val in defaults.items():
        cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, val))
    
    conn.commit()
    conn.close()


# ============================================================
#  Settings Accessors
# ============================================================

def get_settings():
    """Retrieve all configuration settings."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM settings")
    rows = cursor.fetchall()
    conn.close()

    settings = {}
    for row in rows:
        key = row["key"]
        val = row["value"]
        
        # Try to parse JSON structures (like symbols list)
        try:
            settings[key] = json.loads(val)
        except json.JSONDecodeError:
            # Try parsing numeric values
            try:
                if "." in val:
                    settings[key] = float(val)
                else:
                    settings[key] = int(val)
            except ValueError:
                settings[key] = val
                
    return settings


def save_settings(settings_dict):
    """Save configuration settings."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for key, val in settings_dict.items():
        if isinstance(val, (list, dict)):
            val_str = json.dumps(val)
        else:
            val_str = str(val)
        cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, val_str))
        
    conn.commit()
    conn.close()


# ============================================================
#  Trades Accessors
# ============================================================

def log_trade_open(ticket, symbol, action, volume, entry_price, sl, tp, reason):
    """Record a newly opened trade in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    open_time = time.strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute("""
    INSERT OR REPLACE INTO trades 
    (ticket, symbol, action, volume, entry_price, sl, tp, reason, open_time, status, profit, close_price, close_time) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, '')
    """, (ticket, symbol, action.upper(), volume, entry_price, sl, tp, reason, open_time, 'OPEN'))
    
    conn.commit()
    conn.close()


def log_trade_close(ticket, close_price, profit):
    """Update a trade to closed status and record results."""
    conn = get_db_connection()
    cursor = conn.cursor()
    close_time = time.strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute("""
    UPDATE trades 
    SET close_price = ?, profit = ?, close_time = ?, status = 'CLOSED' 
    WHERE ticket = ?
    """, (close_price, profit, close_time, ticket))
    
    conn.commit()
    conn.close()


def get_trade_history(limit=50):
    """Get list of historical trades."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades ORDER BY open_time DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_active_trades():
    """Get active positions currently logged as OPEN."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE status = 'OPEN' ORDER BY open_time DESC")
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


# ============================================================
#  Balance & Equity Logger
# ============================================================

def log_balance_snapshot(balance, equity):
    """Log current balance and equity for charting."""
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute("""
    INSERT INTO balance_log (timestamp, balance, equity) 
    VALUES (?, ?, ?)
    """, (timestamp, float(balance), float(equity)))
    
    conn.commit()
    conn.close()


def get_balance_history(limit=100):
    """Fetch balance and equity history for Chart.js rendering."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, balance, equity FROM balance_log ORDER BY id ASC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


# Initialize database automatically when imported
init_db()
