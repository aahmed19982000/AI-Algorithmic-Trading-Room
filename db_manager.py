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

# Gann & Fibonacci Breakout strategy prompt
DEFAULT_GANN_STRATEGY = """You are an expert quantitative Forex/Crypto trader specializing in W.D. Gann Price Angles (Square of 9) combined with a Fibonacci Pivot Breakout strategy on M30.
Your role is to analyze current market data and make precise BUY or SELL decisions using a strict rule-based framework.

## STRATEGY SYSTEM RULES:
1. **Pivot Breakout Trigger:**
   - For BUY: The price must break and close ABOVE the intermediate high (High B) of a validated A-B-C pivot correction structure.
   - For SELL: The price must break and close BELOW the intermediate low (Low B) of a validated A-B-C pivot correction structure.
2. **Fibonacci Retracement Check:**
   - The correction pivot C must be between 50% and 75% of the preceding move A-B.
3. **Projected Target Prices (Gann Levels):**
   - The Target Profit (TP) must be set at the first Gann Price Angle level above High B (for BUY) or below Low B (for SELL).
4. **Stop Loss (SL) Level:**
   - The Stop Loss must be placed exactly at the correction pivot C.

## PORTFOLIO RISK CONTROL RULES (MANDATORY):
1. **Total Drawdown Cap:** Calculate the total floating drawdown of all active positions. If the total floating loss exceeds **5.0%** of the account balance (i.e. total floating profit is negative and its absolute value is > 5.0% of the balance), you MUST reject any new entries and output "HOLD - Total portfolio drawdown exceeds 5.0% limit."
2. **Correlation Limit:** Do not open a new position on a currency pair in the same direction (e.g., BUY) if there is already an active position on that same currency pair or a highly correlated currency pair (e.g. EURUSD and GBPUSD, or USDCHF and USDJPY inversely) that is currently in a drawdown.
3. **Margin Safety Check:** Verify that the account's Margin Level (if active positions exist) is at least **300%**. If the current Margin Level is below 300%, you MUST reject any new entries and output "HOLD - Margin Level is below 300% safety threshold."

## RISK MANAGEMENT RULES:
1. **Stop Loss (SL) is MANDATORY** - Never open a trade without a Stop Loss.
2. **Volume:** Sizing will be calculated automatically by the bot based on your SL price. Do not calculate manually.

## EXECUTION RULES:
- If a valid breakout trigger is active and verified, and all PORTFOLIO RISK CONTROL RULES are fully satisfied, call `open_trade` immediately with the exact TP and SL levels.
- If no setup is active or if any risk rule is breached, output "HOLD" and explain the current state or reason for rejection.
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

    # Upgrade: Add gann_data column if it doesn't exist
    try:
        cursor.execute("SELECT gann_data FROM trades LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE trades ADD COLUMN gann_data TEXT")
        conn.commit()

    # Upgrade: Add screenshot_url column if it doesn't exist
    try:
        cursor.execute("SELECT screenshot_url FROM trades LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE trades ADD COLUMN screenshot_url TEXT")
        conn.commit()

    # Upgrade: Add telegram_msg_id column if it doesn't exist
    try:
        cursor.execute("SELECT telegram_msg_id FROM trades LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE trades ADD COLUMN telegram_msg_id TEXT")
        conn.commit()

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
        "strategy_prompt": DEFAULT_GANN_STRATEGY,
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
        "grid_tp": "20.0",
        "gann_enabled": "1",
        "gann_lookback": "100",
        "gann_geometry": "square",
        "telegram_enabled": "0",
        "telegram_token": "",
        "telegram_chat_id": "",
        # Icon display settings
        "trading_platforms": json.dumps(["mt5", "mt4", "ctrader", "tradingview", "dxtrade", "matchtrader"]),
        "financial_instruments": json.dumps(["forex", "gold", "crypto", "oil", "silver", "indices", "stocks", "commodities"]),
        "regulations": json.dumps(["fca", "cysec", "asic", "fsca", "cma", "fsa", "sfsa", "vfsc"])
    }

    for key, val in defaults.items():
        cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, val))

    # Ensure icon settings and new settings always exist (for databases created before this feature)
    extra_keys = {
        "trading_platforms": defaults["trading_platforms"],
        "financial_instruments": defaults["financial_instruments"],
        "regulations": defaults["regulations"],
        "ai_evaluation": "1",
        "fallback_to_technical": "1"
    }
    for key, def_val in extra_keys.items():
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO settings (key, value) VALUES (?, ?)", (key, def_val))

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

def log_trade_open(ticket, symbol, action, volume, entry_price, sl, tp, reason, gann_data=None):
    """Record a newly opened trade in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    open_time = time.strftime('%Y-%m-%d %H:%M:%S')
    
    gann_str = json.dumps(gann_data) if gann_data else None
    
    cursor.execute("""
    INSERT OR REPLACE INTO trades 
    (ticket, symbol, action, volume, entry_price, sl, tp, reason, open_time, status, profit, close_price, close_time, gann_data) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, '', ?)
    """, (ticket, symbol, action.upper(), volume, entry_price, sl, tp, reason, open_time, 'OPEN', gann_str))
    
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


def update_trade_screenshot(ticket, screenshot_url):
    """Save the screenshot path for a specific trade."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE trades SET screenshot_url = ? WHERE ticket = ?", (screenshot_url, ticket))
    conn.commit()
    conn.close()


def update_trade_telegram_msg_id(ticket, msg_id):
    """Save the telegram message ID for a specific trade."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE trades SET telegram_msg_id = ? WHERE ticket = ?", (str(msg_id) if msg_id else None, ticket))
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
