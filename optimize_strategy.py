"""
Historical Strategy Optimizer
==============================
Connects to MT5, downloads 1 year of historical candles for a target symbol,
calculates mathematical price action metrics (volatility by hour, trend vs range distributions,
RSI/EMA statistics), and queries Gemini 2.5 Flash to generate an optimized scalping strategy.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
from dotenv import load_dotenv
import google.generativeai as genai

from mt5_connection import connect_mt5, disconnect_mt5
from mt5_data import get_timeframe

# Load environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def fetch_historical_data(symbol, timeframe_str, years=1):
    """
    Downloads historical candle data from MT5 for a specified duration.
    """
    tf_constant = get_timeframe(timeframe_str)
    if tf_constant is None:
        print(f"[ERROR] Invalid timeframe: {timeframe_str}")
        return None

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    print(f"[DATA] Fetching historical data for {symbol} ({timeframe_str})...")
    print(f"  Range: {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")

    # Fetch rates from range
    rates = mt5.copy_rates_range(symbol, tf_constant, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        print(f"[ERROR] Failed to fetch rates from MT5. Error: {error}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    print(f"[DATA] Successfully downloaded {len(df)} candles.")
    return df


def calculate_price_action_stats(df, symbol):
    """
    Calculates detailed price action and volatility statistics.
    """
    # 1. Basic conversions
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"[ERROR] Symbol {symbol} info not found.")
        return None

    point = symbol_info.point
    # Check if forex (usually 5 or 3 digits) to compute standard pips
    is_jpy = symbol.upper().endswith("JPY") or "JPY" in symbol.upper()
    pip_multiplier = 100.0 if is_jpy else 10000.0
    if "XAU" in symbol.upper() or "GOLD" in symbol.upper():
        # Gold pip is usually 0.10 price change (10 points)
        pip_multiplier = 10.0
        asset_class = "Gold"
    else:
        asset_class = "Forex"

    # Candle range in pips
    df['range_pips'] = (df['high'] - df['low']) * pip_multiplier
    df['body_pips'] = (df['close'] - df['open']).abs() * pip_multiplier
    
    # 2. Volatility by Hour of Day
    df['hour'] = df['time'].dt.hour
    hourly_vol = df.groupby('hour')['range_pips'].mean().to_dict()
    hourly_vol_formatted = {f"{h:02d}:00": round(v, 2) for h, v in hourly_vol.items()}

    # 3. Volatility by Day of Week
    df['day_of_week'] = df['time'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    day_vol = df.groupby('day_of_week')['range_pips'].mean().reindex(day_order).to_dict()
    day_vol_formatted = {day: round(v, 2) for day, v in day_vol.items()}

    # 4. Trendiness vs Ranging estimation
    # We estimate trendiness by calculating moving average crossovers
    # EMA 20 & EMA 50
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    trend_up = (df['close'] > df['ema20']) & (df['ema20'] > df['ema50'])
    trend_down = (df['close'] < df['ema20']) & (df['ema20'] < df['ema50'])
    ranging = ~(trend_up | trend_down)
    
    total_bars = len(df)
    trend_up_pct = round((trend_up.sum() / total_bars) * 100, 1)
    trend_down_pct = round((trend_down.sum() / total_bars) * 100, 1)
    ranging_pct = round((ranging.sum() / total_bars) * 100, 1)

    # 5. RSI 14 Overbought/Oversold statistics
    # Simple RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi14'] = 100 - (100 / (1 + rs))
    
    rsi_overbought = (df['rsi14'] >= 70).sum() / total_bars * 100
    rsi_oversold = (df['rsi14'] <= 30).sum() / total_bars * 100
    rsi_mid = ((df['rsi14'] > 30) & (df['rsi14'] < 70)).sum() / total_bars * 100

    # Summary Stats
    stats = {
        "symbol": symbol,
        "asset_class": asset_class,
        "total_candles_analyzed": total_bars,
        "avg_candle_range_pips": round(df['range_pips'].mean(), 2),
        "max_candle_range_pips": round(df['range_pips'].max(), 2),
        "avg_candle_body_pips": round(df['body_pips'].mean(), 2),
        "hourly_volatility_profile": hourly_vol_formatted,
        "weekly_volatility_profile": day_vol_formatted,
        "market_regime": {
            "uptrend_pct": trend_up_pct,
            "downtrend_pct": trend_down_pct,
            "ranging_pct": ranging_pct
        },
        "rsi_distribution": {
            "overbought_pct_ge70": round(rsi_overbought, 1),
            "oversold_pct_le30": round(rsi_oversold, 1),
            "neutral_pct": round(rsi_mid, 1)
        }
    }
    
    return stats


def generate_strategy_blueprint(stats, timeframe_str):
    """
    Submits statistical market data to Gemini to formulate the optimized strategy.
    """
    print(f"\n[AI] Sending analysis metrics to Gemini to formulate optimized strategy...")
    
    prompt = f"""
You are an expert quantitative analyst and algorithmic trading developer specializing in low-timeframe Scalping.
Below are the statistical price action metrics calculated on **1 year of historical market data** for:

**Symbol:** {stats['symbol']}
**Timeframe:** {timeframe_str}
**Asset Class:** {stats['asset_class']}

---

### HISTORICAL STATISTICAL DATA SUMMARY

1. **Volatility Metrics:**
   - Average Bar Range: {stats['avg_candle_range_pips']} pips (High-Low)
   - Average Bar Body Size: {stats['avg_candle_body_pips']} pips (Open-Close)
   - Max Single Bar Swing: {stats['max_candle_range_pips']} pips

2. **Hourly Volatility Profile (Average Pip Range by UTC Hour):**
{json.dumps(stats['hourly_volatility_profile'], indent=2)}

3. **Day of Week Volatility Profile (Average Pip Range):**
{json.dumps(stats['weekly_volatility_profile'], indent=2)}

4. **Market Regime Distribution (Based on EMA 20/50 relation):**
   - Uptrend Conditions: {stats['market_regime']['uptrend_pct']}% of the year
   - Downtrend Conditions: {stats['market_regime']['downtrend_pct']}% of the year
   - Ranging/Consolidation: {stats['market_regime']['ranging_pct']}% of the year

5. **RSI 14 Levels Distribution:**
   - Overbought (RSI >= 70): {stats['rsi_distribution']['overbought_pct_ge70']}% of the year
   - Oversold (RSI <= 30): {stats['rsi_distribution']['oversold_pct_le30']}% of the year
   - Neutral Range (30 < RSI < 70): {stats['rsi_distribution']['neutral_pct']}% of the year

---

### YOUR TASK:
Based on this data, write the **Absolute Best Scalping Strategy Blueprint** for this asset class.
Your primary objective is to design an **ultra-high-probability scalping strategy targeting a Win Rate of 70% or higher** while maintaining a highly positive profit expectancy. 

To achieve this high win rate, you should:
- Use a **Multi-Timeframe Macro Trend Filter** (e.g., only buy if the price is above the 200 EMA on M15 or H1) to ensure we never trade against the dominant market direction.
- Implement a **Volatility Filter** using the Average True Range (ATR) to avoid trading in low-volatility, flat ranging environments where price chops and triggers Stop Losses.
- Incorporate strict entry confirmations (such as candlestick engulfing patterns, pin bars, or momentum indicators like MACD/RSI crosses).
- Set specific and realistic SL and TP in pips.

Your blueprint must contain:
1. **Indicator Configurations:** Which specific indicator settings to use (e.g. EMAs, RSI, Bollinger Bands, ATR) optimized for this pair's volatility.
2. **Session / Hour Filter:** Which specific hours are best to execute trades, and which hours must be completely skipped based on the hourly volatility profile.
3. **Exact Entry Rules (BUY & SELL):** Clear, rule-based criteria (indicator triggers, price action patterns) to generate a trade.
4. **Exact Exit Rules (Stop Loss & Take Profit):** Precise SL and TP distance in pips based on the average bar range.
5. **System Instructions Prompt:** A copy-pasteable **System Prompt** block that we can paste directly into our dashboard settings so the bot executes this exact strategy. Keep the prompt strict, concise, and focused on calling the `open_trade` function under these exact rules.

Make sure the output is professional, highly specific (specify exact indicator periods and pip values), and fully optimized for capital preservation.
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[ERROR] Failed to query Gemini API: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze historical market data and optimize AI strategy.")
    parser.add_argument("--symbol", type=str, default="EURUSDm", help="Trading symbol (e.g., EURUSDm, GBPUSDm)")
    parser.add_argument("--timeframe", type=str, default="M15", help="Timeframe (e.g., M5, M15, H1)")
    parser.add_argument("--years", type=int, default=1, help="Years of data to analyze")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("      HISTORICAL SCALPING STRATEGY OPTIMIZER")
    print("=" * 60)

    # 1. Connect to MT5
    if not connect_mt5():
        print("[FATAL] Could not connect to MT5. Exiting.")
        sys.exit(1)

    try:
        # 2. Fetch data
        df = fetch_historical_data(args.symbol, args.timeframe, args.years)
        if df is None:
            sys.exit(1)

        # 3. Calculate statistics
        stats = calculate_price_action_stats(df, args.symbol)
        if stats is None:
            sys.exit(1)

        print("\n" + "-" * 50)
        print("📊 PRICE ACTION STATISTICS EXTRACTED")
        print("-" * 50)
        print(f"  Asset:               {stats['symbol']} ({stats['asset_class']})")
        print(f"  Avg candle range:    {stats['avg_candle_range_pips']} pips")
        print(f"  Trend vs Range:      Uptrend: {stats['market_regime']['uptrend_pct']}%, Downtrend: {stats['market_regime']['downtrend_pct']}%, Range: {stats['market_regime']['ranging_pct']}%")
        print(f"  RSI Overbought (70): {stats['rsi_distribution']['overbought_pct_ge70']}% of the time")
        print(f"  RSI Oversold (30):   {stats['rsi_distribution']['oversold_pct_le30']}% of the time")

        # Find hours with highest volatility
        vol_sorted = sorted(stats['hourly_volatility_profile'].items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 3 Volatile Hours (UTC): {vol_sorted[0][0]} ({vol_sorted[0][1]} pips), {vol_sorted[1][0]} ({vol_sorted[1][1]} pips), {vol_sorted[2][0]} ({vol_sorted[2][1]} pips)")

        # 4. Generate Strategy via Gemini
        blueprint = generate_strategy_blueprint(stats, args.timeframe)
        
        if blueprint:
            output_file = "optimized_strategy.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(blueprint)
                
            print("\n" + "=" * 60)
            print(f"🎉 SUCCESS! Optimized scalping strategy written to:")
            print(f"   [optimized_strategy.md](file:///{os.path.abspath(output_file)})")
            print("=" * 60)
        else:
            print("[ERROR] Strategy generation failed.")

    finally:
        disconnect_mt5()


if __name__ == "__main__":
    main()
