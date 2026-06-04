As an expert quantitative analyst and algorithmic trading developer specializing in low-timeframe scalping, I have meticulously analyzed the provided historical data for EURUSDm on the M15 timeframe. The objective is to design an ultra-high-probability scalping strategy targeting a 70%+ win rate with positive expectancy.

The key to achieving this lies in:
1.  **Strict Trend Alignment:** Only trading in the direction of the dominant trend.
2.  **Volatility-Driven Entries:** Focusing on active market hours and avoiding chop.
3.  **Precise Pullback Confirmation:** Identifying high-probability entries at dynamic support/resistance levels with strong momentum.
4.  **Realistic SL/TP:** Calibrated to the asset's typical bar movements for quick profits and controlled risk.

---

### EURUSDm M15 Ultra-High-Probability Scalping Strategy Blueprint

**1. Indicator Configurations:**

*   **Macro Trend Filter (M15):**
    *   `EMA_200`: Exponential Moving Average, Period 200, applied to M15 `Close` price.
*   **Intermediate Trend/Pullback Filter (M15):**
    *   `EMA_50`: Exponential Moving Average, Period 50, applied to M15 `Close` price.
*   **Volatility Filter (M15):**
    *   `ATR_14`: Average True Range, Period 14.
*   **Momentum/Entry Confirmation (M15):**
    *   `RSI_14`: Relative Strength Index, Period 14, applied to M15 `Close` price.

**2. Session / Hour Filter (UTC):**

Based on the Hourly Volatility Profile, the most favorable times for trading EURUSDm are during the London and New York overlaps, where volatility is significantly higher, allowing for quick attainment of profit targets.

*   **Optimal Trading Window:** Execute trades only between **07:00 UTC and 17:00 UTC** (inclusive of 07:00, entries cease before 17:00).
*   **Avoid Trading:** All other hours, especially 02:00-05:00 UTC and 20:00-23:00 UTC, due to significantly lower volatility and increased chop.

**3. Exact Entry Rules (BUY & SELL):**

**General Conditions (apply to both BUY & SELL):**
*   **Global Hour Filter:** Current UTC hour must be within 07:00 to 17:00.
*   **Volatility Filter:** Current `ATR_14` (M15) must be greater than **5.0 pips**. This filters out low-volatility periods where price action is typically less decisive and prone to whipsaws.
*   **News Filter:** Do NOT open new trades 30 minutes before and 30 minutes after any scheduled 'High Impact' news events for EUR or USD currencies. This is crucial for capital preservation.

---

**BUY Trade Conditions:**
An M15 BUY trade is executed ONLY if ALL the following conditions are met:

1.  **Macro Trend (Uptrend):** The current M15 `Close` price is above `EMA_200` (M15).
2.  **Intermediate Trend/Pullback Confirmation:**
    *   The current M15 `Close` price is above `EMA_50` (M15).
    *   AND at least one of the previous 3 M15 bars had its `Low` touch or cross `EMA_50` (M15), indicating a pullback to dynamic support.
3.  **Candlestick Entry Confirmation:**
    *   The current M15 bar is a confirmed **Bullish Engulfing Candlestick pattern** (opens below previous close, closes above previous open, engulfs previous body) OR a **Bullish Pin Bar** (long lower wick, small body at the top, close in upper half).
    *   AND the `Close` of the current M15 bar is above `EMA_50` (M15).
4.  **RSI Momentum Confirmation:**
    *   `RSI_14` (M15) value is between **45 and 60** (inclusive).
    *   AND `RSI_14` (M15, current bar) is greater than `RSI_14` (M15, previous bar), indicating increasing upward momentum.

---

**SELL Trade Conditions:**
An M15 SELL trade is executed ONLY if ALL the following conditions are met:

1.  **Macro Trend (Downtrend):** The current M15 `Close` price is below `EMA_200` (M15).
2.  **Intermediate Trend/Pullback Confirmation:**
    *   The current M15 `Close` price is below `EMA_50` (M15).
    *   AND at least one of the previous 3 M15 bars had its `High` touch or cross `EMA_50` (M15), indicating a pullback to dynamic resistance.
3.  **Candlestick Entry Confirmation:**
    *   The current M15 bar is a confirmed **Bearish Engulfing Candlestick pattern** (opens above previous close, closes below previous open, engulfs previous body) OR a **Bearish Pin Bar** (long upper wick, small body at the bottom, close in lower half).
    *   AND the `Close` of the current M15 bar is below `EMA_50` (M15).
4.  **RSI Momentum Confirmation:**
    *   `RSI_14` (M15) value is between **40 and 55** (inclusive).
    *   AND `RSI_14` (M15, current bar) is less than `RSI_14` (M15, previous bar), indicating increasing downward momentum.

---

**4. Exact Exit Rules (Stop Loss & Take Profit):**

Based on an average M15 bar range of 6.44 pips, these targets are designed for high win rates in a scalping context.

*   **Stop Loss (SL):** **9 pips** from the entry price. (Allows for typical M15 noise but keeps risk contained.)
*   **Take Profit (TP):** **6 pips** from the entry price. (Highly achievable within typical M15 bar movements, prioritizing high win rate over large individual trade profits.)

This results in a Reward-to-Risk (R:R) ratio of approximately 1:0.67. With an anticipated win rate of 70%+, the expected profit per trade is highly positive: `(0.70 * 6 pips) - (0.30 * 9 pips) = 4.2 - 2.7 = +1.5 pips`.

---

**5. System Instructions Prompt (Copy-Pasteable):**

```
# SYSTEM INSTRUCTIONS BLOCK FOR ALGORITHMIC TRADING BOT

# Strategy: EURUSDm M15 High Probability Trend Scalper
# Target Win Rate: 70%+
# Asset: EURUSDm
# Primary Timeframe for all indicator calculations and trade execution: M15

# --- GLOBAL FILTERS & CONDITIONS ---
# 1. Trading Session Filter (UTC Hours):
#    Only allow opening new trades when the current UTC hour is between 07 (inclusive) and 17 (exclusive).
#    (e.g., from 07:00:00 to 16:59:59 UTC).
# 2. Volatility Filter:
#    Only allow opening new trades when ATR(M15, 14) > 5.0 pips.
# 3. News Filter:
#    Do NOT open new trades if the current time is within 30 minutes before OR 30 minutes after any scheduled 'High Impact' news events for EUR or USD currencies.
# 4. Max Open Trades: 1 active trade per symbol at any given time.

# --- INDICATOR CONFIGURATIONS ---
# EMA_200: Exponential Moving Average, Period 200, Applied to M15 Close Price.
# EMA_50:  Exponential Moving Average, Period 50, Applied to M15 Close Price.
# RSI_14:  Relative Strength Index, Period 14, Applied to M15 Close Price.
# ATR_14:  Average True Range, Period 14, Applied to M15.

# --- CANDLESTICK PATTERN DEFINITIONS (for Bot interpretation) ---
# Bullish Engulfing: Current candle opens below previous candle's close, closes above previous candle's open, and current candle's body fully engulfs previous candle's body.
# Bearish Engulfing: Current candle opens above previous candle's close, closes below previous candle's open, and current candle's body fully engulfs previous candle's body.
# Bullish Pin Bar: Current candle's total range is at least 3 times its body size. The body is in the upper 1/3 of the candle, and the lower wick is at least 2 times the body length.
# Bearish Pin Bar: Current candle's total range is at least 3 times its body size. The body is in the lower 1/3 of the candle, and the upper wick is at least 2 times the body length.

# --- ENTRY RULES ---

# BUY Trade Conditions:
#   IF:
#     1. Current `M15_Close` > `EMA_200` (M15)  (Macro Trend UP)
#     AND 2. Current `M15_Close` > `EMA_50` (M15)
#     AND 3. At least one of the previous 3 M15 bars had its `Low` touch or cross `EMA_50` (M15) (Pullback to EMA_50)
#     AND 4. The current M15 bar is a confirmed Bullish Engulfing Candlestick pattern OR a Bullish Pin Bar.
#     AND 5. The `Close` of the current M15 bar is above `EMA_50` (M15).
#     AND 6. `RSI_14` (M15) is between 45 and 60 (inclusive).
#     AND 7. `RSI_14` (M15, current bar) > `RSI_14` (M15, previous bar). (RSI momentum trending up)
#     AND 8. All GLOBAL FILTERS & CONDITIONS (1-4) are met.
#   THEN:
#     Call `open_trade(direction='BUY', symbol='EURUSDm', stop_loss_pips=9, take_profit_pips=6)`

# SELL Trade Conditions:
#   IF:
#     1. Current `M15_Close` < `EMA_200` (M15)  (Macro Trend DOWN)
#     AND 2. Current `M15_Close` < `EMA_50` (M15)
#     AND 3. At least one of the previous 3 M15 bars had its `High` touch or cross `EMA_50` (M15) (Pullback to EMA_50)
#     AND 4. The current M15 bar is a confirmed Bearish Engulfing Candlestick pattern OR a Bearish Pin Bar.
#     AND 5. The `Close` of the current M15 bar is below `EMA_50` (M15).
#     AND 6. `RSI_14` (M15) is between 40 and 55 (inclusive).
#     AND 7. `RSI_14` (M15, current bar) < `RSI_14` (M15, previous bar). (RSI momentum trending down)
#     AND 8. All GLOBAL FILTERS & CONDITIONS (1-4) are met.
#   THEN:
#     Call `open_trade(direction='SELL', symbol='EURUSDm', stop_loss_pips=9, take_profit_pips=6)`

# --- EXIT RULES ---
# Stop Loss: 9 pips from entry price.
# Take Profit: 6 pips from entry price.
# (These exits are managed by the platform via the `stop_loss_pips` and `take_profit_pips` parameters passed to `open_trade`.)
# No other explicit exit conditions are used for this specific scalping strategy.
```