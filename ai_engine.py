"""
AI Trading Engine
==================
Gemini AI engine with Function Calling for trading decisions.
Includes comprehensive system prompt with risk management rules.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from db_manager import get_settings

load_dotenv()

# ============================================================
#  Configuration
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("[ERROR] GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)

# ============================================================
#  System Prompt - Trading AI Instructions
# ============================================================

SYSTEM_PROMPT = """
You are a professional Forex trading analyst AI assistant.
Your role is to analyze market data and make precise trading decisions.

## YOUR RESPONSIBILITIES:
1. Analyze candlestick data (OHLCV) provided to you
2. Identify trends, support/resistance levels, and patterns
3. Decide whether to BUY, SELL, or HOLD (do nothing)
4. If trading, call the `open_trade` function with exact parameters

## TECHNICAL ANALYSIS TOOLS TO USE:
- Trend Analysis: Higher highs/lows (uptrend), Lower highs/lows (downtrend)
- Support & Resistance: Key price levels from recent highs/lows
- Candlestick Patterns: Engulfing, Doji, Hammer, Pin Bar, etc.
- Moving Average Logic: Compare recent closes to average close
- Momentum: Is price accelerating or decelerating?

## RISK MANAGEMENT RULES (MANDATORY):
1. **Stop Loss is ALWAYS required** - Never open a trade without SL
2. **Risk/Reward Ratio**: Minimum 1:1.5 (TP must be at least 1.5x the SL distance)
3. **Stop Loss Distance**: 
   - Forex pairs: Between 15-50 pips from entry
   - Gold (XAU): Between 100-500 pips from entry
4. **Volume/Lot Size**: 
   - Use conservative sizing: 0.01 to 0.1 lots maximum
   - Default to 0.01 for accounts under $1000
5. **DO NOT trade if**:
   - Market is ranging/choppy with no clear direction
   - Spread is too wide (> 3 pips for Forex)
   - Not enough data to make a confident decision

## OUTPUT RULES:
- If you decide to trade: Call the `open_trade` function
- If you decide NOT to trade: Respond with text explaining why (e.g., "HOLD - no clear trend")
- Always provide a brief analysis BEFORE making your decision
- Confidence level: Only trade if you are at least 60% confident

## IMPORTANT:
- You are managing REAL MONEY. Be conservative and protect capital.
- When in doubt, DO NOT trade. It's better to miss an opportunity than to lose money.
- Always explain your reasoning briefly.
"""

# ============================================================
#  Function Calling Definition - open_trade
# ============================================================

OPEN_TRADE_FUNCTION = genai.protos.FunctionDeclaration(
    name="open_trade",
    description="Open a new trading position on MetaTrader 5. Only call this when you are confident in the trade setup.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "action": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Trade direction: 'BUY' for long position, 'SELL' for short position",
                enum=["BUY", "SELL"]
            ),
            "symbol": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Trading symbol (e.g., 'EURUSDm', 'GBPUSDm', 'XAUUSDm')"
            ),
            "volume": genai.protos.Schema(
                type=genai.protos.Type.NUMBER,
                description="Lot size for the trade (0.01 to 0.1). Use 0.01 for conservative sizing."
            ),
            "sl": genai.protos.Schema(
                type=genai.protos.Type.NUMBER,
                description="Stop Loss price level (REQUIRED). Must be below entry for BUY, above entry for SELL."
            ),
            "tp": genai.protos.Schema(
                type=genai.protos.Type.NUMBER,
                description="Take Profit price level (REQUIRED). Must be above entry for BUY, below entry for SELL."
            ),
            "reason": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Brief explanation of why this trade is being taken"
            ),
        },
        required=["action", "symbol", "volume", "sl", "tp", "reason"]
    )
)

# Tool wrapper
TRADING_TOOLS = genai.protos.Tool(
    function_declarations=[OPEN_TRADE_FUNCTION]
)

# ============================================================
#  AI Engine Class
# ============================================================

class AITradingEngine:
    """Gemini AI Trading Engine with Function Calling."""

    def __init__(self, model_name="gemini-2.5-flash"):
        """
        Initialize the AI engine.

        Args:
            model_name: Gemini model to use
        """
        # Load custom strategy prompt dynamically from settings database
        try:
            settings = get_settings()
            prompt = settings.get("strategy_prompt", SYSTEM_PROMPT)
        except Exception as e:
            print(f"[WARNING] Could not load strategy prompt from database: {e}. Using fallback prompt.")
            prompt = SYSTEM_PROMPT

        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=prompt,
            tools=[TRADING_TOOLS]
        )
        self.model_name = model_name
        print(f"[OK] AI Engine initialized (model: {model_name})")

    def analyze_market(self, symbol, timeframe, candles_data, account_info, current_price):
        """
        Send market data to Gemini and get a trading decision.

        Args:
            symbol:        Trading symbol (e.g. "EURUSDm")
            timeframe:     Timeframe string (e.g. "H4")
            candles_data:  String of recent OHLCV data
            account_info:  Dict with balance, equity, etc.
            current_price: Dict with bid, ask, spread

        Returns:
            dict: {
                "decision": "BUY" | "SELL" | "HOLD",
                "trade_params": {...} or None,
                "analysis": "text explanation",
                "raw_response": response object
            }
        """
        # Build the prompt with market data
        prompt = self._build_prompt(symbol, timeframe, candles_data, account_info, current_price)

        print(f"\n[AI] Analyzing {symbol} {timeframe}...")
        print(f"[AI] Sending to {self.model_name}...")

        try:
            response = self.model.generate_content(prompt)
            return self._parse_response(response)
        except Exception as e:
            print(f"[ERROR] AI API call failed: {e}")
            return {
                "decision": "ERROR",
                "trade_params": None,
                "analysis": str(e),
                "raw_response": None
            }

    def _build_prompt(self, symbol, timeframe, candles_data, account_info, current_price):
        """Build the market analysis prompt."""
        prompt = f"""
## MARKET DATA FOR ANALYSIS

**Symbol:** {symbol}
**Timeframe:** {timeframe}
**Current Time:** Now

### Current Price:
- Bid: {current_price.get('bid', 'N/A')}
- Ask: {current_price.get('ask', 'N/A')}
- Spread: {current_price.get('spread', 'N/A')} pips

### Account Status:
- Balance: {account_info.get('balance', 'N/A')} {account_info.get('currency', 'USD')}
- Equity: {account_info.get('equity', 'N/A')} {account_info.get('currency', 'USD')}
- Free Margin: {account_info.get('margin_free', 'N/A')} {account_info.get('currency', 'USD')}
- Leverage: 1:{account_info.get('leverage', 'N/A')}

### Recent Candles (newest last):
{candles_data}

---
Analyze the above data. Should I BUY, SELL, or HOLD?
If you decide to trade, use the `open_trade` function.
If you decide to HOLD, explain why in text.
"""
        return prompt

    def _parse_response(self, response):
        """Parse Gemini's response - either function call or text."""
        result = {
            "decision": "HOLD",
            "trade_params": None,
            "analysis": "",
            "raw_response": response
        }

        # Check each part of the response
        for candidate in response.candidates:
            for part in candidate.content.parts:

                # -- Function Call detected --
                if part.function_call:
                    fc = part.function_call
                    if fc.name == "open_trade":
                        args = dict(fc.args)
                        result["decision"] = args.get("action", "UNKNOWN")
                        result["trade_params"] = {
                            "action": args.get("action"),
                            "symbol": args.get("symbol"),
                            "volume": float(args.get("volume", 0.01)),
                            "sl": float(args.get("sl", 0)),
                            "tp": float(args.get("tp", 0)),
                            "reason": args.get("reason", "")
                        }
                        result["analysis"] = args.get("reason", "")
                        print(f"[AI] Decision: {result['decision']}")
                        print(f"[AI] Params: {json.dumps(result['trade_params'], indent=2)}")

                # -- Text response (HOLD) --
                elif part.text:
                    result["analysis"] = part.text
                    # Check if text explicitly says BUY/SELL (shouldn't happen with function calling)
                    if "HOLD" in part.text.upper() or "DO NOT TRADE" in part.text.upper():
                        result["decision"] = "HOLD"
                    print(f"[AI] Decision: HOLD")
                    print(f"[AI] Reason: {part.text[:200]}")

        return result


def format_candles_for_ai(df, last_n=20):
    """
    Format a candles DataFrame into a readable string for the AI.

    Args:
        df: pandas DataFrame from get_candles()
        last_n: Number of recent candles to include

    Returns:
        str: Formatted candle data
    """
    recent = df.tail(last_n)
    lines = ["Time | Open | High | Low | Close | Volume"]
    lines.append("-" * 60)
    for _, row in recent.iterrows():
        lines.append(
            f"{row['Time']} | {row['Open']:.5f} | {row['High']:.5f} | "
            f"{row['Low']:.5f} | {row['Close']:.5f} | {row['Volume']}"
        )
    return "\n".join(lines)
