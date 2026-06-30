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
You are a professional Forex trading risk analyst and portfolio evaluator AI.
Your main role is to assess market conditions and evaluate proposed trade setups to protect account capital.

## YOUR RESPONSIBILITIES:
1. Evaluate proposed trade signals (BUY/SELL) against risk rules and market environments.
2. Analyze candlestick data, support/resistance levels, trend stability, and spreads.
3. Decide whether to APPROVE or REJECT the proposed trade signal.
4. If you APPROVE the trade: call the `open_trade` function.
5. If you REJECT the trade: respond with text starting with "REJECT - [reason]" and explain the risk elements that caused the rejection.

## PORTFOLIO RISK RULES (MANDATORY):
1. **Stop Loss is ALWAYS required** - Never approve a trade without a Stop Loss (SL).
2. **Risk/Reward Ratio**: Minimum 1:1.5 (TP must be at least 1.5x the SL distance) unless overriding parameters are enforced by the system.
3. **Total Drawdown Cap**: If the portfolio has active trades and the total floating loss exceeds 5.0% of the account balance, you MUST reject new entries.
4. **Correlation Limit**: Do not open a new position on a symbol in the same direction if there is already an active position on a highly correlated currency pair that is currently in a drawdown.
5. **Margin Safety**: Verify that the account's Margin Level is at least 300% (if active trades exist). Reject entries if it falls below 300%.
6. **Market Quality**: Do not trade if the spread is too wide, or if volatility is excessively high/unstable (e.g. during major news).

## OUTPUT RULES:
- To APPROVE a trade: Call the `open_trade` function.
- To REJECT a trade: Respond with text explaining why.
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

    def analyze_market(self, symbol, timeframe, candles_data, account_info, current_price, gann_context=None, active_positions=None, proposed_action=None):
        """
        Send market data to Gemini and get a trading decision.

        Args:
            symbol:        Trading symbol (e.g. "EURUSDm")
            timeframe:     Timeframe string (e.g. "H4")
            candles_data:  String of recent OHLCV data
            account_info:  Dict with balance, equity, etc.
            current_price: Dict with bid, ask, spread
            gann_context:  Optional dict with calculated Gann angles & Fibonacci pivot structure
            active_positions: Optional list of open positions from MT5 or mock data
            proposed_action: Optional proposed trade action ('BUY' or 'SELL') to evaluate

        Returns:
            dict: {
                "decision": "BUY" | "SELL" | "HOLD",
                "trade_params": {...} or None,
                "analysis": "text explanation",
                "raw_response": response object
            }
        """
        # Build the prompt with market data
        prompt = self._build_prompt(symbol, timeframe, candles_data, account_info, current_price, gann_context, active_positions, proposed_action)

        print(f"\n[AI] Analyzing {symbol} {timeframe} (Proposed Action: {proposed_action})...")
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

    def _build_prompt(self, symbol, timeframe, candles_data, account_info, current_price, gann_context=None, active_positions=None, proposed_action=None):
        """Build the market analysis prompt."""
        # Format active positions details
        positions_str = "No active open positions."
        if active_positions:
            pos_lines = []
            total_floating_profit = 0.0
            for pos in active_positions:
                if hasattr(pos, 'ticket'):
                    ticket = pos.ticket
                    sym = pos.symbol
                    pos_type = "BUY" if pos.type == 0 else "SELL"
                    vol = pos.volume
                    p_open = pos.price_open
                    p_curr = pos.price_current
                    profit = pos.profit
                    sl = pos.sl
                    tp = pos.tp
                    comment = getattr(pos, 'comment', '')
                else:
                    ticket = pos.get('ticket', 'N/A')
                    sym = pos.get('symbol', 'N/A')
                    pos_type = pos.get('action', pos.get('type', 'BUY')).upper()
                    if pos_type in [0, '0']:
                        pos_type = 'BUY'
                    elif pos_type in [1, '1']:
                        pos_type = 'SELL'
                    vol = pos.get('volume', 0.01)
                    p_open = pos.get('entry_price', pos.get('price_open', 0.0))
                    p_curr = pos.get('close_price', pos.get('price_current', 0.0))
                    profit = pos.get('profit', 0.0)
                    sl = pos.get('sl', 0.0)
                    tp = pos.get('tp', 0.0)
                    comment = pos.get('reason', pos.get('comment', ''))

                total_floating_profit += profit
                pos_lines.append(
                    f"- Ticket #{ticket} | {sym} | {pos_type} | {vol} lots | Entry Price: {p_open:.5f} | Current Price: {p_curr:.5f} | Profit: ${profit:+.2f} | SL: {sl:.5f} | TP: {tp:.5f} | Comment: {comment}"
                )
            positions_str = "\n".join(pos_lines)
            balance = account_info.get('balance', 0.0)
            if balance > 0:
                dd_pct = (-total_floating_profit / balance) * 100.0 if total_floating_profit < 0 else 0.0
            else:
                dd_pct = 0.0
            positions_str += f"\n\nTotal Floating Profit/Loss: ${total_floating_profit:+.2f}\nTotal Portfolio Floating Drawdown: {dd_pct:.2f}%"

        # Format Margin Level string
        margin_level = account_info.get('margin_level')
        if margin_level is None or margin_level == 0.0:
            margin_level_str = "N/A (No active trades, unlimited margin)"
        else:
            try:
                margin_level_str = f"{float(margin_level):.1f}%"
            except Exception:
                margin_level_str = f"{margin_level}%"

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
- Margin Level: {margin_level_str}
- Leverage: 1:{account_info.get('leverage', 'N/A')}

### PORTFOLIO STATE (ACTIVE OPEN POSITIONS)
{positions_str}
"""
        if gann_context:
            prompt += f"""
### W.D. GANN & FIBONACCI ANALYSIS
We detected a market structure setup on M30:
- **Setup Type:** {gann_context.get('type')} Setup
- **Base Pivot (A):** {gann_context.get('A')} (Starting point for Gann Angles, 0° angle)
- **Breakout Trigger (B):** {gann_context.get('B')}
- **Correction Pivot (C):** {gann_context.get('C')}
- **Fibonacci Retracement:** {gann_context.get('retracement_pct'):.1f}% (Confirmed between 50% and 75%)

Dynamically Calculated Gann Levels relative to base pivot A:
- **Detected Entry Angle:** {gann_context.get('entry_angle')}° (closest to breakout trigger B)
- **Dynamic Target Angle:** {gann_context.get('target_angle')}° (double the entry angle)
- **Target Price (TP):** {gann_context.get('target_price'):.5f}
- **Stop Loss Price (SL):** {gann_context.get('sl_price'):.5f} (set at 0° angle / pivot A)

Use these levels to determine your breakout entry and exit. 
If the current price breaks and closes beyond B:
- Entry Price: {current_price.get('ask') if gann_context.get('type') == 'BUY' else current_price.get('bid')}
- Target Profit (TP): {gann_context.get('target_price'):.5f}
- Stop Loss (SL): {gann_context.get('sl_price'):.5f}
"""

        if proposed_action:
            prompt += f"""
### Recent Candles (newest last):
{candles_data}

---
## PROPOSED TRADE SIGNAL (TO EVALUATE)
A technical indicator strategy has generated a PROPOSED TRADE SIGNAL: **{proposed_action}** on **{symbol}**.
Your task is to evaluate this proposed trade setup against all risk rules (drawdown caps, correlation limits, margin level safety, general market trend, volatility, spread) and decide whether to APPROVE or REJECT it.

If you APPROVE: Call the `open_trade` function with the action='{proposed_action}' and the appropriate parameters.
If you REJECT: Do NOT call `open_trade`. Respond with text starting with "REJECT - [reason]" explaining the risk factors that caused you to reject it.
"""
        else:
            prompt += f"""
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
