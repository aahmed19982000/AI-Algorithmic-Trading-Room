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
from db_manager import get_settings, get_trade_history

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
You are a professional Forex trading analyst.
Your main role is to assess market conditions and evaluate proposed trade setups based on technical structures and candlestick patterns.

## YOUR RESPONSIBILITIES:
1. Evaluate proposed trade signals (BUY/SELL) against technical candle behaviors.
2. Analyze candlestick data, support/resistance levels, trend stability, and spreads.
3. Decide whether to APPROVE or REJECT the proposed trade signal.
4. If you APPROVE the trade: call the `open_trade` function.
5. If you REJECT the trade: respond with text starting with "REJECT - [reason]" and explain the technical reason.

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
            self.ai_provider = os.getenv("AI_PROVIDER", settings.get("ai_provider", "gemini")).lower()
            self.ollama_model = os.getenv("OLLAMA_MODEL", settings.get("ollama_model", "phi3")).lower()
            self.ollama_url = os.getenv("OLLAMA_URL", settings.get("ollama_url", "http://localhost:11434")).strip("/")
        except Exception as e:
            print(f"[WARNING] Could not load strategy prompt or provider from database: {e}. Using fallback prompt and Gemini.")
            prompt = SYSTEM_PROMPT
            self.ai_provider = os.getenv("AI_PROVIDER", "gemini").lower()
            self.ollama_model = os.getenv("OLLAMA_MODEL", "phi3").lower()
            self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").strip("/")

        self.system_prompt_text = prompt
        self.model_name = model_name

        if self.ai_provider == "gemini":
            self.model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=prompt,
                tools=[TRADING_TOOLS],
                # Gemini 2.5 Flash spends hidden "thinking" tokens out of this same
                # budget (this SDK version has no thinking_config to cap those
                # separately) — a tight cap here truncates the real answer before it's
                # written (verified: 400 caused finish_reason=MAX_TOKENS with ~380
                # thinking tokens and an empty/cut-off response). 2000 leaves headroom
                # for thinking while still bounding worst-case cost to a fraction of a cent.
                generation_config=genai.GenerationConfig(max_output_tokens=2000)
            )
            print(f"[OK] AI Engine initialized (model: {model_name})")
        else:
            print(f"[OK] AI Engine initialized for Local Ollama (model: {self.ollama_model})")

    def analyze_market(self, symbol, timeframe, candles_data, account_info, current_price, gann_context=None, active_positions=None, proposed_action=None):
        """
        Send market data to AI Provider (Gemini or Ollama) and get a trading decision.

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
        
        if self.ai_provider == "ollama":
            print(f"[AI] Sending to Local Ollama ({self.ollama_model})...")
            import requests
            
            prompt_with_instructions = f"""{prompt}
            
            IMPORTANT: You must respond ONLY with a valid JSON object matching the schema below. Do not write any explanations, Markdown blocks, or prefix text. Just pure JSON.
            JSON Schema:
            {{
                "decision": "APPROVE" or "REJECT",
                "reason": "brief technical explanation why",
                "trade_params": {{
                    "action": "{proposed_action}",
                    "symbol": "{symbol}",
                    "volume": 0.01,
                    "sl": {gann_context.get('sl_price', 0.0) if gann_context else 0.0},
                    "tp": {gann_context.get('target_price', 0.0) if gann_context else 0.0},
                    "reason": "brief reason for trade parameters"
                }}
            }}
            
            Note: If your decision is REJECT, you can set "trade_params" to null, but "decision" and "reason" must be present.
            """
            
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": self.system_prompt_text},
                    {"role": "user", "content": prompt_with_instructions}
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "num_ctx": 1024
                }
            }

            try:
                r = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=180)
                if r.status_code != 200:
                    import sys
                    sys.stderr.write(f"[AI ENGINE] Ollama API error body: {r.text}\n")
                r.raise_for_status()
                res_json = r.json()
                content = res_json.get("message", {}).get("content", "").strip()
                
                # Clean Markdown code blocks if present
                clean_content = content
                if clean_content.startswith("```"):
                    lines = clean_content.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    clean_content = "\n".join(lines).strip()
                
                data = json.loads(clean_content)
                
                decision_val = data.get("decision", "REJECT").upper()
                reason_val = data.get("reason", "Local AI decision")
                
                decision_final = proposed_action if decision_val == "APPROVE" else "HOLD"
                
                trade_params = None
                if decision_final in ["BUY", "SELL"]:
                    tp_data = data.get("trade_params") or {}
                    sl_val = float(tp_data.get("sl", gann_context.get("sl_price", 0.0) if gann_context else 0.0))
                    tp_val = float(tp_data.get("tp", gann_context.get("target_price", 0.0) if gann_context else 0.0))
                    vol_val = float(tp_data.get("volume", 0.01))
                    
                    trade_params = {
                        "action": proposed_action,
                        "symbol": symbol,
                        "volume": vol_val,
                        "sl": sl_val,
                        "tp": tp_val,
                        "reason": f"Ollama {self.ollama_model} Approved: {tp_data.get('reason', reason_val)}"
                    }
                
                print(f"[AI] Local Ollama Decision: {decision_final} (Reason: {reason_val})")
                
                # Extract Token usage metadata from Ollama response if available
                in_tokens = res_json.get("prompt_eval_count", 0)
                out_tokens = res_json.get("eval_count", 0)
                
                result = {
                    "decision": decision_final,
                    "trade_params": trade_params,
                    "analysis": reason_val,
                    "raw_response": content
                }
                
                if in_tokens > 0 or out_tokens > 0:
                    result["usage"] = {
                        "in_tokens": in_tokens,
                        "out_tokens": out_tokens,
                        "cost": 0.0
                    }
                    print(f"[AI USAGE - Ollama] Prompt: {in_tokens} | Candidates: {out_tokens} | Cost: $0.000000")
                
                return result
            except Exception as e:
                print(f"[ERROR] Local Ollama API call failed: {e}")
                return {
                    "decision": "ERROR",
                    "trade_params": None,
                    "analysis": f"Local Ollama failed: {str(e)}",
                    "raw_response": None
                }
        else:
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

    def _get_symbol_performance_summary(self, symbol, days=30, max_trades=10):
        """
        Compute a real recent performance summary for this symbol from the SQLite trade log,
        so the AI evaluates risk against actual numbers instead of guessing.
        """
        import datetime as _dt

        try:
            history = get_trade_history(limit=500)
        except Exception:
            return "No trade history available."

        cutoff = _dt.datetime.now() - _dt.timedelta(days=days)
        wins = losses = considered = 0
        net_profit = 0.0
        worst_trade = 0.0

        for trade in history:
            if trade.get("symbol") != symbol or trade.get("status") != "CLOSED":
                continue
            close_time_str = trade.get("close_time")
            close_dt = None
            if close_time_str:
                try:
                    close_dt = _dt.datetime.strptime(close_time_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    close_dt = None
            if close_dt and close_dt < cutoff:
                continue

            profit = float(trade.get("profit") or 0.0)
            considered += 1
            wins += 1 if profit >= 0 else 0
            losses += 1 if profit < 0 else 0
            net_profit += profit
            worst_trade = min(worst_trade, profit)

            if considered >= max_trades:
                break

        if considered == 0:
            return f"No closed trades on {symbol} in the last {days} days (no real drawdown data to report)."

        return (f"Last {considered} closed trade(s) on {symbol} in the past {days} days: "
                f"{wins}W / {losses}L, Net P/L: ${net_profit:+.2f}, Worst single trade: ${worst_trade:+.2f}.")

    def _build_prompt(self, symbol, timeframe, candles_data, account_info, current_price, gann_context=None, active_positions=None, proposed_action=None):
        """Build the market analysis prompt, grounded in real account/position/history data."""
        account_info = account_info or {}
        active_positions = active_positions or []
        symbol_positions = [p for p in active_positions if getattr(p, "symbol", None) == symbol]
        performance_summary = self._get_symbol_performance_summary(symbol)

        prompt = f"""
## MARKET DATA FOR ANALYSIS

**Symbol:** {symbol}
**Timeframe:** {timeframe}

### Current Price:
- Bid: {current_price.get('bid', 'N/A')}
- Ask: {current_price.get('ask', 'N/A')}
- Spread: {current_price.get('spread', 'N/A')} pips

### Account Status (real data):
- Balance: {account_info.get('balance', 'N/A')} {account_info.get('currency', '')}
- Equity: {account_info.get('equity', 'N/A')}
- Margin Level: {account_info.get('margin_level', 'N/A')}%
- Open Positions (all symbols): {len(active_positions)} | On {symbol}: {len(symbol_positions)}

### Real Recent Performance on {symbol}:
{performance_summary}

IMPORTANT: Only reference the real figures given above. Do not invent statistics, percentages,
or trade counts that are not explicitly stated in this prompt. Keep any explanation or rejection
reason to one short sentence (under 30 words).
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

        # Extract Token usage metadata if available
        try:
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                in_tokens = response.usage_metadata.prompt_token_count
                out_tokens = response.usage_metadata.candidates_token_count
                # Cost calculation: Input is $0.075 / 1M tokens, Output is $0.30 / 1M tokens for Flash 2.5
                cost = (in_tokens * (0.075 / 1000000.0)) + (out_tokens * (0.30 / 1000000.0))
                result["usage"] = {
                    "in_tokens": in_tokens,
                    "out_tokens": out_tokens,
                    "cost": cost
                }
                print(f"[AI USAGE] Prompt: {in_tokens} | Candidates: {out_tokens} | Cost: ${cost:.6f}")
        except Exception as ue:
            print(f"[WARNING] Could not parse token usage metadata: {ue}")

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

    def analyze_exit(self, symbol, pos_type, candles_data):
        """
        Ask the AI to verify whether the M5 trend has indeed reversed.
        
        Args:
            symbol: Trading symbol (e.g. "EURUSDm")
            pos_type: 0 for BUY, 1 for SELL
            candles_data: Formatted string of recent M5 candles
            
        Returns:
            bool: True if AI confirms we should EXIT, False if AI says HOLD (interprets trigger as noise).
        """
        pos_str = "BUY (Long)" if pos_type == 0 else "SELL (Short)"
        signal_str = "Lower Low (LL) breakout" if pos_type == 0 else "Higher High (HH) breakout"
        
        prompt = f"""
You are a professional Forex trading analyst.
We have an active {pos_str} position on {symbol}.
Our technical indicators on the M5 timeframe have triggered a potential exit alert: **{signal_str}**.

Your task is to analyze the recent M5 candlestick data below and decide if this is a genuine trend reversal requiring us to EXIT immediately to protect capital/profits, or if this is just minor market noise (e.g., a brief pullback) and we should HOLD (keep the position open).

## RECENT M5 CANDLES (newest last):
{candles_data}

## YOUR DECISION RULES:
- If the candles confirm a structural change and a clear shift in direction opposing our {pos_str} trade: Recommend EXIT.
- If the candles show it is likely just a minor correction/noise and the primary momentum is still intact: Recommend HOLD.
"""

        # Exit verification always runs on the free local Ollama model, regardless of
        # ai_provider. It's low-stakes (a failure/timeout here just defaults to HOLD,
        # i.e. keep the position open) and fires far more often than the entry decision,
        # so routing it to Gemini would be the single biggest avoidable cost driver.
        print(f"[AI EXIT CHECK] Sending exit analysis to Local Ollama ({self.ollama_model})...")
        import requests

        prompt_with_instructions = f"""{prompt}

        IMPORTANT: You must respond ONLY with a valid JSON object matching the schema below. Do not write any explanations, Markdown blocks, or prefix text. Just pure JSON.
        JSON Schema:
        {{
            "decision": "EXIT" or "HOLD",
            "reason": "brief technical explanation why"
        }}
        """

        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": "You are a professional trading analyst. Return only JSON."},
                {"role": "user", "content": prompt_with_instructions}
            ],
            "stream": False,
            "format": "json",
            "options": {
                "num_ctx": 1024
            }
        }

        try:
            r = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=30)
            r.raise_for_status()
            res_json = r.json()
            content = res_json.get("message", {}).get("content", "").strip()

            # Clean Markdown code blocks if present
            clean_content = content
            if clean_content.startswith("```"):
                lines = clean_content.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                clean_content = "\n".join(lines).strip()

            data = json.loads(clean_content)
            decision_val = data.get("decision", "HOLD").upper()
            reason_val = data.get("reason", "Local AI exit decision")

            print(f"[AI EXIT CHECK] Ollama Decision: {decision_val} (Reason: {reason_val})")
            return decision_val == "EXIT"
        except Exception as e:
            print(f"[ERROR] Ollama exit check failed: {e}. Defaulting to HOLD (keeping position).")
            return False


def format_candles_for_ai(df, last_n=20):
    """
    Format a candles DataFrame into a readable string for the AI.

    Args:
        df: pandas DataFrame from get_candles()
        last_n: Number of recent candles to include

    Returns:
        str: Formatted candle data
    """
    try:
        from db_manager import get_settings
        import os
        ai_prov = os.getenv("AI_PROVIDER", get_settings().get("ai_provider", "gemini")).lower()
        if ai_prov == "ollama":
            last_n = min(last_n, 8)
    except Exception:
        pass

    recent = df.tail(last_n)
    lines = ["Time | Open | High | Low | Close | Volume"]
    lines.append("-" * 60)
    for _, row in recent.iterrows():
        lines.append(
            f"{row['Time']} | {row['Open']:.5f} | {row['High']:.5f} | "
            f"{row['Low']:.5f} | {row['Close']:.5f} | {row['Volume']}"
        )
    return "\n".join(lines)
