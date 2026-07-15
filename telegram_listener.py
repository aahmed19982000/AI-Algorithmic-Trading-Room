"""
Telegram Listener Daemon
========================
Listens for incoming messages from the configured Telegram Chat ID.
Supports commands (/status, /trades, /analyze <symbol>, /help) and handles natural 
language conversations with the AI Trading Analyst, maintaining context of recent trades.
"""

import os
import time
import json
import threading
import traceback
import requests
import MetaTrader5 as mt5
import google.generativeai as genai

from mt5_connection import connect_mt5
from mt5_data import get_candles, get_current_price, get_account_info, get_timeframe
from mt5_orders import get_open_positions
from ai_engine import AITradingEngine, format_candles_for_ai
from gann_helper import detect_market_structure, detect_dynamic_gann_levels
from db_manager import get_settings, get_db_connection, get_trade_history

from telegram_notifier import send_telegram_photo

# Thread controller
listener_running = True
listener_thread = None

# In-memory chat history dictionary mapping chat_id -> list of exchanges
# format: [{"role": "user"|"model", "text": "..."}]
chat_histories = {}

def get_telegram_credentials():
    """Retrieve current Telegram settings from DB."""
    try:
        settings = get_settings()
        enabled = int(settings.get("telegram_enabled", 0)) == 1
        token = settings.get("telegram_token", "")
        chat_id = str(settings.get("telegram_chat_id", ""))
        return enabled, token, chat_id
    except Exception as e:
        print(f"[TG LISTENER] Error reading settings from DB: {e}")
        return False, "", ""

def send_telegram_message(token, chat_id, text):
    """Utility function to send markdown messages back to user."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        data = {
            'chat_id': chat_id, 
            'text': text, 
            'parse_mode': 'Markdown'
        }
        res = requests.post(url, data=data, timeout=15)
        if res.status_code != 200:
            print(f"[TG LISTENER] Failed to send message. Status: {res.status_code}, Body: {res.text}")
        return res.status_code == 200
    except Exception as e:
        print(f"[TG LISTENER] Exception sending message: {e}")
        return False

def send_chat_action(token, chat_id, action="typing"):
    """Send chat action status (e.g. typing, upload_photo) to Telegram user."""
    url = f"https://api.telegram.org/bot{token}/sendChatAction"
    try:
        data = {
            'chat_id': chat_id,
            'action': action
        }
        requests.post(url, json=data, timeout=5)
    except Exception as e:
        print(f"[TG LISTENER] Error sending chat action: {e}")

def build_trades_context():
    """Retrieve recent 20 trades from database and format as a context string for Gemini."""
    try:
        trades = get_trade_history(limit=20)
        if not trades:
            return "No recent trades executed yet."

        context = "RECENT TRADES LIST:\n"
        for t in trades:
            gann_data_str = ""
            if t.get("gann_data"):
                try:
                    gd = json.loads(t["gann_data"])
                    gann_data_str = f", Gann: Type={gd.get('type')}, Pivots (A={gd.get('A')}, B={gd.get('B')}, C={gd.get('C')}), Pullback={gd.get('retracement_pct', 0):.1f}%, TargetAngle={gd.get('target_angle')}°"
                except Exception:
                    pass
            context += (
                f"- Ticket #{t['ticket']}, Symbol: {t['symbol']}, Action: {t['action']}, "
                f"Volume: {t['volume']} lots, Entry Price: {t['entry_price']}, "
                f"Close Price: {t['close_price'] or 'N/A'}, TP: {t['tp']}, SL: {t['sl']}, "
                f"Profit: ${t['profit']:.2f} USD, Open: {t['open_time']}, Close: {t['close_time'] or 'OPEN'}, "
                f"Reason: {t['reason']}{gann_data_str}\n"
            )
        return context
    except Exception as e:
        print(f"[TG LISTENER] Error building trades context: {e}")
        return "Error loading recent trades."

def query_gemini_analyst(user_msg, chat_id):
    """Sends the message to the configured AI provider (Gemini or Ollama) with conversation history and recent trades context."""
    try:
        from db_manager import get_settings
        settings = get_settings()
        ai_provider = os.getenv("AI_PROVIDER", settings.get("ai_provider", "gemini")).lower()
        ollama_model = os.getenv("OLLAMA_MODEL", settings.get("ollama_model", "phi3")).lower()

        # Load API Key if using Gemini
        if ai_provider == "gemini":
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                return "❌ *Error:* `GEMINI_API_KEY` is not set in the server environment variables."

        # Get recent trades context
        trades_context = build_trades_context()

        # Build System Instructions
        system_instruction = (
            "You are a professional AI Trading Analyst Coach. The user is trading Forex and other assets using an MT5 bot powered by a Gann geometry strategy combined with a local AI analysis filter.\n"
            "Below is the history of the 20 most recent trades executed by the bot. Use this history to answer all user queries, analyze specific trades, and explain why trades were opened/closed:\n\n"
            f"{trades_context}\n\n"
            "Your instructions:\n"
            "1. Answer the user's questions about these trades, active positions, and market operations.\n"
            "2. Explain why trades were opened (referencing their reasons, Gann pivots A/B/C, Fibonacci retracement %, and angles) and why they were closed (e.g. hitting TP, SL, or Martingale grid averaging).\n"
            "3. If a user asks about a specific trade, locate it in the list above using its ticket number, symbol, or open time.\n"
            "4. Provide constructive feedback, stats (e.g. win rate, total profit/loss), and trade breakdown.\n"
            "5. ALWAYS respond in the user's language (if they ask in Arabic, respond in Arabic; if English, respond in English). Keep your answers concise, clear, and professional. Use Telegram markdown formatting where appropriate."
        )

        if ai_provider == "ollama":
            import requests
            ollama_url = settings.get("ollama_url", "http://localhost:11434").strip("/")
            
            # Prepare chat history for Ollama format
            history = chat_histories.get(chat_id, [])
            messages = [{"role": "system", "content": system_instruction}]
            for h in history:
                role_name = "assistant" if h.get("role") in ["assistant", "model"] else "user"
                messages.append({"role": role_name, "content": h.get("text", "")})
            messages.append({"role": "user", "content": user_msg})
            
            payload = {
                "model": ollama_model,
                "messages": messages,
                "stream": False
            }
            
            r = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=60)
            r.raise_for_status()
            res_json = r.json()
            response_text = res_json.get("message", {}).get("content", "Error: Empty response from local AI.")
            
            # Save exchange to memory
            if chat_id not in chat_histories:
                chat_histories[chat_id] = []
            chat_histories[chat_id].append({"role": "user", "text": user_msg})
            chat_histories[chat_id].append({"role": "assistant", "text": response_text})
            
            # Keep history bounded to last 10 exchanges (20 messages)
            if len(chat_histories[chat_id]) > 20:
                chat_histories[chat_id] = chat_histories[chat_id][-20:]
                
            return response_text

        else:
            # Initialize Gemini API
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=system_instruction
            )

            # Prepare chat history for Gemini API format
            history = chat_histories.get(chat_id, [])
            gemini_history = []
            for h in history:
                role = h.get("role", "user")
                text = h.get("text", "")
                gemini_history.append({
                    "role": "user" if role == "user" else "model",
                    "parts": [text]
                })

            # Start Chat and send message
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(user_msg)

            # Save exchange to memory
            if chat_id not in chat_histories:
                chat_histories[chat_id] = []
            chat_histories[chat_id].append({"role": "user", "text": user_msg})
            chat_histories[chat_id].append({"role": "model", "text": response.text})

            # Keep history bounded to last 10 exchanges (20 messages)
            if len(chat_histories[chat_id]) > 20:
                chat_histories[chat_id] = chat_histories[chat_id][-20:]

            return response.text
    except Exception as e:
        print(f"[TG LISTENER] Error in chat API: {e}")
        traceback.print_exc()
        return "❌ *حدث خطأ أثناء الاتصال بالمحلل الفني:* \n`" + str(e) + "`"

def handle_status_command(token, chat_id):
    """Formulate and send the account status report."""
    try:
        connect_mt5()
        info = get_account_info()
        if not info:
            send_telegram_message(token, chat_id, "❌ *Error:* Could not fetch account info from MetaTrader 5.")
            return

        # Fetch open positions
        positions = get_open_positions()
        pos_list_str = ""
        
        if not positions:
            pos_list_str = "• _No active open positions._"
        else:
            for p in positions:
                action_str = "🟢 BUY" if p.type == 0 else "🔴 SELL"
                pos_list_str += (
                    f"• *Ticket #{p.ticket}* | `{p.symbol}`\n"
                    f"  Type: {action_str} | Volume: `{p.volume:.2f}` Lots\n"
                    f"  Entry: `{p.price_open:.5f}` ➔ Current: `{p.price_current:.5f}`\n"
                    f"  Profit: *${p.profit:+.2f} USD*\n\n"
                )

        # Fetch config settings to show AI and trade mode
        ai_eval_status = "ENABLED"
        auto_trade_status = "ENABLED"
        try:
            settings = get_settings()
            ai_eval_enabled = int(settings.get("ai_evaluation", 1)) == 1
            auto_trade_enabled = int(settings.get("auto_trade", 1)) == 1
            ai_provider_name = "Ollama" if settings.get("ai_provider", "gemini").lower() == "ollama" else "Gemini"
            ai_eval_status = f"🟢 ACTIVE ({ai_provider_name})" if ai_eval_enabled else "🔴 DISABLED"
            auto_trade_status = "🟢 ACTIVE" if auto_trade_enabled else "🔴 STOPPED (Hold)"
        except Exception:
            pass

        status_msg = (
            f"📊 *SYSTEM STATUS REPORT*\n\n"
            f"• *Account:* `{info['name']}` (`#{info['login']}`)\n"
            f"• *Server:* `{info['server']}`\n"
            f"• *Balance:* `${info['balance']:,.2f} {info['currency']}`\n"
            f"• *Equity:* `${info['equity']:,.2f} {info['currency']}`\n"
            f"• *Free Margin:* `${info['margin_free']:,.2f} {info['currency']}`\n"
            f"• *Margin Level:* `{info['margin_level']:.1f}%`\n"
            f"• *Leverage:* `1:{info['leverage']}`\n"
            f"• *Trade Mode:* `{info['trade_mode']}`\n"
            f"• *Auto-Trading:* {auto_trade_status}\n"
            f"• *AI Analysis:* {ai_eval_status}\n\n"
            f"💼 *ACTIVE OPEN POSITIONS:*\n"
            f"{pos_list_str}"
        )
        send_telegram_message(token, chat_id, status_msg)
    except Exception as e:
        send_telegram_message(token, chat_id, f"❌ *Error fetching status:* `{e}`")

def handle_trades_command(token, chat_id):
    """Format and send the last 5 executed trades."""
    try:
        trades = get_trade_history(limit=5)
        if not trades:
            send_telegram_message(token, chat_id, "📜 *Trades History:* No executed trades logged in the database yet.")
            return

        msg = "📜 *RECENT TRADES HISTORY* (Last 5 Trades)\n\n"
        for t in trades:
            emoji = "🟢" if t['action'].upper() == "BUY" else "🔴"
            status = "OPEN" if not t['close_time'] else "CLOSED"
            close_info = f"➔ Close: `{t['close_price']:.5f}`" if t['close_price'] else ""
            
            msg += (
                f"{emoji} *Ticket #{t['ticket']}* | `{t['symbol']}`\n"
                f"  *Action:* `{t['action'].upper()}` | Volume: `{t['volume']:.2f} Lots`\n"
                f"  *Prices:* Entry `{t['entry_price']:.5f}` {close_info}\n"
                f"  *Profit:* *${t['profit']:+.2f} USD* ({status})\n"
                f"  *Reason:* `{t['reason']}`\n"
                f"  *Time:* Open: `{t['open_time']}`\n\n"
            )
        send_telegram_message(token, chat_id, msg)
    except Exception as e:
        send_telegram_message(token, chat_id, f"❌ *Error fetching trades:* `{e}`")

def resolve_mt5_symbol(symbol_name):
    """Resolve symbol name case-insensitively from available MT5 symbols."""
    try:
        connect_mt5()
        all_symbols = mt5.symbols_get()
        if not all_symbols:
            # Fallback to direct check if symbols_get fails
            if mt5.symbol_info(symbol_name):
                return symbol_name
            return None
            
        symbol_name_lower = symbol_name.strip().lower()
        
        # 1. Direct case-insensitive match
        for sym in all_symbols:
            if sym.name.lower() == symbol_name_lower:
                return sym.name
                
        # 2. Try adding 'm' suffix if not ending with it
        if not symbol_name_lower.endswith("m"):
            with_m = symbol_name_lower + "m"
            for sym in all_symbols:
                if sym.name.lower() == with_m:
                    return sym.name
    except Exception as e:
        print(f"[TG LISTENER] Error resolving symbol: {e}")
    return None

def handle_analyze_command(token, chat_id, symbol_arg):
    """Perform on-demand AI analysis for a specified symbol."""
    if not symbol_arg:
        send_telegram_message(token, chat_id, "⚠️ *الرجاء إدخال اسم الزوج لتحليله.*\nمثال: `/analyze EURUSDm` أو `/analyze BTCUSDm`")
        return

    # Resolve symbol case-insensitively using our helper
    symbol = resolve_mt5_symbol(symbol_arg)
    if not symbol:
        send_telegram_message(token, chat_id, f"❌ *Error:* Symbol `{symbol_arg}` was not found in MT5 Terminal.")
        return

    send_telegram_message(token, chat_id, f"🔍 *جاري تحليل الزوج {symbol}...* الرجاء الانتظار.")

    try:
        connect_mt5()
        
        # Verify symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            send_telegram_message(token, chat_id, f"❌ *Error:* Symbol `{symbol}` was not found in MT5 Terminal.")
            return

        # Fetch settings
        settings = get_settings()
        gann_lookback = int(settings.get("gann_lookback", 100))
        gann_enabled = int(settings.get("gann_enabled", 1)) == 1

        # Fetch candles
        tf_constant = get_timeframe(os.getenv("TRADING_TIMEFRAME", "M30"))
        candles_df = get_candles(symbol=symbol, timeframe=tf_constant, count=gann_lookback + 10)
        
        if candles_df is None or len(candles_df) < 10:
            send_telegram_message(token, chat_id, f"❌ *Error:* Failed to download candles data for `{symbol}`.")
            return

        price_info = get_current_price(symbol)
        candles_text = format_candles_for_ai(candles_df, last_n=20)

        # Detect setup
        gann_context = None
        if gann_enabled:
            setup = detect_market_structure(candles_df, lookback=gann_lookback)
            if setup:
                retr_pct = ((setup["B"] - setup["C"]) / (setup["B"] - setup["A"])) * 100.0 if setup["type"] == "BUY" else ((setup["C"] - setup["B"]) / (setup["A"] - setup["B"])) * 100.0
                dyn = detect_dynamic_gann_levels(setup["A"], setup["B"], mode='bullish' if setup["type"] == 'BUY' else 'bearish')
                gann_context = {
                    "type": setup["type"],
                    "A": setup["A"],
                    "B": setup["B"],
                    "C": setup["C"],
                    "retracement_pct": retr_pct,
                    "entry_angle": dyn["entry_angle"],
                    "target_angle": dyn["target_angle"],
                    "target_price": dyn["target_price"],
                    "sl_price": dyn["sl_price"]
                }

        # Query Gemini via Engine
        engine = AITradingEngine(model_name="gemini-2.5-flash")
        account_info = get_account_info()
        active_positions = get_open_positions()

        # Send typing action right before AI runs
        send_chat_action(token, chat_id, "typing")

        ai_result = engine.analyze_market(
            symbol=symbol,
            timeframe=os.getenv("TRADING_TIMEFRAME", "M30"),
            candles_data=candles_text,
            account_info=account_info,
            current_price=price_info,
            gann_context=gann_context,
            active_positions=active_positions
        )

        decision = ai_result.get("decision", "HOLD")
        analysis = ai_result.get("analysis", "No explanation provided.")
        trade_params = ai_result.get("trade_params", {})

        # Generate chart screenshot in background and prepare files
        photo_path = None
        try:
            send_chat_action(token, chat_id, "upload_photo")
            from chart_generator import generate_chart_screenshot
            
            # Prepare parameters for the chart generator
            ticket_id = f"manual_{symbol}_{int(time.time())}"
            entry_val = price_info['ask'] if decision == "BUY" else price_info['bid']
            
            # Set default stop/target levels for chart lines
            sl_val = entry_val
            tp_val = entry_val
            if decision in ["BUY", "SELL"] and trade_params:
                sl_val = float(trade_params.get("sl", entry_val))
                tp_val = float(trade_params.get("tp", entry_val))
            elif gann_context:
                sl_val = gann_context["sl_price"]
                tp_val = gann_context["target_price"]
                
            generate_chart_screenshot(
                symbol=symbol,
                ticket=ticket_id,
                entry_price=entry_val,
                sl_price=sl_val,
                tp_price=tp_val,
                gann_data=gann_context,
                timeframe_str=os.getenv("TRADING_TIMEFRAME", "M30")
            )
            
            photo_path = os.path.join(os.getcwd(), 'static', 'screenshots', f"{ticket_id}.png")
        except Exception as sc_err:
            print(f"[TG LISTENER] Error generating chart screenshot: {sc_err}")
            traceback.print_exc()

        result_msg = (
            f"📊 *AI ANALYSIS REPORT FOR {symbol}*\n\n"
            f"• *Decision:* `{decision}`\n"
        )
        if decision in ["BUY", "SELL"] and trade_params:
            result_msg += (
                f"• *Entry:* `{price_info['ask'] if decision == 'BUY' else price_info['bid']:.5f}`\n"
                f"• *Stop Loss (SL):* `{trade_params.get('sl', 0.0):.5f}`\n"
                f"• *Take Profit (TP):* `{trade_params.get('tp', 0.0):.5f}`\n\n"
            )
        else:
            result_msg += "• *Action:* `HOLD / NO TRADE`\n\n"

        result_msg += f"📝 *AI Explanation & Reasoning:*\n{analysis}"
        
        # Bounded response length for Telegram limit (4096 characters)
        if len(result_msg) > 4000:
            result_msg = result_msg[:3900] + "\n\n...(truncated due to length)"
            
        # 1. Send the chart photo if successfully generated
        if photo_path and os.path.exists(photo_path):
            caption_title = f"📊 *AI Analysis Chart & Gann levels for {symbol}*"
            send_telegram_photo(token, chat_id, photo_path, caption=caption_title)
            
        # 2. Send the detailed text analysis message
        send_telegram_message(token, chat_id, result_msg)

    except Exception as e:
        traceback.print_exc()
        send_telegram_message(token, chat_id, f"❌ *Error during manual analysis:* `{e}`")

def telegram_listener_loop():
    """Main daemon polling loop."""
    print("[TG LISTENER] Thread started.")
    
    # Wait 7 seconds for system to fully boot
    time.sleep(7)
    
    last_update_id = -1
    loop_start_time = time.time()

    # 1. Start polling
    while listener_running:
        enabled, token, configured_chat_id = get_telegram_credentials()
        if not enabled or not token or not configured_chat_id:
            # Sleep and check again if credentials are set
            time.sleep(10)
            continue
            
        try:
            # Long-poll getUpdates
            url = f"https://api.telegram.org/bot{token}/getUpdates"
            params = {"timeout": 30}
            if last_update_id != -1:
                params["offset"] = last_update_id + 1
                
            res = requests.get(url, params=params, timeout=35)
            if res.status_code != 200:
                time.sleep(5)
                continue
                
            updates = res.json().get("result", [])
            for update in updates:
                last_update_id = update["update_id"]
                
                message = update.get("message")
                if not message:
                    continue
                    
                sender_chat_id = str(message.get("chat", {}).get("id", ""))
                
                # SECURITY RULE: Only respond to the configured Telegram Chat ID
                if sender_chat_id != configured_chat_id:
                    print(f"[TG LISTENER] Unauthorized message from chat {sender_chat_id}. Ignored.")
                    # Optionally send a single unauthorized message
                    send_telegram_message(token, sender_chat_id, "⚠️ *Unauthorized Access:* This bot is locked to a specific administrator account.")
                    continue
                
                # Filter out messages sent more than 5 minutes before the bot started
                msg_date = message.get("date", 0)
                if msg_date < loop_start_time - 300:
                    print(f"[TG LISTENER] Ignoring old message (sent {loop_start_time - msg_date:.0f}s before startup).")
                    continue
                    
                user_msg = message.get("text", "").strip()
                if not user_msg:
                    continue
                
                print(f"[TG LISTENER] Authorized command/message received: '{user_msg}'")
                
                # Show typing action immediately on receipt of any command/message
                send_chat_action(token, configured_chat_id, "typing")
                
                user_msg_lower = user_msg.lower()
                
                # Process commands
                if (user_msg_lower.startswith("/start") and not user_msg_lower.startswith("/start_trade")) or user_msg_lower.startswith("/help"):
                    help_msg = (
                        "🚀 *مرحباً بك في المحلل الفني للذكاء الاصطناعي!* 📊\n\n"
                        "يمكنك مراقبة البوت والتحكم فيه والدردشة مع المحلل مباشرة عبر الأوامر التالية:\n\n"
                        "• `/status` - لعرض الرصيد والصفقات المفتوحة حالياً.\n"
                        "• `/trades` - لعرض آخر 5 صفقات منفذة وتفاصيل أرباحها.\n"
                        "• `/analyze <الرمز>` - لإجراء تحليل فني مباشر وفوري لأي زوج عملات أو رمز مالي.\n"
                        "• `/stop` - لإيقاف التداول التلقائي فوراً (تعطيل الصفقات الجديدة).\n"
                        "• `/start_trade` - لتفعيل التداول التلقائي مجدداً.\n"
                        "• `/toggle_ai` - لتفعيل/تعطيل تقييم الصفقات بواسطة الذكاء الاصطناعي (Gemini).\n"
                        "• `/toggle_gann` - لتفعيل/تعطيل إستراتيجية جان الفنية البرمجية.\n\n"
                        "• *أي رسالة نصية أخرى* - سيتم اعتبارها دردشة وسيجيبك المحلل الفني بالاعتماد على سياق تداولاتك الحالية والتاريخية وبشكل ذكي.\n"
                    )
                    send_telegram_message(token, configured_chat_id, help_msg)
                    
                elif user_msg_lower.startswith("/status"):
                    handle_status_command(token, configured_chat_id)
                    
                elif user_msg_lower.startswith("/trades"):
                    handle_trades_command(token, configured_chat_id)
                    
                elif user_msg_lower.startswith("/analyze"):
                    # Split command and arguments
                    parts = user_msg.split(maxsplit=1)
                    symbol_arg = parts[1] if len(parts) > 1 else ""
                    handle_analyze_command(token, configured_chat_id, symbol_arg)
                    
                elif user_msg_lower.startswith("/stop"):
                    try:
                        from db_manager import save_settings
                        save_settings({"auto_trade": "0"})
                        send_telegram_message(token, configured_chat_id, "⏹️ *تم إيقاف التداول التلقائي بنجاح.* لن يفتح البوت أي صفقات جديدة حتى تقوم بالتفعيل مجدداً.")
                    except Exception as e:
                        send_telegram_message(token, configured_chat_id, f"❌ *فشل إيقاف التداول:* `{str(e)}`")

                elif user_msg_lower.startswith("/start_trade"):
                    try:
                        from db_manager import save_settings
                        save_settings({"auto_trade": "1"})
                        send_telegram_message(token, configured_chat_id, "🚀 *تم تفعيل التداول التلقائي بنجاح!* سيبدأ البوت بالبحث عن صفقات جديدة وتنفيذها تلقائياً.")
                    except Exception as e:
                        send_telegram_message(token, configured_chat_id, f"❌ *فشل تفعيل التداول:* `{str(e)}`")

                elif user_msg_lower.startswith("/toggle_ai"):
                    try:
                        from db_manager import get_settings, save_settings
                        curr_settings = get_settings()
                        curr_ai = int(curr_settings.get("ai_evaluation", 1))
                        new_ai = "0" if curr_ai == 1 else "1"
                        save_settings({"ai_evaluation": new_ai})
                        status_str = "تعطيل 🛑" if new_ai == "0" else "تفعيل 🚀"
                        send_telegram_message(token, configured_chat_id, f"⚙️ *تحديث الإعدادات:*\nتم *{status_str}* فحص وتقييم الصفقات بواسطة الذكاء الاصطناعي (Gemini).")
                    except Exception as e:
                        send_telegram_message(token, configured_chat_id, f"❌ *فشل تحديث الإعدادات:* `{str(e)}`")

                elif user_msg_lower.startswith("/toggle_gann"):
                    try:
                        from db_manager import get_settings, save_settings
                        curr_settings = get_settings()
                        curr_gann = int(curr_settings.get("gann_enabled", 1))
                        new_gann = "0" if curr_gann == 1 else "1"
                        save_settings({"gann_enabled": new_gann})
                        status_str = "تعطيل 🛑 (العودة للمتوسطات الفنية)" if new_gann == "0" else "تفعيل 🚀"
                        send_telegram_message(token, configured_chat_id, f"⚙️ *تحديث الإعدادات:*\nتم *{status_str}* إستراتيجية زوايا جان وفيبوناتشي.")
                    except Exception as e:
                        send_telegram_message(token, configured_chat_id, f"❌ *فشل تحديث الإعدادات:* `{str(e)}`")
                    
                else:
                    # Check if the user is asking to analyze a symbol in natural language
                    # e.g., "حلل لي EURUSDm" or "تحليل زوج BTCUSD"
                    is_analysis_req = any(w in user_msg_lower for w in ["تحليل", "حلل", "analyze", "analyse", "chart", "شارت"])
                    
                    if is_analysis_req:
                        # 1. Try common mappings
                        mappings = {
                            "gold": "XAUUSDm", "الذهب": "XAUUSDm",
                            "btc": "BTCUSDm", "bitcoin": "BTCUSDm", "بيتكوين": "BTCUSDm", "البيتكوين": "BTCUSDm",
                            "eth": "ETHUSDm", "ethereum": "ETHUSDm", "إيثيريوم": "ETHUSDm"
                        }
                        extracted_symbol = None
                        for kw, sym_val in mappings.items():
                            if kw in user_msg_lower:
                                extracted_symbol = sym_val
                                break
                                
                        # 2. If no mapping matched, try regex for currency patterns
                        if not extracted_symbol:
                            import re
                            symbol_match = re.search(r'\b([a-zA-Z]{3,4}USDm?|[a-zA-Z]{6}m?)\b', user_msg)
                            if symbol_match:
                                extracted_symbol = symbol_match.group(1).upper()
                                
                        # 3. If we found a candidate, verify and resolve it with MT5
                        if extracted_symbol:
                            resolved_symbol = resolve_mt5_symbol(extracted_symbol)
                            if resolved_symbol:
                                print(f"[TG LISTENER] Auto-detected natural language analysis request for symbol: {resolved_symbol}")
                                handle_analyze_command(token, configured_chat_id, resolved_symbol)
                                continue
                            
                    # Treat as natural language dialogue with AI Analyst Coach
                    response_text = query_gemini_analyst(user_msg, configured_chat_id)
                    
                    # Bounded length check
                    if len(response_text) > 4000:
                        response_text = response_text[:3900] + "\n\n...(truncated)"
                    send_telegram_message(token, configured_chat_id, response_text)
                    
        except requests.exceptions.RequestException as req_err:
            print(f"[TG LISTENER] Connection warning: Telegram API request timed out or dropped ({req_err}). Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as poll_err:
            print(f"[TG LISTENER] [ERROR] Polling loop exception: {poll_err}")
            traceback.print_exc()
            time.sleep(5)
            
    print("[TG LISTENER] Thread stopped.")

def start_telegram_listener():
    """Spawn the listener thread."""
    global listener_thread, listener_running
    listener_running = True
    listener_thread = threading.Thread(target=telegram_listener_loop)
    listener_thread.daemon = True
    listener_thread.start()
    print("[TG LISTENER] Started background thread daemon successfully.")

def stop_telegram_listener():
    """Signals the listener thread to stop."""
    global listener_running
    listener_running = False
    print("[TG LISTENER] Sent stop signal to polling thread.")
