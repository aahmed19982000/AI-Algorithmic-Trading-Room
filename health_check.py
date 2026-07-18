"""
Health Check
============
Verifies the trading bot's full setup end-to-end: MT5 connection, database
settings, Ollama connectivity, Telegram alerts, market-status detection, and
the currently deployed code version.

Run with: python health_check.py  (or venv\\Scripts\\python.exe health_check.py)
"""
import sys


def check(name, fn):
    try:
        result = fn()
        print(f"[PASS] {name}: {result}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False


def mt5_check():
    from mt5_connection import connect_mt5
    import MetaTrader5 as mt5
    if not connect_mt5():
        raise Exception("connect_mt5() returned False")
    info = mt5.account_info()
    if info is None:
        raise Exception("account_info() returned None")
    return f"Login {info.login}, Balance {info.balance} {info.currency}, Server {info.server}, Type {'Real' if info.trade_mode == 0 else 'Demo'}"


def settings_check():
    from db_manager import get_settings
    s = get_settings()
    symbols = s.get("symbols", [])
    return (f"symbols={len(symbols)}, ai_provider={s.get('ai_provider')}, "
            f"ai_evaluation={s.get('ai_evaluation')}, auto_trade={s.get('auto_trade')}, "
            f"grid_enabled={s.get('grid_enabled')}, ai_cost_cap_daily={s.get('ai_cost_cap_daily')}")


def gemini_check():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY not set in .env")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content("Reply with exactly one word: OK")
    return f"API reachable, response={response.text.strip()!r}"


def ollama_check():
    import requests
    import os
    from db_manager import get_settings
    s = get_settings()
    url = os.getenv("OLLAMA_URL", s.get("ollama_url", "http://localhost:11434")).strip("/")
    r = requests.get(f"{url}/api/tags", timeout=5)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    return f"{url} reachable, models={models}"


def telegram_check():
    from telegram_notifier import get_telegram_config, send_telegram_message
    enabled, token, chat_id = get_telegram_config()
    if not (enabled and token and chat_id):
        raise Exception(f"not configured (enabled={enabled}, token_set={bool(token)}, chat_id={chat_id!r})")
    result = send_telegram_message(token, chat_id, "✅ *Health Check*: test message from the server")
    if result is None:
        raise Exception("send_telegram_message failed (see [TELEGRAM] log line above for the reason)")
    return "test message sent successfully"


def market_status_check():
    from trading_bot import is_market_closed
    closed = is_market_closed()
    return f"market_closed={closed}"


def git_version_check():
    import subprocess
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    return f"current commit: {commit}"


def main():
    print("=" * 60)
    print("  TRADING BOT HEALTH CHECK")
    print("=" * 60)

    results = [
        check("MT5 Connection", mt5_check),
        check("Database Settings", settings_check),
        check("Gemini Connectivity", gemini_check),
        check("Ollama Connectivity", ollama_check),
        check("Telegram", telegram_check),
        check("Market Status Detection", market_status_check),
        check("Git Version", git_version_check),
    ]

    print("\n" + "=" * 60)
    passed = sum(results)
    print(f"  SUMMARY: {passed}/{len(results)} checks passed")
    print("=" * 60)

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
