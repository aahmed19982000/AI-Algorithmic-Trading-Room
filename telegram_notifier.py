import os
import requests
from db_manager import get_settings

def get_telegram_config():
    """Load Telegram config settings from SQLite db."""
    try:
        settings = get_settings()
        enabled = int(settings.get("telegram_enabled", 0)) == 1
        token = settings.get("telegram_token", "")
        chat_id = settings.get("telegram_chat_id", "")
        return enabled, token, chat_id
    except Exception as e:
        print(f"[TELEGRAM] Error loading configuration: {e}")
        return False, "", ""

def send_telegram_photo(token, chat_id, photo_path, caption):
    """Sends a photo with a Markdown caption to Telegram."""
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        if not os.path.exists(photo_path):
            print(f"[TELEGRAM] Photo path does not exist: {photo_path}")
            return None
            
        with open(photo_path, 'rb') as photo_file:
            files = {'photo': photo_file}
            data = {
                'chat_id': chat_id, 
                'caption': caption, 
                'parse_mode': 'Markdown'
            }
            res = requests.post(url, files=files, data=data, timeout=15)
            if res.status_code == 200:
                return res.json()
            else:
                print(f"[TELEGRAM] Failed to send photo. Status: {res.status_code}, Body: {res.text}")
                return None
    except Exception as e:
        print(f"[TELEGRAM] Exception in send_telegram_photo: {e}")
        return None

def send_telegram_message(token, chat_id, text):
    """Sends a text message to Telegram."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        data = {
            'chat_id': chat_id, 
            'text': text, 
            'parse_mode': 'Markdown'
        }
        res = requests.post(url, data=data, timeout=15)
        if res.status_code == 200:
            return res.json()
        else:
            print(f"[TELEGRAM] Failed to send message. Status: {res.status_code}, Body: {res.text}")
            return None
    except Exception as e:
        print(f"[TELEGRAM] Exception in send_telegram_message: {e}")
        return None

def edit_telegram_message_caption(token, chat_id, message_id, caption):
    """Edits the caption of an existing Telegram photo message."""
    url = f"https://api.telegram.org/bot{token}/editMessageCaption"
    try:
        data = {
            'chat_id': chat_id, 
            'message_id': int(message_id), 
            'caption': caption, 
            'parse_mode': 'Markdown'
        }
        res = requests.post(url, data=data, timeout=15)
        if res.status_code == 200:
            return res.json()
        else:
            print(f"[TELEGRAM] Failed to edit caption. Status: {res.status_code}, Body: {res.text}")
            return None
    except Exception as e:
        print(f"[TELEGRAM] Exception in edit_telegram_message_caption: {e}")
        return None

def notify_trade_open(ticket, symbol, action, volume, entry_price, sl, tp, reason, screenshot_url=None):
    """Sends an alert to Telegram that a new trade has been opened."""
    enabled, token, chat_id = get_telegram_config()
    if not enabled or not token or not chat_id:
        print("[TELEGRAM] Notifications are disabled or credentials missing.")
        return None
        
    emoji = "🟢" if action.upper() == "BUY" else "🔴"
    caption = (
        f"🔔 *TRADE OPENED* {emoji}\n\n"
        f"• *Ticket:* `#{ticket}`\n"
        f"• *Symbol:* `{symbol}`\n"
        f"• *Action:* `{action.upper()}`\n"
        f"• *Volume:* `{volume:.2f} Lots`\n"
        f"• *Entry Price:* `{entry_price:.5f}`\n"
        f"• *Target Profit (TP):* `{tp:.5f}`\n"
        f"• *Stop Loss (SL):* `{sl:.5f}`\n"
        f"• *Reason:* `{reason}`"
    )
    
    # Resolve local screenshot file path
    photo_path = None
    if screenshot_url:
        photo_path = screenshot_url.lstrip("/")
        if not os.path.exists(photo_path):
            photo_path = os.path.join(os.getcwd(), 'static', 'screenshots', f"{ticket}.png")
            
    # Send photo if path is valid
    if photo_path and os.path.exists(photo_path):
        res_json = send_telegram_photo(token, chat_id, photo_path, caption)
        if res_json and res_json.get("ok"):
            msg_id = res_json["result"]["message_id"]
            return msg_id
            
    # Fallback to plain text message
    res_json = send_telegram_message(token, chat_id, caption)
    if res_json and res_json.get("ok"):
        return res_json["result"]["message_id"]
    return None

def notify_trade_close(ticket, symbol, action, volume, entry_price, close_price, profit, result, open_time, close_time, screenshot_url=None, original_msg_id=None):
    """Sends an alert that a trade has been closed, and updates the original message caption."""
    enabled, token, chat_id = get_telegram_config()
    if not enabled or not token or not chat_id:
        return None
        
    emoji = "🏆 WIN" if result == "WIN" else "❌ LOSS"
    profit_emoji = "💵" if profit >= 0 else "📉"
    
    caption = (
        f"🏁 *TRADE CLOSED* ({emoji})\n\n"
        f"• *Ticket:* `#{ticket}`\n"
        f"• *Symbol:* `{symbol}`\n"
        f"• *Action:* `{action.upper()}`\n"
        f"• *Volume:* `{volume:.2f} Lots`\n"
        f"• *Entry Price:* `{entry_price:.5f}`\n"
        f"• *Close Price:* `{close_price:.5f}`\n"
        f"• *Profit/Loss:* `{profit_emoji} ${profit:+.2f} USD`\n"
        f"• *Open Time:* `{open_time}`\n"
        f"• *Close Time:* `{close_time}`"
    )
    
    # Resolve local screenshot file path
    photo_path = None
    if screenshot_url:
        photo_path = screenshot_url.lstrip("/")
        if not os.path.exists(photo_path):
            photo_path = os.path.join(os.getcwd(), 'static', 'screenshots', f"{ticket}.png")
            
    # Edit original message caption to mark it CLOSED
    if original_msg_id:
        try:
            original_emoji = "🟢" if action.upper() == "BUY" else "🔴"
            updated_orig_caption = (
                f"📦 *[CLOSED - {result}] TRADE OPENED* {original_emoji}\n\n"
                f"• *Ticket:* `#{ticket}`\n"
                f"• *Symbol:* `{symbol}`\n"
                f"• *Action:* `{action.upper()}`\n"
                f"• *Volume:* `{volume:.2f} Lots`\n"
                f"• *Entry Price:* `{entry_price:.5f}`\n"
                f"• *Close Price:* `{close_price:.5f}`\n"
                f"• *Result:* `*{emoji} (${profit:+.2f})*`"
            )
            edit_telegram_message_caption(token, chat_id, original_msg_id, updated_orig_caption)
        except Exception as edit_ex:
            print(f"[TELEGRAM] Failed to edit original open caption: {edit_ex}")
        
    # Send the new exit message containing the exit screenshot
    if photo_path and os.path.exists(photo_path):
        res_json = send_telegram_photo(token, chat_id, photo_path, caption)
        if res_json and res_json.get("ok"):
            return res_json["result"]["message_id"]
            
    # Fallback to plain text
    res_json = send_telegram_message(token, chat_id, caption)
    if res_json and res_json.get("ok"):
        return res_json["result"]["message_id"]
    return None
