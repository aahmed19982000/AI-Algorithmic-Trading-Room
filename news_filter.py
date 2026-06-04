"""
Economic News Filter Module
===========================
Fetches live macroeconomic events from TradingView's calendar API and checks
if there are upcoming high/medium impact events in major currencies (USD, EUR, GBP)
to temporarily freeze algorithmic trading (e.g., 60 minutes before and 30 minutes after).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import requests
from datetime import datetime, timezone
import json

CALENDAR_URL = "https://economic-calendar.tradingview.com/events"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Origin': 'https://www.tradingview.com',
    'Referer': 'https://www.tradingview.com/'
}

# Major currencies to filter news for our symbols (EURUSDm, GBPUSDm, XAUUSDm)
IMPORTANT_CURRENCIES = ["USD", "EUR", "GBP"]


def fetch_economic_events():
    """
    Fetch raw calendar events from TradingView.
    """
    try:
        response = requests.get(CALENDAR_URL, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"[WARNING] Failed to fetch economic calendar. Status code: {response.status_code}")
            return []
            
        data = response.json()
        if data.get("status") == "ok":
            return data.get("result", [])
        return []
    except Exception as e:
        print(f"[ERROR] Failed to fetch economic events: {e}")
        return []


def get_upcoming_impactful_events(minutes_before=60, minutes_after=30):
    """
    Filter economic events that are close to current time and are of medium/high impact.

    Returns:
        list: Upcoming/recent impactful events
    """
    events = fetch_economic_events()
    active_events = []
    
    # Get current UTC time
    now_utc = datetime.now(timezone.utc)
    
    for event in events:
        importance = event.get("importance", -1)
        # We look for Medium (0) or High (1) importance events
        if importance < 0:
            continue
            
        currency = event.get("currency", "")
        if currency not in IMPORTANT_CURRENCIES:
            continue
            
        # Parse ISO date string to UTC datetime
        date_str = event.get("date")
        if not date_str:
            continue
            
        try:
            # Format usually: 2026-06-04T12:30:00.000Z
            # Clean up the trailing 'Z' and milliseconds if necessary
            clean_date_str = date_str.replace("Z", "+00:00")
            event_time = datetime.fromisoformat(clean_date_str)
            
            # Calculate time difference in minutes
            # positive means event is in the future, negative means event was in the past
            time_diff = (event_time - now_utc).total_seconds() / 60.0
            
            # If the event is within the freeze window
            if -minutes_after <= time_diff <= minutes_before:
                active_events.append({
                    "id": event.get("id"),
                    "title": event.get("title"),
                    "currency": currency,
                    "importance": "HIGH" if importance == 1 else "MEDIUM",
                    "time": event_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    "time_diff_minutes": round(time_diff, 1)
                })
        except Exception as e:
            # Skip invalid date formats
            continue
            
    return active_events


def is_news_time(minutes_before=60, minutes_after=30):
    """
    Check if trading should be frozen due to upcoming/recent high impact news.

    Returns:
        tuple: (bool_should_freeze, list_of_active_news_events)
    """
    active_events = get_upcoming_impactful_events(minutes_before, minutes_after)
    should_freeze = len(active_events) > 0
    return should_freeze, active_events


if __name__ == "__main__":
    print("Testing Economic Calendar News Filter...")
    print("-" * 50)
    
    should_freeze, events = is_news_time()
    
    print(f"Should freeze trading? {should_freeze}")
    print(f"Active news events in window: {len(events)}")
    for e in events:
        print(f"  - [{e['importance']}] {e['currency']} - {e['title']} at {e['time']} (in {e['time_diff_minutes']} mins)")
        
    print("\nListing all upcoming USD/EUR/GBP medium/high impact events today:")
    all_events = fetch_economic_events()
    for e in all_events:
        if e.get("importance", -1) >= 0 and e.get("currency") in IMPORTANT_CURRENCIES:
            print(f"  [{'HIGH' if e.get('importance') == 1 else 'MED'}] {e.get('currency')} - {e.get('title')} ({e.get('date')})")
