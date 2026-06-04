"""
MT5 Connection Module
=====================
Connect to MetaTrader 5 account with comprehensive error handling.
"""

import sys
import os
import time

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import MetaTrader5 as mt5
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Read account data from .env
MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe")


def connect_mt5(max_retries=3, retry_delay=5):
    """
    Connect to MT5 with auto-retry on failure.

    Args:
        max_retries: Number of retry attempts (default: 3)
        retry_delay: Wait between attempts in seconds (default: 5)

    Returns:
        bool: True if connected, False if failed
    """

    # -- Validate .env data --
    if not all([MT5_LOGIN, MT5_PASSWORD, MT5_SERVER]):
        print("[ERROR] Account data missing in .env file")
        print("  Make sure MT5_LOGIN, MT5_PASSWORD, MT5_SERVER are filled")
        return False

    for attempt in range(1, max_retries + 1):
        print(f"\n[ATTEMPT {attempt}/{max_retries}] Connecting...")

        # -- Step 1: Initialize MT5 Terminal --
        if not mt5.initialize(path=MT5_PATH, timeout=10000):
            error_code = mt5.last_error()
            print(f"  [FAIL] MT5 Terminal initialization failed")
            print(f"  Error: {error_code}")

            if attempt < max_retries:
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print("  [ABORT] All initialization attempts failed")
                return False

        print("  [OK] MT5 Terminal initialized")

        # -- Step 2: Login --
        login_result = mt5.login(
            login=int(MT5_LOGIN),
            password=MT5_PASSWORD,
            server=MT5_SERVER
        )

        if login_result:
            print("  [OK] Login successful!")
            _print_account_info()
            return True
        else:
            error_code = mt5.last_error()
            print(f"  [FAIL] Login failed")
            print(f"  Error: {error_code}")
            mt5.shutdown()

            if attempt < max_retries:
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("  [ABORT] All login attempts failed")
                return False

    return False


def _print_account_info():
    """Display account info after successful connection."""
    account_info = mt5.account_info()
    if account_info is not None:
        print("\n" + "=" * 45)
        print("        ACCOUNT INFO")
        print("=" * 45)
        print(f"  Name:      {account_info.name}")
        print(f"  Login:     {account_info.login}")
        print(f"  Server:    {account_info.server}")
        print(f"  Balance:   {account_info.balance} {account_info.currency}")
        print(f"  Leverage:  1:{account_info.leverage}")
        print(f"  Type:      {'Real' if account_info.trade_mode == 0 else 'Demo'}")
        print("=" * 45)


def disconnect_mt5():
    """Safely close MT5 connection."""
    mt5.shutdown()
    print("\n[OK] MT5 disconnected")


# -- Direct run for testing --
if __name__ == "__main__":
    print("Starting MT5 connection...")
    print("-" * 45)

    if connect_mt5():
        print("\n[SUCCESS] Bot is ready!")
        input("\nPress Enter to disconnect...")
        disconnect_mt5()
    else:
        print("\n[FAILED] Could not connect. Check:")
        print("  1. MetaTrader 5 is installed and running")
        print("  2. Account data in .env is correct")
        print("  3. Internet connection is available")
