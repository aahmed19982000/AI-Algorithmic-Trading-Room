import sqlite3
import json
import os

db_path = os.path.join(os.path.dirname(__file__), "database.db")
print("Target Database Path:", os.path.abspath(db_path))

list_syms = [
    "EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm", "USDCADm", "USDCHFm", "NZDUSDm",
    "EURJPYm", "EURGBPm", "GBPJPYm", "EURAUDm", "EURCADm", "GBPAUDm", "EURNZDm",
    "EURCHFm", "GBPCADm", "AUDJPYm", "GBPCHFm", "GBPNZDm", "NZDCADm", "NZDCHFm",
    "NZDJPYm", "AUDCADm", "AUDCHFm", "AUDNZDm", "CADCHFm", "CADJPYm", "CHFJPYm",
    "XAUUSDm", "XAGUSDm", "UKOILm", "USOILm", "BTCUSDm", "ETHUSDm", "LTCUSDm",
    "XRPUSDm", "SOLUSDm", "DOGEUSDm", "BNBUSDm"
]

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", ("symbols", json.dumps(list_syms)))
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", ("grid_enabled", "0"))
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", ("gann_enabled", "1"))
    conn.commit()
    conn.close()
    print("SUCCESS: Database updated to 39 symbols and grid disabled!")
else:
    print("ERROR: database.db not found in this folder!")
