import subprocess
import sys
import os
import winreg

def find_python():
    # 1. Check local venv
    venv_py = os.path.join(os.path.dirname(__file__), "venv", "Scripts", "python.exe")
    if os.path.exists(venv_py):
        return venv_py

    # 2. Try AppData Programs folder
    local_app_data = os.environ.get("LocalAppData", "")
    if local_app_data:
        python_dir = os.path.join(local_app_data, "Programs", "Python")
        if os.path.exists(python_dir):
            for sub in os.listdir(python_dir):
                exe = os.path.join(python_dir, sub, "python.exe")
                if os.path.exists(exe):
                    return exe

    # 3. Check registry
    for ver in ["3.12", "3.11", "3.10", "3.9"]:
        try:
            key_path = rf"Software\Python\PythonCore\{ver}\InstallPath"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                val, _ = winreg.QueryValueEx(key, "")
                exe = os.path.join(val, "python.exe")
                if os.path.exists(exe):
                    return exe
        except Exception:
            continue

    # 4. Fallback to system python
    return "python"

def main():
    py_exe = find_python()
    print("Found Python Executable:", py_exe)
    
    packages = [
        "google-generativeai",
        "MetaTrader5",
        "pandas",
        "flask",
        "requests",
        "matplotlib",
        "python-dotenv"
    ]
    
    print("\n--- Starting package installation ---")
    for pkg in packages:
        print(f"Installing {pkg}...")
        try:
            subprocess.check_call([py_exe, "-m", "pip", "install", pkg])
            print(f"[OK] {pkg} installed successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to install {pkg}: {e}")

    print("\nChecking .env file for Gemini API Key...")
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()
        if "GEMINI_API_KEY" in content:
            print("[OK] GEMINI_API_KEY variable exists in .env file.")
        else:
            print("[WARNING] GEMINI_API_KEY is missing from .env file!")
    else:
        print("[WARNING] .env file not found!")

if __name__ == "__main__":
    main()
