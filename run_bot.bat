@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ===================================================
echo   AI ALGORITHMIC TRADING BOT - LAUNCHER
echo ===================================================

:: Prefer the project's own virtual environment
set "PY_EXE=python"
if exist ".\venv\Scripts\python.exe" (
    set "PY_EXE=.\venv\Scripts\python.exe"
) else (
    set "pythonFolder=%LocalAppData%\Programs\Python"
    if exist "!pythonFolder!" (
        for /d %%d in ("!pythonFolder!\*") do (
            if exist "%%d\python.exe" (
                set "PY_EXE=%%d\python.exe"
            )
        )
    )
)

echo [INFO] Using Python: !PY_EXE!
echo [INFO] Starting bot in continuous loop mode (Ctrl+C to stop)...
echo.

"!PY_EXE!" -u trading_bot.py --loop --interval 300

echo.
echo ===================================================
echo   Bot stopped or crashed. See messages above.
echo ===================================================
pause
