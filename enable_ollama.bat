@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   SWITCHING AI PROVIDER TO LOCAL OLLAMA PHI3
echo ===================================================

:: Find python path
set "PY_EXE=python"
if exist .\venv\Scripts\python.exe (
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

:: Run database update
"!PY_EXE!" -c "import sqlite3; conn = sqlite3.connect('database.db'); c = conn.cursor(); c.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', ('ai_provider', 'ollama')); c.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', ('ollama_model', 'phi3')); conn.commit(); print('SUCCESS: Switched database to local Ollama Phi3!')"

pause
