@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   GEMINI AI & DEPENDENCIES INSTALLER FOR VPS
echo ===================================================

:: Check local venv
if exist .\venv\Scripts\python.exe (
    echo [INFO] Found python in local venv. Running installation...
    .\venv\Scripts\python.exe install_gemini.py
    pause
    exit /b
)

:: Check AppData Programs
set "pythonFolder=%LocalAppData%\Programs\Python"
if exist "%pythonFolder%" (
    for /d %%d in ("%pythonFolder%\*") do (
        if exist "%%d\python.exe" (
            echo [INFO] Found python in AppData: %%d\python.exe
            "%%d\python.exe" install_gemini.py
            pause
            exit /b
        )
    )
)

:: Fallback to default python
echo [INFO] Attempting global python command...
python install_gemini.py
pause
