@echo off
REM ============================================
REM Intelligent Investment Decision System GUI
REM ============================================
echo.
echo ========================================
echo  Investment Decision System GUI Launcher
echo ========================================
echo.

REM Check if virtual environment exists
if exist ".venv\Scripts\python.exe" (
    echo [*] Starting GUI application...
    echo.
    .venv\Scripts\python.exe gui_investment_system.py
) else (
    echo [!] Virtual environment not found.
    echo [*] Please run: python -m venv .venv
    echo [*] Then: .venv\Scripts\pip install -r requirements.txt
    pause
)

