# ============================================
# Intelligent Investment Decision System GUI
# PowerShell Launcher Script
# ============================================

Write-Host ""
Write-Host "========================================"
Write-Host " Investment Decision System GUI Launcher"
Write-Host "========================================"
Write-Host ""

# Check if virtual environment exists
if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "[*] Starting GUI application..."
    Write-Host ""
    & .\.venv\Scripts\python.exe gui_investment_system.py
} else {
    Write-Host "[!] Virtual environment not found."
    Write-Host "[*] Please run: python -m venv .venv"
    Write-Host "[*] Then: .\.venv\Scripts\pip install -r requirements.txt"
    Read-Host "Press Enter to exit"
}

