@echo off
echo ============================================
echo   Keiba AI - Local Training Setup
echo ============================================
echo.

REM Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo.
    echo Install Python 3.9-3.13 from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Check "Add Python to PATH" during install!
    echo.
    pause
    exit /b 1
)

python --version
echo.

REM Create venv
echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Install packages
echo [2/3] Installing packages...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo [3/3] Setup complete!
echo.
echo To start training:
echo   venv\Scripts\activate.bat
echo   python train_advanced.py
echo.
pause
