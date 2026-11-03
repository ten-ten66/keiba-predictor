@echo off
echo ============================================
echo  Keiba AI - Local Setup v7.1
echo ============================================

echo.
echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo [2/4] Activating venv...
call venv\Scripts\activate.bat

echo [3/4] Installing packages...
pip install --upgrade pip
pip install numpy pandas scikit-learn lightgbm xgboost catboost scipy optuna
pip install imbalanced-learn shap
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-tabnet

echo [4/4] Verifying installation...
python -c "import lightgbm; import xgboost; import catboost; print('All packages OK')"
if errorlevel 1 (
    echo WARNING: Some packages failed to install
)

echo.
echo ============================================
echo  Setup complete!
echo  Run: venv\Scripts\activate
echo  Then: python train.py --blend --min-year 2001
echo ============================================
pause
