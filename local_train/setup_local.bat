@echo off
chcp 65001 >nul
echo ============================================================
echo   競馬予測AI - ローカル学習セットアップ (Windows)
echo ============================================================
echo.

REM Pythonチェック
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Pythonがインストールされていません
    echo    https://www.python.org/downloads/ からインストールしてください
    pause
    exit /b 1
)

echo ✅ Python確認OK
echo.
echo 📦 パッケージインストール中...
pip install -r requirements.txt
echo.

if errorlevel 1 (
    echo ❌ パッケージのインストールに失敗しました
    pause
    exit /b 1
)

echo ✅ セットアップ完了！
echo.
echo 使い方:
echo   python train_local.py              ← 学習実行（直近10年）
echo   python train_local.py --full       ← 全データ学習（メモリ16GB以上）
echo   python train_local.py --upload     ← 学習 + VPSアップロード
echo   python train_local.py --years 15   ← 直近15年分で学習
echo.
pause
