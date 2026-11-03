#!/bin/bash
echo "============================================================"
echo "  競馬予測AI - ローカル学習セットアップ (Mac/Linux)"
echo "============================================================"
echo ""

# Pythonチェック
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3がインストールされていません"
    exit 1
fi

echo "✅ Python3: $(python3 --version)"
echo ""
echo "📦 パッケージインストール中..."
pip3 install -r requirements.txt

echo ""
echo "✅ セットアップ完了！"
echo ""
echo "使い方:"
echo "  python3 train_local.py              ← 学習実行（直近10年）"
echo "  python3 train_local.py --full       ← 全データ学習（メモリ16GB以上）"
echo "  python3 train_local.py --upload     ← 学習 + VPSアップロード"
echo "  python3 train_local.py --years 15   ← 直近15年分で学習"
