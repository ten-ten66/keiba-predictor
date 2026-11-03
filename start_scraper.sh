#!/bin/bash
# ============================================================
# 20年分スクレイパー クイックスタート
# ============================================================
# VPSにSSH後、このスクリプトを実行するだけ。
#
# 使い方:
#   chmod +x start_scraper.sh
#   ./start_scraper.sh
#
# 停止:
#   screen -r keiba  (アタッチ)
#   Ctrl+C           (安全停止→データ保存)
#
# 再開:
#   ./start_scraper.sh  (続きから自動再開)
# ============================================================

set -e

APP_DIR="/opt/keiba-predictor"
VENV="${APP_DIR}/venv"
DATA_DIR="${APP_DIR}/data"

echo "🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇"
echo "  競馬データ 大規模スクレイパー"
echo "  対象: 2001年〜2025年 (25年分)"
echo "🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇"

# === 1. 環境チェック ===
if [ ! -d "${VENV}" ]; then
    echo "⚠️ Python環境が見つかりません。先にdeploy_vps.shを実行してください。"
    echo "   → chmod +x deploy_vps.sh && ./deploy_vps.sh"
    exit 1
fi

source ${VENV}/bin/activate
mkdir -p ${DATA_DIR}

# === 2. 必要パッケージ確認 ===
pip install requests beautifulsoup4 lxml pandas -q 2>/dev/null

# === 3. scraper_v2.pyを配置 ===
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp ${SCRIPT_DIR}/scraper_v2.py ${APP_DIR}/scraper_v2.py 2>/dev/null || true

# === 4. 進捗確認 ===
if [ -f "${DATA_DIR}/scrape_progress.json" ]; then
    DONE_RACES=$(python3 -c "
import json
with open('${DATA_DIR}/scrape_progress.json') as f:
    d = json.load(f)
print(d['stats']['total_races'])
" 2>/dev/null || echo "0")
    echo ""
    echo "📊 前回の進捗: ${DONE_RACES} レース完了済み"
    echo "   → 続きから再開します"
else
    echo ""
    echo "📊 新規スクレイピング開始"
fi

# === 5. 所要時間の見積もり ===
echo ""
echo "📐 見積もり:"
echo "   対象: 約85,000レース / 約1,100,000レコード"
echo "   リクエスト間隔: 2〜3秒"
echo "   推定所要時間: 3〜7日"
echo "   ディスク使用量: 約500MB〜1GB"
echo ""

# === 6. Discord Webhook確認 ===
if [ -z "${KEIBA_DISCORD_WEBHOOK}" ]; then
    echo "💡 Discord通知を設定すると進捗がわかって便利です:"
    echo "   export KEIBA_DISCORD_WEBHOOK='<DiscordのWebhook URL>'"
    echo ""
fi

# === 7. 実行方法を選択 ===
echo "実行方法を選んでください:"
echo "  1) screen内で実行（推奨。SSH切断しても続行）"
echo "  2) このまま実行（SSH接続中のみ）"
echo "  3) systemdサービスとして実行（上級者向け）"
echo ""
read -p "選択 [1/2/3]: " CHOICE

case ${CHOICE} in
    1)
        # screen がなければインストール
        which screen > /dev/null 2>&1 || sudo apt-get install -y screen -qq

        echo ""
        echo "🚀 screenセッション 'keiba' で起動します"
        echo ""
        echo "  操作方法:"
        echo "    画面から離れる:  Ctrl+A → D"
        echo "    画面に戻る:      screen -r keiba"
        echo "    停止:            Ctrl+C（安全にデータ保存して終了）"
        echo ""
        read -p "Enterで開始..."

        screen -dmS keiba bash -c "
            cd ${APP_DIR}
            source ${VENV}/bin/activate
            python scraper_v2.py --start 2001 --end 2025 --data-dir ${DATA_DIR}
            echo '完了。Enterで閉じます。'
            read
        "
        echo "✅ バックグラウンドで実行開始！"
        echo "   screen -r keiba で画面を確認できます"
        ;;

    2)
        echo ""
        echo "🚀 フォアグラウンドで実行（Ctrl+C で安全停止）"
        cd ${APP_DIR}
        python scraper_v2.py --start 2001 --end 2025 --data-dir ${DATA_DIR}
        ;;

    3)
        # systemdサービスファイル作成
        sudo tee /etc/systemd/system/keiba-scraper.service > /dev/null << SYSTEMD
[Unit]
Description=Keiba Data Scraper
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=${APP_DIR}
ExecStart=${VENV}/bin/python scraper_v2.py --start 2001 --end 2025 --data-dir ${DATA_DIR}
Restart=on-failure
RestartSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SYSTEMD

        sudo systemctl daemon-reload
        sudo systemctl enable keiba-scraper
        sudo systemctl start keiba-scraper

        echo "✅ systemdサービスとして起動"
        echo "   状態確認: sudo systemctl status keiba-scraper"
        echo "   ログ:     sudo journalctl -u keiba-scraper -f"
        echo "   停止:     sudo systemctl stop keiba-scraper"
        ;;

    *)
        echo "1, 2, 3 のいずれかを選んでください"
        exit 1
        ;;
esac

echo ""
echo "📁 データ保存先: ${DATA_DIR}/races_YYYY.csv"
echo "📁 進捗ファイル: ${DATA_DIR}/scrape_progress.json"
echo "📁 ログ:         ${DATA_DIR}/scraper.log"
