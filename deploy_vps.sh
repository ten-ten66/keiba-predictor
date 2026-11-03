#!/bin/bash
# ==================================================
# 競馬予想AI - VPSデプロイスクリプト
# ==================================================
# 対象: Xserver VPS (Ubuntu/Debian)
#
# 使い方:
#   chmod +x deploy_vps.sh
#   ./deploy_vps.sh
# ==================================================

set -e

APP_DIR="/opt/keiba-predictor"
DATA_DIR="${APP_DIR}/data"
MODEL_DIR="${APP_DIR}/models"
LOG_DIR="${APP_DIR}/logs"
VENV_DIR="${APP_DIR}/venv"

echo "🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇"
echo "  競馬予想AI - VPSデプロイ"
echo "🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇"

# ==================================================
# 1. システムパッケージ
# ==================================================
echo ""
echo "[1/6] システムパッケージインストール..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv \
    build-essential cmake \
    libopenblas-dev liblapack-dev \
    git curl jq

# ==================================================
# 2. ディレクトリ作成
# ==================================================
echo ""
echo "[2/6] ディレクトリ作成..."
sudo mkdir -p ${APP_DIR} ${DATA_DIR} ${MODEL_DIR} ${LOG_DIR}
sudo chown -R $(whoami):$(whoami) ${APP_DIR}

# ==================================================
# 3. Python仮想環境 & パッケージ
# ==================================================
echo ""
echo "[3/6] Python環境セットアップ..."
python3 -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate

pip install --upgrade pip
pip install \
    pandas numpy scikit-learn \
    lightgbm xgboost \
    requests beautifulsoup4 lxml \
    schedule

echo "✅ パッケージインストール完了"
pip list | grep -E "pandas|numpy|scikit-learn|lightgbm|xgboost"

# ==================================================
# 4. スクリプトコピー
# ==================================================
echo ""
echo "[4/6] スクリプト配置..."

# このスクリプトと同じディレクトリにあるPythonファイルをコピー
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp ${SCRIPT_DIR}/*.py ${APP_DIR}/
echo "   コピー済み:"
ls -la ${APP_DIR}/*.py

# ==================================================
# 5. 初回データ生成 & 学習
# ==================================================
echo ""
echo "[5/6] 初回データ生成 & モデル学習..."

cd ${APP_DIR}

# 大規模サンプルデータ生成（2020-2024、約22万レコード）
python generate_large_data.py --start 2020 --end 2024 --output ${DATA_DIR}

# モデル学習
python main.py --train --data ${DATA_DIR}/races_2020_2024.csv --model keiba_model

echo "✅ 初回学習完了"

# ==================================================
# 6. Cron & 自動化セットアップ
# ==================================================
echo ""
echo "[6/6] Cron自動化セットアップ..."

# 日次スクレイピング & 予測スクリプト
cat > ${APP_DIR}/daily_run.sh << 'DAILY_SCRIPT'
#!/bin/bash
# 競馬予想AI 日次実行スクリプト
# cron: 毎週金曜 20:00 にデータ取得、土日 7:00 に予測

APP_DIR="/opt/keiba-predictor"
source ${APP_DIR}/venv/bin/activate
cd ${APP_DIR}

LOG_FILE="${APP_DIR}/logs/daily_$(date +%Y%m%d).log"

echo "=============================" >> ${LOG_FILE}
echo "実行開始: $(date)" >> ${LOG_FILE}
echo "=============================" >> ${LOG_FILE}

MODE=$1

case ${MODE} in
    "scrape")
        echo "📥 データ取得中..." >> ${LOG_FILE}
        YEAR=$(date +%Y)
        python scraper.py --start ${YEAR} --end ${YEAR} --data-dir ${APP_DIR}/data >> ${LOG_FILE} 2>&1
        ;;
    "train")
        echo "🤖 モデル再学習中..." >> ${LOG_FILE}
        python main.py --train --data ${APP_DIR}/data/races_2020_2024.csv --model keiba_model >> ${LOG_FILE} 2>&1
        ;;
    "predict")
        echo "🏇 予測実行中..." >> ${LOG_FILE}
        python main.py --predict --data ${APP_DIR}/data/today_races.csv --model keiba_model >> ${LOG_FILE} 2>&1

        # Discord通知（ウェブフック）
        if [ -f "${APP_DIR}/data/predictions.json" ]; then
            python notify_discord.py >> ${LOG_FILE} 2>&1
        fi
        ;;
    *)
        echo "使い方: daily_run.sh [scrape|train|predict]" >> ${LOG_FILE}
        ;;
esac

echo "完了: $(date)" >> ${LOG_FILE}
DAILY_SCRIPT

chmod +x ${APP_DIR}/daily_run.sh

# Discord通知スクリプト
cat > ${APP_DIR}/notify_discord.py << 'DISCORD_SCRIPT'
"""Discord通知: 予測結果をウェブフックで送信"""
import json
import os
import requests

WEBHOOK_URL = os.environ.get("KEIBA_DISCORD_WEBHOOK", "")

def send_prediction_summary():
    if not WEBHOOK_URL:
        print("⚠️ KEIBA_DISCORD_WEBHOOK 環境変数が未設定")
        return

    try:
        with open("data/predictions.json", "r", encoding="utf-8") as f:
            predictions = json.load(f)
    except FileNotFoundError:
        print("⚠️ predictions.json が見つかりません")
        return

    # 各レースの上位3頭をまとめる
    lines = ["🏇 **競馬AI予想結果** 🏇\n"]

    for race_id, horses in list(predictions.items())[:5]:  # 最大5レース
        lines.append(f"**{race_id}**")
        for h in horses[:3]:
            emoji = ["🥇", "🥈", "🥉"][h["predicted_rank"] - 1] if h["predicted_rank"] <= 3 else "  "
            ev = h["prob_win"] * 10  # オッズがないので概算
            lines.append(
                f"{emoji} {h['predicted_rank']}位: **{h['horse_name']}** "
                f"(勝率{h['prob_win']*100:.1f}% / 複勝{h['prob_top3']*100:.1f}%)"
            )
        lines.append("")

    lines.append("⚠️ AI統計分析です。馬券購入は自己責任で。")

    payload = {
        "content": "\n".join(lines),
        "username": "🏇 KEIBA AI",
    }

    resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
    if resp.status_code == 204:
        print("✅ Discord通知送信完了")
    else:
        print(f"⚠️ Discord通知エラー: {resp.status_code}")

if __name__ == "__main__":
    send_prediction_summary()
DISCORD_SCRIPT

# Cron登録
CRON_FILE="/tmp/keiba_cron"
cat > ${CRON_FILE} << 'CRON'
# 競馬予想AI 自動実行
# 毎週金曜 20:00: データ取得
0 20 * * 5 /opt/keiba-predictor/daily_run.sh scrape
# 毎週土曜 7:00: 予測実行 & Discord通知
0 7 * * 6 /opt/keiba-predictor/daily_run.sh predict
# 毎週日曜 7:00: 予測実行 & Discord通知
0 7 * * 0 /opt/keiba-predictor/daily_run.sh predict
# 毎月1日 3:00: モデル再学習
0 3 1 * * /opt/keiba-predictor/daily_run.sh train
# ログ自動削除（30日以上前）
0 4 * * 1 find /opt/keiba-predictor/logs -name "*.log" -mtime +30 -delete
CRON

crontab ${CRON_FILE}
echo "✅ Cron登録完了:"
crontab -l

# ==================================================
# 完了
# ==================================================
echo ""
echo "=========================================="
echo "✅ デプロイ完了！"
echo "=========================================="
echo ""
echo "📁 アプリ:  ${APP_DIR}"
echo "📁 データ:  ${DATA_DIR}"
echo "📁 モデル:  ${MODEL_DIR}"
echo "📁 ログ:    ${LOG_DIR}"
echo ""
echo "📌 次のステップ:"
echo "  1. Discord Webhook設定:"
echo "     export KEIBA_DISCORD_WEBHOOK='<DiscordのWebhook URL>'"
echo "     echo 'export KEIBA_DISCORD_WEBHOOK=...' >> ~/.bashrc"
echo ""
echo "  2. 実データでスクレイピング開始:"
echo "     cd ${APP_DIR}"
echo "     source venv/bin/activate"
echo "     python scraper.py --start 2020 --end 2025"
echo ""
echo "  3. 手動予測テスト:"
echo "     python main.py --predict --data data/races_2024.csv"
echo ""
echo "  4. ログ確認:"
echo "     tail -f ${LOG_DIR}/daily_*.log"
