#!/bin/bash
# ============================================================
# KEIBA AI - APIサーバー デプロイ
# ============================================================
# VPSにAPIサーバー + リアルタイムジョブを設置。
# systemdで自動起動・自動復旧。
# ============================================================

set -e

APP_DIR="/opt/keiba-predictor"
VENV="${APP_DIR}/venv"
DATA_DIR="${APP_DIR}/data"
PORT=8000

echo "🏇 KEIBA AI サーバーデプロイ"
echo ""

# 1. Python仮想環境確認
if [ ! -d "${VENV}" ]; then
    echo "Python環境を作成..."
    python3 -m venv ${VENV}
fi
source ${VENV}/bin/activate

# 2. パッケージインストール
echo "[1/4] パッケージインストール..."
pip install -q \
    fastapi uvicorn pydantic \
    requests beautifulsoup4 lxml \
    pandas numpy scikit-learn \
    schedule

# 3. スクリプト配置
echo "[2/4] スクリプト配置..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for f in api_server.py realtime_job.py start_server.py \
         race_database.py betting_tracker.py model_trainer.py \
         data_processor.py advanced_engine.py multi_source.py; do
    if [ -f "${SCRIPT_DIR}/${f}" ]; then
        cp "${SCRIPT_DIR}/${f}" "${APP_DIR}/"
    fi
done
mkdir -p ${DATA_DIR}

# 4. systemdサービス作成
echo "[3/4] systemdサービス作成..."

sudo tee /etc/systemd/system/keiba-api.service > /dev/null << EOF
[Unit]
Description=KEIBA AI API Server
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=${APP_DIR}
Environment=KEIBA_DATA_DIR=${DATA_DIR}
Environment=KEIBA_DISCORD_WEBHOOK=${KEIBA_DISCORD_WEBHOOK:-}
ExecStart=${VENV}/bin/python start_server.py --port ${PORT}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable keiba-api
sudo systemctl restart keiba-api

# 5. 確認
echo "[4/4] 起動確認..."
sleep 3

if curl -s http://localhost:${PORT}/api/status > /dev/null 2>&1; then
    STATUS=$(curl -s http://localhost:${PORT}/api/status)
    echo "✅ APIサーバー起動成功"
    echo "   ${STATUS}"
else
    echo "⚠️ 起動確認中...（数秒待ってリトライしてください）"
    echo "   sudo systemctl status keiba-api"
fi

echo ""
echo "=========================================="
echo "✅ デプロイ完了"
echo "=========================================="
echo ""
echo "📡 API: http://$(hostname -I | awk '{print $1}'):${PORT}"
echo ""
echo "操作:"
echo "  状態確認:   sudo systemctl status keiba-api"
echo "  ログ:       sudo journalctl -u keiba-api -f"
echo "  再起動:     sudo systemctl restart keiba-api"
echo "  停止:       sudo systemctl stop keiba-api"
echo ""
echo "APIテスト:"
echo "  curl http://localhost:${PORT}/api/status"
echo "  curl http://localhost:${PORT}/api/venues"
echo "  curl http://localhost:${PORT}/api/races/today"
echo ""
echo "Discord通知設定:"
echo "  echo 'KEIBA_DISCORD_WEBHOOK=https://...' >> /etc/environment"
echo "  sudo systemctl restart keiba-api"
