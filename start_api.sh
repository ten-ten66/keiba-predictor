#!/bin/bash
cd /opt/keiba-predictor
nohup venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 &
echo "API started PID=$!"
