#!/bin/bash
cd /opt/keiba-predictor
exec venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000
