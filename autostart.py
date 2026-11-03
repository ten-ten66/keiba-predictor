#!/usr/bin/env python3
"""Auto-start API server"""
import subprocess, os, sys
os.chdir("/opt/keiba-predictor")
subprocess.Popen(
    ["/opt/keiba-predictor/venv/bin/uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=open("/tmp/api.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True
)
print("API server starting...")
