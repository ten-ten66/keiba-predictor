#!/usr/bin/env python3
"""
Watchdog: comprehensive API health check and auto-restart.

Checks:
  1. Whether the uvicorn process is running
  2. Whether the API responds to /api/health
  3. Whether the API can authenticate (token generation works)
  4. Auto-restart if the process is down
  5. Log rotation if log file gets too large

Usage:
  python watchdog.py          # Run once (check + restart if needed)

Cron (every 5 minutes):
  */5 * * * * /opt/keiba-predictor/venv/bin/python /opt/keiba-predictor/watchdog.py >> /tmp/watchdog.log 2>&1
"""

import subprocess
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

os.chdir("/opt/keiba-predictor")

LOG_FILE = "/tmp/api.log"
MAX_LOG_SIZE_MB = 100
API_URL = "http://127.0.0.1:8000"

logging.basicConfig(
    format="%(asctime)s [watchdog] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("watchdog")


def is_process_running() -> bool:
    """Check if uvicorn api_server is running."""
    result = subprocess.run(
        ["pgrep", "-f", "uvicorn api_server"],
        capture_output=True, timeout=5,
    )
    return result.returncode == 0


def get_process_pids() -> list:
    """Get PIDs of running uvicorn processes."""
    result = subprocess.run(
        ["pgrep", "-f", "uvicorn api_server"],
        capture_output=True, timeout=5,
    )
    if result.returncode == 0:
        return result.stdout.decode().strip().split("\n")
    return []


def is_api_responding() -> bool:
    """Check if the API responds to health check."""
    try:
        import requests
        r = requests.get(f"{API_URL}/api/health", timeout=10)
        return r.status_code in (200, 503)
    except Exception:
        return False


def is_auth_working() -> bool:
    """Check if auth endpoint works."""
    try:
        import requests
        r = requests.post(
            f"{API_URL}/api/auth/login",
            json={"username": "admin", "password": os.environ["KEIBA_ADMIN_PASSWORD"]},
            timeout=10,
        )
        return r.status_code == 200 and "access_token" in r.json()
    except Exception:
        return False


def rotate_logs():
    """Rotate log file if it exceeds size limit."""
    log_path = Path(LOG_FILE)
    if log_path.exists():
        size_mb = log_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_LOG_SIZE_MB:
            rotated = f"{LOG_FILE}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_path.rename(rotated)
            log.info(f"Rotated log ({size_mb:.0f}MB) to {rotated}")
            # Keep only last 3 rotated logs
            rotated_logs = sorted(Path("/tmp").glob("api.log.*"))
            for old_log in rotated_logs[:-3]:
                old_log.unlink()
                log.info(f"Removed old log: {old_log}")


def start_api():
    """Start the API server."""
    log.info("Starting API server...")
    subprocess.Popen(
        [
            "/opt/keiba-predictor/venv/bin/uvicorn",
            "api_server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
        ],
        stdout=open(LOG_FILE, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd="/opt/keiba-predictor",
    )
    # Wait for startup
    for i in range(15):
        time.sleep(2)
        if is_api_responding():
            log.info("API started and responding successfully")
            return True
    log.error("API started but not responding after 30 seconds")
    return False


def kill_api():
    """Kill all uvicorn processes (for restart)."""
    pids = get_process_pids()
    if pids:
        for pid in pids:
            try:
                subprocess.run(["kill", pid.strip()], timeout=5)
            except Exception:
                pass
        time.sleep(2)
        log.info(f"Killed uvicorn processes: {pids}")


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log rotation
    rotate_logs()

    process_running = is_process_running()
    pids = get_process_pids()

    if not process_running:
        log.warning(f"[{now}] API process NOT running - starting...")
        started = start_api()
        if started:
            log.info(f"[{now}] API restarted successfully")
        else:
            log.error(f"[{now}] FAILED to restart API")
        return

    log.info(f"[{now}] API process running (PIDs: {', '.join(pids)})")

    # Check if API is actually responding
    if not is_api_responding():
        log.warning(f"[{now}] API process exists but NOT responding - restarting...")
        kill_api()
        time.sleep(3)
        started = start_api()
        if started:
            log.info(f"[{now}] API restarted after unresponsive state")
        else:
            log.error(f"[{now}] FAILED to restart unresponsive API")
        return

    # Check auth
    if not is_auth_working():
        log.warning(f"[{now}] API responding but auth NOT working")
    else:
        log.info(f"[{now}] All checks passed")


if __name__ == "__main__":
    main()
