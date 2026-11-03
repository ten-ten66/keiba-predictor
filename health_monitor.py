#!/usr/bin/env python3
"""
Health Monitor for KEIBA AI Predictor
======================================
Checks all critical systems and sends Discord alerts on failure.

Usage:
    python health_monitor.py          # Run once
    python health_monitor.py --cron   # Suitable for cron (quiet on success)

Cron entry (every 30 minutes):
    */30 * * * * /opt/keiba-predictor/venv/bin/python /opt/keiba-predictor/health_monitor.py --cron >> /tmp/health_monitor.log 2>&1
"""

import os
import sys
import json
import shutil
import time
import logging
import subprocess
from collections import deque
from datetime import datetime, date, timedelta
from pathlib import Path

# ログ設定
_LOG_DIR = Path("/opt/keiba-predictor/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_hm_logger = logging.getLogger("health_monitor")
if not _hm_logger.handlers:
    _hm_logger.setLevel(logging.INFO)
    _fh = logging.FileHandler(_LOG_DIR / "health_monitor.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _hm_logger.addHandler(_fh)
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("%(asctime)s [HEALTH] %(message)s", datefmt="%H:%M:%S"))
    _hm_logger.addHandler(_ch)

os.chdir("/opt/keiba-predictor")
sys.path.insert(0, "/opt/keiba-predictor")

# ============================================================
# Configuration
# ============================================================

API_BASE = "http://127.0.0.1:8000"
API_TIMEOUT = 15
DISK_THRESHOLD_PCT = 90
MEMORY_THRESHOLD_PCT = 90

# ============================================================
# Check functions
# ============================================================


def check_api_responding() -> dict:
    """Check if the API server is responding."""
    try:
        import requests
        r = requests.get(f"{API_BASE}/api/health", timeout=API_TIMEOUT)
        if r.status_code == 200:
            return {"status": "ok", "message": "API responding (200)"}
        elif r.status_code == 503:
            return {"status": "warning", "message": "API loading (503)"}
        else:
            return {"status": "error", "message": f"API returned {r.status_code}"}
    except Exception as e:
        return {"status": "error", "message": f"API not reachable: {e}"}


def check_model_loaded() -> dict:
    """Check if the prediction model is loaded."""
    try:
        import requests
        # Login to get token
        r = requests.post(
            f"{API_BASE}/api/auth/login",
            json={"username": "admin", "password": os.environ["KEIBA_ADMIN_PASSWORD"]},
            timeout=API_TIMEOUT,
        )
        if r.status_code != 200:
            return {"status": "warning", "message": "Cannot login to check model status"}
        token = r.json().get("access_token")
        r2 = requests.get(
            f"{API_BASE}/api/status",
            headers={"Authorization": f"Bearer {token}"},
            timeout=API_TIMEOUT,
        )
        if r2.status_code == 200:
            data = r2.json()
            loaded = data.get("model_loaded", False)
            return {
                "status": "ok" if loaded else "error",
                "message": "Model loaded" if loaded else "Model NOT loaded",
            }
        return {"status": "warning", "message": f"Status endpoint returned {r2.status_code}"}
    except Exception as e:
        return {"status": "error", "message": f"Model check failed: {e}"}


def check_prediction_cache() -> dict:
    """Check if prediction cache exists for this weekend."""
    cache_dir = Path("/opt/keiba-predictor/data/prediction_cache")
    if not cache_dir.exists():
        return {"status": "warning", "message": "Prediction cache directory missing"}

    today = date.today()
    # Find next Saturday
    days_to_sat = (5 - today.weekday()) % 7
    if days_to_sat == 0 and today.weekday() != 5:
        days_to_sat = 7
    next_sat = today + timedelta(days=days_to_sat)
    next_sun = next_sat + timedelta(days=1)

    sat_files = list(cache_dir.glob(f"{next_sat.strftime('%Y%m%d')}_*.json"))
    sun_files = list(cache_dir.glob(f"{next_sun.strftime('%Y%m%d')}_*.json"))

    total = len(sat_files) + len(sun_files)
    if total > 0:
        return {
            "status": "ok",
            "message": f"Cache: {len(sat_files)} Sat + {len(sun_files)} Sun races",
        }
    # Only warn on Fri-Sun when cache matters
    if today.weekday() in (4, 5, 6):
        return {"status": "warning", "message": "No prediction cache for this weekend"}
    return {"status": "ok", "message": "No weekend cache (not racing period)"}


def check_odds_fetcher() -> dict:
    """Check if the odds fetcher module is importable."""
    try:
        import odds_fetcher
        return {"status": "ok", "message": "odds_fetcher module available"}
    except ImportError:
        return {"status": "warning", "message": "odds_fetcher module not available"}


def check_blood_scraper() -> dict:
    """Check if the blood scraper data is recent enough."""
    blood_dir = Path("/opt/keiba-predictor/data/blood")
    if not blood_dir.exists():
        return {"status": "warning", "message": "Blood data directory missing"}

    json_files = list(blood_dir.glob("*.json"))
    if not json_files:
        return {"status": "warning", "message": "No blood data files found"}

    newest = max(json_files, key=lambda f: f.stat().st_mtime)
    age_days = (time.time() - newest.stat().st_mtime) / 86400
    if age_days < 30:
        return {"status": "ok", "message": f"Latest blood data: {age_days:.0f} days old"}
    return {"status": "warning", "message": f"Blood data stale: {age_days:.0f} days old"}


def check_disk_space() -> dict:
    """Check disk usage."""
    try:
        usage = shutil.disk_usage("/")
        pct = round(usage.used / usage.total * 100, 1)
        free_gb = round(usage.free / (1024**3), 1)
        if pct < DISK_THRESHOLD_PCT:
            return {"status": "ok", "message": f"Disk {pct}% used, {free_gb}GB free"}
        return {"status": "error", "message": f"Disk CRITICAL: {pct}% used, only {free_gb}GB free"}
    except Exception as e:
        return {"status": "error", "message": f"Disk check failed: {e}"}


def check_memory() -> dict:
    """Check memory usage."""
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()
        lines = meminfo.strip().split("\n")
        mem = {}
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip().split()[0]
                mem[key] = int(val) * 1024  # kB -> bytes

        total = mem.get("MemTotal", 1)
        available = mem.get("MemAvailable", total)
        pct = round((1 - available / total) * 100, 1)
        avail_gb = round(available / (1024**3), 1)

        if pct < MEMORY_THRESHOLD_PCT:
            return {"status": "ok", "message": f"Memory {pct}% used, {avail_gb}GB available"}
        return {"status": "error", "message": f"Memory CRITICAL: {pct}% used, only {avail_gb}GB available"}
    except Exception as e:
        return {"status": "error", "message": f"Memory check failed: {e}"}


def check_api_process() -> dict:
    """Check if uvicorn process is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "uvicorn api_server"],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            pids = result.stdout.decode().strip().split("\n")
            return {"status": "ok", "message": f"uvicorn running (PIDs: {', '.join(pids)})"}
        return {"status": "error", "message": "uvicorn process not found"}
    except Exception as e:
        return {"status": "error", "message": f"Process check failed: {e}"}


# ============================================================
# 予測品質の継続監視
# ============================================================

# メモリリーク防止: 直近100レース分の品質指標をメモリに保持
_quality_window: deque = deque(maxlen=100)


def _load_quality_window_from_ledger(n: int = 100) -> deque:
    """
    prediction_ledger.jsonl と結果ファイルを突合して品質ウィンドウを構築する。
    Returns deque of dicts: {race_id, top1_correct, top3_correct, prob_win, odds}
    """
    window: deque = deque(maxlen=n)
    try:
        from prediction_ledger import get_ledger
        entries = get_ledger(limit=n * 2)

        # 結果ファイルの探索
        data_dir = Path("/opt/keiba-predictor/data")
        results_cache: dict = {}
        for rf in sorted(data_dir.glob("results_*.json"), reverse=True)[:30]:
            try:
                with open(rf, encoding="utf-8") as f:
                    day_results = json.load(f)
                results_cache.update(day_results)
            except Exception:
                pass

        for entry in entries:
            race_id = entry.get("race_id", "")
            if race_id not in results_cache:
                continue
            result = results_cache[race_id]
            finish_order = result.get("finish_order", [])
            if not finish_order:
                continue

            winner_num = None
            top3_nums = set()
            for h in finish_order:
                pos = h.get("position", 99)
                if pos == 1:
                    winner_num = h.get("horse_number")
                if pos <= 3:
                    top3_nums.add(h.get("horse_number"))

            top3_preds = entry.get("top3", [])
            if not top3_preds:
                continue

            top1_pred_num = top3_preds[0].get("horse_number")
            top1_correct = int(top1_pred_num == winner_num)
            top3_correct = int(any(p.get("horse_number") in top3_nums for p in top3_preds))
            prob_win = top3_preds[0].get("prob_win", 0.0)

            # オッズは結果ファイルから取得
            odds = 0.0
            for h in finish_order:
                if h.get("horse_number") == top1_pred_num:
                    odds = float(h.get("odds", 0.0))
                    break

            window.append({
                "race_id": race_id,
                "top1_correct": top1_correct,
                "top3_correct": top3_correct,
                "prob_win": prob_win,
                "odds": odds,
            })
    except Exception as e:
        _hm_logger.warning(f"品質ウィンドウ構築エラー: {e}")
    return window


def check_prediction_quality(n: int = 100) -> dict:
    """
    直近 n レースの予測品質指標を計算し、低下時はログアラートを出力する。

    Returns:
        dict with keys:
          - status: "ok" | "warning" | "error"
          - message: human-readable summary
          - metrics: {win_hit_rate, top3_hit_rate, calibration_error, roi, n_samples}
          - drift: {detected, ks_statistic, p_value}
    """
    window = _load_quality_window_from_ledger(n)

    if len(window) < 10:
        return {
            "status": "warning",
            "message": f"品質評価用データ不足 ({len(window)}件 < 10件)",
            "metrics": {},
            "drift": {"detected": False},
        }

    # -- 品質指標計算 --
    top1_hits = [r["top1_correct"] for r in window]
    top3_hits = [r["top3_correct"] for r in window]
    probs = [r["prob_win"] for r in window]
    odds_list = [r["odds"] for r in window]

    n_samples = len(window)
    win_hit_rate = sum(top1_hits) / n_samples
    top3_hit_rate = sum(top3_hits) / n_samples

    # Calibration Error: |mean(prob_win) - win_hit_rate|
    mean_prob = sum(probs) / n_samples
    calibration_error = abs(mean_prob - win_hit_rate)

    # ROI: 単勝100円投資した場合の回収率
    total_bet = n_samples * 100
    total_return = sum(
        r["odds"] * 100 for r in window if r["top1_correct"] and r["odds"] > 0
    )
    roi = (total_return / total_bet) if total_bet > 0 else 0.0

    metrics = {
        "win_hit_rate": round(win_hit_rate, 4),
        "top3_hit_rate": round(top3_hit_rate, 4),
        "calibration_error": round(calibration_error, 4),
        "roi": round(roi, 4),
        "n_samples": n_samples,
    }

    # -- KS検定によるデータドリフト検出 --
    drift_info = {"detected": False, "ks_statistic": 0.0, "p_value": 1.0}
    try:
        from scipy.stats import ks_2samp
        half = n_samples // 2
        if half >= 5:
            old_probs = probs[:half]
            new_probs = probs[half:]
            ks_stat, p_val = ks_2samp(old_probs, new_probs)
            drift_detected = p_val < 0.05
            drift_info = {
                "detected": bool(drift_detected),
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 4),
            }
    except ImportError:
        drift_info["note"] = "scipy が見つかりません。KS検定をスキップしました"
    except Exception as e:
        drift_info["note"] = f"KS検定エラー: {e}"

    # -- 品質低下判定 & アラート --
    issues = []
    if win_hit_rate < 0.05:
        issues.append(f"勝率{win_hit_rate:.1%}が低すぎます（閾値5%）")
    if top3_hit_rate < 0.20:
        issues.append(f"複勝率{top3_hit_rate:.1%}が低すぎます（閾値20%）")
    if calibration_error > 0.15:
        issues.append(f"キャリブレーション誤差{calibration_error:.3f}が大きすぎます（閾値0.15）")
    if roi < 0.5:
        issues.append(f"ROI {roi:.2f}が低すぎます（閾値0.5）")
    if drift_info["detected"]:
        issues.append(
            f"データドリフト検出 KS={drift_info['ks_statistic']:.3f} p={drift_info['p_value']:.3f}"
        )

    if issues:
        alert_msg = "予測品質低下アラート: " + " / ".join(issues)
        _hm_logger.warning(alert_msg)
        status = "error" if len(issues) >= 3 else "warning"
        message = alert_msg
    else:
        status = "ok"
        message = (
            f"予測品質正常 win={win_hit_rate:.1%} top3={top3_hit_rate:.1%} "
            f"cal_err={calibration_error:.3f} roi={roi:.2f} n={n_samples}"
        )
        _hm_logger.info(message)

    return {
        "status": status,
        "message": message,
        "metrics": metrics,
        "drift": drift_info,
    }


# ============================================================
# Alert sending
# ============================================================

def send_alert(failures: list, all_results: dict):
    """Send Discord alert for failures using notifier.py."""
    try:
        from notifier import send_discord, _load_all_settings

        lines = [
            "**KEIBA AI Health Monitor Alert**",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        for name, result in all_results.items():
            icon = {"ok": "OK", "warning": "WARN", "error": "ERROR"}.get(result["status"], "?")
            lines.append(f"[{icon}] {name}: {result['message']}")

        lines.append("")
        error_count = sum(1 for r in all_results.values() if r["status"] == "error")
        warn_count = sum(1 for r in all_results.values() if r["status"] == "warning")
        lines.append(f"Errors: {error_count}, Warnings: {warn_count}")

        message = "\n".join(lines)

        # Try to send via all configured Discord webhooks
        all_settings = _load_all_settings()
        sent = False
        for username, settings in all_settings.items():
            webhook = settings.get("discord_webhook", "").strip()
            if webhook:
                if send_discord(webhook, message):
                    sent = True
                    break

        if not sent:
            # Try environment variable
            env_webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")
            if env_webhook:
                send_discord(env_webhook, message)

    except Exception as e:
        print(f"Failed to send alert: {e}")


# ============================================================
# Main
# ============================================================

def run_all_checks() -> dict:
    """Run all health checks and return results."""
    checks = {
        "api_process": check_api_process,
        "api_responding": check_api_responding,
        "model_loaded": check_model_loaded,
        "prediction_cache": check_prediction_cache,
        "odds_fetcher": check_odds_fetcher,
        "blood_scraper": check_blood_scraper,
        "disk_space": check_disk_space,
        "memory": check_memory,
        "prediction_quality": check_prediction_quality,
    }

    results = {}
    for name, check_fn in checks.items():
        try:
            results[name] = check_fn()
        except Exception as e:
            results[name] = {"status": "error", "message": f"Check crashed: {e}"}

    return results


def main():
    is_cron = "--cron" in sys.argv

    results = run_all_checks()

    # Determine overall status
    failures = [name for name, r in results.items() if r["status"] == "error"]
    warnings = [name for name, r in results.items() if r["status"] == "warning"]

    if not is_cron or failures:
        print(f"\n=== KEIBA AI Health Monitor === {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for name, result in results.items():
            icon = {"ok": "[OK]", "warning": "[WARN]", "error": "[ERROR]"}.get(result["status"], "[?]")
            print(f"  {icon:8s} {name:20s} {result['message']}")
        print(f"\n  Summary: {len(results) - len(failures) - len(warnings)} OK, {len(warnings)} warnings, {len(failures)} errors")

    # Send alert on errors
    if failures:
        send_alert(failures, results)
        if is_cron:
            sys.exit(1)

    # Save last check result
    try:
        state_file = Path("/opt/keiba-predictor/data/health_state.json")
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "errors": failures,
                "warnings": warnings,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
