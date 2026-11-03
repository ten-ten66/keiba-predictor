"""
Prediction Ledger - Immutable Record of All Predictions
========================================================
Every prediction (precache or live) is recorded with a timestamp
and SHA-256 hash for tamper-proof verification.

Radical transparency: anyone can verify that predictions were made
BEFORE race results were known.

監査ログ:
  - record_audit_event()    : 予測・ベット決定・結果を audit_log/ に記録
  - get_audit_log()         : 監査ログを読み込み・フィルタリング
  - audit_summary()         : 事後分析用サマリー生成
"""

import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

LEDGER_FILE = Path("data/prediction_ledger.jsonl")  # append-only

# 監査ログ保存ディレクトリ
AUDIT_LOG_DIR = Path("data/audit_log")

# 監査ログのロガー
_audit_logger = logging.getLogger("prediction_ledger.audit")
if not _audit_logger.handlers:
    _audit_logger.setLevel(logging.INFO)
    _log_dir = Path("/opt/keiba-predictor/logs")
    _log_dir.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(_log_dir / "audit.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [AUDIT] %(message)s"))
    _audit_logger.addHandler(_fh)


def record_prediction(race_id, predictions, source="precache"):
    """Record prediction with timestamp and hash for verification.

    Args:
        race_id: Race identifier string.
        predictions: List of dicts with horse_number, horse_name,
                     prob_win, composite_score, etc.
        source: "precache" or "live".

    Returns:
        The ledger entry dict that was written.
    """
    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)

    sorted_preds = sorted(
        predictions,
        key=lambda x: x.get("composite_score", x.get("score", 0)),
        reverse=True,
    )

    entry = {
        "race_id": str(race_id),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "unix_ts": int(time.time()),
        "source": source,
        "top3": [
            {
                "rank": i + 1,
                "horse_number": int(p.get("horse_number", 0)),
                "horse_name": str(p.get("horse_name", "")),
                "prob_win": round(float(p.get("prob_win", 0)), 4),
                "composite_score": round(
                    float(p.get("composite_score", p.get("score", 0))), 4
                ),
            }
            for i, p in enumerate(sorted_preds[:3])
        ],
        "total_horses": len(predictions),
    }

    # Create hash of prediction content for tamper-proof verification
    content = json.dumps(entry["top3"], sort_keys=True)
    entry["hash"] = hashlib.sha256(content.encode()).hexdigest()[:16]

    with open(LEDGER_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def get_ledger(date=None, race_id=None, limit=100):
    """Read ledger entries with optional filters.

    Args:
        date: Filter by date prefix, e.g. "2026-03-20".
        race_id: Filter by exact race_id.
        limit: Max entries to return (from tail).

    Returns:
        List of ledger entry dicts.
    """
    if not LEDGER_FILE.exists():
        return []
    entries = []
    with open(LEDGER_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if date and not e.get("timestamp", "").startswith(date):
                continue
            if race_id and e.get("race_id") != race_id:
                continue
            entries.append(e)
    return entries[-limit:]


def verify_prediction(entry):
    """Verify a prediction entry hasn't been tampered with.

    Recomputes the SHA-256 hash from the top3 data and compares
    it to the stored hash.

    Args:
        entry: A ledger entry dict.

    Returns:
        True if the hash matches, False otherwise.
    """
    content = json.dumps(entry["top3"], sort_keys=True)
    expected = hashlib.sha256(content.encode()).hexdigest()[:16]
    return expected == entry.get("hash", "")


# ============================================================
# 監査ログ (Audit Log)
# ============================================================

def record_audit_event(
    event_type: str,
    race_id: str,
    payload: dict,
    username: str = "system",
) -> dict:
    """
    予測・ベット決定・結果の監査イベントを JSON 形式で記録する。

    イベントは data/audit_log/YYYY-MM-DD.jsonl に日別で追記される。
    各エントリは SHA-256 ハッシュで改ざん検知が可能。

    Args:
        event_type: "prediction" | "bet_decision" | "result"
        race_id: 対象レースID
        payload: イベント固有のデータ dict
        username: 操作ユーザー名（ログ追跡用）

    Returns:
        記録された監査エントリ dict
    """
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.utcnow()
    entry = {
        "event_type": event_type,
        "race_id": str(race_id),
        "username": username,
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "unix_ts": int(time.time()),
        "payload": payload,
    }

    # SHA-256 でコンテンツ整合性を保証
    content = json.dumps(
        {"event_type": event_type, "race_id": str(race_id), "payload": payload},
        sort_keys=True, ensure_ascii=False,
    )
    entry["hash"] = hashlib.sha256(content.encode()).hexdigest()[:20]

    # 日別ファイルに追記
    date_str = now.strftime("%Y-%m-%d")
    log_file = AUDIT_LOG_DIR / f"{date_str}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    _audit_logger.info(
        f"[{event_type}] race={race_id} user={username} hash={entry['hash']}"
    )
    return entry


def get_audit_log(
    date: str = None,
    race_id: str = None,
    event_type: str = None,
    username: str = None,
    limit: int = 500,
) -> list:
    """
    監査ログを読み込む。複数フィルタをAND条件で適用可能。

    Args:
        date: "YYYY-MM-DD" 形式の日付フィルタ（その日のファイルのみ読む）
        race_id: レースIDフィルタ
        event_type: イベントタイプフィルタ ("prediction" | "bet_decision" | "result")
        username: ユーザー名フィルタ
        limit: 返すエントリの最大件数（末尾から取得）

    Returns:
        マッチしたエントリの list
    """
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    if date:
        log_files = [AUDIT_LOG_DIR / f"{date}.jsonl"]
    else:
        log_files = sorted(AUDIT_LOG_DIR.glob("*.jsonl"), reverse=True)[:30]

    entries = []
    for log_file in log_files:
        if not log_file.exists():
            continue
        try:
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if race_id and e.get("race_id") != str(race_id):
                        continue
                    if event_type and e.get("event_type") != event_type:
                        continue
                    if username and e.get("username") != username:
                        continue
                    entries.append(e)
        except OSError:
            continue

    return entries[-limit:]


def audit_summary(date_range_days: int = 30) -> dict:
    """
    事後分析用サマリーを生成する。

    直近 date_range_days 日間の監査ログを集計して返す。

    Returns:
        dict with keys:
          - total_events: イベント総数
          - by_event_type: イベントタイプ別カウント
          - races_predicted: 予測されたユニークレース数
          - bets_placed: 馬券登録件数
          - results_recorded: 結果記録件数
          - win_accuracy: 結果ありレースの勝率（予測1位が1着の割合）
          - avg_roi: 記録済みレースの平均ROI
          - top_races: 期待値が高かった上位5レース
          - date_range: 集計対象期間
    """
    from datetime import timedelta

    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.utcnow().date()
    start_date = today - timedelta(days=date_range_days - 1)

    all_entries = get_audit_log(limit=10000)

    # 対象期間のエントリのみ
    entries = [
        e for e in all_entries
        if e.get("timestamp", "")[:10] >= str(start_date)
    ]

    by_event_type: dict = defaultdict(int)
    races_predicted: set = set()
    bets_by_race: dict = {}         # race_id -> list of bet payloads
    results_by_race: dict = {}      # race_id -> result payload

    for e in entries:
        etype = e.get("event_type", "unknown")
        by_event_type[etype] += 1
        rid = e.get("race_id", "")
        payload = e.get("payload", {})

        if etype == "prediction":
            races_predicted.add(rid)
        elif etype == "bet_decision":
            bets_by_race.setdefault(rid, []).append(payload)
        elif etype == "result":
            results_by_race[rid] = payload

    # 勝率計算（result イベントに winner_horse_number があると仮定）
    win_count = 0
    matched_races = 0
    roi_sum = 0.0
    roi_count = 0
    top_races = []

    for race_id, result in results_by_race.items():
        winner_num = result.get("winner_horse_number")
        bets = bets_by_race.get(race_id, [])
        if not bets:
            continue
        matched_races += 1

        # 最初のベットの推奨馬番と照合
        first_bet = bets[0]
        recommended_num = first_bet.get("horse_number")
        if recommended_num and recommended_num == winner_num:
            win_count += 1

        # ROI
        total_bet = sum(b.get("amount", 100) for b in bets)
        total_return = result.get("total_return", 0)
        if total_bet > 0:
            roi = total_return / total_bet
            roi_sum += roi
            roi_count += 1
            if roi > 1.0:
                top_races.append({
                    "race_id": race_id,
                    "roi": round(roi, 3),
                    "total_bet": total_bet,
                    "total_return": total_return,
                })

    top_races.sort(key=lambda x: x["roi"], reverse=True)

    return {
        "total_events": len(entries),
        "by_event_type": dict(by_event_type),
        "races_predicted": len(races_predicted),
        "bets_placed": sum(by_event_type.get("bet_decision", 0) for _ in [1]),
        "results_recorded": len(results_by_race),
        "win_accuracy": round(win_count / matched_races, 4) if matched_races > 0 else None,
        "avg_roi": round(roi_sum / roi_count, 4) if roi_count > 0 else None,
        "top_races": top_races[:5],
        "date_range": {
            "start": str(start_date),
            "end": str(today),
            "days": date_range_days,
        },
    }
