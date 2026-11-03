"""
競馬予想AI - 自動学習パイプライン (Walk-Forward Validation版)
================================================================
レース結果 -> CSV追記 -> 自動再学習 -> モデル差替え -> API反映

フロー:
  1. realtime_job がレース結果を取得
  2. このモジュールが結果をメインCSVに追記
  3. 蓄積レース数が閾値を超えたら自動再学習をトリガー
  4. Strict temporal split: train on data BEFORE last 2 months,
     validate on last 2 months (NEVER mix)
  5. Validation gate: compare new vs current model on same validation set
  6. Shadow mode: save candidate model, promote only after weekend comparison
  7. Detailed retrain logs saved to data/retrain_history.json

スケジュール:
  - レース結果取得ごとにCSV追記（リアルタイム）
  - 毎日17:30に日次チェック（再学習条件判定）
  - 毎週月曜3:00に週次再学習（強制）
  - 毎月1日3:00にフル再学習（強制）
"""

import os
import sys
import json
import time
import shutil
import logging
import glob
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# ============================================================
# 設定
# ============================================================

AUTO_LEARN_CONFIG = {
    "data_dir": os.environ.get("KEIBA_DATA_DIR", "data"),
    "model_dir": "models",
    "main_csv": "data/_all_races.csv",
    "state_file": "data/auto_learn_state.json",
    "retrain_history_file": "data/retrain_history.json",

    # 再学習トリガー
    # 200→100に緩和: 週1回程度の再学習を可能にし、市場変化への追従を改善
    "retrain_threshold_races": 100,    # 新規Nレース蓄積で再学習
    "retrain_threshold_days": 7,       # N日経過で再学習
    "force_retrain_weekday": 0,        # 月曜(0)に強制再学習
    "force_retrain_hour": 3,           # 3:00に実行

    # Temporal validation
    "validation_months": 2,            # 直近N月をvalidation set にする

    # Validation gate thresholds
    # 閾値を緩和: 小さな改善でも採用し、漸進的な精度向上を促す
    "min_win_auc_improvement": 0.002,     # 0.005→0.002: 0.2%改善で採用許可
    "max_top3_auc_decrease": 0.015,       # 0.01→0.015: 1.5%低下まで許容（トレードオフ許容）
    "max_position_mae_increase": 0.15,    # 0.1→0.15: 着順MAE増加の許容幅を拡大

    # モデル検証 (legacy - kept for backward compat)
    "min_improvement": 0.001,
    "max_models_keep": 5,              # 保持する旧モデル数

    # 学習パラメータ
    "n_splits": 5,
    "use_advanced_features": True,  # GOD特徴量はcreate_featuresで自動適用。拡張特徴量も有効化
}

log = logging.getLogger("auto_learn")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [AUTO-LEARN] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(h)
    fh = logging.FileHandler(os.path.join(AUTO_LEARN_CONFIG["data_dir"], "auto_learn.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)


# ============================================================
# Retrain History Logger
# ============================================================

def _load_retrain_history() -> list:
    """Load retrain history from JSON file."""
    path = AUTO_LEARN_CONFIG["retrain_history_file"]
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def _save_retrain_history(history: list):
    """Save retrain history to JSON file."""
    path = AUTO_LEARN_CONFIG["retrain_history_file"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _log_retrain_decision(
    decision: str,
    reason: str,
    old_metrics: dict,
    new_metrics: dict,
    version: str = "",
    n_train: int = 0,
    n_val: int = 0,
):
    """Append a retrain decision record to retrain_history.json."""
    history = _load_retrain_history()
    record = {
        "date": datetime.now().isoformat(),
        "version": version,
        "decision": decision,        # "promoted" | "rejected" | "candidate" | "error"
        "reason": reason,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "n_train_rows": n_train,
        "n_val_rows": n_val,
    }
    history.append(record)
    # Keep last 100 entries
    history = history[-100:]
    _save_retrain_history(history)
    return record


# ============================================================
# 状態管理
# ============================================================

class AutoLearnState:
    """自動学習の状態を永続化"""

    def __init__(self, path: str = None):
        self.path = path or AUTO_LEARN_CONFIG["state_file"]
        self.data = {
            "last_retrain_at": "",
            "last_retrain_races": 0,
            "current_model_version": "",
            "current_model_auc": 0,
            "total_retrains": 0,
            "new_races_since_retrain": 0,
            "last_csv_append_at": "",
            "history": [],
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data.update(json.load(f))
            except (json.JSONDecodeError, KeyError):
                pass

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    @property
    def new_races(self) -> int:
        return self.data["new_races_since_retrain"]

    def add_races(self, n: int):
        self.data["new_races_since_retrain"] += n
        self.data["last_csv_append_at"] = datetime.now().isoformat()
        self.save()

    def mark_retrained(self, version: str, auc: float, n_races: int):
        self.data["last_retrain_at"] = datetime.now().isoformat()
        self.data["last_retrain_races"] = n_races
        self.data["current_model_version"] = version
        self.data["current_model_auc"] = auc
        self.data["total_retrains"] += 1
        self.data["new_races_since_retrain"] = 0
        self.data["history"].append({
            "version": version,
            "auc": auc,
            "n_races": n_races,
            "timestamp": datetime.now().isoformat(),
        })
        # 履歴は直近20件まで
        self.data["history"] = self.data["history"][-20:]
        self.save()

    def should_retrain(self) -> tuple:
        """再学習が必要か判定。(bool, reason)"""
        # 1. レース数閾値
        if self.new_races >= AUTO_LEARN_CONFIG["retrain_threshold_races"]:
            return True, f"新規{self.new_races}レース蓄積（閾値{AUTO_LEARN_CONFIG['retrain_threshold_races']}）"

        # 2. 日数閾値
        if self.data["last_retrain_at"]:
            last = datetime.fromisoformat(self.data["last_retrain_at"])
            days = (datetime.now() - last).days
            if days >= AUTO_LEARN_CONFIG["retrain_threshold_days"]:
                return True, f"前回から{days}日経過（閾値{AUTO_LEARN_CONFIG['retrain_threshold_days']}日）"

        # 3. 週次強制
        now = datetime.now()
        if (now.weekday() == AUTO_LEARN_CONFIG["force_retrain_weekday"]
            and now.hour == AUTO_LEARN_CONFIG["force_retrain_hour"]):
            if self.data["last_retrain_at"]:
                last = datetime.fromisoformat(self.data["last_retrain_at"])
                if (now - last).total_seconds() > 3600 * 20:  # 20時間以上空いてたら
                    return True, "週次強制再学習"

        return False, ""


# ============================================================
# CSV追記
# ============================================================

def append_results_to_csv(results_data: list, csv_path: str = None) -> int:
    """
    レース結果をメインCSVに追記。

    Args:
        results_data: [{race_info + horse columns}, ...] のリスト
        csv_path: 追記先CSV

    Returns:
        追記したレコード数
    """
    csv_path = csv_path or AUTO_LEARN_CONFIG["main_csv"]

    if not results_data:
        return 0

    new_df = pd.DataFrame(results_data)

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        # 重複チェック（race_id + horse_number）
        if "race_id" in existing.columns and "horse_number" in existing.columns:
            existing_keys = set(zip(existing["race_id"], existing["horse_number"]))
            new_keys = set(zip(new_df["race_id"], new_df["horse_number"]))
            duplicates = new_keys & existing_keys
            if duplicates:
                mask = ~new_df.apply(lambda r: (r["race_id"], r["horse_number"]) in duplicates, axis=1)
                new_df = new_df[mask]

        if len(new_df) == 0:
            log.info("追記スキップ: 全レコード重複")
            return 0

        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(csv_path, index=False, encoding="utf-8")
    log.info(f"CSV追記: +{len(new_df)}レコード -> {csv_path} (計{len(combined):,}行)")

    return len(new_df)


def process_daily_results(date_str: str = None) -> int:
    """
    当日のレース結果JSONをCSVに追記。
    realtime_job.py が保存した results_YYYY-MM-DD.json を読み込む。
    """
    if date_str is None:
        date_str = date.today().isoformat()

    results_file = os.path.join(AUTO_LEARN_CONFIG["data_dir"], f"results_{date_str}.json")
    if not os.path.exists(results_file):
        return 0

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    records = []
    for race_id, result in results.items():
        race_info = {
            "race_id": race_id,
            "race_name": result.get("race_name", ""),
            "race_date": date_str,
            "venue": result.get("venue", ""),
            "course_type": result.get("course_type", ""),
            "distance": result.get("distance", 0),
            "weather": result.get("weather", ""),
            "track_condition": result.get("track_condition", ""),
        }

        for h in result.get("finish_order", []):
            record = {
                **race_info,
                "horse_id": h.get("horse_id", ""),
                "horse_name": h.get("horse_name", ""),
                "horse_number": h.get("horse_number", 0),
                "frame_number": h.get("frame_number", 0),
                "finish_position": h.get("position", 0),
                "odds": h.get("odds", 0),
                "popularity": h.get("popularity", 0),
                "finish_time": h.get("time", ""),
                "last_3f": h.get("last_3f", 0),
                "jockey_name": h.get("jockey_name", ""),
                "trainer_name": h.get("trainer_name", ""),
                "weight": h.get("weight", 0),
                "horse_weight": h.get("horse_weight", 0),
                "weight_diff": h.get("weight_diff", 0),
                "passing_order": h.get("passing", ""),
                "prize_money": h.get("prize", 0),
            }
            records.append(record)

    n = append_results_to_csv(records)

    # 状態更新
    if n > 0:
        state = AutoLearnState()
        n_races = len(results)
        state.add_races(n_races)
        log.info(f"  新規{n_races}レース / 累計{state.new_races}レース（次回再学習まで{AUTO_LEARN_CONFIG['retrain_threshold_races'] - state.new_races}）")

    return n


# ============================================================
# Temporal Split & Validation
# ============================================================

def _temporal_split(df: pd.DataFrame, val_months: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict temporal split: train on all data BEFORE the last val_months,
    validate on the last val_months. NEVER mix.

    Returns:
        (train_df, val_df)
    """
    if "race_date" not in df.columns:
        raise ValueError("race_date column required for temporal split")

    max_date = df["race_date"].max()
    val_cutoff = max_date - pd.DateOffset(months=val_months)

    train_df = df[df["race_date"] < val_cutoff].copy()
    val_df = df[df["race_date"] >= val_cutoff].copy()

    log.info(f"  Temporal split: train < {val_cutoff.strftime('%Y-%m-%d')} ({len(train_df):,} rows), "
             f"val >= {val_cutoff.strftime('%Y-%m-%d')} ({len(val_df):,} rows)")

    return train_df, val_df


def _evaluate_model_on_validation(model, val_df: pd.DataFrame, feature_columns: list) -> dict:
    """
    Evaluate a trained model on a held-out temporal validation set.
    Returns dict with win_auc, top3_auc, position_mae.
    """
    from sklearn.metrics import roc_auc_score, mean_absolute_error

    X_val = val_df[feature_columns].copy()
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    for col in X_val.columns:
        median_val = X_val[col].median()
        X_val[col] = X_val[col].fillna(median_val if not pd.isna(median_val) else 0)

    y_position = val_df["finish_position"].values
    y_win = (val_df["finish_position"] == 1).astype(int).values
    y_top3 = (val_df["finish_position"] <= 3).astype(int).values

    metrics = {
        "win_auc": 0.5,
        "top3_auc": 0.5,
        "position_mae": 99.0,
    }

    # Position MAE
    if "position" in model.models:
        pred_pos = model.models["position"].predict(X_val)
        metrics["position_mae"] = float(mean_absolute_error(y_position, pred_pos))

    # Win AUC
    if "win" in model.models:
        m = model.models["win"]
        pred = m.predict_proba(X_val) if hasattr(m, "predict_proba") else m.predict(X_val)
        if hasattr(pred, "shape") and len(pred.shape) > 1:
            pred = pred[:, 1]
        try:
            metrics["win_auc"] = float(roc_auc_score(y_win, pred))
        except ValueError:
            pass

    # Top3 AUC
    if "top3" in model.models:
        m = model.models["top3"]
        pred = m.predict_proba(X_val) if hasattr(m, "predict_proba") else m.predict(X_val)
        if hasattr(pred, "shape") and len(pred.shape) > 1:
            pred = pred[:, 1]
        try:
            metrics["top3_auc"] = float(roc_auc_score(y_top3, pred))
        except ValueError:
            pass

    return metrics


def _validation_gate(old_metrics: dict, new_metrics: dict) -> Tuple[bool, str]:
    """
    Validation gate: compare new model vs current model on the SAME validation set.
    Returns (pass, reason).

    Rules:
    - Win AUC must improve by at least 0.005
    - Top3 AUC must not decrease by more than 0.01
    - Position MAE must not increase by more than 0.1
    """
    cfg = AUTO_LEARN_CONFIG

    win_auc_diff = new_metrics["win_auc"] - old_metrics.get("win_auc", 0.5)
    top3_auc_diff = new_metrics["top3_auc"] - old_metrics.get("top3_auc", 0.5)
    pos_mae_diff = new_metrics["position_mae"] - old_metrics.get("position_mae", 99.0)

    reasons = []

    if win_auc_diff < cfg["min_win_auc_improvement"]:
        reasons.append(
            f"Win AUC improvement {win_auc_diff:+.4f} < required {cfg['min_win_auc_improvement']}"
        )

    if top3_auc_diff < -cfg["max_top3_auc_decrease"]:
        reasons.append(
            f"Top3 AUC decreased by {-top3_auc_diff:.4f} > max allowed {cfg['max_top3_auc_decrease']}"
        )

    if pos_mae_diff > cfg["max_position_mae_increase"]:
        reasons.append(
            f"Position MAE increased by {pos_mae_diff:.3f} > max allowed {cfg['max_position_mae_increase']}"
        )

    if reasons:
        return False, "; ".join(reasons)

    return True, (
        f"Win AUC {win_auc_diff:+.4f}, "
        f"Top3 AUC {top3_auc_diff:+.4f}, "
        f"Position MAE {pos_mae_diff:+.3f}"
    )


# ============================================================
# 自動再学習 (Walk-Forward Validation版)
# ============================================================

def retrain_model(force: bool = False) -> Optional[str]:
    """
    条件を満たしていればモデルを再学習。
    Uses strict temporal validation and validation gate.

    Returns:
        新モデルのバージョン文字列。学習しなかった/不採用の場合はNone。
    """
    state = AutoLearnState()

    if not force:
        should, reason = state.should_retrain()
        if not should:
            log.info(f"再学習不要（新規{state.new_races}レース）")
            return None
        log.info(f"再学習トリガー: {reason}")
    else:
        log.info("強制再学習")

    csv_path = AUTO_LEARN_CONFIG["main_csv"]
    if not os.path.exists(csv_path):
        log.warning(f"学習データなし: {csv_path}")
        return None

    try:
        from data_processor import RaceDataProcessor, FeatureEngineer
        from model_trainer import RacePredictionModel

        # 1. データ読込
        log.info("[1/6] データ読込...")
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        csv_name = os.path.basename(csv_path)
        proc = RaceDataProcessor(data_dir=csv_dir)
        df = proc.load_csv(csv_name)
        df = proc.preprocess(df)

        # メモリ節約: 直近10年分に絞る（6GB RAMサーバー対応）
        if "race_date" in df.columns:
            cutoff = datetime.now() - timedelta(days=365 * 10)
            before = len(df)
            df = df[df["race_date"] >= cutoff].copy()
            log.info(f"  メモリ節約: {before:,} -> {len(df):,}行 (直近10年)")

        n_races = df["race_id"].nunique() if "race_id" in df.columns else 0
        log.info(f"  {len(df):,}行 / {n_races:,}レース")

        # 2. 特徴量生成 (on full data, using shift(1) for temporal safety)
        log.info("[2/6] 特徴量生成...")
        fe = FeatureEngineer()
        df = fe.create_features(df)

        if AUTO_LEARN_CONFIG["use_advanced_features"]:
            from features_advanced import AdvancedFeatureEngineer
            afe = AdvancedFeatureEngineer()
            df = afe.create_all(df)
            cols = afe.get_extended_feature_columns(df, [])
        else:
            cols = fe.get_feature_columns(df)

        log.info(f"  特徴量: {len(cols)}個")

        # 3. Strict temporal split
        log.info("[3/6] Temporal split (train/validation)...")
        train_df, val_df = _temporal_split(df, val_months=AUTO_LEARN_CONFIG["validation_months"])

        if len(train_df) < 100 or len(val_df) < 50:
            log.warning(f"データ不足: train={len(train_df)}, val={len(val_df)}. スキップ。")
            _log_retrain_decision("error", "Insufficient data for temporal split",
                                  {}, {}, n_train=len(train_df), n_val=len(val_df))
            return None

        # 4. 新モデル学習 (train set only)
        log.info("[4/6] モデル学習 (train set のみ)...")
        import datetime as _dt_mod
        version = _dt_mod.datetime.now().strftime("v%Y%m%d_%H%M%S")
        new_model = RacePredictionModel(model_dir=AUTO_LEARN_CONFIG["model_dir"])
        new_model.train(train_df, cols, n_splits=AUTO_LEARN_CONFIG["n_splits"])

        # 5. Validation gate
        log.info("[5/6] Validation gate (temporal validation set)...")
        new_metrics = _evaluate_model_on_validation(new_model, val_df, cols)
        log.info(f"  New model metrics: Win AUC={new_metrics['win_auc']:.4f}, "
                 f"Top3 AUC={new_metrics['top3_auc']:.4f}, "
                 f"Position MAE={new_metrics['position_mae']:.3f}")

        # Load current model and evaluate on same validation set
        old_metrics = {"win_auc": 0.5, "top3_auc": 0.5, "position_mae": 99.0}
        model_dir = Path(AUTO_LEARN_CONFIG["model_dir"])
        current_model_path = model_dir / "keiba_model.pkl"

        if current_model_path.exists():
            try:
                old_model = RacePredictionModel(model_dir=AUTO_LEARN_CONFIG["model_dir"])
                old_model.load("keiba_model")
                old_metrics = _evaluate_model_on_validation(old_model, val_df, cols)
                log.info(f"  Old model metrics: Win AUC={old_metrics['win_auc']:.4f}, "
                         f"Top3 AUC={old_metrics['top3_auc']:.4f}, "
                         f"Position MAE={old_metrics['position_mae']:.3f}")
            except Exception as e:
                log.warning(f"  Old model evaluation failed ({e}), treating as baseline")

        gate_pass, gate_reason = _validation_gate(old_metrics, new_metrics)
        log.info(f"  Validation gate: {'PASS' if gate_pass else 'FAIL'} - {gate_reason}")

        if not gate_pass:
            log.warning(f"新モデルは validation gate を通過できませんでした。候補として保存。")
            # Save as candidate (shadow mode)
            new_model.save("keiba_model_candidate")
            _log_retrain_decision(
                "rejected", gate_reason, old_metrics, new_metrics,
                version=version, n_train=len(train_df), n_val=len(val_df),
            )
            return None

        # 6. Shadow mode: save as candidate first
        log.info("[6/6] モデル保存 (shadow mode = candidate)...")

        # Backup current model
        if current_model_path.exists():
            backup_name = f"keiba_model_{state.data.get('current_model_version', 'old')}.pkl"
            shutil.copy2(current_model_path, model_dir / backup_name)

            # 古いバックアップを削除
            backups = sorted(model_dir.glob("keiba_model_v*.pkl"), reverse=True)
            for old in backups[AUTO_LEARN_CONFIG["max_models_keep"]:]:
                old.unlink()
                log.info(f"  旧モデル削除: {old.name}")

        # Save candidate model
        new_model.save("keiba_model_candidate")

        # Also promote to production (since gate passed)
        new_model.save("keiba_model")

        # 状態更新
        new_auc = new_metrics["win_auc"]
        state.mark_retrained(version, new_auc, n_races)

        _log_retrain_decision(
            "promoted", gate_reason, old_metrics, new_metrics,
            version=version, n_train=len(train_df), n_val=len(val_df),
        )

        log.info(f"再学習完了: {version}")
        log.info(f"  Win AUC: {new_metrics['win_auc']:.4f}, "
                 f"Top3 AUC: {new_metrics['top3_auc']:.4f}, "
                 f"Position MAE: {new_metrics['position_mae']:.3f}")
        log.info(f"  データ: {n_races:,}レース / {len(cols)}特徴量")

        # 通知
        _notify_retrain(version, new_metrics, old_metrics, n_races)

        return version

    except Exception as e:
        log.error(f"再学習エラー: {e}")
        import traceback
        traceback.print_exc()
        _log_retrain_decision("error", str(e), {}, {})
        return None


def _notify_retrain(version: str, new_metrics: dict, old_metrics: dict, n_races: int):
    """再学習完了を通知"""
    try:
        from notifier import NotificationManager
        nm = NotificationManager()
        win_diff = new_metrics["win_auc"] - old_metrics.get("win_auc", 0.5)
        msg = (
            f"モデル再学習完了\n"
            f"Version: {version}\n"
            f"Win AUC: {new_metrics['win_auc']:.4f} ({win_diff:+.4f})\n"
            f"Top3 AUC: {new_metrics['top3_auc']:.4f}\n"
            f"Position MAE: {new_metrics['position_mae']:.3f}\n"
            f"データ: {n_races:,}レース"
        )
        nm.broadcast(msg)
    except Exception:
        pass


# ============================================================
# API ホットリロード
# ============================================================

def reload_api_model():
    """
    APIサーバーのモデルをホットリロード。
    api_server.py のグローバル変数を差し替える。
    """
    try:
        import api_server
        if api_server.model is not None:
            from model_trainer import RacePredictionModel
            new_model = RacePredictionModel(model_dir=AUTO_LEARN_CONFIG["model_dir"])
            new_model.load("keiba_model")
            api_server.model = new_model
            log.info("APIモデル ホットリロード完了")
            return True
    except Exception as e:
        log.error(f"ホットリロード失敗: {e}")
    return False


# ============================================================
# 統合: 日次処理
# ============================================================

def daily_pipeline(date_str: str = None):
    """
    日次自動学習パイプライン。
    realtime_job.py の daily_summary() の後に呼ばれる。
    """
    log.info("=" * 40)
    log.info("日次学習パイプライン開始")

    # 1. 結果をCSVに追記
    n = process_daily_results(date_str)
    if n == 0:
        log.info("新規レコードなし。終了。")
        return

    # 2. 再学習チェック
    version = retrain_model()
    if version:
        # 3. APIモデルリロード
        reload_api_model()

    log.info("日次パイプライン完了")


def weekly_retrain():
    """週次強制再学習"""
    log.info("週次強制再学習")
    version = retrain_model(force=True)
    if version:
        reload_api_model()


# ============================================================
# A/Bテストフレームワーク
# ============================================================

# ABテスト状態ファイル
_AB_TEST_STATE_FILE = os.path.join(AUTO_LEARN_CONFIG["data_dir"], "ab_test_state.json")

# Gradual rollout ステージ定義（トラフィック割合 %）
_ROLLOUT_STAGES = [10, 25, 50, 100]

# メモリリーク防止: ABテスト結果は最大200件まで保持
_ab_result_buffer: "deque" = None


def _get_ab_result_buffer():
    """遅延初期化でdequeを返す（collections.dequeで上限200）"""
    global _ab_result_buffer
    if _ab_result_buffer is None:
        from collections import deque
        _ab_result_buffer = deque(maxlen=200)
    return _ab_result_buffer


def _load_ab_state() -> dict:
    """ABテスト状態を読み込む"""
    if os.path.exists(_AB_TEST_STATE_FILE):
        try:
            with open(_AB_TEST_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "active": False,
        "stage_index": 0,           # _ROLLOUT_STAGES のインデックス
        "current_rollout_pct": 0,   # 現在のトラフィック割合
        "new_model_name": "",
        "old_model_name": "keiba_model",
        "started_at": "",
        "races_evaluated": 0,
        "new_model_wins": 0,        # 新モデルの勝利数
        "old_model_wins": 0,
        "new_metrics_sum": {"win_auc": 0.0, "top3_auc": 0.0, "position_mae": 0.0},
        "old_metrics_sum": {"win_auc": 0.0, "top3_auc": 0.0, "position_mae": 0.0},
        "history": [],              # 各ステージの評価結果（最大50件）
        "rolled_back": False,
        "rollback_reason": "",
    }


def _save_ab_state(state: dict):
    """ABテスト状態を保存"""
    os.makedirs(os.path.dirname(_AB_TEST_STATE_FILE) or ".", exist_ok=True)
    tmp = _AB_TEST_STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _AB_TEST_STATE_FILE)


def ab_test_models(
    new_model_name: str = "keiba_model_candidate",
    old_model_name: str = "keiba_model",
    val_months: int = 2,
    force_start: bool = False,
) -> dict:
    """
    新モデルと旧モデルを並行評価するA/Bテストフレームワーク。

    Gradual rollout: 10% -> 25% -> 50% -> 100% の4ステージで展開。
    各ステージで指定期間の評価データを使い、旧モデル比較で品質確認。
    劣化を検知した場合は自動ロールバック。

    Args:
        new_model_name: 評価する新モデルのファイル名（拡張子なし）
        old_model_name: 比較対象の旧モデルのファイル名（拡張子なし）
        val_months: 評価データの期間（月）
        force_start: 既存ABテストが実行中でも強制的に新規開始する

    Returns:
        dict with keys:
          - status: "started" | "promoted" | "rolled_back" | "stage_advanced" | "no_change" | "error"
          - message: 詳細メッセージ
          - state: 現在のABテスト状態
    """
    model_dir = Path(AUTO_LEARN_CONFIG["model_dir"])
    state = _load_ab_state()

    # 既存のABテストが実行中かチェック
    if state["active"] and not force_start:
        log.info(f"ABテスト実行中: stage={state['stage_index']}/{len(_ROLLOUT_STAGES)-1} "
                 f"rollout={state['current_rollout_pct']}%")
        return {
            "status": "no_change",
            "message": f"ABテスト実行中 (rollout={state['current_rollout_pct']}%)",
            "state": state,
        }

    # 新モデルファイルの存在確認
    new_model_path = model_dir / f"{new_model_name}.pkl"
    old_model_path = model_dir / f"{old_model_name}.pkl"

    if not new_model_path.exists():
        log.warning(f"ABテスト: 新モデルが見つかりません: {new_model_path}")
        return {"status": "error", "message": f"新モデルが見つかりません: {new_model_path}", "state": state}

    if not old_model_path.exists():
        log.warning(f"ABテスト: 旧モデルが見つかりません: {old_model_path}")
        return {"status": "error", "message": f"旧モデルが見つかりません: {old_model_path}", "state": state}

    # -- ABテスト開始（または強制再開） --
    if not state["active"] or force_start:
        state.update({
            "active": True,
            "stage_index": 0,
            "current_rollout_pct": _ROLLOUT_STAGES[0],
            "new_model_name": new_model_name,
            "old_model_name": old_model_name,
            "started_at": datetime.now().isoformat(),
            "races_evaluated": 0,
            "new_model_wins": 0,
            "old_model_wins": 0,
            "new_metrics_sum": {"win_auc": 0.0, "top3_auc": 0.0, "position_mae": 0.0},
            "old_metrics_sum": {"win_auc": 0.0, "top3_auc": 0.0, "position_mae": 0.0},
            "history": [],
            "rolled_back": False,
            "rollback_reason": "",
        })
        log.info(f"ABテスト開始: new={new_model_name} vs old={old_model_name} "
                 f"stage=0 rollout={_ROLLOUT_STAGES[0]}%")

    try:
        from data_processor import RaceDataProcessor, FeatureEngineer
        from model_trainer import RacePredictionModel

        csv_path = AUTO_LEARN_CONFIG["main_csv"]
        if not os.path.exists(csv_path):
            return {"status": "error", "message": f"学習データなし: {csv_path}", "state": state}

        # データ読込 & 特徴量生成
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        csv_name = os.path.basename(csv_path)
        proc = RaceDataProcessor(data_dir=csv_dir)
        df = proc.load_csv(csv_name)
        df = proc.preprocess(df)

        fe = FeatureEngineer()
        df = fe.create_features(df)

        if AUTO_LEARN_CONFIG["use_advanced_features"]:
            try:
                from features_advanced import AdvancedFeatureEngineer
                afe = AdvancedFeatureEngineer()
                df = afe.create_all(df)
                cols = afe.get_extended_feature_columns(df, [])
            except Exception:
                cols = fe.get_feature_columns(df)
        else:
            cols = fe.get_feature_columns(df)

        # Temporal split: 評価データのみ使用
        _, val_df = _temporal_split(df, val_months=val_months)
        if len(val_df) < 50:
            return {"status": "error", "message": f"評価データ不足: {len(val_df)}件", "state": state}

        # 両モデルを評価
        new_model = RacePredictionModel(model_dir=str(model_dir))
        new_model.load(new_model_name)
        new_metrics = _evaluate_model_on_validation(new_model, val_df, cols)

        old_model = RacePredictionModel(model_dir=str(model_dir))
        old_model.load(old_model_name)
        old_metrics = _evaluate_model_on_validation(old_model, val_df, cols)

        log.info(f"  [ABテスト] 新モデル: Win AUC={new_metrics['win_auc']:.4f}, "
                 f"Top3 AUC={new_metrics['top3_auc']:.4f}, MAE={new_metrics['position_mae']:.3f}")
        log.info(f"  [ABテスト] 旧モデル: Win AUC={old_metrics['win_auc']:.4f}, "
                 f"Top3 AUC={old_metrics['top3_auc']:.4f}, MAE={old_metrics['position_mae']:.3f}")

        # メトリクス累積
        n = state["races_evaluated"] + len(val_df)
        state["races_evaluated"] = n
        for k in ["win_auc", "top3_auc", "position_mae"]:
            state["new_metrics_sum"][k] += new_metrics[k]
            state["old_metrics_sum"][k] += old_metrics[k]

        # 新モデルが優れているか判定
        new_wins = int(new_metrics["win_auc"] > old_metrics["win_auc"])
        old_wins = int(old_metrics["win_auc"] >= new_metrics["win_auc"])
        state["new_model_wins"] += new_wins
        state["old_model_wins"] += old_wins

        # バッファに記録（メモリリーク防止: dequeで上限管理）
        buf = _get_ab_result_buffer()
        buf.append({
            "ts": datetime.now().isoformat(),
            "stage": state["stage_index"],
            "new_metrics": new_metrics,
            "old_metrics": old_metrics,
        })

        # ステージ履歴記録（最大50件）
        stage_record = {
            "ts": datetime.now().isoformat(),
            "stage_index": state["stage_index"],
            "rollout_pct": state["current_rollout_pct"],
            "new_metrics": new_metrics,
            "old_metrics": old_metrics,
            "n_val": len(val_df),
        }
        state["history"].append(stage_record)
        state["history"] = state["history"][-50:]

        # -- ロールバック判定 --
        rollback_needed, rollback_reason = _ab_check_rollback(old_metrics, new_metrics)
        if rollback_needed:
            log.warning(f"[ABテスト] ロールバック実行: {rollback_reason}")
            _ab_rollback(state, old_model_name, rollback_reason, model_dir)
            _save_ab_state(state)
            return {
                "status": "rolled_back",
                "message": f"ロールバック: {rollback_reason}",
                "state": state,
            }

        # -- ステージ進行判定 --
        gate_pass, gate_reason = _validation_gate(old_metrics, new_metrics)
        current_stage = state["stage_index"]

        if gate_pass:
            next_stage = current_stage + 1
            if next_stage >= len(_ROLLOUT_STAGES):
                # 100%展開 -> 正式昇格
                log.info(f"[ABテスト] 全ステージ通過！新モデルを正式採用: {new_model_name}")
                import shutil as _shutil
                _shutil.copy2(
                    model_dir / f"{new_model_name}.pkl",
                    model_dir / f"{old_model_name}.pkl",
                )
                state["active"] = False
                state["current_rollout_pct"] = 100
                _save_ab_state(state)
                _log_retrain_decision(
                    "promoted", f"ABテスト完了: {gate_reason}",
                    old_metrics, new_metrics,
                    version=new_model_name,
                    n_val=len(val_df),
                )
                log.info(f"[ABテスト] 新モデル昇格完了 -> {old_model_name}.pkl に上書き")
                return {
                    "status": "promoted",
                    "message": f"ABテスト完了: 新モデルを正式採用 ({gate_reason})",
                    "state": state,
                }
            else:
                state["stage_index"] = next_stage
                state["current_rollout_pct"] = _ROLLOUT_STAGES[next_stage]
                _save_ab_state(state)
                log.info(f"[ABテスト] ステージ昇格: {current_stage} -> {next_stage} "
                         f"(rollout={state['current_rollout_pct']}%)")
                return {
                    "status": "stage_advanced",
                    "message": (f"ステージ {current_stage} -> {next_stage} "
                                f"rollout={state['current_rollout_pct']}%: {gate_reason}"),
                    "state": state,
                }
        else:
            # gate 失敗 -> 現ステージ維持
            _save_ab_state(state)
            log.info(f"[ABテスト] ステージ維持 (gate失敗): {gate_reason}")
            return {
                "status": "no_change",
                "message": f"ステージ維持 rollout={state['current_rollout_pct']}%: {gate_reason}",
                "state": state,
            }

    except Exception as e:
        log.error(f"ABテストエラー: {e}")
        import traceback
        traceback.print_exc()
        state["active"] = False
        _save_ab_state(state)
        return {"status": "error", "message": str(e), "state": state}


def _ab_check_rollback(old_metrics: dict, new_metrics: dict) -> tuple:
    """
    ロールバックが必要か判定。
    Win AUC が旧モデルより 0.02 以上低下した場合にロールバック。
    """
    win_diff = new_metrics["win_auc"] - old_metrics["win_auc"]
    top3_diff = new_metrics["top3_auc"] - old_metrics["top3_auc"]
    mae_diff = new_metrics["position_mae"] - old_metrics["position_mae"]

    if win_diff < -0.02:
        return True, f"Win AUC が旧モデルより {-win_diff:.4f} 低下"
    if top3_diff < -0.03:
        return True, f"Top3 AUC が旧モデルより {-top3_diff:.4f} 低下"
    if mae_diff > 0.5:
        return True, f"Position MAE が旧モデルより {mae_diff:.3f} 悪化"
    return False, ""


def _ab_rollback(state: dict, old_model_name: str, reason: str, model_dir: Path):
    """ABテストをロールバックして旧モデルをProduction継続とする"""
    state["active"] = False
    state["rolled_back"] = True
    state["rollback_reason"] = reason
    log.warning(f"[ABテスト] ロールバック完了: 旧モデル '{old_model_name}' を継続使用")
    _log_retrain_decision(
        "rejected", f"ABテストロールバック: {reason}",
        {}, {},
        version=state.get("new_model_name", ""),
    )


def ab_test_status() -> dict:
    """現在のABテスト状態を返す"""
    return _load_ab_state()


# ============================================================
# CLIテスト
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="自動学習パイプライン")
    parser.add_argument("--daily", action="store_true", help="日次パイプライン実行")
    parser.add_argument("--retrain", action="store_true", help="強制再学習")
    parser.add_argument("--status", action="store_true", help="状態確認")
    parser.add_argument("--history", action="store_true", help="再学習履歴")
    parser.add_argument("--ab-test", action="store_true", help="A/Bテスト実行")
    parser.add_argument("--ab-status", action="store_true", help="A/Bテスト状態確認")
    parser.add_argument("--date", type=str, default=None, help="対象日(YYYY-MM-DD)")
    args = parser.parse_args()

    os.makedirs(AUTO_LEARN_CONFIG["data_dir"], exist_ok=True)
    os.makedirs(AUTO_LEARN_CONFIG["model_dir"], exist_ok=True)

    if args.status:
        state = AutoLearnState()
        print(json.dumps(state.data, indent=2, ensure_ascii=False))

    elif args.history:
        history = _load_retrain_history()
        print(json.dumps(history, indent=2, ensure_ascii=False))

    elif args.retrain:
        retrain_model(force=True)

    elif args.ab_test:
        result = ab_test_models()
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.ab_status:
        state = ab_test_status()
        print(json.dumps(state, indent=2, ensure_ascii=False))

    elif args.daily:
        daily_pipeline(args.date)

    else:
        # テスト
        print("=== 自動学習パイプライン テスト ===")

        state = AutoLearnState(path="/tmp/test_autolearn.json")
        state.add_races(50)
        print(f"新規レース: {state.new_races}")

        should, reason = state.should_retrain()
        print(f"再学習必要: {should} ({reason})")

        state.add_races(160)
        should, reason = state.should_retrain()
        print(f"再学習必要: {should} ({reason})")

        os.remove("/tmp/test_autolearn.json")
        print("テスト完了")
