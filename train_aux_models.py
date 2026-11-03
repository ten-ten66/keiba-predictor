"""
Train auxiliary models (segment calibrator / odds residual / quantile rank)
============================================================================
`_all_races.csv` を入力に、本番 predict パイプラインと同じ前処理 + 特徴量を
作り、3つの補助モデルを fit して pkl を書き出す。

出力:
  models/segment_calibrator.pkl
  models/odds_residual.pkl
  models/quantile_rank.pkl

使い方:
  # 全部学習
  venv/bin/python3 train_aux_models.py
  # 一部だけ
  venv/bin/python3 train_aux_models.py --only segment,quantile
  # 直近N日のデータだけ (debug/small)
  venv/bin/python3 train_aux_models.py --since-days 365

`model.predict()` 実行時に自動で検知・適用されるので、retrain 後に
`POST /api/reload-model` で反映可能。

再現性ポイント:
  - 時系列順で先頭90%を train、末尾10%を eval (aux モデルの sanity check 用)
  - 予測対象列 (finish_position / prize_money 等) は **マスクしない** — 学習では必要
  - segment calibrator は train 上の未校正確率 (別モデル出力を想定) が要る
    → ここでは簡易に `normalized_implied_prob` を「未校正確率」として使う
      (本来は model.predict 出力を通したいが API 起動が必要なので)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from data_processor import FeatureEngineer, RaceDataProcessor
from segment_calibration import SegmentCalibrator, derive_grade_group
from odds_residual import OddsResidualModel
from quantile_rank_model import QuantileRankModel

try:
    from features_advanced import AdvancedFeatureEngineer
    _ADV = True
except Exception:
    _ADV = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_aux")

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
ALL_RACES_CSV = DATA_DIR / "_all_races.csv"


def load_and_prepare(since_days: int | None = None) -> pd.DataFrame:
    if not ALL_RACES_CSV.exists():
        raise FileNotFoundError(f"{ALL_RACES_CSV} が存在しません。APIサーバの起動で自動生成されます")
    logger.info("loading %s", ALL_RACES_CSV)
    df = pd.read_csv(ALL_RACES_CSV, low_memory=False)
    logger.info("loaded: %d rows", len(df))

    if since_days is not None and "race_date" in df.columns:
        cutoff = pd.to_datetime(datetime.now() - timedelta(days=since_days))
        df["_rd"] = pd.to_datetime(df["race_date"], errors="coerce", format="mixed")
        df = df[df["_rd"] >= cutoff].copy()
        df = df.drop(columns=["_rd"])
        logger.info("after since_days=%d: %d rows", since_days, len(df))

    proc = RaceDataProcessor()
    proc.raw_data = df
    processed = proc.preprocess()
    logger.info("preprocess done: %d rows, %d cols", len(processed), len(processed.columns))

    fe = FeatureEngineer()
    featured = fe.create_features(processed)
    logger.info("features done: %d cols", len(featured.columns))

    if _ADV:
        try:
            adv = AdvancedFeatureEngineer()
            featured = adv.create_all(featured)
            logger.info("advanced features done: %d cols", len(featured.columns))
        except Exception as e:
            logger.warning("advanced features skipped: %s", e)

    # winner 列を追加
    if "finish_position" in featured.columns:
        featured["winner"] = (featured["finish_position"] == 1).astype(int)

    # grade_group
    if "grade" in featured.columns:
        featured["grade_group"] = featured["grade"].apply(derive_grade_group)
    elif "class_name" in featured.columns:
        featured["grade_group"] = featured["class_name"].apply(derive_grade_group)
    else:
        featured["grade_group"] = "unknown"

    return featured


def train_segment_calibrator(df: pd.DataFrame, out_path: Path) -> None:
    logger.info("=== segment_calibrator ===")
    sub = df.dropna(subset=["odds", "winner", "race_date"]).copy()
    if "course_type" not in sub.columns:
        sub["course_type"] = "unknown"
    # 未校正確率として正規化済み暗黙確率を使う (API 無しで回せるよう)
    from odds_residual import compute_baseline
    sub["prob_win"] = compute_baseline(sub).values
    sub = sub[sub["prob_win"] > 0]
    logger.info("data: %d rows", len(sub))
    cal = SegmentCalibrator(
        segment_cols=("course_type", "grade_group"),
        target_col="prob_win",
        label_col="winner",
        half_life_days=365,
        min_samples=500,
    )
    cal.fit(sub)
    cal.save(out_path)
    logger.info("saved: %s", out_path)
    logger.info("segments fitted: %d", len(cal.models_))


def train_odds_residual(df: pd.DataFrame, out_path: Path) -> None:
    logger.info("=== odds_residual ===")
    sub = df.dropna(subset=["odds", "winner"]).copy()
    # 使用可能な数値特徴量 (オッズ関連は内部で除外される)
    numeric = sub.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        "winner", "finish_position", "prize_money", "finish_time_sec",
        "year", "month", "day_of_week", "horse_number", "frame_number",
        "race_id",
    }
    feature_cols = [c for c in numeric if c not in exclude]
    logger.info("candidate features: %d", len(feature_cols))
    model = OddsResidualModel(n_estimators=200, learning_rate=0.05, max_depth=6)
    model.fit(sub, feature_cols)
    model.save(out_path)
    logger.info("saved: %s (features=%d, mae=%.4f)", out_path, len(model.feature_cols_), model.train_mae_)


def train_quantile_rank(df: pd.DataFrame, out_path: Path) -> None:
    logger.info("=== quantile_rank ===")
    sub = df.dropna(subset=["finish_position"]).copy()
    sub = sub[sub["finish_position"] > 0]
    numeric = sub.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        "finish_position", "prize_money", "finish_time_sec", "winner",
        "race_id", "horse_number", "year",
    }
    feature_cols = [c for c in numeric if c not in exclude]
    logger.info("candidate features: %d", len(feature_cols))
    qm = QuantileRankModel(
        quantiles=(0.1, 0.5, 0.9),
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=100,
    )
    qm.fit(sub, feature_cols=feature_cols, label_col="finish_position")
    qm.save(out_path)
    logger.info("saved: %s (features=%d)", out_path, len(qm.feature_cols_))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list: segment,odds,quantile")
    ap.add_argument("--since-days", type=int, default=None, help="直近N日だけ使う (debug用)")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = ap.parse_args(argv)

    picks = {"segment", "odds", "quantile"}
    if args.only:
        picks = {x.strip() for x in args.only.split(",") if x.strip()}

    args.models_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(since_days=args.since_days)

    if "segment" in picks:
        train_segment_calibrator(df, args.models_dir / "segment_calibrator.pkl")
    if "odds" in picks:
        train_odds_residual(df, args.models_dir / "odds_residual.pkl")
    if "quantile" in picks:
        train_quantile_rank(df, args.models_dir / "quantile_rank.pkl")

    logger.info("done")


if __name__ == "__main__":
    main()
