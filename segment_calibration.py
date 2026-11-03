"""
Segment-aware probability calibration with time-decay weighting
================================================================
RacePredictionModel が出す prob_win/prob_top3 を、course_type (芝/ダート) や
grade (G1/G2/G3/OP/条件戦) 等のセグメント別に isotonic 回帰でキャリブレーションする。

単一の global calibration だと:
  - 芝のG1と条件戦ダートで的中構造が違うため global fit がどちらかに偏る
  - 時間とともに JRA の賞金構造・レース編成が変わるので古い観測の重みを下げたい

使い方:
  cal = SegmentCalibrator()
  cal.fit(predictions_df, actuals_df)   # train
  probs_out = cal.transform(df_with_segment_cols)

ストレージ:
  cal.save("calibration.pkl")
  cal = SegmentCalibrator.load("calibration.pkl")

`fit` は sklearn の IsotonicRegression を使う。sklearn 未インストール時は
binned-mean にフォールバック (テストのみ)。
"""
from __future__ import annotations

import logging
import math
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.isotonic import IsotonicRegression
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

logger = logging.getLogger(__name__)

# セグメントキーに使うカラム
DEFAULT_SEGMENT_COLS = ("course_type", "grade_group")

# 最小サンプル数 — これを下回るセグメントは global fit を使う
MIN_SAMPLES_PER_SEGMENT = 200


def derive_grade_group(grade: Any) -> str:
    """grade文字列を "G1"/"G2G3"/"OP"/"条件戦"/"unknown" にマップ。"""
    if not isinstance(grade, str):
        return "unknown"
    g = grade
    if "G1" in g or "GI" in g.replace("GII", "").replace("GIII", ""):
        return "G1"
    if "G2" in g or "GII" in g.replace("GIII", "") or "G3" in g or "GIII" in g:
        return "G2G3"
    if "OP" in g or "オープン" in g or g.strip() == "L":
        return "OP"
    if any(k in g for k in ("新馬", "未勝利", "1勝", "2勝", "3勝")):
        return "条件戦"
    return "unknown"


def _time_weights(dates: pd.Series, half_life_days: float = 365.0) -> np.ndarray:
    """最新日を0として、half_life_daysごとに重みを半減させる指数減衰。"""
    if dates.empty:
        return np.array([])
    d = pd.to_datetime(dates, errors="coerce")
    latest = d.max()
    if pd.isna(latest):
        return np.ones(len(d))
    days_ago = (latest - d).dt.days.fillna(0).astype(float).values
    # 2^(-days_ago / half_life)
    return np.power(2.0, -days_ago / float(half_life_days))


class _BinnedFallback:
    """sklearn 無しの簡易 isotonic もどき (10bin mean)。"""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_edges: np.ndarray = np.array([])
        self.bin_means: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "_BinnedFallback":
        X = np.asarray(X, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if sample_weight is None:
            sample_weight = np.ones_like(X)
        edges = np.quantile(X, np.linspace(0, 1, self.n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
        means = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            mask = (X >= lo) & (X <= hi if i == len(edges) - 2 else X < hi)
            if mask.sum() == 0:
                means.append((lo + hi) / 2)
            else:
                w = sample_weight[mask]
                means.append(float(np.average(y[mask], weights=w)))
        # 単調化
        for i in range(1, len(means)):
            if means[i] < means[i - 1]:
                means[i] = means[i - 1]
        self.bin_edges = edges
        self.bin_means = np.array(means)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).ravel()
        idx = np.clip(np.searchsorted(self.bin_edges, X, side="right") - 1, 0, len(self.bin_means) - 1)
        return self.bin_means[idx]


class SegmentCalibrator:
    """セグメント別 isotonic calibrator。"""

    def __init__(
        self,
        segment_cols: tuple[str, ...] = DEFAULT_SEGMENT_COLS,
        target_col: str = "prob_win",
        label_col: str = "winner",
        half_life_days: float = 365.0,
        min_samples: int = MIN_SAMPLES_PER_SEGMENT,
    ):
        self.segment_cols = tuple(segment_cols)
        self.target_col = target_col
        self.label_col = label_col
        self.half_life_days = half_life_days
        self.min_samples = min_samples
        self.global_: Optional[Any] = None
        self.models_: dict[tuple, Any] = {}
        self.stats_: dict[str, Any] = {}
        self.fitted_at_: Optional[str] = None

    def _make_regressor(self):
        if _SKLEARN:
            return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        return _BinnedFallback(n_bins=10)

    def _segment_key(self, row: pd.Series) -> tuple:
        return tuple(str(row.get(c, "unknown")) for c in self.segment_cols)

    def _segment_keys(self, df: pd.DataFrame) -> pd.Series:
        if not set(self.segment_cols).issubset(df.columns):
            raise KeyError(f"segment_cols 不足: {set(self.segment_cols) - set(df.columns)}")
        return df[list(self.segment_cols)].astype(str).agg(tuple, axis=1)

    def fit(self, df: pd.DataFrame, date_col: str = "race_date") -> "SegmentCalibrator":
        if self.target_col not in df.columns or self.label_col not in df.columns:
            raise KeyError(f"{self.target_col}/{self.label_col} が df にない")
        df = df.dropna(subset=[self.target_col, self.label_col]).copy()
        if df.empty:
            raise ValueError("fit 用の非欠損データが空")

        # 時間重み
        weights = (
            _time_weights(df[date_col], half_life_days=self.half_life_days)
            if date_col in df.columns
            else np.ones(len(df))
        )

        # global fit (フォールバック用)
        self.global_ = self._make_regressor()
        self.global_.fit(df[self.target_col].values, df[self.label_col].values, sample_weight=weights)

        # セグメント別
        keys = self._segment_keys(df)
        seg_stats: dict[str, Any] = {}
        for k, sub_idx in keys.groupby(keys).groups.items():
            sub = df.loc[sub_idx]
            sw = weights[df.index.get_indexer(sub.index)] if len(weights) == len(df) else None
            if len(sub) < self.min_samples:
                seg_stats[str(k)] = {"n": int(len(sub)), "fitted": False}
                continue
            try:
                reg = self._make_regressor()
                reg.fit(sub[self.target_col].values, sub[self.label_col].values, sample_weight=sw)
                self.models_[k] = reg
                seg_stats[str(k)] = {
                    "n": int(len(sub)),
                    "fitted": True,
                    "pos_rate": round(float(sub[self.label_col].mean()), 4),
                    "pred_mean": round(float(sub[self.target_col].mean()), 4),
                }
            except Exception as e:
                logger.warning("segment %s calibration failed: %s", k, e)
                seg_stats[str(k)] = {"n": int(len(sub)), "fitted": False, "error": str(e)}

        self.stats_ = {
            "total": int(len(df)),
            "segments": seg_stats,
            "weight_sum": float(weights.sum()),
            "half_life_days": self.half_life_days,
        }
        self.fitted_at_ = datetime.now(timezone.utc).isoformat()
        logger.info(
            "SegmentCalibrator fitted: %d segments (global n=%d)",
            sum(1 for v in seg_stats.values() if v.get("fitted")),
            len(df),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.global_ is None:
            raise RuntimeError("fit() を先に呼んでください")
        if self.target_col not in df.columns:
            raise KeyError(f"{self.target_col} が df にない")
        out = np.zeros(len(df))
        raw = df[self.target_col].values
        try:
            keys = self._segment_keys(df)
            for k, sub_idx in keys.groupby(keys).groups.items():
                reg = self.models_.get(k, self.global_)
                pos = df.index.get_indexer(sub_idx)
                out[pos] = reg.predict(raw[pos])
        except KeyError:
            # セグメント列がない場合は global のみ
            out = self.global_.predict(raw)
        return pd.Series(np.clip(out, 0.0, 1.0), index=df.index, name=f"{self.target_col}_calibrated")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "SegmentCalibrator":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, cls)
        return obj
