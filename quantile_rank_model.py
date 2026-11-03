"""
Quantile rank regression for race position CIs
==============================================
現行は point 予測 + Conformal quantile で CI を作っているが、Conformal は
global quantile 幅を全馬に均等に被せるため「13頭立てでCIが全馬に被る」問題が出ている。

ここでは target = finish_position (ordinal) をそのまま回帰し、
LightGBM の `objective="quantile"` で 複数分位点 (10/50/90 等) を直接学習する。
着順自体に条件付き分散が反映されるため、有力馬のCIは狭く・凡走傾向のCIは広く
なることが期待される。

使い方:
  qm = QuantileRankModel(quantiles=[0.1, 0.5, 0.9])
  qm.fit(df, feature_cols=[...], label_col="finish_position")
  preds = qm.predict(df)        # DataFrame: pred_pos_q10, pred_pos_q50, pred_pos_q90
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _LGB = True
except ImportError:
    _LGB = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

logger = logging.getLogger(__name__)

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


class QuantileRankModel:
    """複数分位点の finish_position 予測器。"""

    def __init__(
        self,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_samples_leaf: int = 100,
        random_state: int = 42,
    ):
        assert all(0 < q < 1 for q in quantiles), "quantiles must be in (0,1)"
        self.quantiles = tuple(sorted(quantiles))
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.feature_cols_: list[str] = []
        self.models_: dict[float, Any] = {}
        self.global_median_: float = 0.0
        self.max_field_size_: int = 18

    def _make_regressor(self, q: float):
        if _LGB:
            return lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_child_samples=self.min_samples_leaf,
                random_state=self.random_state,
                verbose=-1,
            )
        if _SKLEARN:
            return GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
        raise RuntimeError("lightgbm or sklearn が必要です")

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "finish_position",
    ) -> "QuantileRankModel":
        if label_col not in df.columns:
            raise KeyError(f"{label_col} が df にない")
        df = df.dropna(subset=[label_col]).copy()
        df = df[df[label_col] > 0]
        if df.empty:
            raise ValueError("fit 用データが空")

        clean_cols = [c for c in feature_cols if c in df.columns]
        if not clean_cols:
            raise ValueError("feature_cols が df に1つも見つからない")

        self.feature_cols_ = clean_cols
        X = df[clean_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        y = df[label_col].astype(float).values

        for q in self.quantiles:
            reg = self._make_regressor(q)
            reg.fit(X, y)
            self.models_[q] = reg

        self.global_median_ = float(np.median(y))
        if "race_id" in df.columns:
            self.max_field_size_ = int(df.groupby("race_id").size().max())
        logger.info(
            "QuantileRankModel fitted: n=%d, features=%d, quantiles=%s",
            len(df), len(clean_cols), self.quantiles,
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.models_:
            raise RuntimeError("fit() してから predict() を呼んでください")
        X_cols = [c for c in self.feature_cols_ if c in df.columns]
        if not X_cols:
            # 特徴量が一つもなければ global median で埋める
            out = {f"pred_pos_q{int(q*100)}": np.full(len(df), self.global_median_) for q in self.quantiles}
            return pd.DataFrame(out, index=df.index)
        X = df[X_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

        result: dict[str, np.ndarray] = {}
        preds_arr: list[np.ndarray] = []
        for q in self.quantiles:
            preds = self.models_[q].predict(X)
            # レース内で分位点を field_size でクリップ
            if "race_id" in df.columns:
                race_size = df.groupby("race_id")["race_id"].transform("count").values
                upper_clip = np.minimum(race_size, self.max_field_size_)
            else:
                upper_clip = self.max_field_size_
            preds = np.clip(preds, 1.0, upper_clip)
            result[f"pred_pos_q{int(q*100)}"] = preds
            preds_arr.append(preds)

        out_df = pd.DataFrame(result, index=df.index)
        # 単調性を保証: q10 <= q50 <= q90 (LightGBMのquantileは独立学習なので順序が壊れることがある)
        if len(preds_arr) >= 2:
            mat = np.stack(preds_arr, axis=1)
            mat = np.maximum.accumulate(mat, axis=1)
            for i, q in enumerate(self.quantiles):
                out_df[f"pred_pos_q{int(q*100)}"] = mat[:, i]
        return out_df

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "QuantileRankModel":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, cls)
        return obj
