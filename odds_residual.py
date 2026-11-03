"""
Odds-baselined residual learning
================================
パリミュチュエルの暗黙確率 (1/odds を race 内で正規化) は極めて強いベースライン。
"AI だけで勝率を予測" するより、市場の予測からどれだけ乖離するかを学習する方が
過学習を避けやすく、サンプル効率が良い (Bayes shrinkage と同じ発想)。

y_win ~ baseline(odds) + f(features_non_odds)

ここで:
  baseline      = normalized_implied_prob (take を引いて正規化済み)
  residual      = y_win - baseline
  f(features)   = LightGBM regressor (lightgbm 未インストール時は GBR に fallback)

予測時は:
  pred = clip(baseline + f(features), 0, 1)

注意: オッズ未確定 (発走数時間前) のときに baseline が 0 になる馬が出るので、
`predict()` は odds 欠損馬には global_mean を使う。
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

ODDS_RELATED = {
    "odds", "implied_prob", "normalized_implied_prob", "log_odds",
    "odds_rank_in_race", "odds_vs_mean", "is_favorite", "popularity",
    "value_gap",  # implied_prob との差 — リークなのでlearn時に外す
}


def compute_baseline(df: pd.DataFrame, odds_col: str = "odds") -> pd.Series:
    """race_id 単位で 1/odds を正規化して暗黙確率を返す。take は自動除去。"""
    if odds_col not in df.columns or "race_id" not in df.columns:
        raise KeyError(f"{odds_col} / race_id 必須")
    inv = 1.0 / df[odds_col].clip(lower=1.0)
    inv = inv.where(df[odds_col] > 0, 0.0)
    race_sum = df.groupby("race_id")[odds_col].transform(lambda s: (1.0 / s.clip(lower=1.0)).where(s > 0, 0.0).sum())
    race_sum = race_sum.replace(0, np.nan)
    return (inv / race_sum).fillna(0.0).clip(0.0, 1.0)


def _drop_odds_features(feature_cols: list[str]) -> list[str]:
    return [c for c in feature_cols if c.lower() not in ODDS_RELATED and not c.lower().startswith("odds_")]


class OddsResidualModel:
    """市場確率 + 残差 の2層モデル。"""

    def __init__(
        self,
        odds_col: str = "odds",
        label_col: str = "winner",
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_samples_leaf: int = 200,
        random_state: int = 42,
    ):
        self.odds_col = odds_col
        self.label_col = label_col
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.feature_cols_: list[str] = []
        self.regressor_: Optional[Any] = None
        self.train_mae_: Optional[float] = None
        self.baseline_mean_: float = 0.05

    def _make_regressor(self):
        if _LGB:
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_child_samples=self.min_samples_leaf,
                random_state=self.random_state,
                verbose=-1,
            )
        if _SKLEARN:
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
        raise RuntimeError("lightgbm or sklearn が必要です")

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "OddsResidualModel":
        if self.odds_col not in df.columns or self.label_col not in df.columns:
            raise KeyError(f"{self.odds_col}/{self.label_col} が df にない")
        df = df.dropna(subset=[self.odds_col, self.label_col]).copy()
        if df.empty:
            raise ValueError("fit 用データが空")

        baseline = compute_baseline(df, self.odds_col)
        y = df[self.label_col].astype(float).values
        residual = y - baseline.values
        self.baseline_mean_ = float(baseline.mean())

        clean_cols = _drop_odds_features(feature_cols)
        missing = [c for c in clean_cols if c not in df.columns]
        if missing:
            logger.warning("feature 不足 %d個: %s...", len(missing), missing[:3])
            clean_cols = [c for c in clean_cols if c in df.columns]
        if not clean_cols:
            raise ValueError("使用可能な non-odds 特徴量がない")

        self.feature_cols_ = clean_cols
        X = df[clean_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

        self.regressor_ = self._make_regressor()
        self.regressor_.fit(X, residual)
        # in-sample MAE
        self.train_mae_ = float(np.mean(np.abs(self.regressor_.predict(X) - residual)))
        logger.info(
            "OddsResidualModel fitted: n=%d, features=%d, mae=%.4f",
            len(df), len(clean_cols), self.train_mae_,
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.regressor_ is None:
            raise RuntimeError("fit() してから predict() を呼んでください")
        if self.odds_col in df.columns:
            base = compute_baseline(df, self.odds_col)
        else:
            base = pd.Series(self.baseline_mean_, index=df.index)
        X_cols = [c for c in self.feature_cols_ if c in df.columns]
        if not X_cols:
            return base.clip(0, 1).rename("prob_win_residual")
        # 足りない列は0埋め
        X = df[X_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        resid = self.regressor_.predict(X)
        out = np.clip(base.values + resid, 0.0, 1.0)
        return pd.Series(out, index=df.index, name="prob_win_residual")

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "OddsResidualModel":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, cls)
        return obj
