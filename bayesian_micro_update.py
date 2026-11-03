"""
Bayesian micro-update (当日オンライン校正)
==========================================
週次 retrain の間に当日の結果を使ってモデル出力の「当日バイアス」だけを
軽量に補正する。馬ごと/レースごとではなくレース全体のキャリブレーションを
ロジットスケールで補正する Platt scaling を online に回すイメージ。

モデル: p_adj = sigmoid(a * logit(p) + b)
  (a, b) を観測した (p, y) ペアから1ステップ SGD で更新する。
  当日リセットして dt 経過で指数減衰させる (半減期2時間程度)。

利用シナリオ:
  - 毎レース結果が確定したら `observe(pred_prob, actual_won)` をまとめて呼ぶ
  - 以降のレース予測には `adjust(pred_prob)` を適用する
  - 翌日 `reset()` でゼロに戻す

ディスク永続化:
  data/bayesian_micro_state.json
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-6


def _logit(p: float) -> float:
    p = max(EPS, min(1.0 - EPS, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        ex = math.exp(-x)
        return 1.0 / (1.0 + ex)
    ex = math.exp(x)
    return ex / (1.0 + ex)


class BayesianMicroUpdater:
    """当日1日スパンの Platt-style online scaler。"""

    def __init__(
        self,
        learning_rate: float = 0.05,
        half_life_hours: float = 2.0,
        prior_a: float = 1.0,
        prior_b: float = 0.0,
        prior_strength: float = 20.0,  # prior を何回分の観測として扱うか
        state_path: Optional[Path] = None,
    ):
        self.lr = learning_rate
        self.half_life_hours = half_life_hours
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.prior_strength = prior_strength
        self.state_path = state_path

        self.a = prior_a
        self.b = prior_b
        self.n_obs = 0
        self.last_update_ts: Optional[float] = None
        self.day_key: Optional[str] = None

        if state_path is not None and Path(state_path).exists():
            self._load()

    def _today_key(self) -> str:
        import datetime as _dt
        return _dt.date.today().isoformat()

    def _decay(self) -> None:
        """最終更新からの経過で prior に向けて指数減衰。"""
        if self.last_update_ts is None:
            return
        now = time.time()
        hours = max(0.0, (now - self.last_update_ts) / 3600.0)
        if self.half_life_hours <= 0:
            return
        # 短すぎる経過 (1分未満) は実質ノイズなので無視
        if hours < 1.0 / 60.0:
            return
        factor = math.pow(0.5, hours / self.half_life_hours)
        pull = 1.0 - factor
        self.a = self.a + pull * (self.prior_a - self.a)
        self.b = self.b + pull * (self.prior_b - self.b)
        self.n_obs = max(0, round(self.n_obs * factor))

    def _maybe_roll_day(self) -> None:
        today = self._today_key()
        if self.day_key != today:
            # 日付が変わったらリセット (prior に戻す)
            self.a = self.prior_a
            self.b = self.prior_b
            self.n_obs = 0
            self.day_key = today
            self.last_update_ts = None

    def observe(self, pred_prob: float, actual: int, weight: float = 1.0) -> None:
        """単一観測 (pred, actual) でa/bを1ステップ更新。"""
        self._maybe_roll_day()
        self._decay()
        x = _logit(float(pred_prob))
        y = float(actual)
        p = _sigmoid(self.a * x + self.b)
        err = (p - y) * float(weight)
        # logistic regression gradient
        self.a -= self.lr * err * x
        self.b -= self.lr * err
        # prior に向けた L2 正則化 (prior_strength / (n+prior_strength))
        alpha = self.prior_strength / (self.n_obs + self.prior_strength + 1)
        self.a = alpha * self.prior_a + (1 - alpha) * self.a
        self.b = alpha * self.prior_b + (1 - alpha) * self.b
        self.n_obs += 1
        self.last_update_ts = time.time()

    def observe_many(self, pairs: list[tuple[float, int]]) -> None:
        for p, y in pairs:
            self.observe(p, y)
        self._save()

    def adjust(self, pred_prob: float) -> float:
        self._maybe_roll_day()
        self._decay()
        return _sigmoid(self.a * _logit(float(pred_prob)) + self.b)

    def adjust_array(self, probs: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(probs, dtype=float)
        logits = np.log(np.clip(arr, EPS, 1 - EPS) / np.clip(1 - arr, EPS, 1 - EPS))
        return 1.0 / (1.0 + np.exp(-(self.a * logits + self.b)))

    def reset(self) -> None:
        self.a = self.prior_a
        self.b = self.prior_b
        self.n_obs = 0
        self.last_update_ts = None
        self.day_key = self._today_key()
        self._save()

    def state(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "n_obs": self.n_obs,
            "last_update_ts": self.last_update_ts,
            "day_key": self.day_key,
            "prior_a": self.prior_a,
            "prior_b": self.prior_b,
            "learning_rate": self.lr,
            "half_life_hours": self.half_life_hours,
        }

    def _save(self) -> None:
        if self.state_path is None:
            return
        try:
            Path(self.state_path).write_text(
                json.dumps(self.state(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("micro-update save failed: %s", e)

    def _load(self) -> None:
        try:
            data = json.loads(Path(self.state_path).read_text(encoding="utf-8"))
            self.a = float(data.get("a", self.prior_a))
            self.b = float(data.get("b", self.prior_b))
            self.n_obs = int(data.get("n_obs", 0))
            self.last_update_ts = data.get("last_update_ts")
            self.day_key = data.get("day_key")
        except Exception as e:
            logger.warning("micro-update load failed: %s", e)
