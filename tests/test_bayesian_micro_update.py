"""Tests for BayesianMicroUpdater (当日オンライン校正)。"""
from __future__ import annotations

import time

import numpy as np
import pytest

from bayesian_micro_update import BayesianMicroUpdater, _logit, _sigmoid


def test_logit_sigmoid_roundtrip():
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        assert _sigmoid(_logit(p)) == pytest.approx(p, abs=1e-6)


def test_adjust_identity_without_observations():
    u = BayesianMicroUpdater()
    for p in [0.1, 0.5, 0.9]:
        assert u.adjust(p) == pytest.approx(p, abs=1e-6)


def test_observe_moves_toward_actual():
    """予測が低めに出ているのに毎回当たるなら、adjust後の確率は上がる。"""
    u = BayesianMicroUpdater(learning_rate=0.3, prior_strength=1.0)
    # 予測0.2, 実勝の観測を30回
    for _ in range(30):
        u.observe(0.2, 1)
    adjusted = u.adjust(0.2)
    assert adjusted > 0.2


def test_observe_moves_toward_zero_when_never_wins():
    u = BayesianMicroUpdater(learning_rate=0.3, prior_strength=1.0)
    for _ in range(30):
        u.observe(0.6, 0)
    assert u.adjust(0.6) < 0.6


def test_adjust_monotone_in_pred():
    u = BayesianMicroUpdater()
    # いくつか観測を入れてから単調性を確認
    u.observe_many([(0.3, 1), (0.4, 0), (0.5, 1), (0.6, 0)])
    xs = [0.05, 0.1, 0.3, 0.5, 0.8, 0.95]
    ys = [u.adjust(x) for x in xs]
    for i in range(1, len(ys)):
        # a > 0 であれば単調増加。prior_a=1 なので SGD の少ない観測後はほぼ常に成立
        assert ys[i] >= ys[i - 1] - 1e-4


def test_adjust_array_matches_scalar():
    u = BayesianMicroUpdater()
    u.observe_many([(0.1, 0), (0.3, 1), (0.5, 0)])
    xs = [0.1, 0.2, 0.3, 0.4]
    arr = u.adjust_array(xs)
    scalar = [u.adjust(x) for x in xs]
    np.testing.assert_allclose(arr, scalar, atol=1e-6)


def test_reset_returns_to_prior():
    u = BayesianMicroUpdater()
    u.observe_many([(0.2, 1)] * 10)
    assert u.n_obs > 0
    u.reset()
    assert u.n_obs == 0
    assert u.a == pytest.approx(u.prior_a)
    assert u.b == pytest.approx(u.prior_b)


def test_state_persistence(tmp_path):
    state = tmp_path / "bmu.json"
    u = BayesianMicroUpdater(state_path=state)
    u.observe_many([(0.3, 1), (0.4, 0), (0.2, 1)])
    # 新しいインスタンスで読み直す
    u2 = BayesianMicroUpdater(state_path=state)
    assert u2.n_obs == u.n_obs
    assert u2.a == pytest.approx(u.a)
    assert u2.b == pytest.approx(u.b)


def test_prior_regularization_prevents_runaway():
    """少数観測で極端に動かないことを確認。"""
    u = BayesianMicroUpdater(prior_strength=50.0, learning_rate=0.1)
    for _ in range(3):
        u.observe(0.1, 1)
    # prior_strength が強いので a/b は prior 付近に留まる
    assert abs(u.a - u.prior_a) < 0.5
    assert abs(u.b - u.prior_b) < 0.5


def test_day_roll_resets():
    u = BayesianMicroUpdater()
    u.observe(0.3, 1)
    u.observe(0.4, 0)
    assert u.n_obs == 2
    # 擬似的に day_key を変えると次の observe/adjust でリセットされる
    u.day_key = "1999-01-01"
    _ = u.adjust(0.5)
    assert u.n_obs == 0
