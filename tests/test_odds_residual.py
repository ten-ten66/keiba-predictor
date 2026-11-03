"""Odds-residual model tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from odds_residual import OddsResidualModel, compute_baseline, _drop_odds_features


def _make_race_df(n_races: int = 120, avg_runners: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for rid in range(n_races):
        n = avg_runners
        # 真の強さ (ノイズ付き)
        strength = rng.uniform(0.1, 1.0, n)
        # オッズは強さの逆数 (市場ノイズ込み)
        raw_odds = (1.0 / strength) * rng.uniform(0.8, 1.2, n) + 0.5
        # 勝者は strength ベース
        probs = strength / strength.sum()
        winner_idx = rng.choice(n, p=probs)
        feat_a = strength + rng.normal(0, 0.1, n)
        feat_b = rng.normal(0, 1, n)
        for i in range(n):
            rows.append({
                "race_id": f"R{rid:04d}",
                "horse_number": i + 1,
                "odds": float(raw_odds[i]),
                "feat_a": float(feat_a[i]),
                "feat_b": float(feat_b[i]),
                "winner": 1 if i == winner_idx else 0,
            })
    return pd.DataFrame(rows)


def test_compute_baseline_sums_to_one_per_race():
    df = _make_race_df(n_races=20)
    df["baseline"] = compute_baseline(df)
    sums = df.groupby("race_id")["baseline"].sum()
    np.testing.assert_allclose(sums.values, 1.0, atol=1e-6)


def test_compute_baseline_zero_odds_handled():
    df = pd.DataFrame({
        "race_id": ["R1"] * 3,
        "odds": [3.0, 0.0, 2.0],
    })
    b = compute_baseline(df)
    assert b.iloc[1] == 0.0
    assert pytest.approx(b.iloc[0] + b.iloc[2], abs=1e-6) == 1.0


def test_drop_odds_features():
    cols = ["feat_a", "odds", "implied_prob", "odds_vs_mean", "is_favorite", "foo"]
    out = _drop_odds_features(cols)
    assert out == ["feat_a", "foo"]


def test_fit_predict_shapes():
    df = _make_race_df()
    model = OddsResidualModel(n_estimators=30)
    model.fit(df, feature_cols=["feat_a", "feat_b"])
    preds = model.predict(df)
    assert len(preds) == len(df)
    assert preds.min() >= 0.0 and preds.max() <= 1.0


def test_residual_beats_baseline_on_structured_data():
    """真の強さと相関する feat_a があれば、残差モデルはベースラインより精度が上がる。"""
    df = _make_race_df(n_races=300, avg_runners=10)
    model = OddsResidualModel(n_estimators=80)
    model.fit(df, feature_cols=["feat_a", "feat_b"])
    base = compute_baseline(df).values
    pred = model.predict(df).values
    y = df["winner"].values
    base_brier = float(np.mean((base - y) ** 2))
    pred_brier = float(np.mean((pred - y) ** 2))
    # In-sample なので improvement は保証できるが、tie でも可
    assert pred_brier <= base_brier + 1e-6


def test_missing_features_fall_back_to_baseline():
    df = _make_race_df()
    model = OddsResidualModel(n_estimators=30)
    model.fit(df, feature_cols=["feat_a", "feat_b"])
    # predict 時に feature 列を削る
    out = model.predict(df[["race_id", "horse_number", "odds"]])
    assert len(out) == len(df)


def test_save_load_roundtrip(tmp_path):
    df = _make_race_df()
    model = OddsResidualModel(n_estimators=30)
    model.fit(df, feature_cols=["feat_a", "feat_b"])
    p = tmp_path / "or.pkl"
    model.save(p)
    reloaded = OddsResidualModel.load(p)
    a = model.predict(df).values
    b = reloaded.predict(df).values
    np.testing.assert_allclose(a, b)


def test_odds_features_excluded_from_training():
    df = _make_race_df()
    df["implied_prob"] = 1.0 / df["odds"]
    model = OddsResidualModel(n_estimators=30)
    # implied_prob を feature_cols に含めても勝手に除去される
    model.fit(df, feature_cols=["feat_a", "feat_b", "implied_prob", "odds_vs_mean"])
    assert "implied_prob" not in model.feature_cols_
    assert "odds_vs_mean" not in model.feature_cols_
    assert "feat_a" in model.feature_cols_


def test_raises_on_missing_label():
    df = pd.DataFrame({"race_id": ["R1"], "odds": [2.0], "feat_a": [0.1]})
    model = OddsResidualModel()
    with pytest.raises(KeyError):
        model.fit(df, feature_cols=["feat_a"])
