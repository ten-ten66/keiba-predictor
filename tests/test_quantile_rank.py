"""QuantileRankModel tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantile_rank_model import QuantileRankModel


def _make_df(n_races: int = 150, n_per_race: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for rid in range(n_races):
        strength = rng.uniform(0.1, 1.0, n_per_race)
        noise = rng.normal(0, 0.2, n_per_race)
        # 強い馬ほど着順が小さい (position=1 が1着)
        scores = -(strength + noise)
        positions = scores.argsort().argsort() + 1
        for i in range(n_per_race):
            rows.append({
                "race_id": f"R{rid:04d}",
                "horse_number": i + 1,
                "feat_strength": float(strength[i]),
                "feat_b": float(rng.normal(0, 1)),
                "finish_position": int(positions[i]),
            })
    return pd.DataFrame(rows)


def test_fit_predict_shape():
    df = _make_df()
    m = QuantileRankModel(quantiles=(0.1, 0.5, 0.9), n_estimators=30)
    m.fit(df, feature_cols=["feat_strength", "feat_b"])
    preds = m.predict(df)
    assert list(preds.columns) == ["pred_pos_q10", "pred_pos_q50", "pred_pos_q90"]
    assert len(preds) == len(df)


def test_quantiles_are_monotonic():
    df = _make_df()
    m = QuantileRankModel(quantiles=(0.1, 0.5, 0.9), n_estimators=30)
    m.fit(df, feature_cols=["feat_strength", "feat_b"])
    preds = m.predict(df)
    # q10 <= q50 <= q90 行ごとに成立
    assert (preds["pred_pos_q10"] <= preds["pred_pos_q50"]).all()
    assert (preds["pred_pos_q50"] <= preds["pred_pos_q90"]).all()


def test_quantiles_clipped_to_field_size():
    df = _make_df(n_per_race=8)
    m = QuantileRankModel(quantiles=(0.1, 0.5, 0.9), n_estimators=30)
    m.fit(df, feature_cols=["feat_strength", "feat_b"])
    preds = m.predict(df)
    # 8頭立てなので最大着順は8
    race_sizes = df.groupby("race_id").size()
    merged = df.assign(**preds)
    for rid, rs in race_sizes.items():
        sub = merged[merged["race_id"] == rid]
        assert (sub["pred_pos_q90"] <= rs).all()
        assert (sub["pred_pos_q10"] >= 1).all()


def test_median_roughly_tracks_true_position():
    """医学ないが、med(q50) ≈ true position 相関が正になる。"""
    df = _make_df(n_races=300, n_per_race=10)
    m = QuantileRankModel(quantiles=(0.5,), n_estimators=60)
    m.fit(df, feature_cols=["feat_strength", "feat_b"])
    preds = m.predict(df)
    corr = np.corrcoef(preds["pred_pos_q50"].values, df["finish_position"].values)[0, 1]
    assert corr > 0.3  # 線形相関がある程度


def test_coverage_approximately_reasonable():
    """in-sample で q10..q90 が実測着順を 40%以上カバーすればOK (in-sampleは通常高め)。"""
    df = _make_df(n_races=300, n_per_race=10)
    m = QuantileRankModel(quantiles=(0.1, 0.9), n_estimators=60)
    m.fit(df, feature_cols=["feat_strength", "feat_b"])
    preds = m.predict(df)
    covered = ((df["finish_position"].values >= preds["pred_pos_q10"].values - 0.5)
               & (df["finish_position"].values <= preds["pred_pos_q90"].values + 0.5))
    cov = covered.mean()
    assert cov > 0.5  # loose sanity: wider than 50%


def test_missing_features_fallback():
    df = _make_df()
    m = QuantileRankModel(quantiles=(0.5,), n_estimators=20)
    m.fit(df, feature_cols=["feat_strength", "feat_b"])
    out = m.predict(df[["race_id", "horse_number", "finish_position"]])
    assert len(out) == len(df)
    assert "pred_pos_q50" in out.columns


def test_save_load_roundtrip(tmp_path):
    df = _make_df()
    m = QuantileRankModel(quantiles=(0.1, 0.5, 0.9), n_estimators=20)
    m.fit(df, feature_cols=["feat_strength", "feat_b"])
    p = tmp_path / "q.pkl"
    m.save(p)
    m2 = QuantileRankModel.load(p)
    a = m.predict(df).values
    b = m2.predict(df).values
    np.testing.assert_allclose(a, b)


def test_raises_on_missing_label():
    df = pd.DataFrame({"race_id": ["R1"], "feat_a": [0.1]})
    m = QuantileRankModel()
    with pytest.raises(KeyError):
        m.fit(df, feature_cols=["feat_a"])
