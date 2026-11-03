"""Segment-aware calibration tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from segment_calibration import (
    SegmentCalibrator,
    _time_weights,
    derive_grade_group,
)


def _make_df(n_per_seg: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    # 2 segments with different miscalibration
    for seg, (course, grade), scale in [
        (0, ("芝", "G1"), 0.9),   # slight under-confidence
        (1, ("ダート", "条件戦"), 1.4),  # over-confidence
    ]:
        pw = rng.uniform(0.01, 0.8, n_per_seg)
        true_p = np.clip(pw * scale, 0.01, 0.99)
        winner = (rng.uniform(size=n_per_seg) < true_p).astype(int)
        dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n_per_seg), unit="D")
        for i in range(n_per_seg):
            rows.append({
                "race_date": dates[i],
                "course_type": course,
                "grade_group": grade,
                "prob_win": pw[i],
                "winner": winner[i],
            })
    return pd.DataFrame(rows)


def test_derive_grade_group():
    assert derive_grade_group("G1") == "G1"
    assert derive_grade_group("GI") == "G1"
    assert derive_grade_group("G2") == "G2G3"
    assert derive_grade_group("G3") == "G2G3"
    assert derive_grade_group("OP") == "OP"
    assert derive_grade_group("オープン") == "OP"
    assert derive_grade_group("3歳未勝利") == "条件戦"
    assert derive_grade_group("1勝クラス") == "条件戦"
    assert derive_grade_group(None) == "unknown"
    assert derive_grade_group("") == "unknown"


def test_time_weights_monotonic():
    dates = pd.to_datetime(["2025-01-01", "2025-06-01", "2026-01-01"])
    w = _time_weights(pd.Series(dates), half_life_days=365)
    # 最新 (2026-01-01) の重み=1, 1年前は0.5
    assert w[2] == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.5, abs=0.02)
    assert w[1] > w[0] and w[1] < w[2]


def test_time_weights_empty():
    w = _time_weights(pd.Series([], dtype="datetime64[ns]"))
    assert len(w) == 0


def test_fit_and_transform_shapes():
    df = _make_df(n_per_seg=300)
    cal = SegmentCalibrator(min_samples=50)
    cal.fit(df)
    out = cal.transform(df)
    assert len(out) == len(df)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_segment_calibration_reduces_miscalibration():
    """セグメント毎に bias が違う場合、キャリブレーション後は mean pred ≈ mean actual に近づく。"""
    df = _make_df(n_per_seg=500)
    cal = SegmentCalibrator(min_samples=50)
    cal.fit(df)
    df["prob_cal"] = cal.transform(df)
    # セグメント別の "予測平均" と "実現勝率" の差
    for seg_key in df[["course_type", "grade_group"]].drop_duplicates().itertuples(index=False):
        sub = df[(df["course_type"] == seg_key.course_type) & (df["grade_group"] == seg_key.grade_group)]
        raw_gap = abs(sub["prob_win"].mean() - sub["winner"].mean())
        cal_gap = abs(sub["prob_cal"].mean() - sub["winner"].mean())
        # Isotonicは校正誤差を減らすべき (gap 同等以下)
        assert cal_gap <= raw_gap + 0.02, f"cal_gap {cal_gap} > raw_gap {raw_gap} in {seg_key}"


def test_segments_below_min_samples_fallback_to_global():
    df = _make_df(n_per_seg=50)  # 小さい
    cal = SegmentCalibrator(min_samples=100)
    cal.fit(df)
    # min_samples 未満ならセグメントモデルは作らず global に fallback
    assert len(cal.models_) == 0
    assert cal.global_ is not None
    # transform はそれでも動く
    out = cal.transform(df)
    assert len(out) == len(df)


def test_save_load_roundtrip(tmp_path):
    df = _make_df(n_per_seg=300)
    cal = SegmentCalibrator(min_samples=50)
    cal.fit(df)
    path = tmp_path / "cal.pkl"
    cal.save(path)
    reloaded = SegmentCalibrator.load(path)
    # Check predictions match
    a = cal.transform(df).values
    b = reloaded.transform(df).values
    np.testing.assert_allclose(a, b)


def test_missing_segment_cols_uses_global():
    df = _make_df(n_per_seg=300)
    cal = SegmentCalibrator(min_samples=50)
    cal.fit(df)
    # transform dropping segment cols should still work using global
    df2 = df[["prob_win", "winner"]].copy()
    out = cal.transform(df2)
    assert len(out) == len(df2)


def test_missing_required_cols_raises():
    cal = SegmentCalibrator()
    with pytest.raises(KeyError):
        cal.fit(pd.DataFrame({"foo": [1, 2]}))
