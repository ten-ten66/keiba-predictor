"""Shadow-eval helper tests."""
from __future__ import annotations

import pandas as pd
import pytest

from shadow_eval import compare_predictions, ShadowLogger, _rank_spearman, _symmetric_kl


def test_rank_spearman_identity():
    assert _rank_spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_rank_spearman_reversed():
    assert _rank_spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_symmetric_kl_identical():
    assert _symmetric_kl([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0, abs=1e-6)


def test_symmetric_kl_divergent():
    kl = _symmetric_kl([0.9, 0.05, 0.05], [0.05, 0.9, 0.05])
    assert kl > 0.5


def test_compare_predictions_identical():
    df = pd.DataFrame({
        "horse_number": [1, 2, 3, 4],
        "predicted_rank": [1, 2, 3, 4],
        "prob_win": [0.4, 0.3, 0.2, 0.1],
    })
    out = compare_predictions(df, df.copy())
    assert out["n_horses"] == 4
    assert out["rank_spearman"] == pytest.approx(1.0)
    assert out["top1_agreement"] is True
    assert out["top3_overlap"] == 3
    assert out["prob_kl"] == pytest.approx(0.0, abs=1e-3)


def test_compare_predictions_top1_disagree():
    primary = pd.DataFrame({
        "horse_number": [1, 2, 3],
        "predicted_rank": [1, 2, 3],
        "prob_win": [0.5, 0.3, 0.2],
    })
    shadow = pd.DataFrame({
        "horse_number": [1, 2, 3],
        "predicted_rank": [2, 1, 3],
        "prob_win": [0.3, 0.5, 0.2],
    })
    out = compare_predictions(primary, shadow)
    assert out["top1_agreement"] is False
    assert out["top3_overlap"] == 3


def test_shadow_logger_roundtrip(tmp_path):
    logger = ShadowLogger(tmp_path)
    logger.log("R_001", {"rank_spearman": 0.9, "top1_agreement": True, "prob_kl": 0.1, "top3_overlap": 3})
    logger.log("R_002", {"rank_spearman": 0.7, "top1_agreement": False, "prob_kl": 0.3, "top3_overlap": 2})
    summary = logger.summarize(days=1)
    assert summary["count"] == 2
    assert summary["avg_rank_spearman"] == pytest.approx(0.8)
    assert summary["top1_agreement_rate"] == pytest.approx(0.5)
    assert summary["avg_top3_overlap"] == pytest.approx(2.5)
