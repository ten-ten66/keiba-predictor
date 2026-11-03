"""Tests for postmortem tag generation."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# api_server のモジュール全体は副作用が多いため、関数だけ import する
import importlib.util
spec = importlib.util.spec_from_file_location("api_server", REPO_ROOT / "api_server.py")
# api_server import は重いので、ロジック部分を直接切り出して検証する


def _extract_fn(src: str, name: str):
    """api_server.py から関数ソースを切り出して exec する。"""
    start = src.index(f"def {name}(")
    tail = src[start:]
    import re
    end = len(tail)
    for m in re.finditer(r"^(?:def |class |@app\.)", tail, flags=re.MULTILINE):
        if m.start() > 0:
            end = m.start()
            break
    body = tail[:end]
    scope: dict = {}
    exec(body, scope)
    return scope[name]


def _load_fn():
    """_generate_postmortem_tags をモジュール全体 import せずに持ってくる。"""
    src = (REPO_ROOT / "api_server.py").read_text(encoding="utf-8")
    return _extract_fn(src, "_generate_postmortem_tags")


_generate = _load_fn()


def _load_odds_history_fn():
    src = (REPO_ROOT / "api_server.py").read_text(encoding="utf-8")
    return _extract_fn(src, "_odds_history_for_postmortem")


_odds_helper = _load_odds_history_fn()


def _mk_pred(num, name, rank, prob_win=0.1, ev=0.9, upper80=None):
    return {
        "horse_number": num,
        "horse_name": name,
        "predicted_rank": rank,
        "prob_win": prob_win,
        "composite_score": 0.3,
        "expected_value": ev,
        "pred_rank_upper80": upper80,
    }


def _mk_actual(num, name, pos):
    return {"horse_number": num, "horse_name": name, "position": pos}


def test_favorite_bust_detected():
    preds = [
        _mk_pred(1, "A", 1, prob_win=0.25),
        _mk_pred(2, "B", 2, prob_win=0.2),
        _mk_pred(3, "C", 3),
    ]
    actual = [_mk_actual(3, "C", 1), _mk_actual(2, "B", 2), _mk_actual(1, "A", 10)]
    tags = _generate(preds, actual)
    keys = [t["key"] for t in tags]
    assert "favorite_bust" in keys


def test_low_confidence_hit_detected():
    preds = [
        _mk_pred(1, "A", 1, prob_win=0.25),
        _mk_pred(2, "B", 2, prob_win=0.2),
        _mk_pred(7, "Dark", 7, prob_win=0.05),
    ]
    actual = [_mk_actual(7, "Dark", 1), _mk_actual(1, "A", 2), _mk_actual(2, "B", 3)]
    tags = _generate(preds, actual)
    keys = [t["key"] for t in tags]
    assert "low_confidence_hit" in keys


def test_our_top3_sweep_detected():
    preds = [
        _mk_pred(1, "A", 1),
        _mk_pred(2, "B", 2),
        _mk_pred(3, "C", 3),
        _mk_pred(4, "D", 4),
    ]
    actual = [_mk_actual(1, "A", 1), _mk_actual(2, "B", 2), _mk_actual(3, "C", 3)]
    tags = _generate(preds, actual)
    keys = [t["key"] for t in tags]
    assert "our_top3_sweep" in keys
    assert "favorite_bust" not in keys


def test_ci_coverage_miss_detected():
    preds = [
        _mk_pred(1, "A", 1, upper80=3),
        _mk_pred(2, "B", 2, upper80=4),
        _mk_pred(3, "C", 3, upper80=5),
        _mk_pred(4, "D", 4, upper80=6),
    ]
    # 全馬が予測範囲より大きく外れる
    actual = [
        _mk_actual(1, "A", 8),
        _mk_actual(2, "B", 7),
        _mk_actual(3, "C", 6),
        _mk_actual(4, "D", 5),
    ]
    tags = _generate(preds, actual)
    keys = [t["key"] for t in tags]
    assert "ci_coverage_miss" in keys


def test_high_ev_missed_detected():
    preds = [
        _mk_pred(1, "A", 1, prob_win=0.3, ev=1.8),  # 高EV
        _mk_pred(2, "B", 2, prob_win=0.2, ev=1.0),
    ]
    actual = [_mk_actual(2, "B", 1), _mk_actual(1, "A", 9)]
    tags = _generate(preds, actual)
    keys = [t["key"] for t in tags]
    assert "high_ev_missed" in keys


def test_no_tags_when_results_empty():
    preds = [_mk_pred(1, "A", 1)]
    assert _generate(preds, []) == []
    assert _generate([], [{"horse_number": 1, "position": 1}]) == []


def test_scratched_detected():
    preds = [_mk_pred(n, f"H{n}", n) for n in range(1, 11)]
    actual = [_mk_actual(n, f"H{n}", n) for n in range(1, 9)]  # 2頭除外
    tags = _generate(preds, actual)
    keys = [t["key"] for t in tags]
    assert "scratched" in keys


# ---------------------------------------------------------------------------
# _odds_history_for_postmortem
# ---------------------------------------------------------------------------
import types as _types  # noqa: E402


def _patch_bayesian(monkeypatch, snapshots):
    fake = _types.ModuleType("bayesian_updater")
    fake.load_odds_history = lambda race_id, date_str: {"snapshots": snapshots}
    monkeypatch.setitem(sys.modules, "bayesian_updater", fake)


def test_odds_helper_returns_none_without_date(monkeypatch):
    _patch_bayesian(monkeypatch, [{"odds": {"1": 3.0}}, {"odds": {"1": 2.5}}])
    assert _odds_helper("R1", None, []) is None
    assert _odds_helper("R1", "", []) is None


def test_odds_helper_returns_none_with_few_snapshots(monkeypatch):
    _patch_bayesian(monkeypatch, [])
    assert _odds_helper("R1", "20260101", []) is None
    _patch_bayesian(monkeypatch, [{"odds": {"1": 3.0}}])
    assert _odds_helper("R1", "20260101", []) is None


def test_odds_helper_sorts_by_final_odds_and_attaches_name(monkeypatch):
    snapshots = [
        {"odds": {"1": 10.0, "2": 5.0, "3": 8.0}},
        {"odds": {"1": 3.0, "2": 5.5, "3": 7.0}},
    ]
    _patch_bayesian(monkeypatch, snapshots)
    preds = [
        {"horse_number": 1, "horse_name": "A"},
        {"horse_number": 2, "horse_name": "B"},
        {"horse_number": 3, "horse_name": "C"},
    ]
    out = _odds_helper("R1", "20260101", preds)
    assert out is not None
    nums = [r["horse_number"] for r in out]
    assert nums == [1, 2, 3]  # final 3.0 < 5.5 < 7.0
    assert out[0]["horse_name"] == "A"
    assert out[0]["odds_early"] == 10.0
    assert out[0]["odds_final"] == 3.0


def test_odds_helper_skips_invalid_entries(monkeypatch):
    snapshots = [
        {"odds": {"1": "bad", "2": 5.0, "3": 0}},  # 1 unparseable, 3 zero early
        {"odds": {"1": 3.0, "2": 4.5, "3": 6.0}},
    ]
    _patch_bayesian(monkeypatch, snapshots)
    out = _odds_helper("R1", "20260101", [])
    assert out is not None
    nums = [r["horse_number"] for r in out]
    assert nums == [2]


def test_odds_helper_handles_load_error(monkeypatch):
    fake = _types.ModuleType("bayesian_updater")

    def _boom(race_id, date_str):
        raise RuntimeError("disk error")

    fake.load_odds_history = _boom
    monkeypatch.setitem(sys.modules, "bayesian_updater", fake)
    assert _odds_helper("R1", "20260101", []) is None


def test_odds_helper_returns_none_when_module_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "bayesian_updater", None)  # forces ImportError
    assert _odds_helper("R1", "20260101", []) is None
