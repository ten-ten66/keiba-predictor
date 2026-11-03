"""Unit tests for backtest metrics enhancements."""
import datetime as dt

from honest_backtest import compute_metrics


def _mk_bet(bet=100, ret=0, hit=False, date=None):
    return {"bet": bet, "return": ret, "hit": hit, "date": date or dt.date(2024, 1, 1)}


def test_empty_returns_error_dict():
    m = compute_metrics([], "empty")
    assert m["total_bets"] == 0
    assert "error" in m


def test_basic_profit_and_hit_rate():
    bets = [
        _mk_bet(bet=100, ret=300, hit=True),
        _mk_bet(bet=100, ret=0, hit=False),
        _mk_bet(bet=100, ret=200, hit=True),
    ]
    m = compute_metrics(bets, "test")
    assert m["total_bets"] == 3
    assert m["hit_count"] == 2
    assert round(m["roi_pct"], 0) == round((500 / 300 - 1) * 100)


def test_longest_losing_streak():
    # pattern: W, L, L, L, W, L, L
    outcomes = [True, False, False, False, True, False, False]
    bets = [_mk_bet(hit=h, ret=200 if h else 0) for h in outcomes]
    m = compute_metrics(bets, "streak")
    assert m["longest_losing_streak"] == 3


def test_profit_factor():
    bets = [
        _mk_bet(bet=100, ret=300, hit=True),  # gross win = 200
        _mk_bet(bet=100, ret=0, hit=False),   # gross loss = 100
    ]
    m = compute_metrics(bets, "pf")
    # profit_factor = 200 / 100 = 2.0
    assert m["profit_factor"] == 2.0


def test_bootstrap_ci_requires_min_30_bets():
    # 20 bets: bootstrap CI should be None
    small = [_mk_bet(hit=True, ret=150) for _ in range(20)]
    m = compute_metrics(small, "small")
    assert m["roi_bootstrap_95ci_lower"] is None
    # 40 bets: bootstrap CI present
    big = [_mk_bet(hit=i % 2 == 0, ret=200 if i % 2 == 0 else 0) for i in range(40)]
    m = compute_metrics(big, "big")
    assert m["roi_bootstrap_95ci_lower"] is not None
    assert m["roi_bootstrap_95ci_upper"] is not None


def test_sample_adequate_threshold():
    # 99 bets: not adequate
    bets99 = [_mk_bet(hit=True, ret=200) for _ in range(99)]
    assert compute_metrics(bets99, "99")["sample_adequate"] is False
    # 100 bets: adequate
    bets100 = [_mk_bet(hit=True, ret=200) for _ in range(100)]
    assert compute_metrics(bets100, "100")["sample_adequate"] is True


def test_sharpe_and_tstat_present_with_varying_monthly_rois():
    """Requires varying monthly ROIs (non-zero std) to produce Sharpe."""
    bets = []
    # Give each month a different hit pattern so monthly ROIs vary
    patterns = {1: 5, 2: 2, 3: 4, 4: 1}  # hits per 10 bets
    for month, hits in patterns.items():
        for i in range(10):
            h = i < hits
            bets.append(_mk_bet(
                hit=h,
                ret=300 if h else 0,
                date=dt.date(2024, month, 15),
            ))
    m = compute_metrics(bets, "varied")
    assert m["sharpe_monthly"] is not None
    assert m["t_stat_monthly"] is not None
