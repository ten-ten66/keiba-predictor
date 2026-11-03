"""Unit tests for RiskManager circuit breakers."""
import tempfile
from pathlib import Path

import pytest

from risk_manager import RiskManager


@pytest.fixture
def rm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rmgr = RiskManager()
    rmgr.state_file = tmp_path / "data" / "risk_state.json"
    return rmgr


def test_allows_normal_bet(rm):
    ok, reason = rm.check_bet_allowed(bet_amount=500, bankroll=100000)
    assert ok, reason


def test_blocks_bet_over_max_size(rm):
    ok, reason = rm.check_bet_allowed(bet_amount=999_999, bankroll=100000)
    assert not ok
    assert "max_bet_size" in reason


def test_per_race_aggregate_cap(rm):
    bankroll = 100_000
    cap = int(bankroll * rm.config["max_race_wager_pct"])  # 10_000
    # First bet under cap: allowed
    ok, _ = rm.check_bet_allowed(bet_amount=5000, bankroll=bankroll, race_total=0)
    assert ok
    # Second bet that would push us over the per-race cap: blocked
    ok, reason = rm.check_bet_allowed(bet_amount=5001, bankroll=bankroll, race_total=5000)
    assert not ok
    assert "per-race" in reason


def test_halt_and_resume(rm):
    rm.halt("test kill switch")
    ok, reason = rm.check_bet_allowed(bet_amount=100, bankroll=100000)
    assert not ok
    assert "halted" in reason
    rm.resume()
    ok, _ = rm.check_bet_allowed(bet_amount=100, bankroll=100000)
    assert ok


def test_daily_bet_count_cap(rm):
    rm.config["max_bets_per_day"] = 2
    rm.record_result(won=True, amount=100, payout=200, bankroll=100100)
    rm.record_result(won=False, amount=100, payout=0, bankroll=100000)
    ok, reason = rm.check_bet_allowed(bet_amount=100, bankroll=100000)
    assert not ok
    assert "daily bet count" in reason


def test_peak_bankroll_tracks_actual_wealth(rm):
    # Win raises bankroll to 110_000, next loss drops to 105_000
    rm.record_result(won=True, amount=1000, payout=11000, bankroll=110_000)
    rm.record_result(won=False, amount=5000, payout=0, bankroll=105_000)
    status = rm.get_status()
    assert status["peak_bankroll"] == 110_000


def test_consecutive_loss_limit_triggers_cooldown(rm):
    rm.config["consecutive_loss_limit"] = 3
    for _ in range(3):
        rm.record_result(won=False, amount=100, payout=0, bankroll=100_000)
    ok, reason = rm.check_bet_allowed(bet_amount=100, bankroll=100_000)
    assert not ok
    assert "consecutive" in reason or "cooldown" in reason


def test_status_exposes_halt_and_bet_count(rm):
    rm.halt("for audit")
    status = rm.get_status()
    assert status["halted"] is True
    assert status["halt_reason"] == "for audit"
    assert "bets_today" in status
    assert "max_race_wager_pct" in status
