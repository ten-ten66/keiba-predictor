"""Unit tests for Kelly sizing and presets in AutoBettor."""
import pytest

from auto_bettor import AutoBettor


@pytest.fixture
def bettor():
    ab = AutoBettor(bankroll=100_000, kelly_fraction=0.25, min_bet=100, max_bet_pct=0.05)
    return ab


def test_kelly_preset_strings():
    assert AutoBettor(kelly_fraction="full").kelly_fraction == 1.0
    assert AutoBettor(kelly_fraction="half").kelly_fraction == 0.5
    assert AutoBettor(kelly_fraction="quarter").kelly_fraction == 0.25
    assert AutoBettor(kelly_fraction="eighth").kelly_fraction == 0.125


def test_kelly_preset_invalid():
    with pytest.raises(ValueError):
        AutoBettor(kelly_fraction="wrong")


def test_kelly_size_zero_when_no_kelly(bettor):
    # kelly_fraction=0 → falls through to min_bet
    assert bettor._kelly_size({"kelly_fraction": 0}) == bettor.min_bet


def test_kelly_size_uses_quarter_kelly(bettor):
    # Full Kelly = 0.2 (20% of bankroll). Quarter Kelly = 5% of bankroll = 5000.
    # But max_bet_pct = 0.05 (5%), so final is bounded by max_bet_pct → 5000.
    bet = {"kelly_fraction": 0.2, "ev_std": 0, "ev_mean": 0}
    amount = bettor._kelly_size(bet)
    assert amount == 5000


def test_uncertainty_discount_reduces_size():
    ab = AutoBettor(bankroll=100_000, kelly_fraction=1.0, max_bet_pct=1.0, uncertainty_discount=True)
    low_unc = ab._kelly_size({"kelly_fraction": 0.1, "ev_mean": 0.20, "ev_std": 0.02})  # rel=10%
    high_unc = ab._kelly_size({"kelly_fraction": 0.1, "ev_mean": 0.20, "ev_std": 0.12})  # rel=60%
    # Higher uncertainty → smaller bet
    assert high_unc < low_unc


def test_uncertainty_discount_off_ignores_std():
    ab = AutoBettor(bankroll=100_000, kelly_fraction=1.0, max_bet_pct=1.0, uncertainty_discount=False)
    low = ab._kelly_size({"kelly_fraction": 0.1, "ev_mean": 0.20, "ev_std": 0.02})
    high = ab._kelly_size({"kelly_fraction": 0.1, "ev_mean": 0.20, "ev_std": 0.12})
    assert low == high  # std ignored


def test_uncertainty_discount_floor_at_30pct():
    """Even extreme uncertainty shouldn't reduce below 30% of original."""
    ab = AutoBettor(bankroll=100_000, kelly_fraction=1.0, max_bet_pct=1.0, uncertainty_discount=True)
    # Massive std: rel ≈ 5, so raw 1-rel = -4, clamped to 0.3
    extreme = ab._kelly_size({"kelly_fraction": 0.1, "ev_mean": 0.10, "ev_std": 0.50})
    baseline = ab._kelly_size({"kelly_fraction": 0.1, "ev_mean": 0.10, "ev_std": 0})
    assert extreme >= baseline * 0.3 * 0.9  # allow 10% rounding slack
