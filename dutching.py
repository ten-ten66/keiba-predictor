"""
Dutching (Spread Bet) Strategy
================================
Instead of betting on a single horse, spread bets proportionally
across multiple selections so that the profit is roughly equal
regardless of which selected horse wins.

Dutch formula: stake_i = total_stake * (1/odds_i) / sum(1/odds_j)

This ensures equal return from any winning selection.
The strategy is profitable when combined_prob > sum(1/odds_i),
i.e., when the market overestimates the chances of non-selected horses.

Endpoint: GET /api/dutch/{race_id}?budget=1000
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def dutch_bet(horses: List[Dict], total_stake: int = 1000,
              bet_type: str = "tansho",
              min_prob: float = 0.05, max_selections: int = 4) -> Optional[Dict]:
    """Calculate optimal stake distribution across multiple horses.

    Goal: equal profit regardless of which horse wins.

    Args:
        horses: list of dicts with keys: horse_number, horse_name, odds, prob_win
        total_stake: total amount to bet (yen)
        bet_type: "tansho" (win) or "fukusho" (place)
        min_prob: minimum prob_win to consider a horse
        max_selections: maximum number of horses to include

    Returns:
        dict with bets list, total_stake, combined_prob, expected_return, dutch_roi
        or None if no valid selections
    """
    # Filter to horses with sufficient probability and valid odds
    selected = [
        h for h in horses
        if h.get("prob_win", 0) > min_prob and h.get("odds", 0) > 1.0
    ]

    # Sort by probability descending, take top N
    selected.sort(key=lambda h: h.get("prob_win", 0), reverse=True)
    selected = selected[:max_selections]

    if not selected:
        return None

    # Check that we have valid odds for all
    valid = [h for h in selected if h.get("odds", 0) > 0]
    if not valid:
        return None

    selected = valid

    # Dutch formula: stake_i = total_stake * (1/odds_i) / sum(1/odds_j)
    inv_odds = [1.0 / h["odds"] for h in selected]
    inv_odds_sum = sum(inv_odds)

    if inv_odds_sum <= 0:
        return None

    bets = []
    for i, h in enumerate(selected):
        stake = total_stake * inv_odds[i] / inv_odds_sum
        # Round to nearest 100 yen (JRA minimum)
        stake_rounded = max(100, int(round(stake / 100) * 100))
        expected_return = stake_rounded * h["odds"]

        bets.append({
            "horse_number": int(h.get("horse_number", 0)),
            "horse_name": str(h.get("horse_name", "")),
            "odds": round(float(h["odds"]), 1),
            "stake": stake_rounded,
            "expected_return": round(expected_return),
            "prob_win": round(float(h.get("prob_win", 0)), 4),
        })

    # Actual total after rounding
    actual_total = sum(b["stake"] for b in bets)

    # Combined probability of any selection winning
    combined_prob = sum(h.get("prob_win", 0) for h in selected)
    combined_prob = min(combined_prob, 0.99)  # Cap at 99%

    # Average expected return (roughly equal for all selections due to dutching)
    avg_return = np.mean([b["expected_return"] for b in bets]) if bets else 0

    # Dutch book margin: if inv_odds_sum < 1.0, the dutch bet has positive expected value
    # from market perspective (before model probabilities)
    market_margin = inv_odds_sum

    # Expected profit = combined_prob * avg_return - actual_total
    expected_profit = combined_prob * avg_return - actual_total

    # ROI calculation
    dutch_roi = (expected_profit / actual_total * 100) if actual_total > 0 else -100

    return {
        "bet_type": bet_type,
        "bets": bets,
        "total_stake": actual_total,
        "budget_requested": total_stake,
        "n_selections": len(bets),
        "combined_prob": round(combined_prob, 4),
        "market_margin": round(market_margin, 4),
        "avg_expected_return": round(avg_return),
        "expected_profit": round(expected_profit),
        "dutch_roi": round(dutch_roi, 1),
        "is_profitable": dutch_roi > 0,
    }


def dutch_top3(horses: List[Dict], total_stake: int = 1000,
               min_prob: float = 0.10, max_selections: int = 3) -> Optional[Dict]:
    """Dutching strategy for place (top 3) bets.

    Uses prob_top3 instead of prob_win, and assumes place odds ~ odds / 3.
    """
    place_horses = []
    for h in horses:
        place_odds = h.get("odds", 0) / 3.0  # Rough place odds estimate
        if place_odds < 1.1:
            place_odds = 1.1
        place_horses.append({
            **h,
            "odds": round(place_odds, 1),
            "prob_win": h.get("prob_top3", h.get("top3_prob", 0)),
        })

    result = dutch_bet(place_horses, total_stake, bet_type="fukusho",
                       min_prob=min_prob, max_selections=max_selections)
    return result


def compute_dutch_for_race(predictions: List[Dict],
                           budget: int = 1000) -> Dict:
    """Compute dutching strategies for a race given predictions.

    Returns both win-dutch and place-dutch recommendations.
    """
    result = {
        "budget": budget,
        "win_dutch": None,
        "place_dutch": None,
        "recommendation": "",
    }

    # Win dutching
    win_dutch = dutch_bet(predictions, total_stake=budget)
    if win_dutch:
        result["win_dutch"] = win_dutch

    # Place dutching
    place_dutch = dutch_top3(predictions, total_stake=budget)
    if place_dutch:
        result["place_dutch"] = place_dutch

    # Recommendation
    if win_dutch and win_dutch["is_profitable"]:
        result["recommendation"] = (
            f"Win dutch recommended: {win_dutch['n_selections']} horses, "
            f"combined {win_dutch['combined_prob']:.0%} win chance, "
            f"expected ROI {win_dutch['dutch_roi']:+.1f}%"
        )
    elif place_dutch and place_dutch["is_profitable"]:
        result["recommendation"] = (
            f"Place dutch recommended: {place_dutch['n_selections']} horses, "
            f"combined {place_dutch['combined_prob']:.0%} place chance, "
            f"expected ROI {place_dutch['dutch_roi']:+.1f}%"
        )
    else:
        result["recommendation"] = "No profitable dutch opportunity found for this race."

    return result
