"""
WIN5 Correlated Optimization
=============================
WIN5 = pick winners of 5 designated races. Huge pools (often 100M+ yen carry).

This module generates optimal WIN5 ticket sets given model predictions for
5 designated races, respecting a budget constraint.

Usage:
    from win5_optimizer import optimize_win5_tickets
    tickets = optimize_win5_tickets(race_predictions, budget=5000)

API:
    GET /api/win5?date=YYYYMMDD&budget=5000 -> optimized WIN5 tickets
"""

import os
import sys
import json
import logging
import math
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from itertools import product

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / os.environ.get("KEIBA_DATA_DIR", "data")
PREDICTION_CACHE_DIR = DATA_DIR / "prediction_cache"


def optimize_win5_tickets(race_predictions: List[Dict],
                          budget: int = 5000,
                          correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Given predictions for 5 races, generate optimal WIN5 ticket set.

    Args:
        race_predictions: list of 5 dicts, each with:
            - race_id: str
            - race_name: str (optional)
            - horses: list of dicts with 'horse_number', 'prob_win', 'horse_name'
        budget: total budget in yen (each ticket = 100 yen)
        correlation_matrix: optional 5x5 matrix of race outcome correlations

    Returns:
        Dict with tickets, expected_value, cost, etc.
    """
    if len(race_predictions) != 5:
        return {"error": f"WIN5 requires exactly 5 races, got {len(race_predictions)}"}

    max_tickets = budget // 100
    if max_tickets < 1:
        return {"error": f"Budget too small: {budget} yen (need at least 100)"}

    # For each race, select top N horses (more in uncertain races)
    selections = []
    for i, race in enumerate(race_predictions):
        horses = race.get("horses", [])
        if not horses:
            return {"error": f"No horses in race {i + 1} ({race.get('race_id', '?')})"}

        # Sort by win probability
        probs = sorted(horses, key=lambda h: float(h.get("prob_win", 0)), reverse=True)
        top_prob = float(probs[0].get("prob_win", 0))

        # In uncertain races (top prob < 15%), include more horses
        if top_prob < 0.10:
            n = min(5, len(probs))
        elif top_prob < 0.15:
            n = min(4, len(probs))
        elif top_prob < 0.25:
            n = min(3, len(probs))
        else:
            n = min(2, len(probs))

        # Ensure we don't exceed budget with too many combinations
        selections.append(probs[:n])

    # Calculate total combinations
    total_combos = 1
    for sel in selections:
        total_combos *= len(sel)

    # If too many combos, reduce selections
    while total_combos > max_tickets and any(len(s) > 1 for s in selections):
        # Find the race with most selections where top prob is highest
        # (i.e., reduce the most certain race first)
        best_race_idx = -1
        best_top_prob = -1
        for i, sel in enumerate(selections):
            if len(sel) > 1:
                top_p = float(sel[0].get("prob_win", 0))
                if top_p > best_top_prob:
                    best_top_prob = top_p
                    best_race_idx = i

        if best_race_idx >= 0:
            selections[best_race_idx] = selections[best_race_idx][:-1]

        total_combos = 1
        for sel in selections:
            total_combos *= len(sel)

    # Generate all combinations
    all_combos = list(product(*selections))

    # Score each combination by joint probability
    scored = []
    for combo in all_combos:
        prob = 1.0
        for horse in combo:
            prob *= float(horse.get("prob_win", 0.01))

        # Apply correlation adjustment if available
        if correlation_matrix is not None:
            # Simplified: boost probability for correlated positive outcomes
            corr_factor = 1.0
            for i in range(5):
                for j in range(i + 1, 5):
                    if correlation_matrix[i][j] > 0:
                        corr_factor *= (1 + correlation_matrix[i][j] * 0.1)
            prob *= corr_factor

        scored.append((combo, prob))

    # Sort by probability (best first)
    scored.sort(key=lambda x: x[1], reverse=True)

    # Select top tickets within budget
    selected = scored[:max_tickets]

    # Format tickets
    tickets = []
    total_prob = 0
    for combo, prob in selected:
        ticket = {
            "races": [],
            "joint_prob": round(prob, 8),
        }
        for i, horse in enumerate(combo):
            ticket["races"].append({
                "race_idx": i + 1,
                "race_id": race_predictions[i].get("race_id", ""),
                "race_name": race_predictions[i].get("race_name", f"Race {i + 1}"),
                "horse_number": int(horse.get("horse_number", 0)),
                "horse_name": horse.get("horse_name", ""),
                "prob_win": round(float(horse.get("prob_win", 0)), 4),
            })
        tickets.append(ticket)
        total_prob += prob

    # Estimate expected payout
    # WIN5 typical pool: 100M yen, typical hit count: 1-5 tickets
    # Rough estimate: payout = pool / n_winning_tickets
    # We estimate the expected number of winning tickets from total ticket sales
    est_pool = 100_000_000  # 1 oku yen typical
    est_ticket_sales = est_pool // 100  # 1M tickets

    cost = len(tickets) * 100

    result = {
        "tickets": tickets,
        "n_tickets": len(tickets),
        "cost": cost,
        "budget": budget,
        "total_hit_probability": round(total_prob, 8),
        "selections_per_race": [len(s) for s in selections],
        "total_combinations": total_combos,
        "races": [],
    }

    # Race summary
    for i, race in enumerate(race_predictions):
        sel_horses = selections[i]
        race_summary = {
            "race_idx": i + 1,
            "race_id": race.get("race_id", ""),
            "race_name": race.get("race_name", ""),
            "selected_horses": [
                {
                    "horse_number": int(h.get("horse_number", 0)),
                    "horse_name": h.get("horse_name", ""),
                    "prob_win": round(float(h.get("prob_win", 0)), 4),
                }
                for h in sel_horses
            ],
            "coverage_prob": round(
                sum(float(h.get("prob_win", 0)) for h in sel_horses), 4
            ),
        }
        result["races"].append(race_summary)

    # Coverage probability (probability of hitting all 5 correctly)
    coverage = 1.0
    for race_summary in result["races"]:
        coverage *= race_summary["coverage_prob"]
    result["coverage_probability"] = round(coverage, 8)

    return result


def get_win5_races(date_str: Optional[str] = None) -> List[str]:
    """
    Get the 5 designated WIN5 race IDs for a given date.
    WIN5 races are typically the last races at 2-3 venues on a race day.

    In JRA, WIN5 target races are:
    - 5 races designated by JRA (usually races 9-12 at different venues)

    Args:
        date_str: YYYYMMDD format

    Returns:
        List of 5 race IDs (or fewer if not all found)
    """
    if date_str is None:
        date_str = date.today().strftime("%Y%m%d")

    # Look for prediction cache files for this date
    cache_files = sorted(PREDICTION_CACHE_DIR.glob(f"{date_str}_*.json"))
    if not cache_files:
        return []

    # Group by venue (race_id format: YYYYVVRRDDNN where VV=venue, RR=round, DD=day, NN=race_num)
    venues = {}
    for f in cache_files:
        race_id = f.stem.split("_", 1)[1]
        # Last 2 digits = race number
        try:
            race_num = int(race_id[-2:])
            venue_key = race_id[:-2]
            if venue_key not in venues:
                venues[venue_key] = []
            venues[venue_key].append((race_num, race_id))
        except (ValueError, IndexError):
            continue

    # WIN5 = last race at each of the 2-3 venues, plus a few before
    # Typically: venues have 12 races, WIN5 uses races 9-12 across venues
    # Select the 5 latest-numbered races across all venues
    all_races = []
    for venue_key, races in venues.items():
        races.sort(reverse=True)
        for race_num, race_id in races:
            if race_num >= 9:  # WIN5 target races are typically R9+
                all_races.append((race_num, race_id, venue_key))

    # Sort by race number descending, take first 5
    all_races.sort(key=lambda x: x[0], reverse=True)
    win5_ids = [r[1] for r in all_races[:5]]

    # If we don't have 5, take whatever we have from the later races
    if len(win5_ids) < 5:
        remaining = [(num, rid) for vk, races in venues.items()
                     for num, rid in races if rid not in win5_ids]
        remaining.sort(key=lambda x: x[0], reverse=True)
        for num, rid in remaining:
            if rid not in win5_ids:
                win5_ids.append(rid)
            if len(win5_ids) >= 5:
                break

    return win5_ids[:5]


def load_predictions_for_race(race_id: str, date_str: Optional[str] = None) -> Optional[Dict]:
    """Load predictions from cache for a single race."""
    if date_str is None:
        date_str = date.today().strftime("%Y%m%d")

    cache_file = PREDICTION_CACHE_DIR / f"{date_str}_{race_id}.json"
    if not cache_file.exists():
        # Try glob
        matches = list(PREDICTION_CACHE_DIR.glob(f"*_{race_id}.json"))
        if matches:
            cache_file = matches[0]
        else:
            return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        predictions = cached.get("predictions", cached.get("horses", []))
        race_name = cached.get("race_info", {}).get("race_name", cached.get("race_name", ""))
        return {
            "race_id": race_id,
            "race_name": race_name,
            "horses": predictions,
        }
    except Exception as e:
        logger.error(f"Failed to load predictions for {race_id}: {e}")
        return None


def generate_win5_tickets(date_str: Optional[str] = None,
                          budget: int = 5000) -> Dict[str, Any]:
    """
    High-level function: find WIN5 races, load predictions, optimize tickets.

    Args:
        date_str: YYYYMMDD format
        budget: total budget in yen

    Returns:
        WIN5 optimization result dict
    """
    if date_str is None:
        date_str = date.today().strftime("%Y%m%d")

    win5_race_ids = get_win5_races(date_str)
    if len(win5_race_ids) < 5:
        return {
            "error": f"Could not find 5 WIN5 races for {date_str}. "
                     f"Found {len(win5_race_ids)} races.",
            "found_race_ids": win5_race_ids,
            "date": date_str,
        }

    race_predictions = []
    for rid in win5_race_ids:
        preds = load_predictions_for_race(rid, date_str)
        if preds is None:
            return {
                "error": f"Missing predictions for race {rid}",
                "date": date_str,
            }
        race_predictions.append(preds)

    result = optimize_win5_tickets(race_predictions, budget)
    result["date"] = date_str
    result["win5_race_ids"] = win5_race_ids

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="WIN5 Optimizer")
    parser.add_argument("--date", type=str, default=None, help="Date YYYYMMDD")
    parser.add_argument("--budget", type=int, default=5000, help="Budget in yen (default: 5000)")
    args = parser.parse_args()

    result = generate_win5_tickets(args.date, args.budget)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"WIN5 Tickets for {result.get('date', 'today')}")
        print(f"Budget: {result['budget']} yen")
        print(f"Tickets: {result['n_tickets']} ({result['cost']} yen)")
        print(f"Coverage probability: {result['coverage_probability']:.6f}")
        print()
        for race_summary in result.get("races", []):
            print(f"Race {race_summary['race_idx']}: {race_summary['race_name']} "
                  f"({race_summary['race_id']})")
            for h in race_summary["selected_horses"]:
                print(f"  #{h['horse_number']} {h['horse_name']} "
                      f"P(win)={h['prob_win']:.3f}")
            print(f"  Coverage: {race_summary['coverage_prob']:.3f}")
