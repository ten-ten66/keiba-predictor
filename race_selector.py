"""
Race Selection Model
====================
Identifies race segments where the AI model has the strongest edge.
Not all races are equal -- this module finds profitable niches by slicing
backtest results across multiple dimensions:

- Field size
- Race grade
- Surface (turf/dirt)
- Distance
- Venue
- Odds range of top pick

Usage:
    python race_selector.py            # Run analysis from backtest data
    python race_selector.py --refresh  # Re-run backtest then analyze

API:
    GET /api/race-selection  -> which race types to focus on
"""

import os
import sys
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / os.environ.get("KEIBA_DATA_DIR", "data")
ANALYSIS_FILE = DATA_DIR / "race_selection_analysis.json"


# ---------------------------------------------------------------------------
# Segment Definitions
# ---------------------------------------------------------------------------

FIELD_SIZE_RANGES = [(2, 8), (9, 12), (13, 16), (17, 18)]

GRADE_KEYWORDS = {
    "G1": ["G1", "GI", "（Ｇ１）"],
    "G2": ["G2", "GII", "（Ｇ２）"],
    "G3": ["G3", "GIII", "（Ｇ３）"],
    "OP": ["オープン", "OP", "L"],
    "3勝": ["3勝"],
    "2勝": ["2勝"],
    "1勝": ["1勝"],
    "未勝利": ["未勝利"],
    "新馬": ["新馬"],
}

SURFACES = ["芝", "ダート"]

DISTANCE_RANGES = [(1000, 1400), (1400, 1800), (1800, 2200), (2200, 3600)]

VENUES = ["東京", "中山", "阪神", "京都", "中京", "小倉", "新潟", "福島",
          "札幌", "函館"]

ODDS_RANGES = [(1, 3), (3, 8), (8, 20), (20, 50), (50, 999)]


# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------

def _compute_segment_roi(bets_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute ROI metrics for a segment of bets."""
    if len(bets_df) == 0:
        return None

    total_bet = bets_df["bet"].sum()
    total_return = bets_df["return"].sum()
    profit = total_return - total_bet
    roi = (total_return / total_bet - 1) * 100 if total_bet > 0 else 0
    hits = bets_df["hit"].sum()
    hit_rate = hits / len(bets_df) * 100 if len(bets_df) > 0 else 0
    n_races = len(bets_df)

    # Statistical significance (z-test for proportion)
    # Null hypothesis: ROI = -20% (typical takeout)
    if n_races >= 10:
        expected_loss_rate = 0.20  # 20% typical house edge
        observed_return_rate = total_return / total_bet if total_bet > 0 else 0
        se = math.sqrt(observed_return_rate * (1 - observed_return_rate) / n_races) if n_races > 1 else 1
        z_score = (observed_return_rate - (1 - expected_loss_rate)) / se if se > 0 else 0
    else:
        z_score = 0

    return {
        "n_bets": n_races,
        "total_bet": int(total_bet),
        "total_return": round(float(total_return), 1),
        "profit": round(float(profit), 1),
        "roi_pct": round(roi, 1),
        "hit_rate_pct": round(hit_rate, 1),
        "hit_count": int(hits),
        "z_score": round(z_score, 2),
    }


def find_profitable_segments(backtest_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Slice the backtest by various dimensions to find profitable niches.

    Args:
        backtest_df: DataFrame with columns:
            race_id, bet, return, hit, n_runners, course_type, distance,
            venue, race_name, odds (top pick odds)

    Returns:
        Dictionary of segment analysis results
    """
    segments = {}

    # --- By field size ---
    if "n_runners" in backtest_df.columns:
        field_size_results = {}
        for lo, hi in FIELD_SIZE_RANGES:
            mask = (backtest_df["n_runners"] >= lo) & (backtest_df["n_runners"] <= hi)
            result = _compute_segment_roi(backtest_df[mask])
            if result:
                field_size_results[f"{lo}-{hi}"] = result
        segments["by_field_size"] = field_size_results

    # --- By race grade ---
    if "race_name" in backtest_df.columns:
        grade_results = {}
        for grade, keywords in GRADE_KEYWORDS.items():
            mask = backtest_df["race_name"].apply(
                lambda name: any(kw in str(name) for kw in keywords)
            )
            result = _compute_segment_roi(backtest_df[mask])
            if result and result["n_bets"] > 0:
                grade_results[grade] = result
        segments["by_grade"] = grade_results

    # --- By surface ---
    if "course_type" in backtest_df.columns:
        surface_results = {}
        for surface in SURFACES:
            mask = backtest_df["course_type"] == surface
            result = _compute_segment_roi(backtest_df[mask])
            if result:
                surface_results[surface] = result
        segments["by_surface"] = surface_results

    # --- By distance ---
    if "distance" in backtest_df.columns:
        distance_results = {}
        for lo, hi in DISTANCE_RANGES:
            mask = (backtest_df["distance"] >= lo) & (backtest_df["distance"] < hi)
            result = _compute_segment_roi(backtest_df[mask])
            if result:
                distance_results[f"{lo}-{hi}m"] = result
        segments["by_distance"] = distance_results

    # --- By venue ---
    if "venue" in backtest_df.columns:
        venue_results = {}
        for venue in VENUES:
            mask = backtest_df["venue"] == venue
            result = _compute_segment_roi(backtest_df[mask])
            if result and result["n_bets"] > 0:
                venue_results[venue] = result
        segments["by_venue"] = venue_results

    # --- By odds range of top pick ---
    if "odds" in backtest_df.columns:
        odds_results = {}
        for lo, hi in ODDS_RANGES:
            mask = (backtest_df["odds"] >= lo) & (backtest_df["odds"] < hi)
            result = _compute_segment_roi(backtest_df[mask])
            if result:
                odds_results[f"{lo}-{hi}" if hi < 999 else f"{lo}+"] = result
        segments["by_odds_range"] = odds_results

    return segments


def compute_edge_score(race_info: Dict, segment_analysis: Dict) -> float:
    """
    Compute an edge score for a specific race based on which segments it
    belongs to and how well those segments have performed historically.

    Args:
        race_info: dict with keys like course_type, distance, venue, race_name,
                   n_runners, top_pick_odds
        segment_analysis: output of find_profitable_segments()

    Returns:
        Edge score (0-100). Higher = more confidence in model edge.
    """
    scores = []
    weights = []

    # Field size
    n_runners = race_info.get("n_runners", 0)
    by_size = segment_analysis.get("by_field_size", {})
    for label, result in by_size.items():
        lo, hi = map(int, label.split("-"))
        if lo <= n_runners <= hi:
            scores.append(result["roi_pct"])
            weights.append(1.0)
            break

    # Surface
    course = race_info.get("course_type", "")
    by_surface = segment_analysis.get("by_surface", {})
    if course in by_surface:
        scores.append(by_surface[course]["roi_pct"])
        weights.append(1.5)

    # Distance
    dist = race_info.get("distance", 0)
    by_dist = segment_analysis.get("by_distance", {})
    for label, result in by_dist.items():
        parts = label.replace("m", "").split("-")
        if len(parts) == 2:
            lo, hi = int(parts[0]), int(parts[1])
            if lo <= dist < hi:
                scores.append(result["roi_pct"])
                weights.append(1.0)
                break

    # Venue
    venue = race_info.get("venue", "")
    by_venue = segment_analysis.get("by_venue", {})
    if venue in by_venue:
        scores.append(by_venue[venue]["roi_pct"])
        weights.append(1.0)

    # Odds range
    top_odds = race_info.get("top_pick_odds", 0)
    by_odds = segment_analysis.get("by_odds_range", {})
    for label, result in by_odds.items():
        if "+" in label:
            lo = int(label.replace("+", ""))
            if top_odds >= lo:
                scores.append(result["roi_pct"])
                weights.append(1.5)
                break
        elif "-" in label:
            parts = label.split("-")
            lo, hi = int(parts[0]), int(parts[1])
            if lo <= top_odds < hi:
                scores.append(result["roi_pct"])
                weights.append(1.5)
                break

    # Grade
    race_name = str(race_info.get("race_name", ""))
    by_grade = segment_analysis.get("by_grade", {})
    for grade, result in by_grade.items():
        keywords = GRADE_KEYWORDS.get(grade, [])
        if any(kw in race_name for kw in keywords):
            scores.append(result["roi_pct"])
            weights.append(2.0)
            break

    if not scores:
        return 50.0  # neutral

    # Weighted average ROI, then map to 0-100 scale
    weighted_roi = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    # Map: ROI of -30% -> 0, ROI of 0% -> 70, ROI of +10% -> 100
    # Linear mapping: score = 70 + (roi * 70/30)
    edge_score = 70 + weighted_roi * (70 / 30)
    edge_score = max(0, min(100, edge_score))

    return round(edge_score, 1)


# ---------------------------------------------------------------------------
# Build analysis from historical backtest
# ---------------------------------------------------------------------------

def run_analysis() -> Dict[str, Any]:
    """
    Run segment analysis using honest backtest data.
    Trains model, generates predictions, then slices by segments.
    """
    sys.path.insert(0, str(BASE_DIR))

    from honest_backtest import (
        load_race_data, load_payouts, create_features, train_model,
        predict, lookup_payout, TRAIN_YEARS, TEST_YEAR, BET_AMOUNT
    )

    logger.info("Loading data for race selection analysis...")
    train_raw = load_race_data(TRAIN_YEARS)
    test_raw = load_race_data([TEST_YEAR])
    payouts = load_payouts()

    # Clean data
    for df_name, df in [("train", train_raw), ("test", test_raw)]:
        df.dropna(subset=["finish_position"], inplace=True)
        mask = df["finish_position"].apply(lambda x: str(x).replace(".", "").isdigit())
        if df_name == "train":
            train_raw = df[mask].copy()
        else:
            test_raw = df[mask].copy()

    train_raw["finish_position"] = train_raw["finish_position"].astype(int)
    test_raw["finish_position"] = test_raw["finish_position"].astype(int)

    # Features and model
    train_df, feature_cols = create_features(train_raw)
    test_df, _ = create_features(test_raw)
    for c in feature_cols:
        if c not in test_df.columns:
            test_df[c] = 0

    model = train_model(train_df, feature_cols)
    preds = predict(model, test_df, feature_cols)

    # Merge race info back
    race_info_cols = ["race_id", "race_name", "venue", "course_type", "distance",
                      "horse_number", "n_runners"]
    # n_runners from test_df
    test_df_info = test_df[["race_id", "horse_number"]].copy()
    if "race_name" in test_raw.columns:
        test_df_info["race_name"] = test_raw["race_name"].values
    if "venue" in test_raw.columns:
        test_df_info["venue"] = test_raw["venue"].values
    if "course_type" in test_raw.columns:
        test_df_info["course_type"] = test_raw["course_type"].values
    if "distance" in test_raw.columns:
        test_df_info["distance"] = test_raw["distance"].values

    preds = preds.merge(test_df_info, on=["race_id", "horse_number"], how="left")

    # Count runners per race
    runner_counts = preds.groupby("race_id")["horse_number"].count().reset_index()
    runner_counts.columns = ["race_id", "n_runners"]
    preds = preds.merge(runner_counts, on="race_id", how="left")

    # Build bet-level dataframe (top-1 tansho strategy)
    bets_records = []
    for race_id, group in preds.groupby("race_id"):
        top = group.sort_values("predicted_rank").iloc[0]
        hit = int(top["finish_position"]) == 1
        ret = 0.0
        if hit:
            payout = lookup_payout(payouts, race_id, "単勝", [int(top["horse_number"])])
            if payout:
                ret = BET_AMOUNT / 100 * payout
            else:
                ret = BET_AMOUNT * float(top["odds"])

        bets_records.append({
            "race_id": race_id,
            "bet": BET_AMOUNT,
            "return": ret,
            "hit": hit,
            "n_runners": int(top.get("n_runners", 0)),
            "course_type": top.get("course_type", ""),
            "distance": top.get("distance", 0),
            "venue": top.get("venue", ""),
            "race_name": top.get("race_name", ""),
            "odds": float(top.get("odds", 0)),
        })

    bets_df = pd.DataFrame(bets_records)
    logger.info(f"Built {len(bets_df)} bet records for segment analysis")

    # Run segment analysis
    segments = find_profitable_segments(bets_df)

    # Find best segments
    best_segments = []
    for dim_name, dim_results in segments.items():
        for label, result in dim_results.items():
            if result["n_bets"] >= 20:  # minimum sample size
                best_segments.append({
                    "dimension": dim_name,
                    "label": label,
                    **result,
                })

    best_segments.sort(key=lambda x: x["roi_pct"], reverse=True)

    # Build summary
    analysis = {
        "generated_at": datetime.now().isoformat(),
        "train_period": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
        "test_period": str(TEST_YEAR),
        "total_races": len(bets_df),
        "overall_roi_pct": round(
            (bets_df["return"].sum() / bets_df["bet"].sum() - 1) * 100, 1
        ) if bets_df["bet"].sum() > 0 else 0,
        "segments": segments,
        "best_segments": best_segments[:20],
        "worst_segments": best_segments[-10:] if len(best_segments) > 10 else [],
        "recommended_focus": [
            s for s in best_segments if s["roi_pct"] > -10 and s["n_bets"] >= 30
        ][:10],
    }

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    logger.info(f"Analysis saved to {ANALYSIS_FILE}")

    # Print summary
    print("\n" + "=" * 70)
    print("RACE SELECTION ANALYSIS")
    print("=" * 70)
    print(f"Total races: {len(bets_df)}")
    print(f"Overall ROI: {analysis['overall_roi_pct']:+.1f}%")
    print(f"\nBest segments (ROI, min 20 bets):")
    for s in best_segments[:15]:
        print(f"  {s['dimension']:<20s} {s['label']:<12s} "
              f"ROI: {s['roi_pct']:+6.1f}%  "
              f"Hit: {s['hit_rate_pct']:5.1f}%  "
              f"N: {s['n_bets']:>4d}")

    print(f"\nRecommended focus areas (ROI > -10%, N >= 30):")
    for s in analysis["recommended_focus"]:
        print(f"  {s['dimension']:<20s} {s['label']:<12s} "
              f"ROI: {s['roi_pct']:+6.1f}%")

    return analysis


def load_analysis() -> Optional[Dict]:
    """Load saved analysis from file."""
    if not ANALYSIS_FILE.exists():
        return None
    with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_edge_score_for_race(race_info: Dict) -> float:
    """
    Convenience function: load analysis and compute edge score for a race.

    Args:
        race_info: dict with course_type, distance, venue, race_name,
                   n_runners, top_pick_odds

    Returns:
        Edge score 0-100
    """
    analysis = load_analysis()
    if analysis is None:
        return 50.0  # neutral if no analysis available

    segments = analysis.get("segments", {})
    return compute_edge_score(race_info, segments)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Race Selection Analysis")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-run backtest analysis")
    args = parser.parse_args()

    if args.refresh or not ANALYSIS_FILE.exists():
        run_analysis()
    else:
        analysis = load_analysis()
        if analysis:
            print(f"Analysis from: {analysis.get('generated_at', 'unknown')}")
            print(f"Overall ROI: {analysis.get('overall_roi_pct', 'N/A')}%")
            print(f"\nRecommended focus areas:")
            for s in analysis.get("recommended_focus", []):
                print(f"  {s['dimension']:<20s} {s['label']:<12s} "
                      f"ROI: {s['roi_pct']:+6.1f}%")
        else:
            print("No analysis found. Run with --refresh to generate.")
