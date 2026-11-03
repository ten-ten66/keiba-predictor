"""
Frame (Post-Position) Bias Model
==================================
Calculates post-position advantages per venue/distance/surface combination.

Inner frames (1-3) often have advantages on turf at shorter distances,
while outer frames can be advantaged on dirt or longer distances.
This module quantifies these biases from historical data and saves
them for use in prediction enrichment.

Output: data/frame_bias.json
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRAME_BIAS_PATH = os.path.join(DATA_DIR, "frame_bias.json")


def _categorize_distance(dist: float) -> str:
    """Categorize distance into sprint/mile/intermediate/long/extended."""
    if dist <= 1200:
        return "sprint"
    elif dist <= 1600:
        return "mile"
    elif dist <= 2000:
        return "intermediate"
    elif dist <= 2400:
        return "long"
    else:
        return "extended"


def compute_frame_bias(history_df: pd.DataFrame,
                       min_races: int = 30) -> Dict:
    """Compute win rate by frame number for each venue/distance/surface combo.

    Args:
        history_df: historical race DataFrame with frame_number, finish_position,
                    venue, distance, course_type, race_id columns
        min_races: minimum number of races in a group to include

    Returns:
        dict keyed by "venue|distance_category|surface" with per-frame stats
    """
    df = history_df.copy()

    # Clean data
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df["frame_number"] = pd.to_numeric(df["frame_number"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df = df.dropna(subset=["finish_position", "frame_number", "distance"])
    df = df[df["finish_position"].apply(lambda x: str(x).replace(".", "").isdigit())]
    df["finish_position"] = df["finish_position"].astype(int)
    df["frame_number"] = df["frame_number"].astype(int)
    df = df[df["frame_number"].between(1, 8)]

    # Add distance category
    df["distance_category"] = df["distance"].apply(_categorize_distance)

    bias = {}

    for (venue, dist_cat, surface), group in df.groupby(
        ["venue", "distance_category", "course_type"]
    ):
        # Need minimum number of races for statistical significance
        n_races = group["race_id"].nunique()
        if n_races < min_races:
            continue

        frame_stats = group.groupby("frame_number").agg(
            total=("finish_position", "count"),
            wins=("finish_position", lambda x: (x == 1).sum()),
            top3=("finish_position", lambda x: (x <= 3).sum()),
        )

        frame_stats["win_rate"] = frame_stats["wins"] / frame_stats["total"]
        frame_stats["top3_rate"] = frame_stats["top3"] / frame_stats["total"]

        # Expected win rate = 1 / avg_field_size
        avg_field = group.groupby("race_id").size().mean()
        expected_win = 1.0 / avg_field if avg_field > 0 else 0.0625
        expected_top3 = 3.0 / avg_field if avg_field > 0 else 0.1875

        # Bias = actual / expected (>1 = advantaged, <1 = disadvantaged)
        frame_stats["win_bias"] = frame_stats["win_rate"] / expected_win
        frame_stats["top3_bias"] = frame_stats["top3_rate"] / expected_top3

        key = f"{venue}|{dist_cat}|{surface}"
        bias[key] = {
            "n_races": int(n_races),
            "avg_field_size": round(float(avg_field), 1),
            "expected_win_rate": round(float(expected_win), 4),
            "frames": {},
        }
        for frame_num, row in frame_stats.iterrows():
            bias[key]["frames"][str(int(frame_num))] = {
                "total": int(row["total"]),
                "wins": int(row["wins"]),
                "top3": int(row["top3"]),
                "win_rate": round(float(row["win_rate"]), 4),
                "top3_rate": round(float(row["top3_rate"]), 4),
                "win_bias": round(float(row["win_bias"]), 3),
                "top3_bias": round(float(row["top3_bias"]), 3),
            }

    return bias


def save_frame_bias(bias: Dict, path: str = FRAME_BIAS_PATH):
    """Save frame bias data to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bias, f, ensure_ascii=False, indent=2)
    logger.info(f"Frame bias saved to {path} ({len(bias)} venue/dist/surface combos)")


def load_frame_bias(path: str = FRAME_BIAS_PATH) -> Dict:
    """Load frame bias data from JSON."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_frame_bias_score(bias_data: Dict, venue: str, distance: float,
                         surface: str, frame_number: int) -> Dict:
    """Get frame bias score for a specific horse position.

    Args:
        bias_data: loaded frame bias dict
        venue: race venue name
        distance: race distance in meters
        surface: course type (e.g. "芝", "ダート")
        frame_number: horse's frame number (1-8)

    Returns:
        dict with win_bias, top3_bias, and adjustment recommendation
    """
    dist_cat = _categorize_distance(distance)
    key = f"{venue}|{dist_cat}|{surface}"

    if key not in bias_data:
        return {"win_bias": 1.0, "top3_bias": 1.0, "adjustment": 0.0, "data_available": False}

    entry = bias_data[key]
    frame_str = str(int(frame_number))
    frame_info = entry.get("frames", {}).get(frame_str)

    if frame_info is None:
        return {"win_bias": 1.0, "top3_bias": 1.0, "adjustment": 0.0, "data_available": False}

    win_bias = frame_info["win_bias"]
    top3_bias = frame_info["top3_bias"]

    # Calculate adjustment: scale bias into a -0.05 to +0.05 range
    # A bias of 1.0 = no adjustment, 1.2 = +0.02, 0.8 = -0.02
    adjustment = (win_bias - 1.0) * 0.1
    adjustment = max(-0.05, min(0.05, adjustment))

    return {
        "win_bias": win_bias,
        "top3_bias": top3_bias,
        "adjustment": round(adjustment, 4),
        "data_available": True,
        "sample_size": frame_info["total"],
        "win_rate": frame_info["win_rate"],
    }


def enrich_predictions_with_frame_bias(predictions: List[Dict],
                                        race_df: pd.DataFrame,
                                        bias_data: Optional[Dict] = None) -> List[Dict]:
    """Add frame bias features to prediction results.

    Args:
        predictions: list of horse prediction dicts
        race_df: DataFrame for the race
        bias_data: pre-loaded bias data (loads from file if None)

    Returns:
        predictions with added frame_bias fields and adjusted probabilities
    """
    if bias_data is None:
        bias_data = load_frame_bias()

    if not bias_data:
        return predictions

    # Get race info
    if race_df.empty:
        return predictions

    first = race_df.iloc[0]
    venue = str(first.get("venue", ""))
    distance = float(first.get("distance", 0))
    surface = str(first.get("course_type", ""))

    for pred in predictions:
        hn = pred.get("horse_number")
        if hn is None:
            continue

        row = race_df[race_df["horse_number"] == hn]
        if row.empty:
            continue

        frame = pd.to_numeric(row.iloc[0].get("frame_number"), errors="coerce")
        if pd.isna(frame):
            continue

        bias_info = get_frame_bias_score(bias_data, venue, distance, surface, int(frame))
        pred["frame_bias"] = {
            "frame_number": int(frame),
            "win_bias": bias_info["win_bias"],
            "top3_bias": bias_info["top3_bias"],
            "data_available": bias_info["data_available"],
        }

        # Apply small probability adjustment
        if bias_info["data_available"] and "prob_win" in pred:
            adj = bias_info["adjustment"]
            pred["prob_win"] = max(0.001, pred["prob_win"] + adj)

    return predictions


if __name__ == "__main__":
    print("Frame Bias Computation")
    print("=" * 60)

    all_path = os.path.join(DATA_DIR, "_all_races.csv")
    if not os.path.exists(all_path):
        print(f"No data file found at {all_path}")
        exit(1)

    df = pd.read_csv(all_path, dtype={"horse_id": str, "race_id": str})
    print(f"Loaded {len(df):,} rows")

    bias = compute_frame_bias(df, min_races=30)
    save_frame_bias(bias)
    print(f"\nComputed bias for {len(bias)} venue/distance/surface combinations")

    # Show top biased frames
    print(f"\nMost biased frames (sorted by abs deviation from 1.0):")
    print(f"{'Key':<35s} {'Frame':>5s} {'WinBias':>8s} {'Top3Bias':>9s} {'Races':>6s}")
    print("-" * 70)

    all_entries = []
    for key, entry in bias.items():
        for frame, stats in entry["frames"].items():
            all_entries.append((key, frame, stats["win_bias"], stats["top3_bias"], stats["total"]))

    all_entries.sort(key=lambda x: abs(x[2] - 1.0), reverse=True)
    for key, frame, wb, tb, total in all_entries[:20]:
        print(f"{key:<35s} {frame:>5s} {wb:>8.3f} {tb:>9.3f} {total:>6d}")
