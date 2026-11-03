"""
Horse Weight Pattern Analyzer
==============================
Analyzes horse weight change patterns to predict form.

Finds each horse's optimal weight range, recent trend direction,
and correlation between weight changes and finishing position.

Integration: import and call from prediction enrichment to add
weight-based features to horse predictions.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


class WeightPatternAnalyzer:
    """Analyze horse weight change patterns to predict form."""

    def __init__(self, history_df: pd.DataFrame):
        self.history = history_df.copy()
        # Ensure types
        self.history["horse_id"] = self.history["horse_id"].astype(str)
        self.history["race_date"] = pd.to_datetime(
            self.history["race_date"], errors="coerce"
        )
        self.history["horse_weight"] = pd.to_numeric(
            self.history["horse_weight"], errors="coerce"
        )
        self.history["weight_diff"] = pd.to_numeric(
            self.history["weight_diff"], errors="coerce"
        )
        self.history["finish_position"] = pd.to_numeric(
            self.history["finish_position"], errors="coerce"
        )

    def analyze_horse_weight_pattern(self, horse_id: str) -> Optional[Dict]:
        """Analyze weight change patterns for a horse.

        Returns dict with:
            optimal_weight: (min, max) weight range where horse performs best
            recent_trend: mean weight_diff over last 3 races (positive = gaining)
            trend_signal: "improving" | "concerning" | "stable"
            weight_performance_corr: correlation between weight_diff and finish_position
            consistency: std dev of weight (lower = more consistent)
            current_weight: most recent recorded weight
            races_analyzed: number of races used
        """
        races = self.history[self.history["horse_id"] == str(horse_id)].sort_values(
            "race_date"
        )
        if len(races) < 3:
            return None

        weights = races["horse_weight"].dropna().values
        diffs = races["weight_diff"].dropna().values
        positions = races["finish_position"].dropna().values

        if len(weights) < 3 or len(positions) < 3:
            return None

        # Find optimal weight range (where horse performs best)
        best_weight_range = None
        if len(weights) >= 5:
            # Create weight-position pairs
            valid = races[["horse_weight", "finish_position"]].dropna()
            if len(valid) >= 5:
                w = valid["horse_weight"].values
                p = valid["finish_position"].values
                # Bin weights into 5kg ranges and find best bin
                bin_size = 6
                min_w = int(w.min())
                max_w = int(w.max())
                best_avg_pos = 999
                for start in range(min_w - bin_size, max_w + 1, 2):
                    mask = (w >= start) & (w < start + bin_size)
                    if mask.sum() >= 2:
                        avg_pos = p[mask].mean()
                        if avg_pos < best_avg_pos:
                            best_avg_pos = avg_pos
                            best_weight_range = (int(start), int(start + bin_size))

        # Weight momentum (gaining vs losing trend)
        recent_trend = float(np.mean(diffs[-3:])) if len(diffs) >= 3 else 0.0

        # Correlation between weight_diff and finish_position
        # Negative correlation = weight gain hurts performance
        corr = 0.0
        valid_corr = races[["weight_diff", "finish_position"]].dropna()
        if len(valid_corr) >= 5:
            try:
                corr = float(valid_corr["weight_diff"].corr(valid_corr["finish_position"]))
                if np.isnan(corr):
                    corr = 0.0
            except Exception:
                corr = 0.0

        # Determine trend signal
        if 0 < recent_trend < 8:
            trend_signal = "improving"
        elif abs(recent_trend) > 10:
            trend_signal = "concerning"
        else:
            trend_signal = "stable"

        # Weight consistency
        consistency = float(np.std(weights)) if len(weights) >= 3 else 0.0

        return {
            "optimal_weight": best_weight_range,
            "recent_trend": round(recent_trend, 1),
            "trend_signal": trend_signal,
            "weight_performance_corr": round(corr, 4),
            "consistency": round(consistency, 1),
            "current_weight": int(weights[-1]) if len(weights) > 0 else None,
            "races_analyzed": len(races),
        }

    def get_weight_score(self, horse_id: str, current_weight: Optional[float] = None,
                         current_diff: Optional[float] = None) -> float:
        """Return a score from -1 to +1 indicating weight-based form signal.

        Positive = weight pattern suggests good form.
        Negative = weight pattern suggests poor form.
        """
        pattern = self.analyze_horse_weight_pattern(horse_id)
        if pattern is None:
            return 0.0

        score = 0.0

        # Trend signal component
        if pattern["trend_signal"] == "improving":
            score += 0.3
        elif pattern["trend_signal"] == "concerning":
            score -= 0.3

        # Optimal weight component
        if pattern["optimal_weight"] is not None and current_weight is not None:
            low, high = pattern["optimal_weight"]
            if low <= current_weight <= high:
                score += 0.4
            elif abs(current_weight - (low + high) / 2) > 10:
                score -= 0.2

        # Correlation component (negative corr with finish_position is good for weight gain)
        # If corr is negative, weight gain predicts lower finish_position (better performance)
        corr = pattern["weight_performance_corr"]
        if current_diff is not None and current_diff > 0 and corr < -0.1:
            score += 0.2  # Gaining weight and historically that helps
        elif current_diff is not None and current_diff > 0 and corr > 0.1:
            score -= 0.15  # Gaining weight but historically that hurts

        # Consistency bonus
        if pattern["consistency"] < 4:
            score += 0.1  # Very consistent weight = reliable horse

        return max(-1.0, min(1.0, round(score, 3)))

    def enrich_predictions(self, predictions: List[Dict],
                           race_df: pd.DataFrame) -> List[Dict]:
        """Add weight pattern features to prediction results.

        Args:
            predictions: list of horse prediction dicts with horse_number
            race_df: DataFrame for the race with horse_id, horse_weight, weight_diff

        Returns:
            predictions with added weight_pattern fields
        """
        for pred in predictions:
            hn = pred.get("horse_number")
            if hn is None:
                continue

            row = race_df[race_df["horse_number"] == hn]
            if row.empty:
                continue

            horse_id = str(row.iloc[0].get("horse_id", ""))
            current_weight = pd.to_numeric(row.iloc[0].get("horse_weight"), errors="coerce")
            current_diff = pd.to_numeric(row.iloc[0].get("weight_diff"), errors="coerce")

            if not horse_id:
                continue

            cw = float(current_weight) if pd.notna(current_weight) else None
            cd = float(current_diff) if pd.notna(current_diff) else None

            weight_score = self.get_weight_score(horse_id, cw, cd)
            pattern = self.analyze_horse_weight_pattern(horse_id)

            pred["weight_score"] = weight_score
            if pattern:
                pred["weight_pattern"] = {
                    "trend_signal": pattern["trend_signal"],
                    "recent_trend": pattern["recent_trend"],
                    "optimal_weight": pattern["optimal_weight"],
                    "in_optimal_range": (
                        pattern["optimal_weight"] is not None
                        and cw is not None
                        and pattern["optimal_weight"][0] <= cw <= pattern["optimal_weight"][1]
                    ),
                }

                # Adjust prob_win slightly based on weight score
                if "prob_win" in pred:
                    adjustment = weight_score * 0.02  # max 2% adjustment
                    pred["prob_win"] = max(0.001, pred["prob_win"] + adjustment)

        return predictions


if __name__ == "__main__":
    print("Weight Pattern Analyzer")
    print("=" * 50)

    # Load data
    all_path = os.path.join(DATA_DIR, "_all_races.csv")
    if not os.path.exists(all_path):
        print(f"No data file found at {all_path}")
        exit(1)

    df = pd.read_csv(all_path, dtype={"horse_id": str, "race_id": str})
    analyzer = WeightPatternAnalyzer(df)

    # Sample analysis: find a horse with enough history
    horse_counts = df["horse_id"].value_counts()
    sample_horses = horse_counts[horse_counts >= 10].head(5).index.tolist()

    for hid in sample_horses:
        result = analyzer.analyze_horse_weight_pattern(hid)
        if result:
            name = df[df["horse_id"] == hid]["horse_name"].iloc[0]
            print(f"\n{name} ({hid}):")
            print(f"  Optimal weight range: {result['optimal_weight']}")
            print(f"  Recent trend: {result['recent_trend']:+.1f}kg ({result['trend_signal']})")
            print(f"  Weight-performance corr: {result['weight_performance_corr']:.4f}")
            print(f"  Weight consistency (std): {result['consistency']:.1f}kg")
            print(f"  Races analyzed: {result['races_analyzed']}")
            score = analyzer.get_weight_score(hid, result["current_weight"])
            print(f"  Weight score: {score:+.3f}")
