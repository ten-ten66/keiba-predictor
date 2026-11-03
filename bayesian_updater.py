"""
Bayesian Odds Updater
=====================
Combines AI model probabilities with live market odds using a Bayesian approach.

- model_weight: trust in the AI model
- market_weight: trust in the market (crowd wisdom)
- Detects smart money signals from odds movements
- Stores odds history per race for tracking
- Calibrates weights from historical data

Usage:
    from bayesian_updater import BayesianOddsUpdater
    updater = BayesianOddsUpdater()
    combined_prob = updater.update(model_prob=0.15, current_odds=8.0, prev_odds=10.0)
"""

import os
import sys
import json
import logging
import math
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / os.environ.get("KEIBA_DATA_DIR", "data")
ODDS_HISTORY_DIR = DATA_DIR / "odds_history"
CALIBRATION_FILE = DATA_DIR / "bayesian_calibration.json"
PREDICTION_CACHE_DIR = DATA_DIR / "prediction_cache"


class BayesianOddsUpdater:
    """Combine AI model prediction with live market odds using Bayesian update."""

    def __init__(self, model_weight: float = 0.5, market_weight: float = 0.5):
        """
        Args:
            model_weight: how much to trust the AI model (0-1)
            market_weight: how much to trust the market odds (0-1)
        These should be calibrated from historical data.
        """
        self.model_weight = model_weight
        self.market_weight = market_weight
        # Try to load calibrated weights
        self._load_calibration()

    def _load_calibration(self):
        """Load calibrated weights from file if available."""
        if CALIBRATION_FILE.exists():
            try:
                with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                    cal = json.load(f)
                self.model_weight = cal.get("model_weight", self.model_weight)
                self.market_weight = cal.get("market_weight", self.market_weight)
                logger.info(f"Loaded calibrated weights: model={self.model_weight:.3f}, "
                            f"market={self.market_weight:.3f}")
            except Exception as e:
                logger.warning(f"Failed to load calibration: {e}")

    def update(self, model_prob: float, current_odds: float,
               prev_odds: Optional[float] = None) -> float:
        """
        Combine model prediction with live odds using Bayesian update.

        If odds moved significantly (smart money), adjust accordingly:
        - Odds dropped (money coming in): increase probability
        - Odds drifted up (money going elsewhere): decrease probability

        Args:
            model_prob: AI model's probability of winning
            current_odds: current market odds
            prev_odds: previous odds snapshot (for movement detection)

        Returns:
            Combined probability (0.001 to 0.99)
        """
        market_prob = 1.0 / current_odds if current_odds > 0 else 0

        # Base combination (weighted average)
        combined = self.model_weight * model_prob + self.market_weight * market_prob

        # Odds movement adjustment
        if prev_odds and prev_odds > 0:
            odds_change = (current_odds - prev_odds) / prev_odds
            # Negative change = odds shortened = money coming in
            # This is a "smart money" signal
            movement_signal = -odds_change * 0.1  # scale factor
            combined = combined * (1 + movement_signal)

        # Clamp to valid range
        return max(0.001, min(0.99, combined))

    def update_race(self, predictions: List[Dict], current_odds: Dict[int, float],
                    prev_odds: Optional[Dict[int, float]] = None) -> List[Dict]:
        """
        Update all horses in a race, then normalize probabilities.

        Args:
            predictions: list of horse dicts with 'horse_number', 'prob_win'
            current_odds: {horse_number: odds}
            prev_odds: {horse_number: previous_odds}

        Returns:
            Updated predictions list with 'bayesian_prob' field added
        """
        raw_probs = {}
        for pred in predictions:
            h_num = int(pred.get("horse_number", 0))
            model_prob = float(pred.get("prob_win", pred.get("win_prob", 0)))
            odds = current_odds.get(h_num, 0)
            prev = prev_odds.get(h_num) if prev_odds else None

            if odds > 0 and model_prob > 0:
                raw_probs[h_num] = self.update(model_prob, odds, prev)
            else:
                raw_probs[h_num] = model_prob

        # Normalize within race so probabilities sum to ~1
        total = sum(raw_probs.values())
        if total > 0:
            for h_num in raw_probs:
                raw_probs[h_num] /= total

        # Apply back to predictions
        for pred in predictions:
            h_num = int(pred.get("horse_number", 0))
            pred["bayesian_prob"] = round(raw_probs.get(h_num, 0), 6)

            # Smart money signal
            odds = current_odds.get(h_num, 0)
            prev = prev_odds.get(h_num) if prev_odds else None
            if prev and prev > 0:
                signal = self.detect_smart_money([prev, odds])
                pred["smart_money_signal"] = round(signal, 3)

        return predictions

    def detect_smart_money(self, odds_history: List[float]) -> float:
        """
        Detect sudden odds movements that suggest insider/informed betting.

        Args:
            odds_history: chronological list of odds snapshots

        Returns:
            Signal strength (-1 to +1, positive = smart money backing this horse)
        """
        if len(odds_history) < 2:
            return 0.0

        recent_change = (odds_history[-1] - odds_history[-2]) / odds_history[-2]

        # Sharp drop (>10% in short time) = likely smart money
        if recent_change < -0.10:
            return min(1.0, abs(recent_change) * 3)
        elif recent_change > 0.15:
            return max(-1.0, -recent_change * 2)
        return 0.0

    def detect_smart_money_extended(self, odds_history: List[float]) -> Dict[str, Any]:
        """
        Extended smart money detection with multiple signals.

        Args:
            odds_history: chronological list of odds snapshots (at least 2)

        Returns:
            Dict with signal, trend, volatility info
        """
        if len(odds_history) < 2:
            return {"signal": 0.0, "trend": "stable", "volatility": 0.0}

        signal = self.detect_smart_money(odds_history)

        # Trend analysis (over full history)
        if len(odds_history) >= 3:
            changes = [(odds_history[i] - odds_history[i - 1]) / odds_history[i - 1]
                       for i in range(1, len(odds_history))]
            avg_change = np.mean(changes)
            if avg_change < -0.05:
                trend = "shortening"
            elif avg_change > 0.05:
                trend = "drifting"
            else:
                trend = "stable"
            volatility = float(np.std(changes))
        else:
            change = (odds_history[-1] - odds_history[0]) / odds_history[0]
            trend = "shortening" if change < -0.05 else "drifting" if change > 0.05 else "stable"
            volatility = abs(change)

        return {
            "signal": round(signal, 3),
            "trend": trend,
            "volatility": round(volatility, 4),
            "latest_odds": odds_history[-1],
            "first_odds": odds_history[0],
            "total_change_pct": round(
                (odds_history[-1] - odds_history[0]) / odds_history[0] * 100, 1
            ),
        }

    def calibrate_weights(self, historical_data: Dict[str, Any]) -> Tuple[float, float]:
        """
        Find optimal model_weight/market_weight from historical data.
        Uses logistic regression: P(win) = sigmoid(w1*model_prob + w2*market_prob)

        Args:
            historical_data: dict with keys 'model_prob', 'market_prob', 'won'
                             (each a list or numpy array)

        Returns:
            (model_weight, market_weight) tuple
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            logger.error("sklearn not available for calibration")
            return self.model_weight, self.market_weight

        model_prob = np.array(historical_data["model_prob"])
        market_prob = np.array(historical_data["market_prob"])
        won = np.array(historical_data["won"])

        # Filter valid entries
        mask = (model_prob > 0) & (market_prob > 0) & np.isfinite(model_prob) & np.isfinite(market_prob)
        if mask.sum() < 100:
            logger.warning(f"Too few valid samples for calibration: {mask.sum()}")
            return self.model_weight, self.market_weight

        X = np.column_stack([model_prob[mask], market_prob[mask]])
        y = won[mask].astype(int)

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y)

        # The coefficients give us relative weights
        total = abs(lr.coef_[0][0]) + abs(lr.coef_[0][1])
        if total > 0:
            self.model_weight = abs(lr.coef_[0][0]) / total
            self.market_weight = abs(lr.coef_[0][1]) / total
        else:
            self.model_weight = 0.3
            self.market_weight = 0.7

        logger.info(f"Calibrated weights: model={self.model_weight:.3f}, "
                    f"market={self.market_weight:.3f}")
        logger.info(f"LR coefficients: {lr.coef_[0]}, intercept: {lr.intercept_[0]:.3f}")

        # Save calibration
        self._save_calibration()

        return self.model_weight, self.market_weight

    def _save_calibration(self):
        """Save calibrated weights to file."""
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        cal = {
            "model_weight": round(self.model_weight, 6),
            "market_weight": round(self.market_weight, 6),
            "calibrated_at": datetime.now().isoformat(),
        }
        with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
            json.dump(cal, f, ensure_ascii=False, indent=2)
        logger.info(f"Calibration saved to {CALIBRATION_FILE}")


# ---------------------------------------------------------------------------
# Odds history storage
# ---------------------------------------------------------------------------

def store_odds_snapshot(race_id: str, odds_dict: Dict[int, float],
                        date_str: Optional[str] = None):
    """
    Store an odds snapshot for a race, appending to the history file.

    Args:
        race_id: race identifier
        odds_dict: {horse_number: odds}
        date_str: YYYYMMDD date string (defaults to today)
    """
    if date_str is None:
        date_str = date.today().strftime("%Y%m%d")

    ODDS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    hist_file = ODDS_HISTORY_DIR / f"{date_str}_{race_id}.json"

    history = {"race_id": race_id, "date": date_str, "snapshots": []}
    if hist_file.exists():
        try:
            with open(hist_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "odds": {str(k): v for k, v in odds_dict.items()},
    }
    history.setdefault("snapshots", []).append(snapshot)

    with open(hist_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    logger.debug(f"Stored odds snapshot for {race_id}: {len(odds_dict)} horses")


def load_odds_history(race_id: str, date_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Load odds history for a race.

    Returns:
        Dict with 'race_id', 'date', 'snapshots' list
    """
    if date_str is None:
        date_str = date.today().strftime("%Y%m%d")

    hist_file = ODDS_HISTORY_DIR / f"{date_str}_{race_id}.json"
    if not hist_file.exists():
        return {"race_id": race_id, "date": date_str, "snapshots": []}

    with open(hist_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_previous_odds(race_id: str, date_str: Optional[str] = None) -> Optional[Dict[int, float]]:
    """
    Get the most recent odds snapshot before the current one.

    Returns:
        {horse_number: odds} or None if no history
    """
    history = load_odds_history(race_id, date_str)
    snapshots = history.get("snapshots", [])
    if len(snapshots) < 2:
        return None

    # Return second-to-last snapshot
    prev = snapshots[-2]["odds"]
    return {int(k): float(v) for k, v in prev.items()}


# ---------------------------------------------------------------------------
# Integration with prediction cache
# ---------------------------------------------------------------------------

def update_predictions_with_bayesian(race_id: str, date_str: Optional[str] = None):
    """
    Apply Bayesian update to cached predictions for a race.
    Uses stored odds history for movement detection.

    Args:
        race_id: race identifier
        date_str: YYYYMMDD date string
    """
    if date_str is None:
        date_str = date.today().strftime("%Y%m%d")

    updater = BayesianOddsUpdater()

    # Load current odds from latest snapshot
    history = load_odds_history(race_id, date_str)
    snapshots = history.get("snapshots", [])
    if not snapshots:
        logger.debug(f"No odds history for {race_id}, skipping Bayesian update")
        return

    current_odds_raw = snapshots[-1]["odds"]
    current_odds = {int(k): float(v) for k, v in current_odds_raw.items()}

    prev_odds = get_previous_odds(race_id, date_str)

    # Find prediction cache file
    cache_files = list(PREDICTION_CACHE_DIR.glob(f"*_{race_id}.json"))
    if not cache_files:
        logger.debug(f"No prediction cache for {race_id}")
        return

    for cache_file in cache_files:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)

            predictions = cached.get("predictions", cached.get("horses", []))
            if not predictions:
                continue

            # Apply Bayesian update
            updated = updater.update_race(predictions, current_odds, prev_odds)

            # Store updated predictions back
            if "predictions" in cached:
                cached["predictions"] = updated
            elif "horses" in cached:
                cached["horses"] = updated

            cached["bayesian_updated_at"] = datetime.now().isoformat()
            cached["bayesian_weights"] = {
                "model": updater.model_weight,
                "market": updater.market_weight,
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached, f, ensure_ascii=False, indent=2)

            logger.info(f"Bayesian update applied: {cache_file.name}")

        except Exception as e:
            logger.error(f"Bayesian update error for {cache_file}: {e}")


# ---------------------------------------------------------------------------
# Calibration from historical data
# ---------------------------------------------------------------------------

def calibrate_from_historical(train_years: List[int] = None):
    """
    Calibrate Bayesian weights from historical race data (where we have
    both model predictions/odds and finish results).

    Args:
        train_years: years to use for calibration (default: 2020-2024)
    """
    import pandas as pd

    if train_years is None:
        train_years = list(range(2020, 2025))

    logger.info(f"Calibrating Bayesian weights from years {train_years}")

    frames = []
    for y in train_years:
        path = DATA_DIR / f"races_{y}.csv"
        if path.exists():
            df = pd.read_csv(path, dtype={"race_id": str, "horse_id": str})
            frames.append(df)
            logger.info(f"  Loaded {path.name}: {len(df):,} rows")

    if not frames:
        logger.error("No historical data found for calibration")
        return

    data = pd.concat(frames, ignore_index=True)

    # Filter valid records
    data = data[data["finish_position"].notna()]
    data = data[data["finish_position"].apply(
        lambda x: str(x).replace(".", "").isdigit()
    )]
    data["finish_position"] = data["finish_position"].astype(int)
    data = data[data["odds"].notna() & (data["odds"] > 0)]

    # Market probability
    data["market_prob"] = 1.0 / data["odds"]

    # Simulate model probability using odds-based features
    # (This is a proxy -- in production, use actual model predictions)
    # We compute a simple probability from odds rank within race
    race_groups = data.groupby("race_id")
    data["odds_rank"] = race_groups["odds"].rank(method="min")
    data["n_runners"] = race_groups["horse_number"].transform("count")

    # Simple model proxy: inverse rank probability (normalized)
    data["model_prob_raw"] = 1.0 / data["odds_rank"]
    data["model_prob"] = data.groupby("race_id")["model_prob_raw"].transform(
        lambda x: x / x.sum()
    )

    data["won"] = (data["finish_position"] == 1).astype(int)

    # Calibrate
    updater = BayesianOddsUpdater()
    historical = {
        "model_prob": data["model_prob"].values,
        "market_prob": data["market_prob"].values,
        "won": data["won"].values,
    }

    model_w, market_w = updater.calibrate_weights(historical)
    logger.info(f"Calibration complete: model_weight={model_w:.4f}, market_weight={market_w:.4f}")

    return model_w, market_w


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Bayesian Odds Updater")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate weights from historical data")
    parser.add_argument("--update", type=str, default=None, help="Update predictions for race_id")
    parser.add_argument("--date", type=str, default=None, help="Date YYYYMMDD")
    args = parser.parse_args()

    if args.calibrate:
        calibrate_from_historical()
    elif args.update:
        update_predictions_with_bayesian(args.update, args.date)
    else:
        print("Usage: python bayesian_updater.py --calibrate")
        print("       python bayesian_updater.py --update RACE_ID [--date YYYYMMDD]")
