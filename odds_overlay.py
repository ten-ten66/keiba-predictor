"""
Odds Overlay Model
===================
Finds VALUE bets by combining a base prediction model with live odds.

The key insight: to be profitable, we need model_prob > 1/odds (EV > 1.0).
A base model that ignores odds just finds favorites, not mispriced horses.

This overlay model:
1. Trains a base model on historical features (excluding odds) to get model_prob
2. Trains an overlay model that combines model_prob + odds features
3. Uses isotonic regression for calibration so probabilities are accurate
4. Identifies value bets where calibrated_prob * odds > threshold

Usage:
    python odds_overlay.py              # Train + backtest
    python odds_overlay.py --threshold 1.15  # Custom EV threshold
"""

import os
import sys
import json
import warnings
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_race_data(years: list) -> pd.DataFrame:
    """Load race CSVs for given years."""
    frames = []
    for y in years:
        path = os.path.join(DATA_DIR, f"races_{y}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, dtype={"race_id": str, "horse_id": str,
                                           "jockey_id": str, "trainer_id": str})
            frames.append(df)
    if not frames:
        # Fallback: try _all_races.csv
        all_path = os.path.join(DATA_DIR, "_all_races.csv")
        if os.path.exists(all_path):
            df = pd.read_csv(all_path, dtype={"race_id": str, "horse_id": str,
                                               "jockey_id": str, "trainer_id": str})
            df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
            df["year"] = df["race_date"].dt.year
            df = df[df["year"].isin(years)]
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No race data found for years {years}")
    combined = pd.concat(frames, ignore_index=True)
    combined["race_date"] = pd.to_datetime(combined["race_date"], errors="coerce")
    return combined


def clean_finish_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean finish positions."""
    df = df[df["finish_position"].notna()].copy()
    df = df[df["finish_position"].apply(
        lambda x: str(x).replace(".", "").isdigit()
    )]
    df["finish_position"] = df["finish_position"].astype(int)
    # Filter valid odds
    df = df[df["odds"].notna() & (df["odds"] > 0)].copy()
    df = df[df["popularity"].notna()].copy()
    df["popularity"] = df["popularity"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def create_base_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Create features for the BASE model (no odds/popularity -- leak-free for overlay)."""
    df = df.copy()

    for col, mapping in [
        ("course_type", {"芝": 0, "ダート": 1, "障害": 2}),
        ("weather", {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "小雪": 4, "雪": 5}),
        ("track_condition", {"良": 0, "稍重": 1, "重": 2, "不良": 3}),
        ("sex", {"牡": 0, "牝": 1, "セ": 2}),
    ]:
        if col in df.columns:
            df[col + "_enc"] = df[col].map(mapping).fillna(0).astype(int)

    df["horse_weight"] = df["horse_weight"].fillna(470)
    df["weight_diff"] = df["weight_diff"].fillna(0)
    df["distance_norm"] = df["distance"].fillna(1600) / 1000.0
    df["age"] = df["age"].fillna(3)
    df["frame_number"] = df["frame_number"].fillna(4)

    race_groups = df.groupby("race_id")
    df["n_runners"] = race_groups["horse_number"].transform("count")
    df["horse_number_norm"] = df["horse_number"] / df["n_runners"]

    # Weight (impost)
    df["weight"] = df["weight"].fillna(55)

    base_feature_cols = [
        "horse_weight", "weight_diff", "weight", "distance_norm",
        "age", "frame_number", "horse_number", "horse_number_norm",
        "distance", "n_runners",
    ]
    for c in ["course_type_enc", "weather_enc", "track_condition_enc", "sex_enc"]:
        if c in df.columns:
            base_feature_cols.append(c)

    base_feature_cols = [c for c in base_feature_cols if c in df.columns]

    for c in base_feature_cols:
        df[c] = df[c].fillna(0)
        df[c] = df[c].replace([np.inf, -np.inf], 0)

    return df, base_feature_cols


def create_overlay_features(df: pd.DataFrame, base_probs: np.ndarray) -> Tuple[pd.DataFrame, list]:
    """
    Create features for the OVERLAY model.
    Combines base model predictions with odds/market info.
    """
    df = df.copy()
    df["model_prob_win"] = base_probs

    # Market features
    df["odds"] = df["odds"].clip(lower=1.0)
    df["implied_prob"] = 1.0 / df["odds"]
    df["log_odds"] = np.log1p(df["odds"])
    df["inv_popularity"] = 1.0 / df["popularity"].clip(lower=1)

    # Edge = model thinks higher than market
    df["edge"] = df["model_prob_win"] - df["implied_prob"]
    df["edge_ratio"] = df["model_prob_win"] / df["implied_prob"].clip(lower=0.001)

    # Race-level ranks
    race_groups = df.groupby("race_id")
    df["odds_rank_in_race"] = race_groups["odds"].rank(method="min")
    df["model_rank_in_race"] = race_groups["model_prob_win"].rank(ascending=False, method="min")
    df["n_runners"] = race_groups["horse_number"].transform("count")
    df["odds_rank_pct"] = df["odds_rank_in_race"] / df["n_runners"]
    df["model_rank_pct"] = df["model_rank_in_race"] / df["n_runners"]

    # Rank disagreement: negative = model rates horse higher than market
    df["rank_diff"] = df["model_rank_in_race"] - df["odds_rank_in_race"]

    # Is favorite
    df["is_favorite"] = (df["popularity"] == 1).astype(int)

    # Odds spread in race (market uncertainty)
    df["odds_std_in_race"] = race_groups["odds"].transform("std").fillna(0)
    df["odds_mean_in_race"] = race_groups["odds"].transform("mean").fillna(10)
    df["odds_relative"] = df["odds"] / df["odds_mean_in_race"].clip(lower=1)

    overlay_feature_cols = [
        "model_prob_win",
        "odds", "log_odds", "implied_prob",
        "popularity", "inv_popularity",
        "edge", "edge_ratio",
        "odds_rank_in_race", "model_rank_in_race",
        "odds_rank_pct", "model_rank_pct",
        "rank_diff",
        "n_runners",
        "is_favorite",
        "odds_std_in_race", "odds_mean_in_race", "odds_relative",
    ]

    for c in overlay_feature_cols:
        df[c] = df[c].fillna(0)
        df[c] = df[c].replace([np.inf, -np.inf], 0)

    return df, overlay_feature_cols


# ---------------------------------------------------------------------------
# Odds Overlay Model
# ---------------------------------------------------------------------------
class OddsOverlayModel:
    """
    Two-stage model:
    1. Base model: predicts P(win) from horse/race features (no odds)
    2. Overlay model: predicts calibrated P(win) from base_prob + odds features

    Value bet = calibrated_prob * odds > threshold
    """

    def __init__(self):
        self.base_model = None         # LightGBM for base P(win)
        self.overlay_model = None      # LightGBM for calibrated P(win | base_prob, odds)
        self.calibrator = None         # Isotonic regression for calibration
        self.base_feature_cols = None
        self.overlay_feature_cols = None
        self.model_type = None
        self.training_stats = {}

    def train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """
        Train the two-stage overlay model.

        Args:
            train_df: Training data with odds, popularity, finish_position
            val_df: Optional validation data
        """
        print("\n--- Stage 1: Training base model (no odds features) ---")
        train_df = clean_finish_positions(train_df)
        train_feat, self.base_feature_cols = create_base_features(train_df)

        y_win = (train_feat["finish_position"] == 1).astype(int).values
        X_base = train_feat[self.base_feature_cols].values

        try:
            import lightgbm as lgb
            self.model_type = "lgb"

            dtrain = lgb.Dataset(X_base, label=y_win)
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 5,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.7,
                "bagging_freq": 5,
                "verbose": -1,
                "seed": 42,
                "min_data_in_leaf": 50,
            }
            self.base_model = lgb.train(params, dtrain, num_boost_round=200)
            base_probs = self.base_model.predict(X_base)
        except ImportError:
            print("  LightGBM not available, using sklearn GBM")
            self.model_type = "sklearn"
            from sklearn.ensemble import GradientBoostingClassifier
            self.base_model = GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                subsample=0.7, random_state=42
            )
            self.base_model.fit(X_base, y_win)
            base_probs = self.base_model.predict_proba(X_base)[:, 1]

        base_auc = self._calc_auc(y_win, base_probs)
        print(f"  Base model AUC: {base_auc:.4f}")
        print(f"  Base model mean P(win): {base_probs.mean():.4f}")

        # Stage 2: Overlay model
        print("\n--- Stage 2: Training overlay model (base_prob + odds) ---")
        train_overlay, self.overlay_feature_cols = create_overlay_features(train_feat, base_probs)

        X_overlay = train_overlay[self.overlay_feature_cols].values
        y_overlay = y_win  # Same target

        if self.model_type == "lgb":
            import lightgbm as lgb
            dtrain_ov = lgb.Dataset(X_overlay, label=y_overlay)
            params_ov = {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.03,
                "num_leaves": 31,
                "max_depth": 5,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "seed": 42,
                "min_data_in_leaf": 100,
                "lambda_l1": 1.0,
                "lambda_l2": 1.0,
            }
            self.overlay_model = lgb.train(params_ov, dtrain_ov, num_boost_round=300)
            overlay_probs = self.overlay_model.predict(X_overlay)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.overlay_model = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.03,
                subsample=0.8, random_state=42
            )
            self.overlay_model.fit(X_overlay, y_overlay)
            overlay_probs = self.overlay_model.predict_proba(X_overlay)[:, 1]

        overlay_auc = self._calc_auc(y_overlay, overlay_probs)
        print(f"  Overlay model AUC: {overlay_auc:.4f}")

        # Stage 3: Calibration via isotonic regression
        print("\n--- Stage 3: Calibrating probabilities ---")
        from sklearn.isotonic import IsotonicRegression
        self.calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        self.calibrator.fit(overlay_probs, y_overlay)
        calibrated = self.calibrator.predict(overlay_probs)
        cal_auc = self._calc_auc(y_overlay, calibrated)
        print(f"  Calibrated AUC: {cal_auc:.4f}")
        print(f"  Calibrated mean P(win): {calibrated.mean():.4f}")
        print(f"  Actual win rate: {y_overlay.mean():.4f}")

        # Calibration quality check
        self._check_calibration(calibrated, y_overlay, train_overlay["odds"].values)

        self.training_stats = {
            "base_auc": round(base_auc, 4),
            "overlay_auc": round(overlay_auc, 4),
            "calibrated_auc": round(cal_auc, 4),
            "train_size": len(train_df),
            "actual_win_rate": round(float(y_overlay.mean()), 4),
            "calibrated_mean": round(float(calibrated.mean()), 4),
            "trained_at": datetime.now().isoformat(),
        }

        # Validate if val_df provided
        if val_df is not None:
            print("\n--- Validation ---")
            self._validate(val_df)

    def predict_value(self, df: pd.DataFrame,
                      odds_dict: Optional[Dict[int, float]] = None) -> List[Dict]:
        """
        Find value bets by combining model predictions with odds.

        Args:
            df: DataFrame with horse/race data (must include odds, popularity columns)
            odds_dict: Optional dict of {horse_number: live_odds} to override stored odds

        Returns:
            List of dicts sorted by EV, each containing value analysis
        """
        if self.base_model is None or self.overlay_model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        df = df.copy()

        # Override odds if live odds provided
        if odds_dict:
            df["odds"] = df["horse_number"].map(
                lambda hn: odds_dict.get(int(hn), df.loc[df["horse_number"] == hn, "odds"].iloc[0]
                                         if not df.loc[df["horse_number"] == hn, "odds"].empty else 0)
            )

        # Clean
        df["odds"] = pd.to_numeric(df["odds"], errors="coerce").fillna(0)
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(10).astype(int)
        df = df[df["odds"] > 0].copy()

        if df.empty:
            return []

        # Stage 1: Base predictions
        feat_df, _ = create_base_features(df)
        X_base = feat_df[self.base_feature_cols].values

        if self.model_type == "lgb":
            base_probs = self.base_model.predict(X_base)
        else:
            base_probs = self.base_model.predict_proba(X_base)[:, 1]

        # Stage 2: Overlay predictions
        overlay_df, _ = create_overlay_features(feat_df, base_probs)
        X_overlay = overlay_df[self.overlay_feature_cols].values

        if self.model_type == "lgb":
            overlay_probs = self.overlay_model.predict(X_overlay)
        else:
            overlay_probs = self.overlay_model.predict_proba(X_overlay)[:, 1]

        # Stage 3: Calibrate
        calibrated_probs = self.calibrator.predict(overlay_probs)

        # Build results
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            odds = float(row["odds"])
            if odds <= 1:
                continue

            implied_prob = 1.0 / odds
            model_prob = float(base_probs[i])
            cal_prob = float(calibrated_probs[i])
            edge = cal_prob - implied_prob
            ev = cal_prob * odds

            # Kelly fraction: (bp - q) / b where b = odds-1, p = cal_prob, q = 1-cal_prob
            b = odds - 1
            kelly = max(0, (b * cal_prob - (1 - cal_prob)) / b) if b > 0 else 0

            # Classification
            if ev > 1.3:
                bet_type = "STRONG_VALUE"
            elif ev > 1.15:
                bet_type = "VALUE"
            elif ev > 1.0:
                bet_type = "MARGINAL"
            elif implied_prob > cal_prob * 1.5:
                bet_type = "TRAP"  # Market overrates this horse
            else:
                bet_type = "PASS"

            results.append({
                "horse_number": int(row.get("horse_number", 0)),
                "horse_name": str(row.get("horse_name", "")),
                "odds": round(odds, 1),
                "popularity": int(row.get("popularity", 0)),
                "model_prob": round(model_prob, 4),
                "market_prob": round(implied_prob, 4),
                "calibrated_prob": round(cal_prob, 4),
                "edge": round(edge, 4),
                "ev": round(ev, 3),
                "kelly_fraction": round(kelly, 4),
                "bet_type": bet_type,
                "is_value_bet": ev > 1.15,
            })

        return sorted(results, key=lambda x: x["ev"], reverse=True)

    def predict_from_base_predictions(self, base_predictions: List[Dict],
                                       odds_dict: Dict[int, float]) -> List[Dict]:
        """
        Alternative entry point: takes pre-computed base model predictions and live odds.
        Used when the GOD model has already run and we just need the overlay.

        Args:
            base_predictions: List of dicts with keys: horse_number, horse_name, prob_win,
                            prob_top3, composite_score
            odds_dict: {horse_number: odds}

        Returns:
            List of value analysis dicts sorted by EV
        """
        if self.overlay_model is None:
            raise RuntimeError("Overlay model not trained")

        results = []
        # Build a mini-dataframe for overlay features
        records = []
        for horse in base_predictions:
            hn = int(horse["horse_number"])
            odds = odds_dict.get(hn, 0)
            if odds <= 0:
                continue
            records.append({
                "horse_number": hn,
                "horse_name": horse.get("horse_name", ""),
                "model_prob_win": horse.get("prob_win", 0),
                "prob_top3": horse.get("prob_top3", 0),
                "composite_score": horse.get("composite_score", 0),
                "odds": odds,
                "popularity": horse.get("popularity", 0),
            })

        if not records:
            return []

        tmp_df = pd.DataFrame(records)
        n = len(tmp_df)

        # Compute overlay features manually
        tmp_df["odds"] = tmp_df["odds"].clip(lower=1.0)
        tmp_df["implied_prob"] = 1.0 / tmp_df["odds"]
        tmp_df["log_odds"] = np.log1p(tmp_df["odds"])
        tmp_df["inv_popularity"] = 1.0 / tmp_df["popularity"].clip(lower=1)
        tmp_df["edge"] = tmp_df["model_prob_win"] - tmp_df["implied_prob"]
        tmp_df["edge_ratio"] = tmp_df["model_prob_win"] / tmp_df["implied_prob"].clip(lower=0.001)
        tmp_df["odds_rank_in_race"] = tmp_df["odds"].rank(method="min")
        tmp_df["model_rank_in_race"] = tmp_df["model_prob_win"].rank(ascending=False, method="min")
        tmp_df["n_runners"] = n
        tmp_df["odds_rank_pct"] = tmp_df["odds_rank_in_race"] / n
        tmp_df["model_rank_pct"] = tmp_df["model_rank_in_race"] / n
        tmp_df["rank_diff"] = tmp_df["model_rank_in_race"] - tmp_df["odds_rank_in_race"]
        tmp_df["is_favorite"] = (tmp_df["popularity"] == 1).astype(int)
        tmp_df["odds_std_in_race"] = tmp_df["odds"].std()
        tmp_df["odds_mean_in_race"] = tmp_df["odds"].mean()
        tmp_df["odds_relative"] = tmp_df["odds"] / tmp_df["odds_mean_in_race"].clip(lower=1)

        X_overlay = tmp_df[self.overlay_feature_cols].values

        if self.model_type == "lgb":
            overlay_probs = self.overlay_model.predict(X_overlay)
        else:
            overlay_probs = self.overlay_model.predict_proba(X_overlay)[:, 1]

        calibrated_probs = self.calibrator.predict(overlay_probs)

        for i, (_, row) in enumerate(tmp_df.iterrows()):
            odds = float(row["odds"])
            implied_prob = 1.0 / odds
            model_prob = float(row["model_prob_win"])
            cal_prob = float(calibrated_probs[i])
            edge = cal_prob - implied_prob
            ev = cal_prob * odds

            b = odds - 1
            kelly = max(0, (b * cal_prob - (1 - cal_prob)) / b) if b > 0 else 0

            if ev > 1.3:
                bet_type = "STRONG_VALUE"
            elif ev > 1.15:
                bet_type = "VALUE"
            elif ev > 1.0:
                bet_type = "MARGINAL"
            elif implied_prob > cal_prob * 1.5:
                bet_type = "TRAP"
            else:
                bet_type = "PASS"

            results.append({
                "horse_number": int(row["horse_number"]),
                "horse_name": str(row["horse_name"]),
                "odds": round(odds, 1),
                "popularity": int(row["popularity"]),
                "model_prob": round(model_prob, 4),
                "market_prob": round(implied_prob, 4),
                "calibrated_prob": round(cal_prob, 4),
                "edge": round(edge, 4),
                "ev": round(ev, 3),
                "kelly_fraction": round(kelly, 4),
                "bet_type": bet_type,
                "is_value_bet": ev > 1.15,
            })

        return sorted(results, key=lambda x: x["ev"], reverse=True)

    def save(self, path: Optional[str] = None):
        """Save trained model to disk."""
        if path is None:
            path = os.path.join(MODEL_DIR, "odds_overlay.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "base_model": self.base_model,
                "overlay_model": self.overlay_model,
                "calibrator": self.calibrator,
                "base_feature_cols": self.base_feature_cols,
                "overlay_feature_cols": self.overlay_feature_cols,
                "model_type": self.model_type,
                "training_stats": self.training_stats,
            }, f)
        print(f"  Model saved to {path}")

    @classmethod
    def load(cls, path: Optional[str] = None) -> "OddsOverlayModel":
        """Load trained model from disk."""
        if path is None:
            path = os.path.join(MODEL_DIR, "odds_overlay.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.base_model = data["base_model"]
        obj.overlay_model = data["overlay_model"]
        obj.calibrator = data["calibrator"]
        obj.base_feature_cols = data["base_feature_cols"]
        obj.overlay_feature_cols = data["overlay_feature_cols"]
        obj.model_type = data["model_type"]
        obj.training_stats = data.get("training_stats", {})
        return obj

    def _validate(self, val_df: pd.DataFrame):
        """Run validation and print metrics."""
        val_df = clean_finish_positions(val_df)
        if val_df.empty:
            print("  No valid validation data")
            return

        val_feat, _ = create_base_features(val_df)
        X_base = val_feat[self.base_feature_cols].values
        y_val = (val_feat["finish_position"] == 1).astype(int).values

        if self.model_type == "lgb":
            base_probs = self.base_model.predict(X_base)
        else:
            base_probs = self.base_model.predict_proba(X_base)[:, 1]

        val_overlay, _ = create_overlay_features(val_feat, base_probs)
        X_overlay = val_overlay[self.overlay_feature_cols].values

        if self.model_type == "lgb":
            overlay_probs = self.overlay_model.predict(X_overlay)
        else:
            overlay_probs = self.overlay_model.predict_proba(X_overlay)[:, 1]

        calibrated = self.calibrator.predict(overlay_probs)

        base_auc = self._calc_auc(y_val, base_probs)
        overlay_auc = self._calc_auc(y_val, overlay_probs)
        cal_auc = self._calc_auc(y_val, calibrated)

        print(f"  Validation base AUC:       {base_auc:.4f}")
        print(f"  Validation overlay AUC:    {overlay_auc:.4f}")
        print(f"  Validation calibrated AUC: {cal_auc:.4f}")

        self._check_calibration(calibrated, y_val, val_overlay["odds"].values)

        self.training_stats["val_base_auc"] = round(base_auc, 4)
        self.training_stats["val_overlay_auc"] = round(overlay_auc, 4)
        self.training_stats["val_calibrated_auc"] = round(cal_auc, 4)

    def _check_calibration(self, probs: np.ndarray, actuals: np.ndarray, odds: np.ndarray):
        """Check calibration quality by probability bins."""
        df = pd.DataFrame({
            "prob": probs,
            "actual": actuals,
            "odds": odds,
        })
        bins = [0, 0.02, 0.05, 0.08, 0.12, 0.2, 0.35, 1.0]
        labels = ["<2%", "2-5%", "5-8%", "8-12%", "12-20%", "20-35%", "35%+"]
        df["bin"] = pd.cut(df["prob"], bins=bins, labels=labels)

        print(f"\n  {'Prob Bin':<10s} {'Count':>7s} {'Pred%':>7s} {'Actual%':>8s} {'AvgOdds':>8s} {'ROI%':>7s}")
        print("  " + "-" * 50)
        for label, group in df.groupby("bin", observed=True):
            if len(group) < 10:
                continue
            pred_mean = group["prob"].mean() * 100
            actual_mean = group["actual"].mean() * 100
            avg_odds = group["odds"].mean()
            # ROI if we bet on all in this bin
            roi = (group["actual"] * group["odds"]).mean() * 100 - 100
            print(f"  {str(label):<10s} {len(group):>7d} {pred_mean:>6.2f}% {actual_mean:>7.2f}% "
                  f"{avg_odds:>7.1f} {roi:>+6.1f}%")

    @staticmethod
    def _calc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calculate AUC-ROC."""
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_score)
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------
def backtest_overlay(model: OddsOverlayModel, test_df: pd.DataFrame,
                     thresholds: List[float] = None,
                     bet_amount: int = 100) -> Dict:
    """
    Backtest the overlay model on test data.
    Tries multiple EV thresholds to find optimal.
    """
    if thresholds is None:
        thresholds = [1.0, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50, 2.0]

    test_df = clean_finish_positions(test_df)
    if test_df.empty:
        return {"error": "No valid test data"}

    # Get predictions
    value_results_by_race = {}
    for race_id, group in test_df.groupby("race_id"):
        try:
            values = model.predict_value(group)
            for v in values:
                v["actual_win"] = int(
                    group[group["horse_number"] == v["horse_number"]]["finish_position"].iloc[0]
                ) == 1
            value_results_by_race[race_id] = values
        except Exception as e:
            continue

    all_predictions = []
    for race_id, values in value_results_by_race.items():
        for v in values:
            v["race_id"] = race_id
            all_predictions.append(v)

    if not all_predictions:
        return {"error": "No predictions generated"}

    pred_df = pd.DataFrame(all_predictions)
    total_races = len(value_results_by_race)
    total_horses = len(pred_df)

    print(f"\n  Total races: {total_races}")
    print(f"  Total horses evaluated: {total_horses}")

    # Strategy comparison
    results = {}

    # Baseline: always bet favorite
    fav_df = pred_df[pred_df["popularity"] == 1].copy()
    if not fav_df.empty:
        fav_bets = len(fav_df)
        fav_hits = fav_df["actual_win"].sum()
        fav_returns = (fav_df["actual_win"].astype(float) * fav_df["odds"]).sum() * bet_amount
        fav_cost = fav_bets * bet_amount
        fav_roi = (fav_returns / fav_cost - 1) * 100 if fav_cost > 0 else 0
        results["favorite_baseline"] = {
            "bets": fav_bets,
            "hits": int(fav_hits),
            "hit_rate": round(fav_hits / fav_bets * 100, 1) if fav_bets > 0 else 0,
            "cost": fav_cost,
            "returns": round(fav_returns, 0),
            "profit": round(fav_returns - fav_cost, 0),
            "roi_pct": round(fav_roi, 1),
            "avg_odds_bet": round(fav_df["odds"].mean(), 1),
        }

    # Baseline: bet on model top-1 per race
    top1_mask = pred_df.groupby("race_id")["calibrated_prob"].transform("max") == pred_df["calibrated_prob"]
    top1_df = pred_df[top1_mask].drop_duplicates(subset=["race_id"], keep="first")
    if not top1_df.empty:
        t1_bets = len(top1_df)
        t1_hits = top1_df["actual_win"].sum()
        t1_returns = (top1_df["actual_win"].astype(float) * top1_df["odds"]).sum() * bet_amount
        t1_cost = t1_bets * bet_amount
        t1_roi = (t1_returns / t1_cost - 1) * 100 if t1_cost > 0 else 0
        results["model_top1"] = {
            "bets": t1_bets,
            "hits": int(t1_hits),
            "hit_rate": round(t1_hits / t1_bets * 100, 1) if t1_bets > 0 else 0,
            "cost": t1_cost,
            "returns": round(t1_returns, 0),
            "profit": round(t1_returns - t1_cost, 0),
            "roi_pct": round(t1_roi, 1),
            "avg_odds_bet": round(top1_df["odds"].mean(), 1),
        }

    # EV-threshold strategies
    for threshold in thresholds:
        ev_df = pred_df[pred_df["ev"] >= threshold].copy()
        if ev_df.empty:
            results[f"ev_{threshold}"] = {
                "bets": 0, "hits": 0, "hit_rate": 0,
                "cost": 0, "returns": 0, "profit": 0, "roi_pct": 0,
                "avg_odds_bet": 0,
            }
            continue

        n_bets = len(ev_df)
        n_hits = ev_df["actual_win"].sum()
        total_returns = (ev_df["actual_win"].astype(float) * ev_df["odds"]).sum() * bet_amount
        total_cost = n_bets * bet_amount
        roi = (total_returns / total_cost - 1) * 100 if total_cost > 0 else 0

        results[f"ev_{threshold}"] = {
            "bets": n_bets,
            "hits": int(n_hits),
            "hit_rate": round(n_hits / n_bets * 100, 1) if n_bets > 0 else 0,
            "cost": total_cost,
            "returns": round(total_returns, 0),
            "profit": round(total_returns - total_cost, 0),
            "roi_pct": round(roi, 1),
            "avg_odds_bet": round(ev_df["odds"].mean(), 1),
        }

    # Kelly-weighted strategy (bet proportional to kelly fraction)
    kelly_df = pred_df[pred_df["kelly_fraction"] > 0.01].copy()
    if not kelly_df.empty:
        bankroll = 100000  # 100k yen simulated bankroll
        kelly_bankroll = bankroll
        kelly_profit = 0
        kelly_bets = 0
        kelly_hits = 0
        for _, row in kelly_df.iterrows():
            # Quarter-kelly for safety
            bet_size = kelly_bankroll * row["kelly_fraction"] * 0.25
            bet_size = min(bet_size, kelly_bankroll * 0.05)  # Cap at 5% of bankroll
            bet_size = max(bet_size, 100)  # Minimum bet
            if bet_size > kelly_bankroll:
                continue
            kelly_bets += 1
            if row["actual_win"]:
                kelly_hits += 1
                kelly_bankroll += bet_size * (row["odds"] - 1)
                kelly_profit += bet_size * (row["odds"] - 1)
            else:
                kelly_bankroll -= bet_size
                kelly_profit -= bet_size

        results["kelly_quarter"] = {
            "bets": kelly_bets,
            "hits": kelly_hits,
            "hit_rate": round(kelly_hits / kelly_bets * 100, 1) if kelly_bets > 0 else 0,
            "initial_bankroll": bankroll,
            "final_bankroll": round(kelly_bankroll, 0),
            "profit": round(kelly_profit, 0),
            "roi_pct": round(kelly_profit / bankroll * 100, 1),
        }

    # Print results table
    print(f"\n  {'Strategy':<25s} {'Bets':>6s} {'Hits':>5s} {'Hit%':>6s} {'AvgOdds':>8s} {'Profit':>10s} {'ROI%':>8s}")
    print("  " + "-" * 70)
    for name, r in sorted(results.items(), key=lambda x: x[1].get("roi_pct", -999), reverse=True):
        print(f"  {name:<25s} {r['bets']:>6d} {r['hits']:>5d} "
              f"{r['hit_rate']:>5.1f}% {r.get('avg_odds_bet', 0):>7.1f} "
              f"{r['profit']:>+10.0f} {r['roi_pct']:>+7.1f}%")

    return {
        "total_races": total_races,
        "total_horses": total_horses,
        "strategies": results,
    }


# ---------------------------------------------------------------------------
# Main: Train + Backtest
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Odds Overlay Model")
    parser.add_argument("--threshold", type=float, default=1.15, help="EV threshold for value bets")
    parser.add_argument("--train-years", type=str, default="2020-2024",
                        help="Training years range (e.g., 2020-2024)")
    parser.add_argument("--test-year", type=int, default=2025, help="Test year")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtest")
    args = parser.parse_args()

    # Parse year range
    if "-" in args.train_years:
        start, end = args.train_years.split("-")
        train_years = list(range(int(start), int(end) + 1))
    else:
        train_years = [int(args.train_years)]

    print("=" * 70)
    print("ODDS OVERLAY MODEL")
    print(f"  Train: {train_years[0]}-{train_years[-1]}")
    print(f"  Test:  {args.test_year}")
    print(f"  EV Threshold: {args.threshold}")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    train_df = load_race_data(train_years)
    test_df = load_race_data([args.test_year])
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")

    # Train
    print("\n[2/4] Training overlay model...")
    model = OddsOverlayModel()
    model.train(train_df, val_df=test_df)

    # Save
    print("\n[3/4] Saving model...")
    model.save()

    # Save training stats
    stats_path = os.path.join(DATA_DIR, "odds_overlay_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(model.training_stats, f, ensure_ascii=False, indent=2)
    print(f"  Stats saved to {stats_path}")

    # Backtest
    if not args.skip_backtest:
        print("\n[4/4] Backtesting...")
        bt_results = backtest_overlay(
            model, test_df,
            thresholds=[1.0, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50, 2.0],
        )

        bt_results["train_years"] = f"{train_years[0]}-{train_years[-1]}"
        bt_results["test_year"] = args.test_year
        bt_results["training_stats"] = model.training_stats

        bt_path = os.path.join(DATA_DIR, "overlay_backtest_results.json")
        with open(bt_path, "w", encoding="utf-8") as f:
            json.dump(bt_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  Backtest results saved to {bt_path}")

        # Find optimal threshold
        best_strat = None
        best_roi = -999
        for name, r in bt_results.get("strategies", {}).items():
            if name.startswith("ev_") and r["bets"] >= 20:
                if r["roi_pct"] > best_roi:
                    best_roi = r["roi_pct"]
                    best_strat = name

        if best_strat:
            print(f"\n  Best EV strategy: {best_strat} (ROI: {best_roi:+.1f}%)")
        else:
            print("\n  No EV strategy had enough bets for reliable assessment")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
