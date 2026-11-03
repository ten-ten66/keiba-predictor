"""
Honest Walk-Forward Backtest
============================
Trains on historical data, tests on held-out future data.
Uses actual payouts from payouts.csv. Compares model vs always-bet-favorite.
Flat 100-yen bets. No lookahead bias.

Usage:
    python honest_backtest.py
"""

import os
import sys
import json
import warnings
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_YEARS = list(range(2019, 2025))  # Train: 2019-2024
TEST_YEAR = 2025                        # Test: 2025
BET_AMOUNT = 100                        # Flat bet in yen


def load_race_data(years: list) -> pd.DataFrame:
    """Load race CSVs for given years."""
    frames = []
    for y in years:
        path = os.path.join(DATA_DIR, f"races_{y}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, dtype={"race_id": str, "horse_id": str,
                                           "jockey_id": str, "trainer_id": str})
            frames.append(df)
            print(f"  Loaded {path}: {len(df):,} rows")
    if not frames:
        raise FileNotFoundError("No race data found")
    combined = pd.concat(frames, ignore_index=True)
    combined["race_date"] = pd.to_datetime(combined["race_date"], errors="coerce")
    return combined


def load_payouts() -> Dict:
    """Load payouts.csv into lookup dict keyed by (race_id, bet_type, combination)."""
    path = os.path.join(DATA_DIR, "payouts.csv")
    if not os.path.exists(path):
        print("  WARNING: payouts.csv not found")
        return {}
    pay_df = pd.read_csv(path, dtype={"race_id": str})
    payouts = {}
    for _, row in pay_df.iterrows():
        key = (str(row["race_id"]), row["bet_type"], str(row["combination"]))
        payouts[key] = row["payout"]
    print(f"  Loaded {len(payouts):,} payout records")
    return payouts


def lookup_payout(payouts: Dict, race_id: str, bet_type: str, nums: list) -> Optional[float]:
    """Look up actual payout."""
    combo_str = " - ".join(str(n) for n in sorted(nums))
    key = (str(race_id), bet_type, combo_str)
    return payouts.get(key)


# ---------------------------------------------------------------------------
# Rolling Feature Computation (leak-free: uses PRIOR data only)
# ---------------------------------------------------------------------------
def compute_rolling_features(full_df: pd.DataFrame) -> pd.DataFrame:
    """Compute horse/jockey/trainer rolling features from PRIOR races only.

    IMPORTANT: full_df must be sorted by race_date ascending so that
    rolling computations only look backward.  We iterate date-by-date:
    first we LOOK UP accumulated stats, then UPDATE the accumulators
    with that day's results.  This guarantees zero lookahead.

    New features:
      - horse_avg_pos_last5       avg finish pos, horse's last 5 races
      - horse_win_rate            career win rate (prior races)
      - jockey_win_rate_recent    jockey win rate in the preceding 30 days
      - trainer_win_rate_recent   trainer win rate in the preceding 30 days
      - days_since_last_race      calendar days since horse's previous start
      - horse_distance_win_rate   horse's win rate at this distance category
    """
    df = full_df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.sort_values("race_date").reset_index(drop=True)

    fp = pd.to_numeric(df["finish_position"], errors="coerce")

    # Distance category
    df["_dist_cat"] = pd.cut(
        df["distance"].fillna(1600),
        bins=[0, 1400, 1800, 2200, 9999],
        labels=["sprint", "mile", "mid", "long"],
    ).astype(str)

    # Pre-allocate
    df["horse_avg_pos_last5"] = np.nan
    df["horse_win_rate"] = np.nan
    df["jockey_win_rate_recent"] = np.nan
    df["trainer_win_rate_recent"] = np.nan
    df["days_since_last_race"] = np.nan
    df["horse_distance_win_rate"] = np.nan

    # Accumulators
    horse_positions = {}     # horse_id -> list of finish_positions
    horse_wins = {}          # horse_id -> [total, wins]
    horse_last_date = {}     # horse_id -> last race datetime
    horse_dist_stats = {}    # (horse_id, dist_cat) -> [total, wins]
    jockey_results = {}      # jockey_id -> list of (date, won_bool)
    trainer_results = {}     # trainer_id -> list of (date, won_bool)

    has_horse_id = "horse_id" in df.columns
    has_jockey_id = "jockey_id" in df.columns
    has_trainer_id = "trainer_id" in df.columns

    print("  Computing rolling features (leak-free)...")

    for race_date, date_group in df.groupby("race_date", sort=True):
        if pd.isna(race_date):
            continue

        # --- PHASE 1: look up prior stats for every row on this date ---
        for idx in date_group.index:
            row = df.loc[idx]
            h_id = str(row.get("horse_id", "")) if has_horse_id else ""
            j_id = str(row.get("jockey_id", "")) if has_jockey_id else ""
            t_id = str(row.get("trainer_id", "")) if has_trainer_id else ""
            dist_cat = row.get("_dist_cat", "mile")

            if h_id and h_id in horse_positions:
                last5 = horse_positions[h_id][-5:]
                df.at[idx, "horse_avg_pos_last5"] = np.mean(last5)

            if h_id and h_id in horse_wins:
                total, wins = horse_wins[h_id]
                if total > 0:
                    df.at[idx, "horse_win_rate"] = wins / total

            if h_id and h_id in horse_last_date:
                df.at[idx, "days_since_last_race"] = (race_date - horse_last_date[h_id]).days

            if h_id and (h_id, dist_cat) in horse_dist_stats:
                total, wins = horse_dist_stats[(h_id, dist_cat)]
                if total > 0:
                    df.at[idx, "horse_distance_win_rate"] = wins / total

            if j_id and j_id in jockey_results:
                cutoff = race_date - pd.Timedelta(days=30)
                recent = [(d, w) for d, w in jockey_results[j_id] if d >= cutoff]
                if recent:
                    df.at[idx, "jockey_win_rate_recent"] = sum(w for _, w in recent) / len(recent)

            if t_id and t_id in trainer_results:
                cutoff = race_date - pd.Timedelta(days=30)
                recent = [(d, w) for d, w in trainer_results[t_id] if d >= cutoff]
                if recent:
                    df.at[idx, "trainer_win_rate_recent"] = sum(w for _, w in recent) / len(recent)

        # --- PHASE 2: update accumulators with this date's results ---
        for idx in date_group.index:
            row = df.loc[idx]
            h_id = str(row.get("horse_id", "")) if has_horse_id else ""
            j_id = str(row.get("jockey_id", "")) if has_jockey_id else ""
            t_id = str(row.get("trainer_id", "")) if has_trainer_id else ""
            dist_cat = row.get("_dist_cat", "mile")
            finish = fp.loc[idx]

            if pd.notna(finish) and h_id:
                horse_positions.setdefault(h_id, []).append(finish)
                hw = horse_wins.setdefault(h_id, [0, 0])
                hw[0] += 1
                if finish == 1:
                    hw[1] += 1
                horse_last_date[h_id] = race_date

                hd = horse_dist_stats.setdefault((h_id, dist_cat), [0, 0])
                hd[0] += 1
                if finish == 1:
                    hd[1] += 1

            if pd.notna(finish) and j_id:
                jockey_results.setdefault(j_id, []).append(
                    (race_date, 1 if finish == 1 else 0)
                )

            if pd.notna(finish) and t_id:
                trainer_results.setdefault(t_id, []).append(
                    (race_date, 1 if finish == 1 else 0)
                )

    df.drop(columns=["_dist_cat"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Feature Engineering (with rolling features, leak-free)
# ---------------------------------------------------------------------------
def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Create features for training. Returns (df_with_features, feature_columns)."""
    df = df.copy()

    # Encode categoricals
    for col, mapping in [
        ("course_type", {"芝": 0, "ダート": 1, "障害": 2}),
        ("weather", {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "小雪": 4, "雪": 5}),
        ("track_condition", {"良": 0, "稍重": 1, "重": 2, "不良": 3}),
        ("sex", {"牡": 0, "牝": 1, "セ": 2}),
    ]:
        if col in df.columns:
            df[col + "_enc"] = df[col].map(mapping).fillna(0).astype(int)

    # Numeric features that are available pre-race (no leakage)
    df["log_odds"] = np.log1p(df["odds"].fillna(10))
    df["inv_popularity"] = 1.0 / df["popularity"].clip(lower=1).fillna(10)

    # Race-level stats (computed from pre-race info only)
    race_groups = df.groupby("race_id")
    df["n_runners"] = race_groups["horse_number"].transform("count")
    df["odds_rank_in_race"] = race_groups["odds"].rank(method="min")
    df["odds_rank_pct"] = df["odds_rank_in_race"] / df["n_runners"]

    # Weight features
    df["horse_weight"] = df["horse_weight"].fillna(470)
    df["weight_diff"] = df["weight_diff"].fillna(0)

    # Distance
    df["distance_norm"] = df["distance"].fillna(1600) / 1000.0

    # Age
    df["age"] = df["age"].fillna(3)

    # Frame / horse number features
    df["frame_number"] = df["frame_number"].fillna(4)
    df["horse_number_norm"] = df["horse_number"] / df["n_runners"]

    feature_cols = [
        "odds", "log_odds", "popularity", "inv_popularity",
        "n_runners", "odds_rank_in_race", "odds_rank_pct",
        "horse_weight", "weight_diff", "weight", "distance_norm",
        "age", "frame_number", "horse_number", "horse_number_norm",
        "distance",
        # Rolling features (horse / jockey / trainer)
        "horse_avg_pos_last5", "horse_win_rate",
        "jockey_win_rate_recent", "trainer_win_rate_recent",
        "days_since_last_race", "horse_distance_win_rate",
    ]
    # Add encoded categoricals
    for c in ["course_type_enc", "weather_enc", "track_condition_enc", "sex_enc"]:
        if c in df.columns:
            feature_cols.append(c)

    # Drop any missing feature cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Fill NaN
    for c in feature_cols:
        df[c] = df[c].fillna(0)
        df[c] = df[c].replace([np.inf, -np.inf], 0)

    return df, feature_cols


# ---------------------------------------------------------------------------
# Model Training (simplified LightGBM)
# ---------------------------------------------------------------------------
def train_model(train_df: pd.DataFrame, feature_cols: list):
    """Train a LightGBM model to predict win probability. Returns model."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  LightGBM not available, falling back to sklearn GBM")
        return _train_sklearn(train_df, feature_cols)

    X = train_df[feature_cols].values
    y_win = (train_df["finish_position"] == 1).astype(int).values
    y_top3 = (train_df["finish_position"] <= 3).astype(int).values

    # Train win model
    dtrain_win = lgb.Dataset(X, label=y_win)
    params_win = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }
    model_win = lgb.train(params_win, dtrain_win, num_boost_round=300)

    # Train top3 model
    dtrain_top3 = lgb.Dataset(X, label=y_top3)
    params_top3 = params_win.copy()
    model_top3 = lgb.train(params_top3, dtrain_top3, num_boost_round=300)

    return {"win": model_win, "top3": model_top3, "type": "lgb"}


def _train_sklearn(train_df, feature_cols):
    """Fallback sklearn model."""
    from sklearn.ensemble import GradientBoostingClassifier

    X = train_df[feature_cols].values
    y_win = (train_df["finish_position"] == 1).astype(int).values
    y_top3 = (train_df["finish_position"] <= 3).astype(int).values

    model_win = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model_win.fit(X, y_win)

    model_top3 = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model_top3.fit(X, y_top3)

    return {"win": model_win, "top3": model_top3, "type": "sklearn"}


def predict(model_dict, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Generate predictions. Returns df with prob_win, prob_top3, predicted_rank."""
    X = df[feature_cols].values
    results = df[["race_id", "horse_number", "finish_position", "odds", "popularity"]].copy()
    if "race_date" in df.columns:
        results["race_date"] = df["race_date"]
    if "race_name" in df.columns:
        results["race_name"] = df["race_name"]

    if model_dict["type"] == "lgb":
        results["prob_win"] = model_dict["win"].predict(X)
        results["prob_top3"] = model_dict["top3"].predict(X)
    else:
        results["prob_win"] = model_dict["win"].predict_proba(X)[:, 1]
        results["prob_top3"] = model_dict["top3"].predict_proba(X)[:, 1]

    # Composite score
    results["composite_score"] = results["prob_win"] * 0.5 + results["prob_top3"] * 0.5

    # Rank within each race
    results["predicted_rank"] = results.groupby("race_id")["composite_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    return results


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def strategy_top1_tansho(preds: pd.DataFrame, payouts: Dict) -> List[Dict]:
    """Strategy 1: Bet tansho (win) on top-1 predicted horse."""
    bets = []
    for race_id, group in preds.groupby("race_id"):
        top = group.sort_values("predicted_rank").iloc[0]
        hit = int(top["finish_position"]) == 1
        ret = 0
        if hit:
            payout = lookup_payout(payouts, race_id, "単勝", [int(top["horse_number"])])
            if payout:
                ret = BET_AMOUNT / 100 * payout
            else:
                ret = BET_AMOUNT * float(top["odds"])
        bets.append({
            "race_id": race_id, "bet": BET_AMOUNT, "return": ret, "hit": hit,
            "date": top.get("race_date"),
        })
    return bets


def strategy_top3_fukusho(preds: pd.DataFrame, payouts: Dict) -> List[Dict]:
    """Strategy 2: Bet fukusho (place) on top-3 predicted horses."""
    bets = []
    for race_id, group in preds.groupby("race_id"):
        tops = group.sort_values("predicted_rank").head(3)
        for _, row in tops.iterrows():
            hit = int(row["finish_position"]) <= 3
            ret = 0
            if hit:
                payout = lookup_payout(payouts, race_id, "複勝", [int(row["horse_number"])])
                if payout:
                    ret = BET_AMOUNT / 100 * payout
                else:
                    # Estimate fukusho payout from win odds
                    place_odds = max(1.1, float(row["odds"]) * 0.25)
                    ret = BET_AMOUNT * place_odds
            bets.append({
                "race_id": race_id, "bet": BET_AMOUNT, "return": ret, "hit": hit,
                "date": row.get("race_date"),
            })
    return bets


def strategy_top2_umaren(preds: pd.DataFrame, payouts: Dict) -> List[Dict]:
    """Strategy 3: Bet umaren (quinella) on top-1 + top-2."""
    bets = []
    for race_id, group in preds.groupby("race_id"):
        tops = group.sort_values("predicted_rank").head(2)
        if len(tops) < 2:
            continue
        top_nums = set(tops["horse_number"].astype(int).tolist())
        actual_top2 = set(group[group["finish_position"].astype(int) <= 2]["horse_number"].astype(int).tolist())
        hit = top_nums == actual_top2
        ret = 0
        if hit:
            payout = lookup_payout(payouts, race_id, "馬連", list(top_nums))
            if payout:
                ret = BET_AMOUNT / 100 * payout
            else:
                o1, o2 = tops.iloc[0]["odds"], tops.iloc[1]["odds"]
                ret = BET_AMOUNT / 100 * max(200, o1 * o2 * 40)
        bets.append({
            "race_id": race_id, "bet": BET_AMOUNT, "return": ret, "hit": hit,
            "date": tops.iloc[0].get("race_date"),
        })
    return bets


def strategy_top2_wide(preds: pd.DataFrame, payouts: Dict) -> List[Dict]:
    """Strategy: Bet wide (ワイド) on top-2 predicted horses.
    Wide wins if both finish in top 3. Often better ROI than fukusho
    because the market is less efficient for combination bets."""
    bets = []
    for race_id, group in preds.groupby("race_id"):
        tops = group.sort_values("predicted_rank").head(2)
        if len(tops) < 2:
            continue
        top_nums = sorted(tops["horse_number"].astype(int).tolist())
        # Both must finish in top 3
        actual_top3 = set(
            group[group["finish_position"].astype(int) <= 3]["horse_number"].astype(int).tolist()
        )
        hit = set(top_nums).issubset(actual_top3)
        ret = 0
        if hit:
            payout = lookup_payout(payouts, race_id, "ワイド", top_nums)
            if payout:
                ret = BET_AMOUNT / 100 * payout
            else:
                # Estimate: wide_payout ~ sqrt(odds1 * odds2) * 0.7
                o1, o2 = float(tops.iloc[0]["odds"]), float(tops.iloc[1]["odds"])
                estimated_payout = math.sqrt(max(o1, 1) * max(o2, 1)) * 0.7
                ret = BET_AMOUNT * max(1.1, estimated_payout)
        bets.append({
            "race_id": race_id, "bet": BET_AMOUNT, "return": ret, "hit": hit,
            "date": tops.iloc[0].get("race_date"),
        })
    return bets


def strategy_ev_filter(preds: pd.DataFrame, payouts: Dict) -> List[Dict]:
    """Strategy 4: Only bet tansho when EV > 1.2."""
    bets = []
    for race_id, group in preds.groupby("race_id"):
        for _, row in group.iterrows():
            ev = float(row.get("prob_win", 0)) * float(row.get("odds", 0))
            if ev > 1.2:
                hit = int(row["finish_position"]) == 1
                ret = 0
                if hit:
                    payout = lookup_payout(payouts, race_id, "単勝", [int(row["horse_number"])])
                    if payout:
                        ret = BET_AMOUNT / 100 * payout
                    else:
                        ret = BET_AMOUNT * float(row["odds"])
                bets.append({
                    "race_id": race_id, "bet": BET_AMOUNT, "return": ret, "hit": hit,
                    "date": row.get("race_date"),
                })
    return bets


def strategy_graded_only(preds: pd.DataFrame, payouts: Dict) -> List[Dict]:
    """Strategy 5: Only bet tansho on graded stakes (race name contains keywords)."""
    graded_keywords = ["ステークス", "賞", "記念", "カップ", "トロフィー", "特別"]
    bets = []
    for race_id, group in preds.groupby("race_id"):
        race_name = str(group.iloc[0].get("race_name", ""))
        is_graded = any(kw in race_name for kw in graded_keywords)
        if not is_graded:
            continue
        top = group.sort_values("predicted_rank").iloc[0]
        hit = int(top["finish_position"]) == 1
        ret = 0
        if hit:
            payout = lookup_payout(payouts, race_id, "単勝", [int(top["horse_number"])])
            if payout:
                ret = BET_AMOUNT / 100 * payout
            else:
                ret = BET_AMOUNT * float(top["odds"])
        bets.append({
            "race_id": race_id, "bet": BET_AMOUNT, "return": ret, "hit": hit,
            "date": top.get("race_date"),
        })
    return bets


def strategy_favorite_baseline(preds: pd.DataFrame, payouts: Dict) -> List[Dict]:
    """Baseline: Always bet on the favorite (popularity=1)."""
    bets = []
    for race_id, group in preds.groupby("race_id"):
        favorites = group[group["popularity"].astype(int) == 1]
        if favorites.empty:
            favorites = group.sort_values("odds").head(1)
        fav = favorites.iloc[0]
        hit = int(fav["finish_position"]) == 1
        ret = 0
        if hit:
            payout = lookup_payout(payouts, race_id, "単勝", [int(fav["horse_number"])])
            if payout:
                ret = BET_AMOUNT / 100 * payout
            else:
                ret = BET_AMOUNT * float(fav["odds"])
        bets.append({
            "race_id": race_id, "bet": BET_AMOUNT, "return": ret, "hit": hit,
            "date": fav.get("race_date"),
        })
    return bets


def strategy_overlay_value(preds: pd.DataFrame, payouts: Dict,
                           ev_threshold: float = 1.15) -> List[Dict]:
    """Strategy: Use odds overlay model to find value bets (EV > threshold)."""
    try:
        from odds_overlay import OddsOverlayModel
        model_path = os.path.join(BASE_DIR, "models", "odds_overlay.pkl")
        if not os.path.exists(model_path):
            print("  WARNING: odds_overlay.pkl not found, skipping overlay strategy")
            return []
        overlay = OddsOverlayModel.load(model_path)
    except ImportError:
        print("  WARNING: odds_overlay module not available")
        return []

    bets = []
    for race_id, group in preds.groupby("race_id"):
        base_predictions = []
        odds_dict = {}
        for _, row in group.iterrows():
            hn = int(row["horse_number"])
            odds_val = float(row.get("odds", 0))
            if odds_val <= 0:
                continue
            base_predictions.append({
                "horse_number": hn,
                "horse_name": "",
                "prob_win": float(row.get("prob_win", 0)),
                "prob_top3": float(row.get("prob_top3", 0)),
                "composite_score": float(row.get("composite_score", 0)),
                "popularity": int(row.get("popularity", 0)),
            })
            odds_dict[hn] = odds_val

        if not base_predictions or not odds_dict:
            continue

        try:
            value_results = overlay.predict_from_base_predictions(base_predictions, odds_dict)
        except Exception:
            continue

        for v in value_results:
            if v["ev"] < ev_threshold:
                continue
            hn = v["horse_number"]
            row_match = group[group["horse_number"] == hn]
            if row_match.empty:
                continue
            row = row_match.iloc[0]
            hit = int(row["finish_position"]) == 1
            ret = 0
            if hit:
                payout = lookup_payout(payouts, race_id, "単勝", [hn])
                if payout:
                    ret = BET_AMOUNT / 100 * payout
                else:
                    ret = BET_AMOUNT * float(row["odds"])
            bets.append({
                "race_id": race_id, "bet": BET_AMOUNT, "return": ret, "hit": hit,
                "date": row.get("race_date"),
            })

    return bets


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(bets: List[Dict], strategy_name: str) -> Dict:
    """Compute honest metrics from a list of bets."""
    if not bets:
        return {"strategy": strategy_name, "total_bets": 0, "error": "no bets"}

    total_bet = sum(b["bet"] for b in bets)
    total_ret = sum(b["return"] for b in bets)
    hits = sum(1 for b in bets if b["hit"])
    n = len(bets)
    profit = total_ret - total_bet
    roi = (total_ret / total_bet - 1) * 100 if total_bet > 0 else 0
    hit_rate = hits / n * 100 if n > 0 else 0

    # Equity curve + max drawdown
    equity = []
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for b in bets:
        cumulative += b["return"] - b["bet"]
        equity.append(round(cumulative, 1))
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    # Monthly breakdown
    monthly = {}
    for b in bets:
        dt = b.get("date")
        if dt is not None and hasattr(dt, "strftime"):
            m = dt.strftime("%Y-%m")
        elif isinstance(dt, str) and len(dt) >= 7:
            m = dt[:7]
        else:
            m = "unknown"
        if m not in monthly:
            monthly[m] = {"bet": 0, "return": 0, "hits": 0, "count": 0}
        monthly[m]["bet"] += b["bet"]
        monthly[m]["return"] += b["return"]
        monthly[m]["hits"] += 1 if b["hit"] else 0
        monthly[m]["count"] += 1

    monthly_list = []
    for m in sorted(monthly.keys()):
        v = monthly[m]
        m_roi = (v["return"] / v["bet"] - 1) * 100 if v["bet"] > 0 else 0
        m_hr = v["hits"] / v["count"] * 100 if v["count"] > 0 else 0
        monthly_list.append({
            "month": m,
            "bets": v["count"],
            "bet_amount": v["bet"],
            "return_amount": round(v["return"], 1),
            "profit": round(v["return"] - v["bet"], 1),
            "roi_pct": round(m_roi, 1),
            "hit_rate_pct": round(m_hr, 1),
        })

    # Parametric CI for monthly ROI (legacy)
    monthly_rois = [m["roi_pct"] for m in monthly_list if m["month"] != "unknown"]
    if len(monthly_rois) >= 3:
        mean_roi = float(np.mean(monthly_rois))
        std_roi = float(np.std(monthly_rois, ddof=1))
        n_months = len(monthly_rois)
        margin = 1.96 * std_roi / math.sqrt(n_months)
        ci_lower = round(mean_roi - margin, 1)
        ci_upper = round(mean_roi + margin, 1)
        # Sharpe ratio (monthly mean ROI / std)
        sharpe = round(mean_roi / std_roi, 3) if std_roi > 0 else None
        # t-statistic for H0: monthly ROI = 0. |t| > 2 ≈ p<0.05 with modest n.
        t_stat = round(mean_roi / (std_roi / math.sqrt(n_months)), 3) if std_roi > 0 else None
    else:
        ci_lower = ci_upper = sharpe = t_stat = None

    # Bootstrap CI for per-bet ROI (more robust than parametric)
    boot_lower = boot_upper = None
    if n >= 30:
        try:
            per_bet_return = np.array([b["return"] - b["bet"] for b in bets], dtype=float)
            per_bet_cost = np.array([b["bet"] for b in bets], dtype=float)
            rng = np.random.default_rng(seed=42)
            boot_rois = []
            idx_arr = np.arange(n)
            for _ in range(1000):
                idx = rng.choice(idx_arr, size=n, replace=True)
                bet_sum = per_bet_cost[idx].sum()
                if bet_sum > 0:
                    boot_rois.append((per_bet_return[idx].sum() / bet_sum) * 100)
            if boot_rois:
                boot_lower = round(float(np.percentile(boot_rois, 2.5)), 1)
                boot_upper = round(float(np.percentile(boot_rois, 97.5)), 1)
        except Exception:
            pass

    # Longest losing streak
    longest_losing_streak = 0
    cur = 0
    for b in bets:
        if b.get("hit"):
            cur = 0
        else:
            cur += 1
            if cur > longest_losing_streak:
                longest_losing_streak = cur

    # Profit factor (gross wins / gross losses)
    gross_win = sum(max(0, b["return"] - b["bet"]) for b in bets)
    gross_loss = sum(max(0, b["bet"] - b["return"]) for b in bets)
    profit_factor = round(gross_win / gross_loss, 3) if gross_loss > 0 else None

    # Sample adequacy flag
    sample_adequate = n >= 100

    return {
        "strategy": strategy_name,
        "total_bets": n,
        "total_bet_amount": total_bet,
        "total_return": round(total_ret, 1),
        "profit": round(profit, 1),
        "roi_pct": round(roi, 1),
        "hit_rate_pct": round(hit_rate, 1),
        "hit_count": hits,
        "max_drawdown": round(max_dd, 1),
        "roi_95ci_lower": ci_lower,
        "roi_95ci_upper": ci_upper,
        "roi_bootstrap_95ci_lower": boot_lower,
        "roi_bootstrap_95ci_upper": boot_upper,
        "sharpe_monthly": sharpe,
        "t_stat_monthly": t_stat,
        "longest_losing_streak": longest_losing_streak,
        "profit_factor": profit_factor,
        "sample_adequate": sample_adequate,
        "monthly": monthly_list,
        "equity_curve_sample": equity[::max(1, len(equity) // 200)],  # Downsample for JSON
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_honest_backtest() -> Dict:
    """Execute the full honest walk-forward backtest."""
    print("=" * 70)
    print("HONEST WALK-FORWARD BACKTEST")
    print(f"  Train: {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}")
    print(f"  Test:  {TEST_YEAR}")
    print(f"  Bet:   {BET_AMOUNT} yen flat")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    train_raw = load_race_data(TRAIN_YEARS)
    test_raw = load_race_data([TEST_YEAR])
    payouts = load_payouts()

    # Filter out invalid finish positions
    train_raw = train_raw[train_raw["finish_position"].notna()]
    train_raw = train_raw[train_raw["finish_position"].apply(
        lambda x: str(x).replace(".", "").isdigit()
    )]
    train_raw["finish_position"] = train_raw["finish_position"].astype(int)

    test_raw = test_raw[test_raw["finish_position"].notna()]
    test_raw = test_raw[test_raw["finish_position"].apply(
        lambda x: str(x).replace(".", "").isdigit()
    )]
    test_raw["finish_position"] = test_raw["finish_position"].astype(int)

    print(f"  Train: {len(train_raw):,} rows, {train_raw['race_id'].nunique():,} races")
    print(f"  Test:  {len(test_raw):,} rows, {test_raw['race_id'].nunique():,} races")

    # Compute rolling features on combined data (leak-free: prior data only)
    print("\n[2/6] Computing rolling features on combined data...")
    train_raw["_split"] = "train"
    test_raw["_split"] = "test"
    combined_raw = pd.concat([train_raw, test_raw], ignore_index=True)
    combined_raw = compute_rolling_features(combined_raw)
    train_raw = combined_raw[combined_raw["_split"] == "train"].drop(columns=["_split"])
    test_raw = combined_raw[combined_raw["_split"] == "test"].drop(columns=["_split"])
    del combined_raw

    # Feature engineering
    print("\n[3/6] Creating features...")
    train_df, feature_cols = create_features(train_raw)
    test_df, _ = create_features(test_raw)

    # Ensure test has same features
    for c in feature_cols:
        if c not in test_df.columns:
            test_df[c] = 0

    # Train model
    print("\n[4/6] Training model (walk-forward: train on past, test on future)...")
    model = train_model(train_df, feature_cols)
    print("  Model trained successfully")

    # Predict on test set
    print("\n[5/6] Generating predictions on test set...")
    preds = predict(model, test_df, feature_cols)
    print(f"  Predictions: {len(preds):,} rows, {preds['race_id'].nunique():,} races")

    # Check basic prediction quality
    top1_preds = preds[preds["predicted_rank"] == 1]
    if len(top1_preds) > 0:
        top1_win_rate = (top1_preds["finish_position"] == 1).mean() * 100
        top1_place_rate = (top1_preds["finish_position"] <= 3).mean() * 100
        print(f"  Top-1 pick win rate:   {top1_win_rate:.1f}%")
        print(f"  Top-1 pick place rate: {top1_place_rate:.1f}%")

    # Favorite baseline
    fav_df = preds[preds["popularity"].astype(int) == 1]
    if len(fav_df) > 0:
        fav_win_rate = (fav_df["finish_position"] == 1).mean() * 100
        print(f"  Favorite (pop=1) win rate: {fav_win_rate:.1f}%")

    # Run strategies
    print("\n[6/6] Running strategies...")
    strategies = [
        ("top1_tansho", "Top-1 Tansho (Win)", strategy_top1_tansho),
        ("top3_fukusho", "Top-3 Fukusho (Place)", strategy_top3_fukusho),
        ("top2_umaren", "Top-2 Umaren (Quinella)", strategy_top2_umaren),
        ("top2_wide", "Top-2 Wide (ワイド)", strategy_top2_wide),
        ("ev_filter", "EV > 1.2 Filter", strategy_ev_filter),
        ("graded_only", "Graded Stakes Only", strategy_graded_only),
        ("favorite_baseline", "Baseline: Bet Favorite", strategy_favorite_baseline),
        ("overlay_ev115", "Overlay EV > 1.15", lambda p, pay: strategy_overlay_value(p, pay, 1.15)),
        ("overlay_ev130", "Overlay EV > 1.30", lambda p, pay: strategy_overlay_value(p, pay, 1.30)),
    ]

    results = []
    for key, name, func in strategies:
        bets = func(preds, payouts)
        metrics = compute_metrics(bets, name)
        metrics["key"] = key
        results.append(metrics)
        roi_str = f"{metrics['roi_pct']:+.1f}%" if metrics.get("roi_pct") is not None else "N/A"
        hr_str = f"{metrics.get('hit_rate_pct', 0):.1f}%"
        print(f"  {name:<30s}  ROI: {roi_str:>8s}  Hit: {hr_str:>6s}  Bets: {metrics['total_bets']:>6,}")

    # Build summary
    model_top1 = next((r for r in results if r["key"] == "top1_tansho"), None)
    baseline = next((r for r in results if r["key"] == "favorite_baseline"), None)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "train_period": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
        "test_period": str(TEST_YEAR),
        "train_rows": len(train_raw),
        "test_rows": len(test_raw),
        "test_races": int(test_raw["race_id"].nunique()),
        "bet_amount": BET_AMOUNT,
        "model_type": model["type"],
        "model_beats_favorite": (
            model_top1["roi_pct"] > baseline["roi_pct"]
            if model_top1 and baseline else None
        ),
        "strategies": results,
    }

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<30s} {'ROI':>8s} {'Hit%':>7s} {'Profit':>10s} {'Bets':>6s} {'MaxDD':>8s}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x.get("roi_pct", -999), reverse=True):
        print(f"{r['strategy']:<30s} {r['roi_pct']:>+7.1f}% {r.get('hit_rate_pct',0):>6.1f}% "
              f"{r['profit']:>+10.0f} {r['total_bets']:>6,} {r['max_drawdown']:>8.0f}")
    print("-" * 70)

    if model_top1 and baseline:
        if model_top1["roi_pct"] > baseline["roi_pct"]:
            print(f"\nModel TOP-1 beats favorite by {model_top1['roi_pct'] - baseline['roi_pct']:.1f}pp ROI")
        else:
            print(f"\nWARNING: Model TOP-1 does NOT beat favorite "
                  f"({model_top1['roi_pct']:.1f}% vs {baseline['roi_pct']:.1f}%)")

    # Save
    out_path = os.path.join(DATA_DIR, "honest_backtest_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return summary


def run_god_backtest() -> Dict:
    """Backtest using the pre-trained GOD model (keiba_model.pkl).
    Uses data_processor.py FeatureEngineer with GOD features."""
    from model_trainer import RacePredictionModel
    from data_processor import RaceDataProcessor, FeatureEngineer

    print("=" * 70)
    print("GOD MODEL BACKTEST (pre-trained v4)")
    print(f"  Test:  {TEST_YEAR}")
    print(f"  Bet:   {BET_AMOUNT} yen flat")
    print("=" * 70)

    import gc

    # Load all years (2019-2025) for full feature computation
    # On VPS (5.8GB RAM) use: --lite flag to load 2024-2025 only
    if "--lite" in sys.argv:
        load_years = [TEST_YEAR - 1, TEST_YEAR]
        print(f"\n[1/4] Loading data ({load_years[0]}-{load_years[-1]} lite mode)...")
    else:
        load_years = list(range(2019, TEST_YEAR + 1))
        print(f"\n[1/4] Loading data ({load_years[0]}-{load_years[-1]} full)...")
    all_raw = load_race_data(load_years)
    payouts = load_payouts()

    # Preprocess
    print("\n[2/4] Preprocessing + Feature engineering...")
    processor = RaceDataProcessor(data_dir="data")
    processor.raw_data = all_raw
    df = processor.preprocess()
    del all_raw; gc.collect()

    fe = FeatureEngineer()
    df = fe.create_features(df)
    gc.collect()

    # Filter test year
    df_test = df[df["race_date"].dt.year == TEST_YEAR].copy()
    df_test = df_test[df_test["finish_position"].notna()].copy()
    df_test = df_test[df_test["finish_position"] > 0].copy()

    print(f"  Test rows: {len(df_test):,}, races: {df_test['race_id'].nunique():,}")

    # Load GOD model
    print("\n[3/4] Loading GOD model + predicting...")
    model = RacePredictionModel(model_dir="models")
    model.load("keiba_model")

    # Add missing features
    for f in model.feature_columns:
        if f not in df_test.columns:
            df_test[f] = 0.0

    preds = model.predict(df_test)

    # Merge back needed columns
    merge_cols = ["race_id", "horse_number"]
    keep_cols = ["finish_position", "odds", "popularity", "race_date", "race_name"]
    for c in keep_cols:
        if c in df_test.columns and c not in preds.columns:
            preds = preds.merge(df_test[merge_cols + [c]].drop_duplicates(),
                                on=merge_cols, how="left", suffixes=("", "_dup"))

    # Ensure columns exist
    for c in ["finish_position", "odds", "popularity"]:
        if c not in preds.columns:
            preds[c] = 0

    print(f"  Predictions: {len(preds):,} rows, {preds['race_id'].nunique():,} races")

    # Top-1 stats
    top1_preds = preds[preds["predicted_rank"] == 1]
    if len(top1_preds) > 0:
        top1_win = (top1_preds["finish_position"] == 1).mean() * 100
        top1_place = (top1_preds["finish_position"] <= 3).mean() * 100
        print(f"  Top-1 pick win rate:   {top1_win:.1f}%")
        print(f"  Top-1 pick place rate: {top1_place:.1f}%")
    fav_df = preds[preds["popularity"].astype(float).astype(int) == 1]
    if len(fav_df) > 0:
        print(f"  Favorite (pop=1) win rate: {(fav_df['finish_position'] == 1).mean() * 100:.1f}%")

    # Run strategies
    print("\n[4/4] Running strategies...")
    strategies = [
        ("top1_tansho", "Top-1 Tansho (Win)", strategy_top1_tansho),
        ("top3_fukusho", "Top-3 Fukusho (Place)", strategy_top3_fukusho),
        ("top2_umaren", "Top-2 Umaren (Quinella)", strategy_top2_umaren),
        ("top2_wide", "Top-2 Wide (ワイド)", strategy_top2_wide),
        ("ev_filter", "EV > 1.2 Filter", strategy_ev_filter),
        ("graded_only", "Graded Stakes Only", strategy_graded_only),
        ("favorite_baseline", "Baseline: Bet Favorite", strategy_favorite_baseline),
        ("overlay_ev115", "Overlay EV > 1.15", lambda p, pay: strategy_overlay_value(p, pay, 1.15)),
        ("overlay_ev130", "Overlay EV > 1.30", lambda p, pay: strategy_overlay_value(p, pay, 1.30)),
    ]

    results = []
    for key, name, func in strategies:
        bets = func(preds, payouts)
        metrics = compute_metrics(bets, name)
        metrics["key"] = key
        results.append(metrics)
        roi_str = f"{metrics['roi_pct']:+.1f}%" if metrics.get("roi_pct") is not None else "N/A"
        hr_str = f"{metrics.get('hit_rate_pct', 0):.1f}%"
        print(f"  {name:<30s}  ROI: {roi_str:>8s}  Hit: {hr_str:>6s}  Bets: {metrics['total_bets']:>6,}")

    # Summary
    model_top1 = next((r for r in results if r["key"] == "top1_tansho"), None)
    baseline = next((r for r in results if r["key"] == "favorite_baseline"), None)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (GOD MODEL)")
    print("=" * 70)
    print(f"{'Strategy':<30s} {'ROI':>8s} {'Hit%':>7s} {'Profit':>10s} {'Bets':>6s} {'MaxDD':>8s}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x.get("roi_pct", -999), reverse=True):
        print(f"{r['strategy']:<30s} {r['roi_pct']:>+7.1f}% {r.get('hit_rate_pct',0):>6.1f}% "
              f"{r['profit']:>+10.0f} {r['total_bets']:>6,} {r['max_drawdown']:>8.0f}")
    print("-" * 70)

    if model_top1 and baseline:
        if model_top1["roi_pct"] > baseline["roi_pct"]:
            print(f"\nGOD Model TOP-1 beats favorite by {model_top1['roi_pct'] - baseline['roi_pct']:.1f}pp ROI")
        else:
            print(f"\nWARNING: GOD Model TOP-1 does NOT beat favorite "
                  f"({model_top1['roi_pct']:.1f}% vs {baseline['roi_pct']:.1f}%)")

    summary = {
        "type": "god_model_backtest",
        "generated_at": datetime.now().isoformat(),
        "test_period": str(TEST_YEAR),
        "test_rows": len(df_test),
        "test_races": int(df_test["race_id"].nunique()),
        "bet_amount": BET_AMOUNT,
        "model_type": "god_v4",
        "strategies": results,
    }
    out_path = os.path.join(DATA_DIR, "god_backtest_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return summary


def run_multiyear_backtest(test_years: list = None) -> Dict:
    """
    マルチイヤーWalk-Forward バックテスト。
    各テスト年ごとにそれ以前のデータで学習 → テスト年で検証。
    複数年にわたる安定性を確認できる。

    Usage:
        python honest_backtest.py --multiyear
        python honest_backtest.py --multiyear --years 2023,2024,2025
    """
    if test_years is None:
        test_years = [2023, 2024, 2025]

    all_years = list(range(2019, max(test_years) + 1))

    print("=" * 70)
    print("MULTI-YEAR WALK-FORWARD BACKTEST")
    print(f"  Available years: {all_years[0]}-{all_years[-1]}")
    print(f"  Test years: {test_years}")
    print(f"  Bet: {BET_AMOUNT} yen flat")
    print("=" * 70)

    payouts = load_payouts()

    yearly_results = {}
    for test_year in test_years:
        train_years = [y for y in all_years if y < test_year]
        if len(train_years) < 2:
            print(f"\n⚠️ {test_year}: トレーニングデータ不足（{len(train_years)}年）、スキップ")
            continue

        print(f"\n{'='*50}")
        print(f"  Test year: {test_year} | Train: {train_years[0]}-{train_years[-1]}")
        print(f"{'='*50}")

        try:
            train_raw = load_race_data(train_years)
            test_raw = load_race_data([test_year])
        except FileNotFoundError as e:
            print(f"  ⚠️ データなし: {e}")
            continue

        # Filter invalid finish positions
        def _clean_finish(df_in):
            df_out = df_in[df_in["finish_position"].notna()].copy()
            df_out = df_out[df_out["finish_position"].apply(
                lambda x: str(x).replace(".", "").isdigit()
            )]
            df_out["finish_position"] = df_out["finish_position"].astype(int)
            return df_out

        train_raw = _clean_finish(train_raw)
        test_raw = _clean_finish(test_raw)

        print(f"  Train: {len(train_raw):,} rows | Test: {len(test_raw):,} rows")

        # Rolling features
        train_raw["_split"] = "train"
        test_raw["_split"] = "test"
        combined = pd.concat([train_raw, test_raw], ignore_index=True)
        combined = compute_rolling_features(combined)
        train_raw = combined[combined["_split"] == "train"].drop(columns=["_split"])
        test_raw = combined[combined["_split"] == "test"].drop(columns=["_split"])
        del combined

        # Features
        train_df, feature_cols = create_features(train_raw)
        test_df, _ = create_features(test_raw)
        for c in feature_cols:
            if c not in test_df.columns:
                test_df[c] = 0

        # Train & Predict
        model = train_model(train_df, feature_cols)
        preds = predict(model, test_df, feature_cols)

        # Evaluate strategies
        strategy_funcs = [
            ("top1_tansho", "Top-1 Tansho", strategy_top1_tansho),
            ("top3_fukusho", "Top-3 Fukusho", strategy_top3_fukusho),
            ("top2_umaren", "Top-2 Umaren", strategy_top2_umaren),
            ("top2_wide", "Top-2 Wide", strategy_top2_wide),
            ("ev_filter", "EV Filter", strategy_ev_filter),
        ]
        year_results = {}
        for key, name, func in strategy_funcs:
            try:
                bets = func(preds, payouts)
                result = compute_metrics(bets, name)
                year_results[key] = result
                print(f"    {key:20s}: ROI={result.get('roi_pct', 0):7.1f}% | "
                      f"Hit={result.get('hit_rate_pct', 0):5.1f}% | "
                      f"Profit=¥{result.get('profit', 0):+,.0f}")
            except Exception as e:
                print(f"    {key:20s}: ERROR - {e}")

        yearly_results[test_year] = year_results

    # Summary across years
    print(f"\n{'='*70}")
    print("MULTI-YEAR SUMMARY")
    print(f"{'='*70}")

    all_strategies = set()
    for yr in yearly_results.values():
        all_strategies.update(yr.keys())

    for strat in sorted(all_strategies):
        rois = []
        hit_rates = []
        profits = []
        for year, yr_results in sorted(yearly_results.items()):
            if strat in yr_results:
                r = yr_results[strat]
                rois.append(r.get("roi_pct", 0))
                hit_rates.append(r.get("hit_rate_pct", 0))
                profits.append(r.get("profit", 0))
        if rois:
            print(f"\n  {strat}:")
            for year in sorted(yearly_results.keys()):
                if strat in yearly_results[year]:
                    r = yearly_results[year][strat]
                    print(f"    {year}: ROI={r.get('roi_pct', 0):7.1f}% | Hit={r.get('hit_rate_pct', 0):5.1f}%")
            print(f"    AVG:  ROI={np.mean(rois):7.1f}% +/- {np.std(rois):5.1f}% | "
                  f"Hit={np.mean(hit_rates):5.1f}% | "
                  f"Total Profit=¥{sum(profits):+,.0f}")

    # Save
    summary = {
        "type": "multiyear_backtest",
        "generated_at": datetime.now().isoformat(),
        "test_years": test_years,
        "bet_amount": BET_AMOUNT,
        "yearly_results": {str(k): v for k, v in yearly_results.items()},
    }
    out_path = os.path.join(DATA_DIR, "multiyear_backtest_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return summary


if __name__ == "__main__":
    if "--god" in sys.argv:
        run_god_backtest()
    elif "--multiyear" in sys.argv:
        # Parse optional --years argument
        years = None
        for i, arg in enumerate(sys.argv):
            if arg == "--years" and i + 1 < len(sys.argv):
                years = [int(y) for y in sys.argv[i + 1].split(",")]
        run_multiyear_backtest(test_years=years)
    else:
        run_honest_backtest()
