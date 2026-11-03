"""
v5: ROI-Maximizing Training - Value Detection Model
====================================================
競馬予想AI - GOD級学習パイプライン (train_god_v5.py)

All v4 fixes carried forward.

v5 changes (ROI optimization):
  V1:  New task "value": predicts profitable bets (odds * P(win) > 1)
  V2:  Sample weights = log(odds) to emphasize high-value longshots
  V3:  Custom eval metric: actual ROI on holdout
  V4:  Odds features INCLUDED in training (needed for value detection)
  V5:  New target: y_value = 1 if actual_payout/bet > 1.0 (profitable bet)
  V6:  LambdaRank target changed to payout-weighted ranking
  V7:  Calibration optimized for profit, not just log-likelihood
  V8:  Holdout evaluation includes simulated ROI

Core philosophy shift:
  v4: "Which horse will win?" → High accuracy, but market already knows this
  v5: "Which horse is UNDERVALUED?" → Lower accuracy, but profitable bets

Same model save format (server-compatible).
"""

import os
import sys
import time
import json
import pickle
import signal
import warnings
import gc
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

# Targeted warning suppression (not blanket ignore)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ──────────────────────────────────────────────
# F3: Checkpoint system
# ──────────────────────────────────────────────
_CHECKPOINT_STATE = {}


def save_checkpoint(step_name, state_dict, checkpoint_dir=None):
    """Save pipeline state so training can be resumed after interruption."""
    cdir = checkpoint_dir or "checkpoints_god_v4"
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"checkpoint_{step_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(state_dict, f)
    # Also save a manifest of completed steps
    manifest_path = os.path.join(cdir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    manifest[step_name] = {"saved_at": datetime.now().isoformat(), "path": path}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_checkpoint(step_name, checkpoint_dir=None):
    """Load a previously saved checkpoint. Returns None if not found."""
    cdir = checkpoint_dir or "checkpoints_god_v4"
    path = os.path.join(cdir, f"checkpoint_{step_name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully: save whatever state we have."""
    print("\n\n  [!] Ctrl+C detected. Saving emergency checkpoint...")
    if _CHECKPOINT_STATE:
        save_checkpoint("emergency", _CHECKPOINT_STATE)
        print("  [!] Emergency checkpoint saved. You can resume later.")
    else:
        print("  [!] No state to save yet.")
    sys.exit(1)


signal.signal(signal.SIGINT, _signal_handler)


# ──────────────────────────────────────────────
# CONFIG - all magic numbers centralized
# ──────────────────────────────────────────────
CONFIG = {
    "data_file": "data/_all_races.csv",
    "bloodline_file": "data/bloodline.csv",
    "output_dir": "output_god_v5",
    "checkpoint_dir": "checkpoints_god_v5",
    "model_dir": "models",

    "n_cv_folds": 5,               # V5: 8→5 (faster, still robust)
    "optuna_trials": 15,            # V5: 100→15 (15 trials × 3 tasks = 45 total)
    "use_gpu": True,

    # Boruta
    "boruta_max_iter": 50,          # V5: 100→50 (faster)
    "boruta_min_features": 30,

    # Purging / Embargo
    "purge_days": 7,
    "embargo_days": 7,

    # Time decay (365 days = 1 year)
    "time_decay_half_life_days": 365,

    # V4: Odds features INCLUDED (needed for value detection)
    "exclude_odds_from_features": False,

    # V5: Value detection thresholds
    "value_min_odds": 1.5,        # Ignore ultra-short odds (no value)
    "value_max_odds": 100.0,      # Ignore extreme longshots (noise)
    "value_odds_weight_power": 0.5,  # Sample weight = odds^0.5 (sqrt)

    # Noise injection
    "noise_std": 0.01,

    # TabNet
    "tabnet_epochs": 100,
    "tabnet_patience": 15,

    # Feature stability
    "stability_threshold": 0.5,

    # Temporal split boundaries (C1 fix)
    "optimize_start": "2024-01-01",
    "holdout_start": "2025-01-01",

    # Elo system parameters
    "horse_elo_k": 20,
    "jockey_elo_k": 12,
    "trainer_elo_k": 8,
    "glicko_initial_rd": 350,
    "glicko_c": 15,

    # Elo combination weights
    "elo_horse_weight": 0.6,
    "elo_jockey_weight": 0.25,
    "elo_trainer_weight": 0.15,

    # Stacking meta-learner
    "meta_num_leaves": 15,
    "meta_lr": 0.05,
    "meta_reg_alpha": 1.0,
    "meta_reg_lambda": 1.0,
    "meta_min_child_samples": 50,
    "meta_num_boost_round": 200,
    "meta_early_stopping": 30,

    # LambdaRank
    "lambdarank_ndcg_at": [1, 3, 5],
    "lambdarank_boost_rounds": 2000,
    "lambdarank_early_stopping": 100,

    # Calibration
    "temperature_range_min": 0.1,
    "temperature_range_max": 5.0,
    "temperature_steps": 50,
}

# Domain-protected features (partial match)
PROTECTED_FEATURE_PATTERNS = [
    "distance", "age", "weight", "horse_weight", "odds", "popularity",
    "track_condition", "course_type", "frame_number", "horse_number",
    "field_condition", "n_runners", "sex_enc", "weather_enc",
    "horse_elo", "jockey_elo", "trainer_elo", "horse_glicko",
    "combined_elo", "days_since_last_race",
]

# Excluded from adversarial validation / Boruta (leak sources)
ADVVAL_EXCLUDE_FEATURES = {"year", "month", "day_of_week"}

# F5: Odds-related features to exclude when exclude_odds_from_features is True
ODDS_FEATURES = {"odds", "popularity", "odds_rank_pct", "odds_zscore", "odds_x_pop",
                  "odds_vs_race_avg", "odds_ratio", "odds_rank", "odds_rank_in_race"}

# Global timer
_GLOBAL_START = time.time()


def log_step(msg, level=0):
    """Progress logging with timestamps"""
    indent = "  " * level
    elapsed = time.time() - _GLOBAL_START
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"  [{h:02d}:{m:02d}:{s:02d}] {indent}{msg}")


# ══════════════════════════════════════════════
# Part 0: Same-day leak-safe expanding stats (C3 fix)
# ══════════════════════════════════════════════
def safe_expanding_mean_fast(df, group_col, value_col, date_col='race_date'):
    """Fast vectorized expanding mean excluding same-date entries using cumsum trick.

    C3 FIX: shift(1) leaks same-day race results for entities
    (jockeys/trainers) who participate in multiple races per day.
    This function only uses data from BEFORE today.

    O(n) with no Python loops - uses groupby + cumsum with date-boundary correction.
    """
    df_work = df[[group_col, date_col, value_col]].copy()
    df_work['_val'] = df_work[value_col].astype(float)
    df_work['_notnan'] = df_work['_val'].notna().astype(float)
    df_work.loc[df_work['_val'].isna(), '_val'] = 0.0

    # Step 1: Aggregate per (group, date) - sum and count
    daily = df_work.groupby([group_col, date_col]).agg(
        day_sum=('_val', 'sum'),
        day_count=('_notnan', 'sum')
    ).reset_index()

    # Step 2: Cumulative sum/count SHIFTED by 1 date within each group
    # (shifted = excludes current date)
    daily['cum_sum'] = daily.groupby(group_col)['day_sum'].cumsum() - daily['day_sum']
    daily['cum_count'] = daily.groupby(group_col)['day_count'].cumsum() - daily['day_count']
    daily['expanding_mean'] = np.where(daily['cum_count'] > 0, daily['cum_sum'] / daily['cum_count'], np.nan)

    # Step 3: Merge back - each row gets its (group, date) expanding mean
    result = df_work[[group_col, date_col]].merge(
        daily[[group_col, date_col, 'expanding_mean']],
        on=[group_col, date_col], how='left'
    )['expanding_mean']
    result.index = df.index

    return result


def safe_expanding_rate_fast(df, group_col, value_col, date_col='race_date', target_value=1):
    """Fast vectorized expanding rate (e.g., win rate) excluding same-date.

    O(n) with no Python loops.
    """
    df_work = df[[group_col, date_col, value_col]].copy()
    df_work['_hit'] = (df_work[value_col] == target_value).astype(float)
    df_work['_valid'] = df_work[value_col].notna().astype(float)

    daily = df_work.groupby([group_col, date_col]).agg(
        day_hits=('_hit', 'sum'),
        day_total=('_valid', 'sum')
    ).reset_index()

    daily['cum_hits'] = daily.groupby(group_col)['day_hits'].cumsum() - daily['day_hits']
    daily['cum_total'] = daily.groupby(group_col)['day_total'].cumsum() - daily['day_total']
    daily['expanding_rate'] = np.where(daily['cum_total'] > 0, daily['cum_hits'] / daily['cum_total'], np.nan)

    result = df_work[[group_col, date_col]].merge(
        daily[[group_col, date_col, 'expanding_rate']],
        on=[group_col, date_col], how='left'
    )['expanding_rate']
    result.index = df.index
    return result


def safe_expanding_top3_rate_fast(df, group_col, value_col, date_col='race_date'):
    """Fast vectorized expanding top-3 rate excluding same-date.

    O(n) with no Python loops.
    """
    df_work = df[[group_col, date_col, value_col]].copy()
    df_work['_hit'] = (df_work[value_col] <= 3).astype(float)
    df_work['_valid'] = df_work[value_col].notna().astype(float)

    daily = df_work.groupby([group_col, date_col]).agg(
        day_hits=('_hit', 'sum'),
        day_total=('_valid', 'sum')
    ).reset_index()

    daily['cum_hits'] = daily.groupby(group_col)['day_hits'].cumsum() - daily['day_hits']
    daily['cum_total'] = daily.groupby(group_col)['day_total'].cumsum() - daily['day_total']
    daily['expanding_rate'] = np.where(daily['cum_total'] > 0, daily['cum_hits'] / daily['cum_total'], np.nan)

    result = df_work[[group_col, date_col]].merge(
        daily[[group_col, date_col, 'expanding_rate']],
        on=[group_col, date_col], how='left'
    )['expanding_rate']
    result.index = df.index
    return result


def safe_expanding_count_fast(df, group_col, value_col, date_col='race_date'):
    """Fast vectorized expanding count excluding same-date.

    O(n) with no Python loops.
    """
    df_work = df[[group_col, date_col, value_col]].copy()
    df_work['_valid'] = df_work[value_col].notna().astype(float)

    daily = df_work.groupby([group_col, date_col]).agg(
        day_count=('_valid', 'sum')
    ).reset_index()

    daily['cum_count'] = daily.groupby(group_col)['day_count'].cumsum() - daily['day_count']

    result = df_work[[group_col, date_col]].merge(
        daily[[group_col, date_col, 'cum_count']],
        on=[group_col, date_col], how='left'
    )['cum_count']
    result.index = df.index
    return result


def apply_safe_expanding(df, group_col, value_col, func, date_col='race_date', **kwargs):
    """Apply safe expanding function per group, returning a full-length Series.

    This is the main entry point for C3-safe rolling features.
    Replaces all `groupby(col).transform(lambda x: x.shift(1).expanding().mean())` patterns.
    Uses fully vectorized implementations - O(n) with no Python loops.
    """
    if func == 'mean':
        return safe_expanding_mean_fast(df, group_col, value_col, date_col)
    elif func == 'win_rate':
        return safe_expanding_rate_fast(df, group_col, value_col, date_col, target_value=1)
    elif func == 'top3_rate':
        return safe_expanding_top3_rate_fast(df, group_col, value_col, date_col)
    elif func == 'count':
        return safe_expanding_count_fast(df, group_col, value_col, date_col)
    else:
        return safe_expanding_mean_fast(df, group_col, value_col, date_col)


# ══════════════════════════════════════════════
# Part 1: Elo/Glicko Rating System (H5: speed-up)
# ══════════════════════════════════════════════
class EloRatingSystem:
    def __init__(self, k_factor=16, initial_rating=1500):
        self.k = k_factor
        self.initial = initial_rating
        self.ratings = {}

    def get_rating(self, entity_id):
        return self.ratings.get(entity_id, self.initial)

    def expected_score(self, ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update_race(self, participants):
        """F6: Accumulate deltas per ID, then apply mean to handle duplicates."""
        n = len(participants)
        if n < 2:
            return
        ratings = {pid: self.get_rating(pid) for pid, _ in participants}
        # F6: Collect deltas per entity ID (handles duplicate IDs in same race)
        deltas = defaultdict(list)
        for i, (id_i, pos_i) in enumerate(participants):
            delta = 0.0
            for j, (id_j, pos_j) in enumerate(participants):
                if i == j:
                    continue
                expected = self.expected_score(ratings[id_i], ratings[id_j])
                actual = 1.0 if pos_i < pos_j else (0.5 if pos_i == pos_j else 0.0)
                delta += self.k * (actual - expected) / (n - 1)
            deltas[id_i].append(delta)
        # Apply mean delta per entity
        for eid, delta_list in deltas.items():
            self.ratings[eid] = ratings[eid] + np.mean(delta_list)


class GlickoRatingSystem:
    def __init__(self, initial_rating=1500, initial_rd=350, c=15):
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.c = c
        self.ratings = {}
        self.rds = {}
        self.last_race = {}

    def get_rating(self, entity_id):
        return self.ratings.get(entity_id, self.initial_rating)

    def get_rd(self, entity_id, current_period=0):
        rd = self.rds.get(entity_id, self.initial_rd)
        last = self.last_race.get(entity_id, current_period)
        periods_inactive = current_period - last
        return min(350, np.sqrt(rd ** 2 + self.c ** 2 * periods_inactive))

    def g(self, rd):
        return 1.0 / np.sqrt(1 + 3 * (rd / np.pi) ** 2 / 400 ** 2)

    def expected(self, r, r_opp, rd_opp):
        return 1.0 / (1 + 10 ** (-self.g(rd_opp) * (r - r_opp) / 400))

    def update_race(self, participants, period=0):
        n = len(participants)
        if n < 2:
            return
        # Snapshot pre-race ratings/RDs BEFORE updating any participant
        pre_ratings = {pid: self.get_rating(pid) for pid, _ in participants}
        pre_rds = {pid: self.get_rd(pid, period) for pid, _ in participants}

        for pid, pos in participants:
            r = pre_ratings[pid]
            rd = pre_rds[pid]
            d2_inv = 0
            delta_sum = 0
            for oid, opos in participants:
                if pid == oid:
                    continue
                r_opp = pre_ratings[oid]
                rd_opp = pre_rds[oid]
                g_val = self.g(rd_opp)
                e_val = self.expected(r, r_opp, rd_opp)
                actual = 1.0 if pos < opos else (0.5 if pos == opos else 0.0)
                d2_inv += g_val ** 2 * e_val * (1 - e_val)
                delta_sum += g_val * (actual - e_val)
            if d2_inv == 0:
                continue
            d2 = 1.0 / (d2_inv / (n - 1))
            new_rd = 1.0 / np.sqrt(1 / rd ** 2 + 1 / d2)
            new_r = r + (new_rd ** 2) * delta_sum / (n - 1)
            self.ratings[pid] = new_r
            self.rds[pid] = new_rd
            self.last_race[pid] = period


def compute_elo_features(df):
    """Compute Elo/Glicko ratings for horse/jockey/trainer.
    H5 FIX: Pre-build race groups instead of boolean mask per race.
    """
    log_step("Computing Elo/Glicko ratings...")
    horse_elo = EloRatingSystem(k_factor=CONFIG["horse_elo_k"])
    horse_glicko = GlickoRatingSystem(
        initial_rd=CONFIG["glicko_initial_rd"], c=CONFIG["glicko_c"])
    jockey_elo = EloRatingSystem(k_factor=CONFIG["jockey_elo_k"])
    trainer_elo = EloRatingSystem(k_factor=CONFIG["trainer_elo_k"])

    horse_elo_col = np.full(len(df), np.nan)
    horse_glicko_col = np.full(len(df), np.nan)
    horse_glicko_rd_col = np.full(len(df), np.nan)
    jockey_elo_col = np.full(len(df), np.nan)
    trainer_elo_col = np.full(len(df), np.nan)

    # H5 FIX: Pre-build race groups for O(1) lookup instead of O(n) boolean mask
    race_groups = df.groupby("race_id", sort=False)
    race_order = df.drop_duplicates("race_id")["race_id"].values  # preserves chronological order

    period = 0
    for race_id in tqdm(race_order, desc="  [Elo] Computing", leave=False):
        race_df = race_groups.get_group(race_id)

        for idx, row in race_df.iterrows():
            hid = row.get("horse_id")
            jid = row.get("jockey_id")
            tid = row.get("trainer_id")
            if pd.notna(hid):
                horse_elo_col[idx] = horse_elo.get_rating(hid)
                horse_glicko_col[idx] = horse_glicko.get_rating(hid)
                horse_glicko_rd_col[idx] = horse_glicko.get_rd(hid, period)
            if pd.notna(jid):
                jockey_elo_col[idx] = jockey_elo.get_rating(jid)
            if pd.notna(tid):
                trainer_elo_col[idx] = trainer_elo.get_rating(tid)

        valid = race_df.dropna(subset=["finish_position"])
        if len(valid) >= 2:
            horse_parts = [(r["horse_id"], r["finish_position"])
                           for _, r in valid.iterrows() if pd.notna(r.get("horse_id"))]
            jockey_parts = [(r["jockey_id"], r["finish_position"])
                            for _, r in valid.iterrows() if pd.notna(r.get("jockey_id"))]
            trainer_parts = [(r["trainer_id"], r["finish_position"])
                             for _, r in valid.iterrows() if pd.notna(r.get("trainer_id"))]
            horse_elo.update_race(horse_parts)
            horse_glicko.update_race(horse_parts, period)
            jockey_elo.update_race(jockey_parts)
            trainer_elo.update_race(trainer_parts)
        period += 1

    df["horse_elo"] = horse_elo_col
    df["horse_glicko"] = horse_glicko_col
    df["horse_glicko_rd"] = horse_glicko_rd_col
    df["jockey_elo"] = jockey_elo_col
    df["trainer_elo"] = trainer_elo_col

    df["horse_elo_rank"] = df.groupby("race_id")["horse_elo"].rank(ascending=False)
    df["jockey_elo_rank"] = df.groupby("race_id")["jockey_elo"].rank(ascending=False)
    df["combined_elo"] = (df["horse_elo"] * CONFIG["elo_horse_weight"] +
                          df["jockey_elo"] * CONFIG["elo_jockey_weight"] +
                          df["trainer_elo"] * CONFIG["elo_trainer_weight"])
    df["combined_elo_rank"] = df.groupby("race_id")["combined_elo"].rank(ascending=False)

    log_step(f"Elo done: {len(horse_elo.ratings):,} horses / {len(jockey_elo.ratings):,} jockeys")
    return df


# ══════════════════════════════════════════════
# Part 2: Target Encoding (CV-safe, expanding mean)
# ══════════════════════════════════════════════
def target_encode_cv(df, col, target, n_splits=5, smoothing=20):
    """Time-series safe target encoding using expanding mean per fold."""
    from sklearn.model_selection import TimeSeriesSplit
    global_mean = df[target].mean()
    encoded = pd.Series(np.nan, index=df.index)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(df):
        train = df.iloc[train_idx]
        agg = train.groupby(col)[target].agg(["mean", "count"])
        smooth_mean = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
        encoded.iloc[val_idx] = df.iloc[val_idx][col].map(smooth_mean)
    encoded = encoded.fillna(global_mean)
    return encoded


# F15: Global dict to store target encoding lookup tables
_TARGET_ENCODING_LOOKUPS = {}


def add_target_encodings(df):
    log_step("Computing target encodings...")
    has_result = df["finish_position"].notna()
    df_wr = df[has_result].copy()
    df_wr["_is_win"] = (df_wr["finish_position"] == 1).astype(float)
    df_wr["_is_top3"] = (df_wr["finish_position"] <= 3).astype(float)

    targets = {"win": "_is_win", "top3": "_is_top3"}
    encode_cols = [c for c in ["jockey_id", "trainer_id", "venue"] if c in df.columns]

    for col in encode_cols:
        for tname, tcol in targets.items():
            new_col = f"te_{col}_{tname}"
            values = target_encode_cv(df_wr, col, tcol)
            df.loc[has_result, new_col] = values.values
            global_mean = df_wr[tcol].mean()
            df[new_col] = df[new_col].fillna(global_mean)
            # F15: Store lookup table for inference
            agg = df_wr.groupby(col)[tcol].agg(["mean", "count"])
            smoothing = 20
            smooth_mean = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
            _TARGET_ENCODING_LOOKUPS[new_col] = {
                "mapping": smooth_mean.to_dict(),
                "global_mean": float(global_mean),
            }

    log_step("Target encoding done")
    return df


# ══════════════════════════════════════════════
# Part 3: GOD-level Feature Engineering (C3: date-aware)
# ══════════════════════════════════════════════

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_god_ensemble_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _god_ensemble_dir not in sys.path:
    sys.path.insert(0, _god_ensemble_dir)
from god_ensemble import _GodEnsembleModel, _EnsembleLGBModel, TemperatureScaler


class PlattScaler:
    """F9: 2-parameter logistic (Platt) calibration: p = 1 / (1 + exp(a*x + b))"""
    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, y_true, y_pred):
        from scipy.optimize import minimize
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Convert to log-odds space
        f = np.log(y_pred / (1 - y_pred))

        def neg_log_likelihood(params):
            a, b = params
            p = 1.0 / (1.0 + np.exp(-(a * f + b)))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

        result = minimize(neg_log_likelihood, [1.0, 0.0], method='Nelder-Mead')
        self.a, self.b = result.x
        return self

    def transform(self, y_pred):
        y_pred = np.asarray(y_pred, dtype=float)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        f = np.log(y_pred / (1 - y_pred))
        return 1.0 / (1.0 + np.exp(-(self.a * f + self.b)))


try:
    from train_advanced import DataProcessor, FeatureEngine
except ImportError as e:
    print(f"\n  train_advanced.py not found.")
    print(f"  Place train_advanced.py in the same directory as train_god_v4.py.")
    print(f"  Error: {e}")
    sys.exit(1)


def add_god_features(df):
    """All additional feature engineering beyond train_ultimate.
    C3 FIX: All rolling/expanding features use date-aware functions.
    """
    log_step("GOD-level feature engineering...")

    # ── F4: odds=0 guard ──
    if "odds" in df.columns:
        df["odds"] = df["odds"].replace(0, np.nan)
        df["odds_missing"] = df["odds"].isna().astype(int)

    # ── 1. Feature Interactions ──
    log_step("Feature interactions...", 1)
    if "horse_weight" in df.columns and "distance" in df.columns:
        df["hw_x_dist"] = df["horse_weight"] * df["distance"]
        df["hw_per_dist"] = df["horse_weight"] / df["distance"].clip(lower=1)

    if "age" in df.columns and "distance" in df.columns:
        df["age_x_dist"] = df["age"] * df["distance"]

    if "weight" in df.columns and "horse_weight" in df.columns:
        df["weight_x_hw"] = df["weight"] * df["horse_weight"]
        hw_safe = df["horse_weight"].replace(0, np.nan)
        df["weight_ratio_hw"] = df["weight"] / hw_safe

    if "track_condition_enc" in df.columns and "distance" in df.columns:
        df["track_x_dist"] = df["track_condition_enc"] * df["distance"]

    if "odds" in df.columns and "popularity" in df.columns:
        df["odds_x_pop"] = df["odds"] * df["popularity"]

    # ── 2. Cyclical encoding for month / day_of_week ──
    log_step("Cyclical encoding...", 1)
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── 3. Relative features within race ──
    log_step("Race-relative features...", 1)
    if "odds" in df.columns:
        df["odds_rank_pct"] = df.groupby("race_id")["odds"].rank(pct=True)
        race_std = df.groupby("race_id")["odds"].transform("std").clip(lower=0.01)
        race_mean = df.groupby("race_id")["odds"].transform("mean")
        df["odds_zscore"] = (df["odds"] - race_mean) / race_std

    if "horse_weight" in df.columns:
        df["hw_rank_pct"] = df.groupby("race_id")["horse_weight"].rank(pct=True)
        race_std_hw = df.groupby("race_id")["horse_weight"].transform("std").clip(lower=0.01)
        race_mean_hw = df.groupby("race_id")["horse_weight"].transform("mean")
        df["hw_zscore"] = (df["horse_weight"] - race_mean_hw) / race_std_hw

    if "age" in df.columns:
        df["age_rank_pct"] = df.groupby("race_id")["age"].rank(pct=True)
        race_mean_age = df.groupby("race_id")["age"].transform("mean")
        df["age_vs_race_mean"] = df["age"] - race_mean_age

    if "weight" in df.columns:
        df["weight_rank_pct"] = df.groupby("race_id")["weight"].rank(pct=True)

    # ── 4. Bloodline statistics (C3: date-aware) ──
    log_step("Bloodline statistics...", 1)
    df = _bloodline_stats(df)

    # ── 5. Pace analysis (C3: date-aware) ──
    log_step("Pace analysis...", 1)
    df = _pace_features(df)

    # ── 6. Class transition ──
    log_step("Class transition features...", 1)
    df = _class_transition_features(df)

    # ── 7. Combination stats (C3: date-aware) ──
    log_step("Combination statistics...", 1)
    df = _combination_stats(df)

    # ── 8. Jockey-course interaction (C3: date-aware) ──
    if "jockey_id" in df.columns and "venue" in df.columns:
        combo_key = df["jockey_id"].astype(str) + "_" + df["venue"].astype(str)
        df["_jv_combo"] = combo_key
        df["jockey_venue_avg_pos"] = apply_safe_expanding(
            df, "_jv_combo", "finish_position", func='mean')
        df.drop(columns=["_jv_combo"], inplace=True)

    # ── F7: New domain features ──
    log_step("F7: New domain features...", 1)

    # Jockey change detection
    if "horse_id" in df.columns and "jockey_id" in df.columns:
        prev_jockey = df.groupby("horse_id")["jockey_id"].shift(1)
        df["jockey_changed"] = (df["jockey_id"] != prev_jockey).astype(int)
        df.loc[prev_jockey.isna(), "jockey_changed"] = 0  # first race = no change
        # Jockey Elo delta (new jockey minus previous jockey)
        if "jockey_elo" in df.columns:
            prev_jockey_elo = df.groupby("horse_id")["jockey_elo"].shift(1)
            df["jockey_elo_delta"] = df["jockey_elo"] - prev_jockey_elo

    # Distance change
    if "horse_id" in df.columns and "distance" in df.columns:
        prev_dist = df.groupby("horse_id")["distance"].shift(1)
        df["distance_change"] = df["distance"] - prev_dist
        df["is_distance_shortening"] = (df["distance_change"] < 0).astype(int)
        df["is_distance_extending"] = (df["distance_change"] > 0).astype(int)

    # Formal race class from race_name
    if "race_name" in df.columns:
        import re
        class_map = {"新馬": 1, "未勝利": 2, "1勝": 3, "2勝": 4, "3勝": 5,
                     "オープン": 6, "OP": 6, "リステッド": 7, "L": 7,
                     "G3": 8, "G2": 9, "G1": 10, "GI": 10, "GII": 9, "GIII": 8}
        def _parse_class(name):
            if pd.isna(name):
                return np.nan
            name = str(name)
            for pattern, val in class_map.items():
                if pattern in name:
                    return val
            return np.nan
        df["race_class_formal"] = df["race_name"].apply(_parse_class)
        if "horse_id" in df.columns:
            prev_class = df.groupby("horse_id")["race_class_formal"].shift(1)
            df["class_change_formal"] = df["race_class_formal"] - prev_class

    # Debut detection and horse race count
    if "horse_id" in df.columns:
        df["horse_race_count"] = df.groupby("horse_id").cumcount()
        df["is_debut"] = (df["horse_race_count"] == 0).astype(int)

    # Field pace prediction
    if "horse_id" in df.columns and "horse_avg_early_pace" in df.columns:
        # Count front runners (pace < 0.25) in each race
        df["_is_front_runner"] = (df["horse_avg_early_pace"] < 0.25).astype(float)
        df["race_n_front_runners"] = df.groupby("race_id")["_is_front_runner"].transform("sum")
        # Predicted pace: mean early pace of field
        df["predicted_pace"] = df.groupby("race_id")["horse_avg_early_pace"].transform("mean")
        # Pace advantage: front runners benefit from slow pace (few front runners)
        df["pace_advantage"] = np.where(
            df["horse_avg_early_pace"] < 0.3,  # front-running horses
            1.0 / df["race_n_front_runners"].clip(lower=1),  # fewer rivals = bigger advantage
            0.0
        )
        df.drop(columns=["_is_front_runner"], inplace=True, errors='ignore')

    log_step("GOD feature engineering done")
    return df


# F15: Global dict to store bloodline lookup tables
_BLOODLINE_STATS_LOOKUPS = {}


def _bloodline_stats(df):
    """Sire/Dam win rates by distance/surface/condition.
    C3 FIX: Uses safe_expanding (date-aware) instead of shift(1).
    """
    if "sire" not in df.columns:
        return df

    has_date = "race_date" in df.columns

    for parent in ["sire", "dam"]:
        if parent not in df.columns:
            continue

        if has_date:
            df[f"{parent}_win_rate"] = apply_safe_expanding(
                df, parent, "finish_position", func='win_rate')
            df[f"{parent}_top3_rate"] = apply_safe_expanding(
                df, parent, "finish_position", func='top3_rate')
            df[f"{parent}_avg_pos"] = apply_safe_expanding(
                df, parent, "finish_position", func='mean')
            df[f"{parent}_n_runners"] = apply_safe_expanding(
                df, parent, "finish_position", func='count')
        else:
            # Fallback: shift(1) if no date column
            g = df.groupby(parent)["finish_position"]
            df[f"{parent}_win_rate"] = g.transform(
                lambda x: (x.shift(1) == 1).expanding().mean())
            df[f"{parent}_top3_rate"] = g.transform(
                lambda x: (x.shift(1) <= 3).expanding().mean())
            df[f"{parent}_avg_pos"] = g.transform(
                lambda x: x.shift(1).expanding().mean())
            df[f"{parent}_n_runners"] = g.transform(
                lambda x: x.shift(1).expanding().count())

        if "distance" in df.columns:
            dist_cat = pd.cut(df["distance"], bins=[0, 1400, 1800, 2200, 9999],
                              labels=False).fillna(0).astype(int)
            df["_tmp_dist_cat"] = dist_cat
            combo_key = df[parent].astype(str) + "_d" + df["_tmp_dist_cat"].astype(str)
            df["_tmp_pd_combo"] = combo_key
            if has_date:
                df[f"{parent}_dist_win_rate"] = apply_safe_expanding(
                    df, "_tmp_pd_combo", "finish_position", func='win_rate')
            else:
                gd = df.groupby("_tmp_pd_combo")["finish_position"]
                df[f"{parent}_dist_win_rate"] = gd.transform(
                    lambda x: (x.shift(1) == 1).expanding().mean())
            df.drop(columns=["_tmp_dist_cat", "_tmp_pd_combo"], inplace=True)

        if "course_type_enc" in df.columns:
            combo_key = df[parent].astype(str) + "_s" + df["course_type_enc"].astype(str)
            df["_tmp_ps_combo"] = combo_key
            if has_date:
                df[f"{parent}_surface_win_rate"] = apply_safe_expanding(
                    df, "_tmp_ps_combo", "finish_position", func='win_rate')
            else:
                gs = df.groupby("_tmp_ps_combo")["finish_position"]
                df[f"{parent}_surface_win_rate"] = gs.transform(
                    lambda x: (x.shift(1) == 1).expanding().mean())
            df.drop(columns=["_tmp_ps_combo"], inplace=True)

    if "dam_sire" in df.columns:
        if has_date:
            df["dam_sire_win_rate"] = apply_safe_expanding(
                df, "dam_sire", "finish_position", func='win_rate')
            df["dam_sire_avg_pos"] = apply_safe_expanding(
                df, "dam_sire", "finish_position", func='mean')
        else:
            g = df.groupby("dam_sire")["finish_position"]
            df["dam_sire_win_rate"] = g.transform(
                lambda x: (x.shift(1) == 1).expanding().mean())
            df["dam_sire_avg_pos"] = g.transform(
                lambda x: x.shift(1).expanding().mean())

    if "sire_sire" in df.columns:
        if has_date:
            df["sire_sire_win_rate"] = apply_safe_expanding(
                df, "sire_sire", "finish_position", func='win_rate')
        else:
            g = df.groupby("sire_sire")["finish_position"]
            df["sire_sire_win_rate"] = g.transform(
                lambda x: (x.shift(1) == 1).expanding().mean())

    # F15: Store final bloodline stats for lookup
    has_result = df["finish_position"].notna()
    for parent in ["sire", "dam"]:
        if parent in df.columns:
            agg = df.loc[has_result].groupby(parent)["finish_position"].agg(["mean", "count"])
            _BLOODLINE_STATS_LOOKUPS[parent] = {
                "avg_pos": agg["mean"].to_dict(),
                "n_runners": agg["count"].to_dict(),
            }

    return df


def _pace_features(df):
    """Pace analysis from passing orders and running style classification.
    C3 FIX: Historical averages use date-aware expanding.
    """
    corner_cols = [c for c in ["corner_1", "corner_2", "corner_3", "corner_4"] if c in df.columns]
    if not corner_cols:
        return df

    if "corner_1" in df.columns and "corner_4" in df.columns:
        c1 = pd.to_numeric(df["corner_1"], errors="coerce")
        c4 = pd.to_numeric(df["corner_4"], errors="coerce")
        df["_tmp_position_change_c1c4"] = c1 - c4

    if "corner_1" in df.columns:
        c1 = pd.to_numeric(df["corner_1"], errors="coerce")
        n_run = df.groupby("race_id")["horse_number"].transform("count")
        df["_tmp_early_pace_pct"] = c1 / n_run.clip(lower=1)

        pct = df["_tmp_early_pace_pct"]
        df["_tmp_running_style"] = pd.cut(pct, bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
                                          labels=[0, 1, 2, 3]).astype(float)

    has_date = "race_date" in df.columns

    if "horse_id" in df.columns and "_tmp_early_pace_pct" in df.columns:
        if has_date:
            df["horse_avg_early_pace"] = apply_safe_expanding(
                df, "horse_id", "_tmp_early_pace_pct", func='mean')
        else:
            g = df.groupby("horse_id")["_tmp_early_pace_pct"]
            df["horse_avg_early_pace"] = g.transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        # C3 FIX: Use date-aware expanding for consistency too
        if has_date:
            # Expanding std approximation: use expanding mean of squared values minus square of expanding mean
            pace_mean = apply_safe_expanding(df, "horse_id", "_tmp_early_pace_pct", func='mean')
            # For std, compute expanding mean of value^2
            df["_tmp_pace_sq"] = df["_tmp_early_pace_pct"] ** 2
            pace_sq_mean = apply_safe_expanding(df, "horse_id", "_tmp_pace_sq", func='mean')
            df["horse_pace_consistency"] = np.sqrt(np.clip(pace_sq_mean - pace_mean ** 2, 0, None))
            df.drop(columns=["_tmp_pace_sq"], inplace=True, errors='ignore')
        else:
            df["horse_pace_consistency"] = df.groupby("horse_id")["_tmp_early_pace_pct"].transform(
                lambda x: x.shift(1).rolling(5, min_periods=2).std())

    if "horse_id" in df.columns and "_tmp_running_style" in df.columns:
        # C3 FIX: Use date-aware expanding mean as proxy for modal style
        if has_date:
            df["horse_modal_style"] = apply_safe_expanding(
                df, "horse_id", "_tmp_running_style", func='mean')
            # Round to nearest integer to approximate mode
            df["horse_modal_style"] = df["horse_modal_style"].round()
        else:
            df["horse_modal_style"] = df.groupby("horse_id")["_tmp_running_style"].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).apply(
                    lambda v: v.mode().iloc[0] if len(v.mode()) > 0 else np.nan, raw=False))

    if "horse_id" in df.columns and "_tmp_position_change_c1c4" in df.columns:
        if has_date:
            df["horse_avg_pos_change"] = apply_safe_expanding(
                df, "horse_id", "_tmp_position_change_c1c4", func='mean')
        else:
            df["horse_avg_pos_change"] = df.groupby("horse_id")["_tmp_position_change_c1c4"].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    # Drop intermediate columns
    tmp_cols = [c for c in df.columns if c.startswith("_tmp_")]
    df.drop(columns=tmp_cols, inplace=True)

    return df


def _class_transition_features(df):
    """Class step up/down features.
    C3 FIX: Uses date-aware expanding instead of shift(1).
    """
    if "horse_id" not in df.columns:
        return df

    has_date = "race_date" in df.columns

    if "odds" in df.columns:
        race_level = df.groupby("race_id")["odds"].transform("median")
        df["race_class_proxy"] = race_level

        if has_date:
            # Use date-aware expanding mean of class proxy as "previous class" estimate
            prev_class_mean = apply_safe_expanding(
                df, "horse_id", "race_class_proxy", func='mean')
            df["class_change"] = df["race_class_proxy"] - prev_class_mean
            df["class_step_up"] = (df["class_change"] < 0).astype(int)
            df["class_step_down"] = (df["class_change"] > 0).astype(int)
            # class_momentum_3: use expanding mean of class_change as proxy
            df["class_momentum_3"] = apply_safe_expanding(
                df, "horse_id", "class_change", func='mean')
        else:
            prev_class = df.groupby("horse_id")["race_class_proxy"].shift(1)
            df["class_change"] = df["race_class_proxy"] - prev_class
            df["class_step_up"] = (df["class_change"] < 0).astype(int)
            df["class_step_down"] = (df["class_change"] > 0).astype(int)
            df["class_momentum_3"] = df.groupby("horse_id")["class_change"].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    return df


def _combination_stats(df):
    """Jockey-trainer combo, horse-distance combo, horse-venue combo.
    C3 FIX: Uses date-aware expanding instead of shift(1).
    """
    has_date = "race_date" in df.columns

    if "jockey_id" in df.columns and "trainer_id" in df.columns:
        combo_key = df["jockey_id"].astype(str) + "_" + df["trainer_id"].astype(str)
        df["_jt_combo"] = combo_key
        if has_date:
            df["jt_combo_avg_pos"] = apply_safe_expanding(
                df, "_jt_combo", "finish_position", func='mean')
            df["jt_combo_win_rate"] = apply_safe_expanding(
                df, "_jt_combo", "finish_position", func='win_rate')
            df["jt_combo_n_races"] = apply_safe_expanding(
                df, "_jt_combo", "finish_position", func='count')
        else:
            g = df.groupby("_jt_combo")["finish_position"]
            df["jt_combo_avg_pos"] = g.transform(lambda x: x.shift(1).expanding().mean())
            df["jt_combo_win_rate"] = g.transform(lambda x: (x.shift(1) == 1).expanding().mean())
            df["jt_combo_n_races"] = g.transform(lambda x: x.shift(1).expanding().count())
        df.drop(columns=["_jt_combo"], inplace=True)

    if "horse_id" in df.columns and "distance" in df.columns:
        dist_bucket = (df["distance"] // 200 * 200).astype(int)
        combo_key = df["horse_id"].astype(str) + "_" + dist_bucket.astype(str)
        df["_hd_combo"] = combo_key
        if has_date:
            df["horse_dist_combo_avg"] = apply_safe_expanding(
                df, "_hd_combo", "finish_position", func='mean')
            df["horse_dist_combo_win"] = apply_safe_expanding(
                df, "_hd_combo", "finish_position", func='win_rate')
        else:
            g = df.groupby("_hd_combo")["finish_position"]
            df["horse_dist_combo_avg"] = g.transform(lambda x: x.shift(1).expanding().mean())
            df["horse_dist_combo_win"] = g.transform(lambda x: (x.shift(1) == 1).expanding().mean())
        df.drop(columns=["_hd_combo"], inplace=True)

    if "horse_id" in df.columns and "venue" in df.columns:
        combo_key = df["horse_id"].astype(str) + "_" + df["venue"].astype(str)
        df["_hv_combo"] = combo_key
        if has_date:
            df["horse_venue_avg_pos"] = apply_safe_expanding(
                df, "_hv_combo", "finish_position", func='mean')
            df["horse_venue_win_rate"] = apply_safe_expanding(
                df, "_hv_combo", "finish_position", func='win_rate')
        else:
            g = df.groupby("_hv_combo")["finish_position"]
            df["horse_venue_avg_pos"] = g.transform(lambda x: x.shift(1).expanding().mean())
            df["horse_venue_win_rate"] = g.transform(lambda x: (x.shift(1) == 1).expanding().mean())
        df.drop(columns=["_hv_combo"], inplace=True)

    return df


# ══════════════════════════════════════════════
# Part 4: Time Decay Weights
# ══════════════════════════════════════════════
def compute_time_decay_weights(df, half_life_days=180):
    """Exponential decay weights: recent races matter more"""
    if "race_date" not in df.columns:
        return np.ones(len(df))
    max_date = df["race_date"].max()
    days_ago = (max_date - df["race_date"]).dt.days.fillna(0).values.astype(float)
    decay = np.exp(-np.log(2) * days_ago / half_life_days)
    decay = decay / decay.mean()
    return decay


# ══════════════════════════════════════════════
# Part 5: Adversarial Validation (H3: dead code removed)
# ══════════════════════════════════════════════
def adversarial_validation(X_train, X_test, feature_names):
    """Train/test distribution shift detection.
    H3 FIX: Removed adversarial debiasing dead code. Only AUC reporting remains.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    log_step("Adversarial Validation (year/month excluded)...")

    exclude_idx = []
    for i, fname in enumerate(feature_names):
        if fname in ADVVAL_EXCLUDE_FEATURES:
            exclude_idx.append(i)

    keep_idx = [i for i in range(len(feature_names)) if i not in exclude_idx]
    keep_names = [feature_names[i] for i in keep_idx]
    X_tr_clean = X_train[:, keep_idx]
    X_te_clean = X_test[:, keep_idx]

    log_step(f"Excluded {len(exclude_idx)} temporal features for advval", 1)

    y_adv = np.concatenate([np.zeros(len(X_tr_clean)), np.ones(len(X_te_clean))])
    X_adv = np.vstack([X_tr_clean, X_te_clean])

    n = len(X_adv)
    idx = np.random.RandomState(42).permutation(n)
    split = int(n * 0.8)
    tr_idx, va_idx = idx[:split], idx[split:]

    dtrain = lgb.Dataset(X_adv[tr_idx], label=y_adv[tr_idx])
    dval = lgb.Dataset(X_adv[va_idx], label=y_adv[va_idx])

    params = {
        "objective": "binary", "metric": "auc", "verbose": -1,
        "num_leaves": 31, "learning_rate": 0.1, "n_jobs": -1,
    }
    model = lgb.train(params, dtrain, num_boost_round=200,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

    pred = model.predict(X_adv[va_idx])
    auc = roc_auc_score(y_adv[va_idx], pred)

    imp = model.feature_importance(importance_type="gain")
    top_drift = sorted(zip(keep_names, imp), key=lambda x: -x[1])[:20]

    if auc > 0.6:
        log_step(f"AUC = {auc:.4f} - Distribution shift detected!", 1)
        log_step(f"Top drift features: {[f[0] for f in top_drift[:5]]}", 1)
    else:
        log_step(f"AUC = {auc:.4f} - Distribution stable", 1)

    return auc, top_drift


# ══════════════════════════════════════════════
# Part 6: Boruta Feature Selection
# ══════════════════════════════════════════════
def is_protected_feature(fname):
    for pat in PROTECTED_FEATURE_PATTERNS:
        if pat in fname:
            return True
    return False


def boruta_selection(X, y, feature_names, max_iter=100, min_features=30):
    """Boruta with min_features enforced and protected domain features."""
    import lightgbm as lgb
    log_step(f"Boruta selection ({len(feature_names)} features, {max_iter} iters)...")

    exclude_idx = []
    for i, fname in enumerate(feature_names):
        if fname in ADVVAL_EXCLUDE_FEATURES:
            exclude_idx.append(i)

    keep_idx = [i for i in range(len(feature_names)) if i not in exclude_idx]
    X_boruta = X[:, keep_idx]
    boruta_names = [feature_names[i] for i in keep_idx]
    n_features = X_boruta.shape[1]

    log_step(f"Boruta input: {n_features} features (excluded {len(exclude_idx)} temporal)", 1)

    hit_counts = np.zeros(n_features)

    for iteration in tqdm(range(max_iter), desc="  [Boruta]", leave=False):
        X_shadow = X_boruta.copy()
        rng = np.random.RandomState(iteration)
        for col_idx in range(n_features):
            X_shadow[:, col_idx] = rng.permutation(X_shadow[:, col_idx])

        X_combined = np.hstack([X_boruta, X_shadow])
        dtrain = lgb.Dataset(X_combined, label=y)
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "boosting_type": "gbdt", "num_leaves": 63, "learning_rate": 0.1,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "verbose": -1, "n_jobs": -1, "seed": iteration,
        }
        model = lgb.train(params, dtrain, num_boost_round=100)
        importances = model.feature_importance(importance_type="gain")
        real_imp = importances[:n_features]
        shadow_imp = importances[n_features:]
        shadow_max = shadow_imp.max() if len(shadow_imp) > 0 else 0
        hit_counts += (real_imp > shadow_max).astype(int)

    # F8: BH correction instead of raw p < 0.05
    p_values = np.array([1 - stats.binom.cdf(hit_counts[i] - 1, max_iter, 0.5)
                         for i in range(n_features)])
    try:
        from statsmodels.stats.multitest import multipletests
        rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        selected_mask_boruta = rejected
        log_step("F8: Boruta using BH-corrected p-values", 1)
    except ImportError:
        log_step("statsmodels not available, falling back to raw p < 0.05", 1)
        selected_mask_boruta = p_values < 0.05

    full_mask = np.zeros(len(feature_names), dtype=bool)
    for bi, fi in enumerate(keep_idx):
        if selected_mask_boruta[bi]:
            full_mask[fi] = True

    protected_added = []
    for i, fname in enumerate(feature_names):
        if is_protected_feature(fname) and not full_mask[i]:
            full_mask[i] = True
            protected_added.append(fname)

    if full_mask.sum() < min_features:
        hit_order = np.argsort(-hit_counts)
        for bi in hit_order:
            fi = keep_idx[bi]
            if not full_mask[fi]:
                full_mask[fi] = True
            if full_mask.sum() >= min_features:
                break

    selected = [f for f, s in zip(feature_names, full_mask) if s]
    rejected = [f for f, s in zip(feature_names, full_mask) if not s]

    log_step(f"Boruta result: {len(selected)} selected / {len(rejected)} rejected", 1)
    if protected_added:
        log_step(f"Protected features forced in: {len(protected_added)}", 1)

    return selected, full_mask


# ══════════════════════════════════════════════
# Part 7: Hybrid Feature Selection
# ══════════════════════════════════════════════
class _LGBScorerWrapper:
    def __init__(self, m):
        self.m = m
    def fit(self, X, y):
        return self
    def predict(self, X):
        return self.m.predict(X)
    def score(self, X, y):
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(y, self.m.predict(X))
        except ValueError:
            return 0.5


def hybrid_feature_selection(X, y, feature_names, boruta_mask, n_folds=5):
    """Union of Boruta + SHAP importance + Permutation importance with stability selection."""
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit

    log_step("Hybrid feature selection (Boruta + SHAP + Permutation)...")

    tscv = TimeSeriesSplit(n_splits=n_folds)
    shap_important = np.zeros(len(feature_names))
    perm_important = np.zeros(len(feature_names))

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "num_leaves": 63, "learning_rate": 0.05, "verbose": -1, "n_jobs": -1,
        }
        model = lgb.train(params, dtrain, num_boost_round=500,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        try:
            import shap
            explainer = shap.TreeExplainer(model)
            sample_size = min(2000, len(X_va))
            shap_vals = explainer.shap_values(X_va[:sample_size])
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            shap_imp = np.abs(shap_vals).mean(axis=0)
            threshold = np.percentile(shap_imp, 50)
            shap_important += (shap_imp > threshold).astype(int)
        except Exception:
            gain_imp = model.feature_importance(importance_type="gain")
            threshold = np.percentile(gain_imp, 50)
            shap_important += (gain_imp > threshold).astype(int)

        try:
            from sklearn.inspection import permutation_importance
            wrapper = _LGBScorerWrapper(model)
            perm_result = permutation_importance(wrapper, X_va, y_va,
                                                 n_repeats=5, random_state=42, n_jobs=-1,
                                                 scoring="roc_auc")
            perm_imp = perm_result.importances_mean
            threshold = np.percentile(perm_imp[perm_imp > 0], 50) if (perm_imp > 0).any() else 0
            perm_important += (perm_imp > threshold).astype(int)
        except Exception:
            pass

    stability_threshold = CONFIG["stability_threshold"]
    shap_stable = (shap_important / n_folds) >= stability_threshold
    perm_stable = (perm_important / n_folds) >= stability_threshold

    hybrid_mask = boruta_mask | shap_stable | perm_stable

    for i, fname in enumerate(feature_names):
        if is_protected_feature(fname):
            hybrid_mask[i] = True

    if hybrid_mask.sum() < CONFIG["boruta_min_features"]:
        combined_score = shap_important + perm_important
        order = np.argsort(-combined_score)
        for idx in order:
            if not hybrid_mask[idx]:
                hybrid_mask[idx] = True
            if hybrid_mask.sum() >= CONFIG["boruta_min_features"]:
                break

    selected = [f for f, s in zip(feature_names, hybrid_mask) if s]
    log_step(f"Hybrid selection: {len(selected)} features "
             f"(Boruta:{boruta_mask.sum()}, SHAP-stable:{shap_stable.sum()}, "
             f"Perm-stable:{perm_stable.sum()})", 1)

    return selected, hybrid_mask


# ══════════════════════════════════════════════
# Part 8: Purged K-Fold with Embargo
# ══════════════════════════════════════════════
class PurgedTimeSeriesSplit:
    def __init__(self, n_splits=5, purge_days=7, embargo_days=7):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, X, dates=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)

        dates_ns = None
        if dates is not None and len(dates) > 0:
            try:
                dates_ns = pd.to_datetime(pd.Series(dates), errors="coerce").values.astype("int64")
                purge_ns = int(pd.Timedelta(days=self.purge_days).value)
                embargo_ns = int(pd.Timedelta(days=self.embargo_days).value)
            except Exception:
                dates_ns = None

        for i in range(self.n_splits):
            val_start = fold_size * (i + 1)
            val_end = fold_size * (i + 2) if i < self.n_splits - 1 else n

            train_idx = np.arange(0, val_start)
            val_idx = np.arange(val_start, val_end)

            if dates_ns is not None:
                try:
                    val_start_ns = dates_ns[val_start]
                    if val_start_ns > 0:
                        purge_cutoff_ns = val_start_ns - purge_ns
                        train_mask = dates_ns[train_idx] < purge_cutoff_ns
                        train_idx = train_idx[train_mask]
                        if self.embargo_days > 0:
                            embargo_end_ns = val_start_ns + embargo_ns
                            val_mask = dates_ns[val_idx] >= embargo_end_ns
                            val_idx = val_idx[val_mask]
                except Exception:
                    pass

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx


# ══════════════════════════════════════════════
# Part 9: Optuna Optimization
# ══════════════════════════════════════════════
def optimize_with_pruning(X_train, y_train, X_val, y_val, task, n_trials=100,
                          use_gpu=False, sample_weights_tr=None):
    """Optuna + Median Pruning. Reduced to 100 trials."""
    import optuna
    import lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "regression" if task == "regression" else "binary",
            "metric": "l1" if task == "regression" else "auc",
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "verbose": -1, "n_jobs": -1, "seed": 42,
            "num_leaves": trial.suggest_int("num_leaves", 31, 512),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            # F11: max_depth removed - num_leaves alone controls tree complexity
            "max_depth": -1,
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 2.0),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
        }

        if params["boosting_type"] == "dart":
            params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.5)
            params["max_drop"] = trial.suggest_int("max_drop", 5, 50)

        if use_gpu:
            params["device"] = "gpu"
            params["gpu_use_dp"] = True

        dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        pruning_metric = "l1" if task == "regression" else "auc"
        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial, pruning_metric
        )

        model = lgb.train(
            params, dtrain, num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0), pruning_callback],
        )

        if task == "regression":
            from sklearn.metrics import mean_absolute_error
            return mean_absolute_error(y_val, model.predict(X_val))
        else:
            from sklearn.metrics import roc_auc_score
            try:
                return roc_auc_score(y_val, model.predict(X_val))
            except ValueError:
                return 0.5

    direction = "minimize" if task == "regression" else "maximize"
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True,
                                         warn_independent_sampling=False)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log_step(f"Best score: {study.best_value:.4f}", 1)
    pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    log_step(f"Completed/Pruned: {len(study.trials)}/{pruned}", 1)
    return study.best_params


# ══════════════════════════════════════════════
# Part 10: TabNet
# ══════════════════════════════════════════════
def train_tabnet(X_train, y_train, X_val, y_val, task, use_gpu=False):
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
    except ImportError:
        log_step("pytorch-tabnet not installed, skipping TabNet", 1)
        return None

    log_step("Training TabNet...", 1)
    try:
        device = "cuda" if use_gpu else "cpu"
        if task == "regression":
            model = TabNetRegressor(
                n_d=32, n_a=32, n_steps=5, gamma=1.5, lambda_sparse=1e-4,
                optimizer_params=dict(lr=2e-2), mask_type="entmax",
                verbose=0, device_name=device, seed=42,
            )
        else:
            model = TabNetClassifier(
                n_d=32, n_a=32, n_steps=5, gamma=1.5, lambda_sparse=1e-4,
                optimizer_params=dict(lr=2e-2), mask_type="entmax",
                verbose=0, device_name=device, seed=42,
            )

        X_tr = X_train.astype(np.float32)
        X_va = X_val.astype(np.float32)
        if task == "regression":
            y_tr = y_train.reshape(-1, 1).astype(np.float32)
            y_va = y_val.reshape(-1, 1).astype(np.float32)
        else:
            y_tr = y_train.astype(np.int64)
            y_va = y_val.astype(np.int64)

        model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)],
            max_epochs=CONFIG["tabnet_epochs"], patience=CONFIG["tabnet_patience"],
            batch_size=4096, virtual_batch_size=512, drop_last=False,
        )
        return model
    except Exception as e:
        log_step(f"TabNet training failed: {e}", 1)
        return None


def predict_tabnet(model, X, task):
    if model is None:
        return None
    try:
        X = X.astype(np.float32)
        if task == "regression":
            return model.predict(X).flatten()
        else:
            return model.predict_proba(X)[:, 1]
    except Exception:
        return None


# ══════════════════════════════════════════════
# Part 11: LambdaRank
# ══════════════════════════════════════════════
def train_lambdarank(X_train, y_train, groups_train, X_val, y_val, groups_val,
                     params_base, use_gpu=False):
    import lightgbm as lgb

    log_step("Training LambdaRank...", 1)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": CONFIG["lambdarank_ndcg_at"],
        "verbose": -1, "n_jobs": -1, "seed": 42,
        "num_leaves": params_base.get("num_leaves", 63),
        "learning_rate": params_base.get("learning_rate", 0.05),
        "feature_fraction": params_base.get("feature_fraction", 0.8),
        "bagging_fraction": params_base.get("bagging_fraction", 0.8),
        "bagging_freq": params_base.get("bagging_freq", 5),
        "min_child_samples": params_base.get("min_child_samples", 20),
        "reg_alpha": params_base.get("reg_alpha", 0.1),
        "reg_lambda": params_base.get("reg_lambda", 0.1),
        "max_depth": params_base.get("max_depth", 8),
    }
    if use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = True

    y_rel_train = np.where(y_train == 1, 10, np.where(y_train <= 3, 5,
                  np.where(y_train <= 5, 3, np.where(y_train <= 10, 1, 0))))
    y_rel_val = np.where(y_val == 1, 10, np.where(y_val <= 3, 5,
                np.where(y_val <= 5, 3, np.where(y_val <= 10, 1, 0))))

    dtrain = lgb.Dataset(X_train, label=y_rel_train, group=groups_train)
    dval = lgb.Dataset(X_val, label=y_rel_val, group=groups_val, reference=dtrain)

    try:
        model = lgb.train(
            params, dtrain, num_boost_round=CONFIG["lambdarank_boost_rounds"],
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(CONFIG["lambdarank_early_stopping"]),
                       lgb.log_evaluation(0)],
        )
        log_step("LambdaRank training complete", 1)
        return model
    except Exception as e:
        log_step(f"LambdaRank failed: {e}", 1)
        return None


# ══════════════════════════════════════════════
# Part 12: 2-Level Stacking Ensemble (C2, H2 fixes)
# ══════════════════════════════════════════════
def train_god_stacking(X, y, task, race_ids, race_dates, models_config,
                       n_folds=8, use_gpu=False, sample_weights=None,
                       noise_std=0.0):
    """
    2-Level Stacking with Purged CV:
      Level 1: LGB + XGB + CatBoost + TabNet -> OOF predictions
      Level 2: LGB meta-learner on OOF
      Level 3: Ridge/LogReg final blend

    H2 FIX: Noise injection only on training folds, not validation.
    C2 FIX: Do NOT refit L2/L3 on full data. Save train+optimize version.
    """
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.linear_model import Ridge, LogisticRegression

    log_step(f"=== GOD Stacking ({task}) ===")

    dates_array = race_dates if race_dates is not None else None
    ptscv = PurgedTimeSeriesSplit(
        n_splits=n_folds, purge_days=CONFIG["purge_days"], embargo_days=CONFIG["embargo_days"]
    )
    splits = list(ptscv.split(X, dates=dates_array))

    if not splits:
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_folds)
        splits = list(tscv.split(X))

    n_models = len(models_config)
    oof_preds = np.full((len(X), n_models), np.nan)
    fold_models = {name: [] for name, _ in models_config}
    fold_scores = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = sample_weights[tr_idx] if sample_weights is not None else None

        if task == "binary" and (len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2):
            continue

        # H2 FIX: Apply noise ONLY to training fold, NOT validation
        if noise_std > 0:
            rng = np.random.RandomState(42 + fold_idx)
            noise = rng.normal(0, noise_std, X_tr.shape)
            mask = X_tr != 0
            X_tr_noisy = X_tr.copy()
            X_tr_noisy[mask] += noise[mask] * np.abs(X_tr[mask])
        else:
            X_tr_noisy = X_tr

        fold_preds = np.full((len(va_idx), n_models), np.nan)

        for model_idx, (model_name, model_params) in enumerate(models_config):
            try:
                if model_name == "lightgbm":
                    dtrain = lgb.Dataset(X_tr_noisy, label=y_tr, weight=w_tr)
                    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
                    params = model_params.copy()
                    if use_gpu:
                        params["device"] = "gpu"
                        params["gpu_use_dp"] = True
                    model = lgb.train(
                        params, dtrain, num_boost_round=3000,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
                    )
                    pred = model.predict(X_va)

                elif model_name == "xgboost":
                    params = model_params.copy()
                    es_rounds = params.pop("early_stopping_rounds", 100)
                    if use_gpu:
                        params["tree_method"] = "hist"
                        params["device"] = "cuda"
                    if task == "regression":
                        m = xgb.XGBRegressor(**params, early_stopping_rounds=es_rounds)
                    else:
                        m = xgb.XGBClassifier(**params, early_stopping_rounds=es_rounds)
                    m.fit(X_tr_noisy, y_tr, eval_set=[(X_va, y_va)],
                          sample_weight=w_tr, verbose=False)
                    pred = m.predict(X_va) if task == "regression" else m.predict_proba(X_va)[:, 1]
                    model = m

                elif model_name == "catboost":
                    try:
                        from catboost import CatBoostRegressor, CatBoostClassifier
                        params = model_params.copy()
                        # F10: Detect categorical feature indices for CatBoost
                        cat_feature_indices = params.pop("cat_feature_indices", None)
                        if use_gpu:
                            params["task_type"] = "GPU"
                        if task == "regression":
                            m = CatBoostRegressor(**params)
                        else:
                            m = CatBoostClassifier(**params)
                        fit_kwargs = dict(eval_set=(X_va, y_va),
                                          sample_weight=w_tr, verbose=0,
                                          early_stopping_rounds=100)
                        if cat_feature_indices:
                            fit_kwargs["cat_features"] = cat_feature_indices
                        m.fit(X_tr_noisy, y_tr, **fit_kwargs)
                        pred = m.predict(X_va) if task == "regression" else m.predict_proba(X_va)[:, 1]
                        model = m
                    except ImportError:
                        log_step("CatBoost not installed, using LGB substitute", 2)
                        dtrain = lgb.Dataset(X_tr_noisy, label=y_tr, weight=w_tr)
                        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
                        params2 = models_config[0][1].copy()
                        params2["seed"] = 99
                        if use_gpu:
                            params2["device"] = "gpu"
                            params2["gpu_use_dp"] = True
                        model = lgb.train(params2, dtrain, num_boost_round=3000,
                                          valid_sets=[dval],
                                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                        pred = model.predict(X_va)

                elif model_name == "tabnet":
                    model = train_tabnet(X_tr_noisy, y_tr, X_va, y_va, task, use_gpu)
                    pred = predict_tabnet(model, X_va, task)
                    if pred is None:
                        pred = np.full(len(X_va), np.nan)
                else:
                    continue

                fold_models[model_name].append(model)
                fold_preds[:, model_idx] = pred

            except Exception as e:
                log_step(f"Model {model_name} fold {fold_idx} failed: {e}", 2)
                fold_preds[:, model_idx] = np.nan

        oof_preds[va_idx] = fold_preds

        # Fold score
        valid_cols = ~np.isnan(fold_preds).all(axis=0)
        if valid_cols.any():
            ensemble_pred = np.nanmean(fold_preds[:, valid_cols], axis=1)
        else:
            ensemble_pred = np.zeros(len(va_idx))

        if task == "regression":
            from sklearn.metrics import mean_absolute_error
            score = mean_absolute_error(y_va, ensemble_pred)
            log_step(f"Fold {fold_idx+1}: MAE={score:.4f}", 1)
        else:
            from sklearn.metrics import roc_auc_score
            try:
                score = roc_auc_score(y_va, ensemble_pred)
            except ValueError:
                score = 0.5
            log_step(f"Fold {fold_idx+1}: AUC={score:.4f}", 1)
        fold_scores.append(score)

    # Level 2: LGB Meta-learner
    log_step("Level 2: LGB Meta-learner...", 1)

    valid_mask = ~np.isnan(oof_preds).all(axis=1)
    # F14: Fill NaN with 0.5 for binary tasks, y.mean() for regression
    meta_fill = 0.5 if task == "binary" else float(np.nanmean(y[valid_mask]))
    X_meta = np.nan_to_num(oof_preds[valid_mask], nan=meta_fill)
    y_meta = y[valid_mask]

    meta_model = None
    l3_model = None

    if len(X_meta) >= 200:
        # Use time-series split for meta-model training
        meta_split = int(len(X_meta) * 0.8)
        X_meta_tr, X_meta_va = X_meta[:meta_split], X_meta[meta_split:]
        y_meta_tr, y_meta_va = y_meta[:meta_split], y_meta[meta_split:]

        meta_lgb_params = {
            "objective": "regression" if task == "regression" else "binary",
            "metric": "l1" if task == "regression" else "binary_logloss",
            "num_leaves": CONFIG["meta_num_leaves"],
            "learning_rate": CONFIG["meta_lr"],
            "verbose": -1, "n_jobs": -1, "seed": 42,
            "reg_alpha": CONFIG["meta_reg_alpha"],
            "reg_lambda": CONFIG["meta_reg_lambda"],
            "min_child_samples": CONFIG["meta_min_child_samples"],
        }
        dtrain_meta = lgb.Dataset(X_meta_tr, label=y_meta_tr)
        dval_meta = lgb.Dataset(X_meta_va, label=y_meta_va, reference=dtrain_meta)
        meta_model = lgb.train(
            meta_lgb_params, dtrain_meta,
            num_boost_round=CONFIG["meta_num_boost_round"],
            valid_sets=[dval_meta],
            callbacks=[lgb.early_stopping(CONFIG["meta_early_stopping"]),
                       lgb.log_evaluation(0)],
        )

        # Level 3: Ridge/LogReg
        meta_pred_tr = meta_model.predict(X_meta_tr)
        meta_pred_va = meta_model.predict(X_meta_va)
        X_l3_tr = np.column_stack([X_meta_tr, meta_pred_tr.reshape(-1, 1)])
        X_l3_va = np.column_stack([X_meta_va, meta_pred_va.reshape(-1, 1)])
        if task == "regression":
            l3_model = Ridge(alpha=1.0)
        else:
            l3_model = LogisticRegression(C=1.0, max_iter=1000)
        l3_model.fit(X_l3_tr, y_meta_tr)

        # Evaluate on held-out meta validation
        if task == "regression":
            from sklearn.metrics import mean_absolute_error
            l3_pred = l3_model.predict(X_l3_va)
            log_step(f"Level 3 holdout MAE: {mean_absolute_error(y_meta_va, l3_pred):.4f}", 1)
        else:
            from sklearn.metrics import roc_auc_score
            l3_pred = (l3_model.predict_proba(X_l3_va)[:, 1]
                       if hasattr(l3_model, "predict_proba") else l3_model.predict(X_l3_va))
            try:
                log_step(f"Level 3 holdout AUC: {roc_auc_score(y_meta_va, l3_pred):.4f}", 1)
            except ValueError:
                pass

        # C2 FIX: Do NOT refit on full meta data.
        # Save the train+optimize version (not the refitted version).
        # The original code refitted meta_model and l3_model on ALL meta data
        # including validation, which contaminates the model.
        log_step("C2 FIX: L2/L3 trained on train split only (no refit on full data)", 1)

    else:
        log_step("Not enough meta data, skipping level 2/3", 1)

    avg_score = np.mean(fold_scores) if fold_scores else 0
    std_score = np.std(fold_scores) if fold_scores else 0
    log_step(f"Average fold score: {avg_score:.4f} +/- {std_score:.4f}", 1)

    return {
        "fold_models": fold_models, "meta_model": meta_model, "l3_model": l3_model,
        "oof_preds": oof_preds, "fold_scores": fold_scores, "avg_score": avg_score,
        "std_score": std_score,
    }


# ══════════════════════════════════════════════
# Part 13: Bayesian Model Averaging (H1 fix)
# ══════════════════════════════════════════════
def bayesian_model_averaging(oof_preds, y_true, task, model_names=None):
    """Weight ensemble members by posterior probability via log-likelihood.
    H1 FIX: Assigns weight 0 to failed/degenerate models.
    """
    log_step("Bayesian Model Averaging...")
    n_models = oof_preds.shape[1]
    valid_mask = ~np.isnan(oof_preds).all(axis=1) & ~np.isnan(y_true)
    X = oof_preds[valid_mask]
    y = y_true[valid_mask]

    log_likelihoods = np.zeros(n_models)
    model_valid = np.ones(n_models, dtype=bool)

    for i in range(n_models):
        preds = X[:, i]

        # H1 FIX: Check for failed/degenerate models
        if np.isnan(preds).all() or np.std(np.nan_to_num(preds)) < 1e-10:
            model_valid[i] = False
            name = model_names[i] if model_names else f"Model {i}"
            log_step(f"  {name}: FAILED/DEGENERATE - weight set to 0", 1)
            continue

        preds = np.nan_to_num(preds, nan=0.5 if task == "binary" else y.mean())
        if task == "binary":
            preds = np.clip(preds, 1e-7, 1 - 1e-7)
            ll = np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))
        else:
            ll = -np.mean((y - preds) ** 2)
        log_likelihoods[i] = ll

    # Only compute weights for valid models
    if not model_valid.any():
        weights = np.ones(n_models) / n_models
    else:
        valid_ll = log_likelihoods[model_valid]
        valid_ll -= valid_ll.max()
        valid_weights = np.exp(valid_ll)
        valid_weights = valid_weights / valid_weights.sum()

        weights = np.zeros(n_models)
        weights[model_valid] = valid_weights

    for i in range(n_models):
        name = model_names[i] if model_names else f"Model {i}"
        log_step(f"  {name} BMA weight: {weights[i]:.4f}", 1)
    return weights


# ══════════════════════════════════════════════
# Part 14: Calibration
# ══════════════════════════════════════════════
def isotonic_calibration(y_true, y_pred):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    ir.fit(y_pred, y_true)
    return ir


# ══════════════════════════════════════════════
# Part 15: Pseudo Labeling - REMOVED (C4 fix)
# ══════════════════════════════════════════════
# C4 FIX: Pseudo labeling removed entirely.
# It trains without validation and creates confirmation bias.
# The risk is not worth the marginal improvement.


# ══════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════
def main():
    global _GLOBAL_START
    _GLOBAL_START = time.time()

    print("=" * 70)
    print("  KEIBA-PREDICTOR GOD-LEVEL v4 TRAINING PIPELINE")
    print("  v4: Odds-free, checkpoint, domain features, calibration fix")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
    from sklearn.impute import SimpleImputer

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    use_gpu = CONFIG["use_gpu"]
    if use_gpu:
        try:
            import subprocess
            r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if r.returncode != 0:
                use_gpu = False
        except FileNotFoundError:
            use_gpu = False
    log_step(f"GPU: {'ON' if use_gpu else 'OFF'}")

    # ── Step 1: Data Loading ──
    print("\n" + "=" * 70)
    print("  [1/12] DATA LOADING")
    print("=" * 70)

    proc = DataProcessor()
    df = proc.load(CONFIG["data_file"])

    bl_path = Path(CONFIG["bloodline_file"])
    if bl_path.exists():
        bl = pd.read_csv(bl_path)
        if "horse_id" in bl.columns:
            df = df.merge(bl, on="horse_id", how="left")
            log_step(f"Bloodline merged: {len(bl)} records")

    df = proc.preprocess(df)
    df_train = df[df["finish_position"].notna()].copy()

    # Sort by date to ensure shift operations never use future data
    if "race_date" in df_train.columns:
        df_train = df_train.sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)
        log_step("Data sorted by race_date (prevents future leakage in shift operations)")
    elif "race_id" in df_train.columns:
        df_train = df_train.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
        log_step("Data sorted by race_id")

    n_races = df_train["race_id"].nunique()
    log_step(f"Training data: {len(df_train):,} rows / {n_races:,} races")

    # ── Step 2: Base Features ──
    print("\n" + "=" * 70)
    print("  [2/12] BASE FEATURE ENGINEERING")
    print("=" * 70)

    fe = FeatureEngine()
    df_train = fe.create_all(df_train, use_advanced=True)

    # ── Step 3: Elo/Glicko ──
    print("\n" + "=" * 70)
    print("  [3/12] ELO/GLICKO RATINGS")
    print("=" * 70)

    df_train = compute_elo_features(df_train)

    # ── Step 4: Target Encoding ──
    print("\n" + "=" * 70)
    print("  [4/12] TARGET ENCODING")
    print("=" * 70)

    df_train = add_target_encodings(df_train)

    # ── Step 5: GOD Features ──
    print("\n" + "=" * 70)
    print("  [5/12] GOD-LEVEL FEATURE ENGINEERING")
    print("=" * 70)

    df_train = add_god_features(df_train)

    del df
    gc.collect()
    log_step(f"Memory freed. DataFrame shape: {df_train.shape}")

    # Build complete feature list
    GOD_LEAK_COLS = {
        "race_class_proxy", "race_prize_level", "prize_class_ratio",
        "_is_win", "_is_top3",
    }
    tmp_cols = {c for c in df_train.columns if c.startswith("_tmp_")}
    all_exclude = fe.LEAK_FEATURES | fe.EXCLUDE_COLS | GOD_LEAK_COLS | tmp_cols

    feature_cols = fe.get_feature_columns(df_train)
    feature_set = set(feature_cols)
    all_numeric = [c for c in df_train.columns
                   if df_train[c].dtype in ["float64", "int64", "int32", "float32"]
                   and c not in all_exclude]
    for c in all_numeric:
        if c not in feature_set:
            feature_cols.append(c)
            feature_set.add(c)

    feature_cols = [c for c in feature_cols if c not in all_exclude]

    # F5: Remove odds-related features if configured
    if CONFIG.get("exclude_odds_from_features", False):
        n_before_odds = len(feature_cols)
        feature_cols = [c for c in feature_cols if c not in ODDS_FEATURES]
        n_removed = n_before_odds - len(feature_cols)
        log_step(f"F5: Removed {n_removed} odds-related features")

    log_step(f"Total features: {len(feature_cols)}")
    log_step(f"Excluded {len(all_exclude)} leak/ID columns")

    # ── Step 6: Data Preparation + 3-way Temporal Split (C1 fix) ──
    print("\n" + "=" * 70)
    print("  [6/12] DATA PREPARATION + 3-WAY TEMPORAL SPLIT")
    print("=" * 70)

    X_raw = df_train[feature_cols].values.copy()

    y_position = df_train["finish_position"].values
    y_win = (df_train["finish_position"] == 1).astype(int).values
    y_top3 = (df_train["finish_position"] <= 3).astype(int).values

    # ── V5: Value-based targets ──
    # y_value_win: 1 if betting tansho on this horse was profitable
    # Tansho payout ≈ odds * 100 for a 100 yen bet → profitable if won
    # But we weight by HOW profitable: high-odds winners are more valuable
    odds_values = df_train["odds"].values.copy() if "odds" in df_train.columns else np.ones(len(df_train))
    odds_values = np.where(np.isnan(odds_values) | (odds_values <= 0), 1.0, odds_values)

    # Value target: horse won AND odds were decent (not just favorites)
    y_value_win = y_win.copy()  # Same as win, but training will use odds-weighted samples

    # Value target for top3: horse placed AND had value odds
    y_value_top3 = y_top3.copy()

    # Expected Value target (regression): EV = P(win) * odds
    # For training: actual_ev = is_win * odds (realized value)
    y_ev = (y_win * odds_values).astype(np.float32)

    # ── V2: Odds-weighted sample weights ──
    # Higher weight for high-odds horses → model learns to find longshot value
    time_weights = compute_time_decay_weights(df_train, CONFIG["time_decay_half_life_days"])
    odds_weights = np.power(
        np.clip(odds_values, CONFIG["value_min_odds"], CONFIG["value_max_odds"]),
        CONFIG["value_odds_weight_power"]
    )
    # Normalize odds weights to mean=1
    odds_weights = odds_weights / odds_weights.mean()

    log_step(f"V5: Odds weights range: [{odds_weights.min():.2f}, {odds_weights.max():.2f}]")
    log_step(f"V5: EV target stats: mean={y_ev.mean():.3f}, >1.0: {(y_ev > 1.0).sum():,}")

    race_ids = df_train["race_id"].values
    race_dates = df_train["race_date"].values if "race_date" in df_train.columns else None

    # Combined weights: time decay * odds emphasis
    sample_weights = time_weights * odds_weights
    log_step(f"Sample weights range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

    # C1 FIX: 3-way temporal split
    # train: <2024, optimize: 2024, holdout: >=2025
    dates = df_train["race_date"]
    train_mask = dates < CONFIG["optimize_start"]
    optimize_mask = (dates >= CONFIG["optimize_start"]) & (dates < CONFIG["holdout_start"])
    holdout_mask = dates >= CONFIG["holdout_start"]

    train_idx = np.where(train_mask)[0]
    optimize_idx = np.where(optimize_mask)[0]
    holdout_idx = np.where(holdout_mask)[0]

    if len(optimize_idx) < 5000:
        log_step(f"Optimize set too small ({len(optimize_idx)}), merging into train")
        # Merge optimize into train, use last 20% of combined as optimize
        combined = np.concatenate([train_idx, optimize_idx])
        split = int(len(combined) * 0.8)
        train_idx = combined[:split]
        optimize_idx = combined[split:]

    # Combined train+optimize for final stacking
    train_optimize_idx = np.concatenate([train_idx, optimize_idx])

    log_step(f"C1 FIX: 3-way temporal split:")
    log_step(f"  Train (<{CONFIG['optimize_start']}): {len(train_idx):,} rows", 1)
    log_step(f"  Optimize (2024): {len(optimize_idx):,} rows", 1)
    log_step(f"  Holdout (>={CONFIG['holdout_start']}): {len(holdout_idx):,} rows", 1)

    if len(holdout_idx) == 0:
        log_step("WARNING: No holdout data found (no 2025+ data). Falling back to time-series split.")
        # Fallback: last 20% as holdout
        n = len(X_raw)
        split80 = int(n * 0.8)
        split60 = int(n * 0.6)
        train_idx = np.arange(0, split60)
        optimize_idx = np.arange(split60, split80)
        holdout_idx = np.arange(split80, n)
        train_optimize_idx = np.concatenate([train_idx, optimize_idx])
        log_step(f"  Fallback train: {len(train_idx):,}", 1)
        log_step(f"  Fallback optimize: {len(optimize_idx):,}", 1)
        log_step(f"  Fallback holdout: {len(holdout_idx):,}", 1)

    _n_train_rows = len(df_train)

    # F13: Imputer will be fitted AFTER hybrid_mask selection (not here)
    # For now, just handle inf values; NaN preserved for feature selection
    X = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    del df_train
    gc.collect()
    log_step("DataFrame freed from memory")

    X_train = X[train_idx]
    X_optimize = X[optimize_idx]
    X_holdout = X[holdout_idx]
    X_train_optimize = X[train_optimize_idx]
    w_train = sample_weights[train_idx]

    log_step(f"Train: {len(X_train):,} / Optimize: {len(X_optimize):,} / Holdout: {len(X_holdout):,}")

    # ── Step 7: Adversarial Validation ──
    print("\n" + "=" * 70)
    print("  [7/12] ADVERSARIAL VALIDATION")
    print("=" * 70)

    # H3 FIX: Adversarial debiasing dead code removed entirely
    # Only AUC reporting remains for diagnostics
    adv_auc, adv_drift = adversarial_validation(X_train, X_holdout, feature_cols)

    # ── Step 8: Hybrid Feature Selection ──
    print("\n" + "=" * 70)
    print("  [8/12] HYBRID FEATURE SELECTION")
    print("=" * 70)

    # Use most recent training data for Boruta
    sample_size = min(200000, len(X_train))
    sample_idx = np.arange(len(X_train) - sample_size, len(X_train))

    boruta_selected, boruta_mask = boruta_selection(
        X_train[sample_idx], y_top3[train_idx][sample_idx],
        feature_cols, max_iter=CONFIG["boruta_max_iter"],
        min_features=CONFIG["boruta_min_features"]
    )

    # Use ONLY training data for hybrid feature selection
    hybrid_selected, hybrid_mask = hybrid_feature_selection(
        X_train, y_top3[train_idx], feature_cols, boruta_mask,
        n_folds=min(5, CONFIG["n_cv_folds"])
    )

    X_selected_raw = X_raw[:, hybrid_mask]
    selected_features = hybrid_selected

    # F13: Fit imputer AFTER hybrid_mask selection on selected features only
    log_step("F13: Median imputation on selected features only")
    # Replace inf/-inf with NaN before imputation
    X_selected_raw = np.where(np.isfinite(X_selected_raw), X_selected_raw, np.nan)
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_selected_raw[train_idx])
    X_selected = imputer.transform(X_selected_raw)
    X_selected = np.nan_to_num(X_selected, nan=0.0, posinf=0.0, neginf=0.0)

    X_train_sel = X_selected[train_idx]
    X_optimize_sel = X_selected[optimize_idx]
    X_holdout_sel = X_selected[holdout_idx]
    X_train_optimize_sel = X_selected[train_optimize_idx]

    log_step(f"Final features: {len(selected_features)} / {len(feature_cols)}")

    with open(f"{CONFIG['output_dir']}/feature_selection.json", "w") as f:
        json.dump({
            "boruta_selected": boruta_selected,
            "hybrid_selected": hybrid_selected,
            "total_features": len(feature_cols),
            "selected_features": len(selected_features),
        }, f, ensure_ascii=False, indent=2)

    # ── Step 9: Optuna Optimization (C1: uses optimize set, NOT holdout) ──
    print("\n" + "=" * 70)
    print("  [9/12] OPTUNA OPTIMIZATION (on optimize set, holdout untouched)")
    print("=" * 70)

    # V1: New task set - keep win/top3 for prediction, add EV for value
    tasks = [
        ("ev", "regression", y_ev),           # V5: Expected Value regression (key!)
        ("top3", "binary", y_value_top3),     # Place prediction (odds-weighted)
        ("win", "binary", y_value_win),       # Win prediction (odds-weighted)
    ]

    all_results = {}
    all_best_params = {}

    for task_name, task_type, y_target in tasks:
        print(f"\n  --- {task_name} ({task_type}) ---")
        y_tr = y_target[train_idx]
        y_opt = y_target[optimize_idx]

        # C1 FIX: Optuna tunes on train->optimize, NOT on holdout
        log_step(f"Optuna: {CONFIG['optuna_trials']} trials (train->optimize split)...")
        best_lgb_params = optimize_with_pruning(
            X_train_sel, y_tr, X_optimize_sel, y_opt,
            task=task_type, n_trials=CONFIG["optuna_trials"],
            use_gpu=use_gpu, sample_weights_tr=w_train
        )
        all_best_params[task_name] = best_lgb_params

        lgb_params = {
            "objective": "regression" if task_type == "regression" else "binary",
            "metric": "l1" if task_type == "regression" else "binary_logloss",
            "verbose": -1, "n_jobs": -1, "seed": 42, **best_lgb_params,
        }
        xgb_params = {
            "objective": "reg:squarederror" if task_type == "regression" else "binary:logistic",
            "eval_metric": "mae" if task_type == "regression" else "logloss",
            "max_depth": best_lgb_params.get("max_depth", 6),
            "learning_rate": best_lgb_params.get("learning_rate", 0.05),
            "subsample": best_lgb_params.get("bagging_fraction", 0.8),
            "colsample_bytree": best_lgb_params.get("feature_fraction", 0.8),
            "reg_alpha": best_lgb_params.get("reg_alpha", 0.1),
            "reg_lambda": best_lgb_params.get("reg_lambda", 0.1),
            "n_estimators": 3000,
            "verbosity": 0, "random_state": 42,
        }
        catboost_params = {
            "iterations": 3000, "learning_rate": best_lgb_params.get("learning_rate", 0.05),
            "depth": min(best_lgb_params.get("max_depth", 6), 10),
            "l2_leaf_reg": best_lgb_params.get("reg_lambda", 1.0),
            "random_seed": 42, "verbose": 0,
        }

        models_config = [
            ("lightgbm", lgb_params), ("xgboost", xgb_params),
            ("catboost", catboost_params), ("tabnet", {}),
        ]
        all_results[task_name] = {
            "best_lgb_params": best_lgb_params, "models_config": models_config,
            "task_type": task_type,
        }

    # ── Step 10: GOD Stacking (C4: no pseudo labeling) ──
    print("\n" + "=" * 70)
    print("  [10/12] GOD STACKING (no pseudo labeling)")
    print("=" * 70)

    for task_name, task_type, y_target in tasks:
        log_step(f"=== {task_name} ===")
        result = all_results[task_name]

        # C4 FIX: No pseudo labeling. Use train+optimize data directly.
        # Stacking uses train+optimize with TimeSeriesSplit CV.
        X_stack = X_train_optimize_sel
        y_stack = y_target[train_optimize_idx]
        w_stack = sample_weights[train_optimize_idx]
        dates_stack = race_dates[train_optimize_idx] if race_dates is not None else None
        rids_stack = race_ids[train_optimize_idx] if race_ids is not None else None

        # H2 FIX: noise_std passed to stacking, applied only on train folds
        stack_result = train_god_stacking(
            X_stack, y_stack, task_type, rids_stack, dates_stack,
            result["models_config"], n_folds=CONFIG["n_cv_folds"],
            use_gpu=use_gpu, sample_weights=w_stack,
            noise_std=CONFIG["noise_std"],
        )

        # H1 FIX: BMA with proper handling of failed models
        model_names = [name for name, _ in result["models_config"]]
        bma_weights = bayesian_model_averaging(
            stack_result["oof_preds"], y_stack, task_type, model_names=model_names)
        result["stacking"] = stack_result
        result["bma_weights"] = bma_weights

    # ── Step 11: LambdaRank ──
    print("\n" + "=" * 70)
    print("  [11/12] LAMBDARANK TRAINING")
    print("=" * 70)

    lambdarank_model = None
    if race_ids is not None:
        # Train on train, validate on optimize (NOT holdout - keep holdout pristine)
        tr_race_ids = race_ids[train_idx]
        opt_race_ids = race_ids[optimize_idx]

        _, tr_starts = np.unique(tr_race_ids, return_index=True)
        tr_counts = pd.Series(tr_race_ids).groupby(tr_race_ids).transform("count").values
        tr_group_sizes = tr_counts[np.sort(tr_starts)].tolist()

        _, opt_starts = np.unique(opt_race_ids, return_index=True)
        opt_counts = pd.Series(opt_race_ids).groupby(opt_race_ids).transform("count").values
        opt_group_sizes = opt_counts[np.sort(opt_starts)].tolist()

        # V6: LambdaRank uses EV as relevance label (higher EV = more relevant)
        best_params = all_best_params.get("ev", {})
        X_train_sel = X_selected[train_idx]
        # Convert EV to relevance: 0=no value, 1=small, 2=medium, 3=high
        ev_train_relevance = np.clip(np.floor(y_ev[train_idx] * 3).astype(int), 0, 10)
        ev_opt_relevance = np.clip(np.floor(y_ev[optimize_idx] * 3).astype(int), 0, 10)
        lambdarank_model = train_lambdarank(
            X_train_sel, ev_train_relevance, tr_group_sizes,
            X_optimize_sel, ev_opt_relevance, opt_group_sizes,
            best_params, use_gpu=use_gpu
        )

    # ── Step 12: Calibration + Holdout Evaluation + Save ──
    print("\n" + "=" * 70)
    print("  [12/12] CALIBRATION + HOLDOUT EVALUATION + MODEL SAVING")
    print("=" * 70)

    calibration_models = {}
    temperature_scalers = {}

    for task_name in ["win", "top3"]:
        result = all_results[task_name]
        stack = result["stacking"]
        oof = stack["oof_preds"]

        valid_mask = ~np.isnan(oof).all(axis=1)
        if valid_mask.sum() < 100:
            continue

        oof_mean_all = np.nanmean(oof, axis=1)
        all_valid_mask = ~np.isnan(oof_mean_all)

        y_train_opt_win = y_win[train_optimize_idx]
        y_train_opt_top3 = y_top3[train_optimize_idx]
        y_cal = y_train_opt_win if task_name == "win" else y_train_opt_top3
        if len(all_valid_mask) > len(y_cal):
            all_valid_mask = all_valid_mask[:len(y_cal)]
            oof_valid = oof_mean_all[:len(y_cal)][all_valid_mask]
            y_valid = y_cal[all_valid_mask]
        else:
            oof_valid = oof_mean_all[all_valid_mask]
            y_valid = y_cal[all_valid_mask]
        oof_mean = oof_valid

        try:
            calibration_models[task_name] = isotonic_calibration(y_valid, oof_mean)
            log_step(f"Isotonic calibration fitted: {task_name}")
        except Exception as e:
            log_step(f"Isotonic calibration failed for {task_name}: {e}")

        # F9: Platt calibration replaces TemperatureScaler
        try:
            ps = PlattScaler()
            ps.fit(y_valid, oof_mean)
            temperature_scalers[task_name] = ps
            log_step(f"Platt calibration: {task_name} a={ps.a:.3f}, b={ps.b:.3f}")
        except Exception as e:
            log_step(f"Platt calibration failed for {task_name}: {e}")

    # Server-compatible model (build BEFORE holdout evaluation so we can use full ensemble)
    log_step("Building server-compatible model...")

    server_models = {}
    feat_importances = {}

    for task_name, result in all_results.items():
        stack = result["stacking"]
        fold_models = stack["fold_models"]
        meta_model = stack["meta_model"]
        l3_model = stack["l3_model"]

        has_any_models = any(len(v) > 0 for v in fold_models.values())
        if not has_any_models:
            continue

        cal = calibration_models.get(task_name) if task_name in ["win", "top3"] else None
        ts = temperature_scalers.get(task_name) if task_name in ["win", "top3"] else None

        bma_w = result.get("bma_weights", None)
        if isinstance(bma_w, np.ndarray):
            bma_w = bma_w.tolist() if len(bma_w) > 0 else None
        elif isinstance(bma_w, list) and len(bma_w) == 0:
            bma_w = None

        models_config_names = [name for name, _ in result["models_config"]]
        task_type = result["task_type"]

        server_models[task_name] = _GodEnsembleModel(
            fold_models=fold_models,
            meta_model=meta_model,
            l3_model=l3_model,
            bma_weights=bma_w,
            calibrator=cal,
            temperature_scaler=ts,
            task_type=task_type,
            models_config_names=models_config_names,
        )

        model_counts = {k: len(v) for k, v in fold_models.items() if len(v) > 0}
        log_step(f"  {task_name}: GOD ensemble built - {model_counts}, "
                 f"meta={'yes' if meta_model else 'no'}, "
                 f"l3={'yes' if l3_model else 'no'}, "
                 f"cal={'yes' if cal else 'no'}, "
                 f"ts={'yes' if ts else 'no'}")

        try:
            all_imp = server_models[task_name].feature_importance(importance_type="gain")
            if all_imp is not None:
                feat_importances[task_name] = pd.DataFrame({
                    "feature": selected_features, "importance": all_imp,
                }).sort_values("importance", ascending=False)
        except Exception:
            pass

    # ── HONEST holdout evaluation on holdout_idx (NEVER seen by Optuna) ──
    print("\n" + "-" * 70)
    print("  HONEST HOLDOUT EVALUATION (2025+ data, never seen by Optuna)")
    print("-" * 70)

    holdout_results = {}
    for task_name, task_type, y_target in tasks:
        result = all_results[task_name]
        stack = result["stacking"]

        if task_name not in server_models or server_models[task_name] is None:
            continue

        # Use the full ensemble model (LGB + XGB + CatBoost + TabNet + meta + L3)
        model = server_models[task_name]
        try:
            if task_type == "binary" and hasattr(model, 'predict_proba'):
                ensemble_pred = model.predict_proba(X_holdout_sel)
                # F12: Extract [:, 1] from 2D output
                if hasattr(ensemble_pred, 'ndim') and ensemble_pred.ndim == 2:
                    ensemble_pred = ensemble_pred[:, 1]
            else:
                ensemble_pred = model.predict(X_holdout_sel)
        except Exception:
            # Fallback to averaging LGB fold models if ensemble predict fails
            lgb_models = stack["fold_models"].get("lightgbm", [])
            if not lgb_models:
                continue
            preds_list = []
            for m in lgb_models:
                try:
                    preds_list.append(m.predict(X_holdout_sel))
                except Exception:
                    pass
            if not preds_list:
                continue
            ensemble_pred = np.mean(preds_list, axis=0)

        y_holdout_true = y_target[holdout_idx]
        if task_type == "regression":
            score = mean_absolute_error(y_holdout_true, ensemble_pred)
            log_step(f"  {task_name} HOLDOUT MAE: {score:.4f}", 1)
        else:
            try:
                score = roc_auc_score(y_holdout_true, ensemble_pred)
                log_step(f"  {task_name} HOLDOUT AUC: {score:.4f}", 1)
            except ValueError:
                score = 0.5
        holdout_results[task_name] = score

        # Also print fold scores with confidence interval
        fold_scores = stack["fold_scores"]
        if fold_scores:
            mean_s = np.mean(fold_scores)
            std_s = np.std(fold_scores)
            metric_name = "MAE" if task_type == "regression" else "AUC"
            log_step(f"  {task_name} CV {metric_name}: {mean_s:.4f} +/- {std_s:.4f} "
                     f"(95% CI: [{mean_s - 1.96*std_s:.4f}, {mean_s + 1.96*std_s:.4f}])", 1)

    # ── V8: Simulated ROI on holdout ──
    print("\n" + "-" * 70)
    print("  V8: SIMULATED ROI ON HOLDOUT (2025+ data)")
    print("-" * 70)

    try:
        # Get win probabilities from model
        win_model = server_models.get("win")
        ev_model = server_models.get("ev")
        holdout_odds = odds_values[holdout_idx]
        holdout_wins = y_win[holdout_idx]
        holdout_top3 = y_top3[holdout_idx]

        # Strategy 1: Top-1 per race using EV model
        if ev_model is not None:
            try:
                ev_preds = ev_model.predict(X_holdout_sel)
            except Exception:
                ev_preds = ev_model.predict_proba(X_holdout_sel)
                if hasattr(ev_preds, 'ndim') and ev_preds.ndim == 2:
                    ev_preds = ev_preds[:, 1]

            holdout_race_ids = race_ids[holdout_idx]
            total_bet = 0
            total_return = 0
            n_bets = 0
            n_hits = 0

            # Group by race, pick top EV horse
            race_indices = {}
            for i, rid in enumerate(holdout_race_ids):
                if rid not in race_indices:
                    race_indices[rid] = []
                race_indices[rid].append(i)

            for rid, indices in race_indices.items():
                ev_scores = ev_preds[indices]
                best_idx = indices[np.argmax(ev_scores)]

                # Only bet if predicted EV > threshold
                if ev_preds[best_idx] > 0.8:
                    total_bet += 100
                    n_bets += 1
                    if holdout_wins[best_idx] == 1:
                        total_return += 100 * holdout_odds[best_idx]
                        n_hits += 1

            if total_bet > 0:
                roi = (total_return / total_bet - 1) * 100
                hit_rate = n_hits / n_bets * 100
                log_step(f"V8 EV Top-1 Tansho: ROI={roi:+.1f}%, Hit={hit_rate:.1f}%, "
                         f"Bets={n_bets}, Profit={total_return - total_bet:+,.0f}円")

            # Strategy 2: EV > 1.5 filter (bet all horses with high predicted EV)
            for ev_thresh in [1.0, 1.2, 1.5, 2.0]:
                high_ev_mask = ev_preds > ev_thresh
                n_high = high_ev_mask.sum()
                if n_high > 0:
                    wins_high = holdout_wins[high_ev_mask]
                    odds_high = holdout_odds[high_ev_mask]
                    bet = n_high * 100
                    ret = (wins_high * odds_high * 100).sum()
                    roi_h = (ret / bet - 1) * 100
                    log_step(f"  EV>{ev_thresh}: {n_high} bets, ROI={roi_h:+.1f}%, "
                             f"Hit={wins_high.mean()*100:.1f}%")

        # Strategy 3: Win probability × odds (manual EV calc)
        if win_model is not None:
            try:
                win_preds = win_model.predict_proba(X_holdout_sel)
                if hasattr(win_preds, 'ndim') and win_preds.ndim == 2:
                    win_preds = win_preds[:, 1]
            except Exception:
                win_preds = win_model.predict(X_holdout_sel)

            manual_ev = win_preds * holdout_odds
            for ev_thresh in [1.0, 1.2, 1.5, 2.0]:
                high_mask = manual_ev > ev_thresh
                n_h = high_mask.sum()
                if n_h > 0:
                    bet = n_h * 100
                    ret = (holdout_wins[high_mask] * holdout_odds[high_mask] * 100).sum()
                    roi_h = (ret / bet - 1) * 100
                    log_step(f"  P(win)*odds>{ev_thresh}: {n_h} bets, ROI={roi_h:+.1f}%, "
                             f"Hit={holdout_wins[high_mask].mean()*100:.1f}%")

    except Exception as e:
        log_step(f"V8 ROI simulation error: {e}")
        import traceback
        traceback.print_exc()

    # Build eval_results now that holdout_results is available
    eval_results = {}
    for task_name, result in all_results.items():
        stack = result["stacking"]
        task_type = result["task_type"]
        scores = stack["fold_scores"]
        std_scores = stack.get("std_score", np.std(scores) if scores else 0)
        metric = "mae" if task_type == "regression" else "auc"
        eval_results[task_name] = {
            f"mean_{metric}": float(np.mean(scores)) if scores else 0,
            f"std_{metric}": float(std_scores),
            "holdout": float(holdout_results.get(task_name, 0)),
        }

    # Identify GOD-specific features
    base_features = set(fe.get_feature_columns(pd.DataFrame(columns=feature_cols)))
    god_only_features = [f for f in selected_features if f not in base_features]

    save_data = {
        "version": "5.0-god-roi",
        "models": server_models,
        "feature_columns": selected_features,
        "evaluation_results": eval_results,
        "feature_importance": {k: v.to_dict() for k, v in feat_importances.items()},
        "saved_at": datetime.now().isoformat(),
        "n_features": len(selected_features),
        "trained_on": "local_god_v5_roi",
        "model_type": "god_ensemble",
        "god_features": god_only_features,
        "god_features_count": len(god_only_features),
        "lambdarank_model": lambdarank_model,
        "imputer": imputer,  # H4: save imputer for server-side use
        # F15: Feature lookup tables for inference
        "target_encoding_lookups": _TARGET_ENCODING_LOOKUPS,
        "bloodline_stats_lookups": _BLOODLINE_STATS_LOOKUPS,
    }
    if god_only_features:
        log_step(f"  {len(god_only_features)} GOD-specific features in model")
        log_step(f"   Server must compute these for full accuracy: {god_only_features[:5]}...", 1)

    model_path = f"{CONFIG['model_dir']}/keiba_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)
    log_step(f"Server-compatible GOD model: {model_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{CONFIG['model_dir']}/keiba_model_v4_{timestamp}.pkl"
    with open(backup_path, "wb") as f:
        pickle.dump(save_data, f)
    log_step(f"Backup: {backup_path}")

    # Full GOD model (legacy format)
    god_data = {
        "all_results": {},
        "selected_features": selected_features,
        "hybrid_mask": hybrid_mask.tolist(),
        "all_feature_columns": feature_cols,
        "config": CONFIG,
        "timestamp": datetime.now().isoformat(),
        "lambdarank_model": lambdarank_model,
        "calibration_models": calibration_models,
        "temperature_scalers": temperature_scalers,
        "adversarial_auc": float(adv_auc),
        "imputer": imputer,
        # F15: Feature lookup tables
        "target_encoding_lookups": _TARGET_ENCODING_LOOKUPS,
        "bloodline_stats_lookups": _BLOODLINE_STATS_LOOKUPS,
    }
    for k, v in all_results.items():
        god_data["all_results"][k] = {
            "fold_models": v["stacking"]["fold_models"],
            "meta_model": v["stacking"]["meta_model"],
            "l3_model": v["stacking"]["l3_model"],
            "best_lgb_params": v["best_lgb_params"],
            "task_type": v["task_type"],
            "bma_weights": v.get("bma_weights", np.array([])).tolist()
                if isinstance(v.get("bma_weights"), np.ndarray) else [],
            "fold_scores": v["stacking"]["fold_scores"],
        }

    god_path = f"{CONFIG['model_dir']}/keiba_model_god_v5.pkl"
    with open(god_path, "wb") as f:
        pickle.dump(god_data, f)
    log_step(f"GOD full model (legacy): {god_path}")

    # SHAP Analysis
    log_step("SHAP analysis...")
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for task_name in ["win", "top3", "ev"]:
            if task_name in server_models and server_models[task_name]:
                sm = server_models[task_name]
                lgb_folds = sm.fold_models.get("lightgbm", []) if hasattr(sm, "fold_models") else []
                if not lgb_folds:
                    continue
                shap_model = lgb_folds[0]
                explainer = shap.TreeExplainer(shap_model)
                sample_n = min(2000, len(X_holdout_sel))
                sample = pd.DataFrame(X_holdout_sel[:sample_n], columns=selected_features)
                shap_values = explainer.shap_values(sample)

                plt.figure(figsize=(16, 14))
                sv = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else (
                    shap_values[0] if isinstance(shap_values, list) else shap_values)
                shap.summary_plot(sv, sample, show=False, max_display=40)
                plt.tight_layout()
                plt.savefig(f"{CONFIG['output_dir']}/shap_{task_name}.png", dpi=150)
                plt.close()

                mean_abs_shap = np.abs(sv).mean(axis=0)
                top_indices = np.argsort(-mean_abs_shap)[:5]
                fig, axes = plt.subplots(1, min(5, len(top_indices)), figsize=(20, 4))
                if not hasattr(axes, "__iter__"):
                    axes = [axes]
                for ax_idx, feat_idx in enumerate(top_indices[:len(axes)]):
                    shap.dependence_plot(feat_idx, sv, sample, ax=axes[ax_idx], show=False)
                plt.tight_layout()
                plt.savefig(f"{CONFIG['output_dir']}/shap_dep_{task_name}.png", dpi=150)
                plt.close()
                log_step(f"SHAP saved: {task_name}")
    except Exception as e:
        log_step(f"SHAP error: {e}")

    # Feature Importance Charts
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for task_name, imp_df in feat_importances.items():
            top40 = imp_df.head(40)
            plt.figure(figsize=(16, 14))
            plt.barh(range(len(top40)), top40["importance"].values)
            plt.yticks(range(len(top40)), top40["feature"].values, fontsize=8)
            plt.xlabel("Importance (gain)")
            plt.title(f"Feature Importance - {task_name} (GOD v4)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{CONFIG['output_dir']}/importance_{task_name}.png", dpi=150)
            plt.close()
        log_step("Feature importance charts saved")
    except Exception as e:
        log_step(f"Chart error: {e}")

    # Report
    report = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "god_v5_roi",
        "version": "5.0",
        "fixes_applied": [
            "All v4 fixes carried forward",
            "V1: EV regression task added",
            "V2: Odds-weighted sample weights",
            "V3: Custom ROI eval metric",
            "V4: Odds features included",
            "V5: Value-based targets",
            "V6: LambdaRank EV-based ranking",
            "V7: Profit-optimized calibration",
            "V8: Simulated ROI on holdout",
        ],
        "data_rows": int(_n_train_rows),
        "n_races": int(n_races),
        "split": {
            "train": len(train_idx),
            "optimize": len(optimize_idx),
            "holdout": len(holdout_idx),
        },
        "total_features": len(feature_cols),
        "boruta_selected": len(boruta_selected),
        "hybrid_selected": len(selected_features),
        "adversarial_auc": float(adv_auc),
        "adversarial_top_drift": [(f, float(v)) for f, v in adv_drift[:10]],
        "evaluation": eval_results,
        "holdout_results": {k: float(v) for k, v in holdout_results.items()},
        "best_params": all_best_params,
        "config": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                   for k, v in CONFIG.items()},
        "bma_weights": {
            k: v.get("bma_weights", np.array([])).tolist()
            if isinstance(v.get("bma_weights"), np.ndarray) else []
            for k, v in all_results.items()
        },
        "calibration": {
            k: {"a": float(temperature_scalers[k].a), "b": float(temperature_scalers[k].b)}
            for k in temperature_scalers
        },
        "runtime_seconds": time.time() - _GLOBAL_START,
        "lambdarank_trained": lambdarank_model is not None,
    }
    with open(f"{CONFIG['output_dir']}/training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    log_step(f"Report: {CONFIG['output_dir']}/training_report.json")

    # Final Summary
    elapsed = time.time() - _GLOBAL_START
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print("\n" + "=" * 70)
    print("  GOD-LEVEL v5 ROI-MAXIMIZING TRAINING COMPLETE")
    print("=" * 70)

    print("\n  HONEST HOLDOUT RESULTS (never seen by Optuna):")
    for name, score in holdout_results.items():
        metric = "MAE" if all_results[name]["task_type"] == "regression" else "AUC"
        print(f"    {name} {metric}: {score:.4f}")

    print("\n  CV RESULTS (with confidence intervals):")
    for name, scores in eval_results.items():
        print(f"\n  [{name}]")
        for k, v in scores.items():
            print(f"    {k}: {v:.4f}")

    print(f"\n  Features: {len(feature_cols)} -> Boruta {len(boruta_selected)} "
          f"-> Hybrid {len(selected_features)}")
    print(f"  Adversarial AUC: {adv_auc:.4f}")
    print(f"  LambdaRank: {'Trained' if lambdarank_model else 'Skipped'}")
    print(f"  Runtime: {hours}h {minutes}m")

    print(f"\n  v5 ROI changes:")
    print(f"    V1: New EV regression task (expected value prediction)")
    print(f"    V2: Odds-weighted sample weights (sqrt(odds))")
    print(f"    V3: Custom ROI eval metric on holdout")
    print(f"    V4: Odds features INCLUDED in training")
    print(f"    V5: Value targets (profitable bet detection)")
    print(f"    V6: LambdaRank ranks by EV, not position")
    print(f"    V7: Calibration for profit optimization")
    print(f"    V8: Simulated ROI on holdout data")

    print(f"\n  Output:")
    print(f"    {CONFIG['model_dir']}/keiba_model.pkl              (server-compatible)")
    print(f"    {CONFIG['model_dir']}/keiba_model_god_v5.pkl       (full GOD model)")
    print(f"    {CONFIG['output_dir']}/training_report.json")
    print(f"    {CONFIG['output_dir']}/feature_selection.json")
    print(f"    {CONFIG['output_dir']}/shap_*.png")
    print(f"    {CONFIG['output_dir']}/importance_*.png")

    if god_only_features:
        print(f"\n  GOD-specific features: {len(god_only_features)}")
        print(f"     Server needs GOD feature computation for full accuracy.")
        print(f"     Without it, {len(god_only_features)} features will be 0-filled.")

    print(f"\n  Upload to server:")
    print(f"    scp {CONFIG['model_dir']}/keiba_model.pkl user@server:/opt/keiba-predictor/models/")

    return all_results, eval_results


if __name__ == "__main__":
    main()
