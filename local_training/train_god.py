"""
競馬予想AI - GOD級学習パイプライン (train_god.py)
===================================================
train_ultimate.py の全問題を修正し、最先端の手法を全投入。

修正点:
  - Boruta 最低30特徴量保証 + ドメイン保護特徴量
  - Adversarial Validation: year/month 除外でリーク防止
  - Pruning callback metric 名修正 ("l1")

追加手法:
  1. Feature Interactions (交互作用)
  2. 血統統計 (sire/dam 勝率 by distance/surface/condition)
  3. ペース分析 (通過順パターン, 脚質分類)
  4. 時間減衰重み付け (half-life 730日)
  5. クラス遷移特徴量
  6. Cyclical encoding (sin/cos)
  7. レース内相対ランク・偏差
  8. 組み合わせ統計 (騎手x調教師, 馬x距離, 馬x会場)
  9. LambdaRank (LightGBM)
  10. TabNet (pytorch-tabnet)
  11. Entity Embeddings (via TabNet)
  12. 2-Level Stacking (L1: LGB+XGB+CB+TabNet -> L2: LGB meta -> L3: Ridge)
  13. Ranking-aware loss (LambdaRank)
  14. Time decay sample weighting
  15. Pseudo Labeling
  16. Adversarial debiasing
  17. Protected feature list for Boruta
  18. Hybrid feature selection (Boruta + SHAP + Permutation)
  19. Bayesian Model Averaging
  20. Cross-validated blending (OOF)
  21. Isotonic regression calibration
  22. Temperature scaling
  23. Race-level normalization
  24. K-fold with purging and embargo
  25. Noise injection
  26. Feature stability selection
  27. Comprehensive output + SHAP

推定所要時間: 10-16時間 (i9-12900K + RTX 3070 8GB + 64GB RAM)
対応: Windows + Python 3.12 + CUDA
"""

import os
import sys
import time
import json
import pickle
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

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
CONFIG = {
    "data_file": "data/_all_races.csv",
    "bloodline_file": "data/bloodline.csv",
    "output_dir": "output_god",
    "model_dir": "models",

    "n_cv_folds": 8,
    "optuna_trials": 200,
    "use_gpu": True,

    # Boruta
    "boruta_max_iter": 100,
    "boruta_min_features": 30,

    # Purging / Embargo
    "purge_days": 7,
    "embargo_days": 7,

    # Time decay (730 days = 2 years: 3-year-old data has weight ~0.35, not ~0.01)
    "time_decay_half_life_days": 730,

    # Pseudo labeling
    "pseudo_label_threshold": 0.90,
    "pseudo_label_rounds": 2,

    # Noise injection
    "noise_std": 0.01,

    # TabNet
    "tabnet_epochs": 100,
    "tabnet_patience": 15,

    # Feature stability
    "stability_threshold": 0.5,  # must appear in >50% of folds

}

# ドメイン知識で絶対に残す特徴量 (部分一致)
PROTECTED_FEATURE_PATTERNS = [
    "distance", "age", "weight", "horse_weight", "odds", "popularity",
    "track_condition", "course_type", "frame_number", "horse_number",
    "field_condition", "n_runners", "sex_enc", "weather_enc",
    "horse_elo", "jockey_elo", "trainer_elo", "horse_glicko",
    "combined_elo", "days_since_last_race",
]

# Adversarial Validation / Boruta から除外する特徴量 (リーク源)
# NOTE: exact match only. Cyclical features (month_sin, month_cos, dow_sin, dow_cos)
# are NOT excluded because they are time-invariant by design.
ADVVAL_EXCLUDE_FEATURES = {"year", "month", "day_of_week"}

# Global timer
_GLOBAL_START = time.time()


# _EnsembleLGBModel and TemperatureScaler are imported from god_ensemble.py
# _GodEnsembleModel is also imported from god_ensemble.py for the full stacking server model


def log_step(msg, level=0):
    """Progress logging with timestamps"""
    indent = "  " * level
    elapsed = time.time() - _GLOBAL_START
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"  [{h:02d}:{m:02d}:{s:02d}] {indent}{msg}")


# ══════════════════════════════════════════════
# Part 1: Elo/Glicko Rating System
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
        n = len(participants)
        if n < 2:
            return
        ratings = {pid: self.get_rating(pid) for pid, _ in participants}
        for i, (id_i, pos_i) in enumerate(participants):
            delta = 0.0
            for j, (id_j, pos_j) in enumerate(participants):
                if i == j:
                    continue
                expected = self.expected_score(ratings[id_i], ratings[id_j])
                actual = 1.0 if pos_i < pos_j else (0.5 if pos_i == pos_j else 0.0)
                delta += self.k * (actual - expected) / (n - 1)
            self.ratings[id_i] = ratings[id_i] + delta


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
        # FIX: Snapshot pre-race ratings/RDs BEFORE updating any participant
        # (Original code updated during iteration, causing order-dependent results)
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
    """Compute Elo/Glicko ratings for horse/jockey/trainer"""
    log_step("Computing Elo/Glicko ratings...")
    horse_elo = EloRatingSystem(k_factor=20)
    horse_glicko = GlickoRatingSystem()
    jockey_elo = EloRatingSystem(k_factor=12)
    trainer_elo = EloRatingSystem(k_factor=8)

    horse_elo_col = np.full(len(df), np.nan)
    horse_glicko_col = np.full(len(df), np.nan)
    horse_glicko_rd_col = np.full(len(df), np.nan)
    jockey_elo_col = np.full(len(df), np.nan)
    trainer_elo_col = np.full(len(df), np.nan)

    race_ids = df["race_id"].unique()
    period = 0

    for race_id in tqdm(race_ids, desc="  [Elo] Computing", leave=False):
        mask = df["race_id"] == race_id
        race_df = df[mask]

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
    df["combined_elo"] = df["horse_elo"] * 0.6 + df["jockey_elo"] * 0.25 + df["trainer_elo"] * 0.15
    df["combined_elo_rank"] = df.groupby("race_id")["combined_elo"].rank(ascending=False)

    log_step(f"Elo done: {len(horse_elo.ratings):,} horses / {len(jockey_elo.ratings):,} jockeys")
    return df


# ══════════════════════════════════════════════
# Part 2: Target Encoding (CV-safe)
# ══════════════════════════════════════════════
def target_encode_cv(df, col, target, n_splits=5, smoothing=20):
    """Time-series safe target encoding: only uses past data for each fold."""
    from sklearn.model_selection import TimeSeriesSplit
    global_mean = df[target].mean()
    encoded = pd.Series(np.nan, index=df.index)
    # Use TimeSeriesSplit to prevent future data leaking into past encoding
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(df):
        train = df.iloc[train_idx]
        agg = train.groupby(col)[target].agg(["mean", "count"])
        smooth_mean = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
        encoded.iloc[val_idx] = df.iloc[val_idx][col].map(smooth_mean)
    # For the first fold's training portion (no prior data), use global mean
    encoded = encoded.fillna(global_mean)
    return encoded


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
            df[new_col] = df[new_col].fillna(df_wr[tcol].mean())

    # _is_win / _is_top3 were added to df_wr (copy), not df itself.
    # No cleanup needed on df.

    log_step("Target encoding done")
    return df


# ══════════════════════════════════════════════
# Part 3: GOD-level Feature Engineering
# ══════════════════════════════════════════════

# Import base classes from train_advanced
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import shared GOD ensemble classes (server-compatible, pickle-safe)
_god_ensemble_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _god_ensemble_dir not in sys.path:
    sys.path.insert(0, _god_ensemble_dir)
from god_ensemble import _GodEnsembleModel, _EnsembleLGBModel, TemperatureScaler
try:
    from train_advanced import DataProcessor, FeatureEngine
except ImportError as e:
    print(f"\n❌ train_advanced.py が見つかりません。")
    print(f"   train_god.py と同じディレクトリに train_advanced.py を配置してください。")
    print(f"   Error: {e}")
    sys.exit(1)


def add_god_features(df):
    """All additional feature engineering beyond train_ultimate"""
    log_step("GOD-level feature engineering...")

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

    # ── 4. Bloodline statistics ──
    log_step("Bloodline statistics...", 1)
    df = _bloodline_stats(df)

    # ── 5. Pace analysis ──
    log_step("Pace analysis...", 1)
    df = _pace_features(df)

    # ── 6. Class transition ──
    log_step("Class transition features...", 1)
    df = _class_transition_features(df)

    # ── 7. Combination stats ──
    log_step("Combination statistics...", 1)
    df = _combination_stats(df)

    # ── 8. Jockey-course interaction ──
    if "jockey_id" in df.columns and "venue" in df.columns:
        jc_stats = df.groupby(["jockey_id", "venue"])["finish_position"].transform(
            lambda x: x.shift(1).expanding().mean())
        df["jockey_venue_avg_pos"] = jc_stats

    log_step("GOD feature engineering done")
    return df


def _bloodline_stats(df):
    """Sire/Dam win rates by distance/surface/condition.
    IMPORTANT: Data must be sorted by date before calling this function.
    All shift(1) operations rely on chronological order to avoid future leakage.
    """
    if "sire" not in df.columns:
        return df

    for parent in ["sire", "dam"]:
        if parent not in df.columns:
            continue

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
            gd = df.groupby([parent, "_tmp_dist_cat"])["finish_position"]
            df[f"{parent}_dist_win_rate"] = gd.transform(
                lambda x: (x.shift(1) == 1).expanding().mean())
            df.drop(columns=["_tmp_dist_cat"], inplace=True)

        if "course_type_enc" in df.columns:
            gs = df.groupby([parent, "course_type_enc"])["finish_position"]
            df[f"{parent}_surface_win_rate"] = gs.transform(
                lambda x: (x.shift(1) == 1).expanding().mean())

    if "dam_sire" in df.columns:
        g = df.groupby("dam_sire")["finish_position"]
        df["dam_sire_win_rate"] = g.transform(
            lambda x: (x.shift(1) == 1).expanding().mean())
        df["dam_sire_avg_pos"] = g.transform(
            lambda x: x.shift(1).expanding().mean())

    if "sire_sire" in df.columns:
        g = df.groupby("sire_sire")["finish_position"]
        df["sire_sire_win_rate"] = g.transform(
            lambda x: (x.shift(1) == 1).expanding().mean())

    return df


def _pace_features(df):
    """Pace analysis from passing orders and running style classification.
    IMPORTANT: corner_1~4 are RACE RESULTS (not available at prediction time).
    Only HISTORICAL averages (shift(1)) can be used as features.
    Raw values (position_change_c1c4, early_pace_pct, running_style) are
    intermediate columns only — they are added to LEAK_EXCLUDE to prevent
    them from being used as features.
    """
    corner_cols = [c for c in ["corner_1", "corner_2", "corner_3", "corner_4"] if c in df.columns]
    if not corner_cols:
        return df

    # These are intermediate columns (current race results) — NOT features
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

    # Only HISTORICAL (shift(1)) features are safe to use
    if "horse_id" in df.columns and "_tmp_early_pace_pct" in df.columns:
        g = df.groupby("horse_id")["_tmp_early_pace_pct"]
        df["horse_avg_early_pace"] = g.transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        df["horse_pace_consistency"] = g.transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).std())

    if "horse_id" in df.columns and "_tmp_running_style" in df.columns:
        df["horse_modal_style"] = df.groupby("horse_id")["_tmp_running_style"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).apply(
                lambda v: v.mode().iloc[0] if len(v.mode()) > 0 else np.nan, raw=False))

    if "horse_id" in df.columns and "_tmp_position_change_c1c4" in df.columns:
        df["horse_avg_pos_change"] = df.groupby("horse_id")["_tmp_position_change_c1c4"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    # Drop intermediate columns (race results — NOT allowed as features)
    tmp_cols = [c for c in df.columns if c.startswith("_tmp_")]
    df.drop(columns=tmp_cols, inplace=True)

    return df


def _class_transition_features(df):
    """Class step up/down features.
    FIX: class_momentum_3 now uses shift(1) to exclude current row.
    FIX: prize_money is a race RESULT (leak) — removed entirely.
    """
    if "horse_id" not in df.columns:
        return df

    if "odds" in df.columns:
        race_level = df.groupby("race_id")["odds"].transform("median")
        # race_class_proxy is available at prediction time (based on odds)
        df["race_class_proxy"] = race_level
        prev_class = df.groupby("horse_id")["race_class_proxy"].shift(1)
        df["class_change"] = df["race_class_proxy"] - prev_class
        df["class_step_up"] = (df["class_change"] < 0).astype(int)
        df["class_step_down"] = (df["class_change"] > 0).astype(int)
        # FIX: shift(1) to exclude current row from rolling window
        df["class_momentum_3"] = df.groupby("horse_id")["class_change"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    # REMOVED: prize_money is a RACE RESULT (not available at prediction time)
    # Using it would be data leakage.

    return df


def _combination_stats(df):
    """Jockey-trainer combo, horse-distance combo, horse-venue combo"""
    if "jockey_id" in df.columns and "trainer_id" in df.columns:
        combo_key = df["jockey_id"].astype(str) + "_" + df["trainer_id"].astype(str)
        df["_jt_combo"] = combo_key
        g = df.groupby("_jt_combo")["finish_position"]
        df["jt_combo_avg_pos"] = g.transform(lambda x: x.shift(1).expanding().mean())
        df["jt_combo_win_rate"] = g.transform(lambda x: (x.shift(1) == 1).expanding().mean())
        df["jt_combo_n_races"] = g.transform(lambda x: x.shift(1).expanding().count())
        df.drop(columns=["_jt_combo"], inplace=True)

    if "horse_id" in df.columns and "distance" in df.columns:
        dist_bucket = (df["distance"] // 200 * 200).astype(int)
        combo_key = df["horse_id"].astype(str) + "_" + dist_bucket.astype(str)
        df["_hd_combo"] = combo_key
        g = df.groupby("_hd_combo")["finish_position"]
        df["horse_dist_combo_avg"] = g.transform(lambda x: x.shift(1).expanding().mean())
        df["horse_dist_combo_win"] = g.transform(lambda x: (x.shift(1) == 1).expanding().mean())
        df.drop(columns=["_hd_combo"], inplace=True)

    if "horse_id" in df.columns and "venue" in df.columns:
        combo_key = df["horse_id"].astype(str) + "_" + df["venue"].astype(str)
        df["_hv_combo"] = combo_key
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
# Part 5: Adversarial Validation (FIXED)
# ══════════════════════════════════════════════
def adversarial_validation(X_train, X_test, feature_names):
    """
    Train/test distribution shift detection.
    FIX: Remove year/month before advval to prevent AUC=1.0 leakage.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    log_step("Adversarial Validation (fixed: year/month excluded)...")

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

    return auc, top_drift, keep_names, imp


def adversarial_debiasing(feature_names, adv_importances, adv_feature_names, threshold_pct=90):
    """Soft-downweight features with high distribution shift."""
    log_step("Adversarial debiasing...")
    adv_imp_dict = dict(zip(adv_feature_names, adv_importances))

    if len(adv_importances) > 0:
        threshold = np.percentile(adv_importances, threshold_pct)
    else:
        threshold = float("inf")

    feature_weights = np.ones(len(feature_names))
    debiased = []
    for i, fname in enumerate(feature_names):
        imp = adv_imp_dict.get(fname, 0)
        if imp > threshold:
            feature_weights[i] = 0.5
            debiased.append(fname)

    if debiased:
        log_step(f"Downweighted {len(debiased)} features: {debiased[:5]}...", 1)

    return feature_weights


# ══════════════════════════════════════════════
# Part 6: Boruta Feature Selection (FIXED)
# ══════════════════════════════════════════════
def is_protected_feature(fname):
    for pat in PROTECTED_FEATURE_PATTERNS:
        if pat in fname:
            return True
    return False


def boruta_selection(X, y, feature_names, max_iter=100, min_features=30):
    """
    Boruta with FIXES:
    - min_features enforced (default 30)
    - Protected domain features always kept
    - year/month excluded from Boruta input
    """
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

    # Statistical test
    selected_mask_boruta = np.zeros(n_features, dtype=bool)
    for i in range(n_features):
        p_value = 1 - stats.binom.cdf(hit_counts[i] - 1, max_iter, 0.5)
        selected_mask_boruta[i] = p_value < 0.05

    # Map back to full feature set
    full_mask = np.zeros(len(feature_names), dtype=bool)
    for bi, fi in enumerate(keep_idx):
        if selected_mask_boruta[bi]:
            full_mask[fi] = True

    # Force-include protected features
    protected_added = []
    for i, fname in enumerate(feature_names):
        if is_protected_feature(fname) and not full_mask[i]:
            full_mask[i] = True
            protected_added.append(fname)

    # Enforce minimum feature count
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
    """Wraps LGB Booster for sklearn permutation_importance compatibility."""
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

        # SHAP importance
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

        # Permutation importance
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

    # Stability selection
    stability_threshold = CONFIG["stability_threshold"]
    shap_stable = (shap_important / n_folds) >= stability_threshold
    perm_stable = (perm_important / n_folds) >= stability_threshold

    # Union
    hybrid_mask = boruta_mask | shap_stable | perm_stable

    # Always keep protected features
    for i, fname in enumerate(feature_names):
        if is_protected_feature(fname):
            hybrid_mask[i] = True

    # Minimum features
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

        # FIX: Pre-convert dates to int64 (nanoseconds) ONCE, avoiding per-element pd.Timestamp
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
                    if val_start_ns > 0:  # not NaT
                        # Purge: remove training rows within purge_days before val start
                        purge_cutoff_ns = val_start_ns - purge_ns
                        train_mask = dates_ns[train_idx] < purge_cutoff_ns
                        train_idx = train_idx[train_mask]
                        # Embargo: remove val rows within embargo_days after val start
                        if self.embargo_days > 0:
                            embargo_end_ns = val_start_ns + embargo_ns
                            val_mask = dates_ns[val_idx] >= embargo_end_ns
                            val_idx = val_idx[val_mask]
                except Exception:
                    pass

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx


# ══════════════════════════════════════════════
# Part 9: Optuna Optimization (FIXED)
# ══════════════════════════════════════════════
def optimize_with_pruning(X_train, y_train, X_val, y_val, task, n_trials=200,
                          use_gpu=False, sample_weights_tr=None):
    """Optuna + Median Pruning. FIX: metric name 'l1' for new LightGBM."""
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
            "max_depth": trial.suggest_int("max_depth", 3, 15),
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

        # Pruning metric must match study direction: minimize l1, maximize auc
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
    """Train TabNet as additional base learner"""
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
    """LightGBM LambdaRank for direct ranking optimization"""
    import lightgbm as lgb

    log_step("Training LambdaRank...", 1)
    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 3, 5],
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

    # Relevance labels
    y_rel_train = np.where(y_train == 1, 10, np.where(y_train <= 3, 5,
                  np.where(y_train <= 5, 3, np.where(y_train <= 10, 1, 0))))
    y_rel_val = np.where(y_val == 1, 10, np.where(y_val <= 3, 5,
                np.where(y_val <= 5, 3, np.where(y_val <= 10, 1, 0))))

    dtrain = lgb.Dataset(X_train, label=y_rel_train, group=groups_train)
    dval = lgb.Dataset(X_val, label=y_rel_val, group=groups_val, reference=dtrain)

    try:
        model = lgb.train(
            params, dtrain, num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        log_step("LambdaRank training complete", 1)
        return model
    except Exception as e:
        log_step(f"LambdaRank failed: {e}", 1)
        return None


# ══════════════════════════════════════════════
# Part 12: 2-Level Stacking Ensemble
# ══════════════════════════════════════════════
def train_god_stacking(X, y, task, race_ids, race_dates, models_config,
                       n_folds=8, use_gpu=False, sample_weights=None):
    """
    2-Level Stacking with Purged CV:
      Level 1: LGB + XGB + CatBoost + TabNet -> OOF predictions
      Level 2: LGB meta-learner on OOF
      Level 3: Ridge/LogReg final blend
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

        fold_preds = np.full((len(va_idx), n_models), np.nan)

        for model_idx, (model_name, model_params) in enumerate(models_config):
            try:
                if model_name == "lightgbm":
                    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
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
                    # Pass early_stopping_rounds to constructor for XGBoost 2.0+ compatibility
                    es_rounds = params.pop("early_stopping_rounds", 100)
                    if use_gpu:
                        params["tree_method"] = "hist"
                        params["device"] = "cuda"
                    if task == "regression":
                        m = xgb.XGBRegressor(**params, early_stopping_rounds=es_rounds)
                    else:
                        m = xgb.XGBClassifier(**params, early_stopping_rounds=es_rounds)
                    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                          sample_weight=w_tr, verbose=False)
                    pred = m.predict(X_va) if task == "regression" else m.predict_proba(X_va)[:, 1]
                    model = m

                elif model_name == "catboost":
                    try:
                        from catboost import CatBoostRegressor, CatBoostClassifier
                        params = model_params.copy()
                        if use_gpu:
                            params["task_type"] = "GPU"
                        if task == "regression":
                            m = CatBoostRegressor(**params)
                        else:
                            m = CatBoostClassifier(**params)
                        m.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                              sample_weight=w_tr, verbose=0, early_stopping_rounds=100)
                        pred = m.predict(X_va) if task == "regression" else m.predict_proba(X_va)[:, 1]
                        model = m
                    except ImportError:
                        log_step("CatBoost not installed, using LGB substitute", 2)
                        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
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
                    model = train_tabnet(X_tr, y_tr, X_va, y_va, task, use_gpu)
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

    # Level 2: LGB Meta-learner with proper train/val split
    log_step("Level 2: LGB Meta-learner...", 1)

    valid_mask = ~np.isnan(oof_preds).all(axis=1)
    X_meta = np.nan_to_num(oof_preds[valid_mask], nan=0.0)
    y_meta = y[valid_mask]

    meta_model = None
    l3_model = None

    if len(X_meta) >= 200:
        # FIX: Use time-series split to avoid overfitting meta-model
        meta_split = int(len(X_meta) * 0.8)
        X_meta_tr, X_meta_va = X_meta[:meta_split], X_meta[meta_split:]
        y_meta_tr, y_meta_va = y_meta[:meta_split], y_meta[meta_split:]

        meta_lgb_params = {
            "objective": "regression" if task == "regression" else "binary",
            "metric": "l1" if task == "regression" else "binary_logloss",
            "num_leaves": 15, "learning_rate": 0.05, "verbose": -1,
            "n_jobs": -1, "seed": 42, "reg_alpha": 1.0, "reg_lambda": 1.0,
            "min_child_samples": 50,
        }
        dtrain_meta = lgb.Dataset(X_meta_tr, label=y_meta_tr)
        dval_meta = lgb.Dataset(X_meta_va, label=y_meta_va, reference=dtrain_meta)
        meta_model = lgb.train(
            meta_lgb_params, dtrain_meta, num_boost_round=200,
            valid_sets=[dval_meta],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        # Level 3: Ridge/LogReg — train on train split, evaluate on val split
        meta_pred_tr = meta_model.predict(X_meta_tr)
        meta_pred_va = meta_model.predict(X_meta_va)
        X_l3_tr = np.column_stack([X_meta_tr, meta_pred_tr.reshape(-1, 1)])
        X_l3_va = np.column_stack([X_meta_va, meta_pred_va.reshape(-1, 1)])
        if task == "regression":
            l3_model = Ridge(alpha=1.0)
        else:
            l3_model = LogisticRegression(C=1.0, max_iter=1000)
        l3_model.fit(X_l3_tr, y_meta_tr)

        # Evaluate on held-out validation (not same data used for training)
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

        # Refit on full meta data for final model
        dtrain_full = lgb.Dataset(X_meta, label=y_meta)
        meta_model = lgb.train(meta_lgb_params, dtrain_full, num_boost_round=meta_model.best_iteration)
        meta_pred_full = meta_model.predict(X_meta)
        X_l3_full = np.column_stack([X_meta, meta_pred_full.reshape(-1, 1)])
        l3_model.fit(X_l3_full, y_meta)
    else:
        log_step("Not enough meta data, skipping level 2/3", 1)

    avg_score = np.mean(fold_scores) if fold_scores else 0
    log_step(f"Average fold score: {avg_score:.4f}", 1)

    return {
        "fold_models": fold_models, "meta_model": meta_model, "l3_model": l3_model,
        "oof_preds": oof_preds, "fold_scores": fold_scores, "avg_score": avg_score,
    }


# ══════════════════════════════════════════════
# Part 13: Bayesian Model Averaging
# ══════════════════════════════════════════════
def bayesian_model_averaging(oof_preds, y_true, task):
    """Weight ensemble members by posterior probability via log-likelihood."""
    log_step("Bayesian Model Averaging...")
    n_models = oof_preds.shape[1]
    valid_mask = ~np.isnan(oof_preds).all(axis=1) & ~np.isnan(y_true)
    X = oof_preds[valid_mask]
    y = y_true[valid_mask]

    log_likelihoods = np.zeros(n_models)
    for i in range(n_models):
        preds = np.nan_to_num(X[:, i], nan=0.5 if task == "binary" else y.mean())
        if task == "binary":
            preds = np.clip(preds, 1e-7, 1 - 1e-7)
            ll = np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))
        else:
            ll = -np.mean((y - preds) ** 2)
        log_likelihoods[i] = ll

    log_likelihoods -= log_likelihoods.max()
    weights = np.exp(log_likelihoods)
    weights = weights / weights.sum()

    for i in range(n_models):
        log_step(f"Model {i} BMA weight: {weights[i]:.4f}", 1)
    return weights


# ══════════════════════════════════════════════
# Part 14: Calibration
# ══════════════════════════════════════════════
def isotonic_calibration(y_true, y_pred):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    ir.fit(y_pred, y_true)
    return ir


# TemperatureScaler is imported from god_ensemble.py



# ══════════════════════════════════════════════
# Part 15: Pseudo Labeling
# ══════════════════════════════════════════════
def pseudo_label_round(X_labeled, y_labeled, task, use_gpu=False, threshold=0.90):
    """Self-training: re-label low-confidence samples using high-confidence model.

    FIX: Original version duplicated labeled data. Now uses proper self-training:
    1. Train on 80% (confident data)
    2. Predict on remaining 20%
    3. For high-confidence predictions, REPLACE original labels with model predictions
       (soft relabeling for noise reduction)
    4. Return same-sized dataset with refined labels
    """
    import lightgbm as lgb
    log_step(f"Self-training (threshold={threshold})...", 1)

    n = len(X_labeled)
    split = int(n * 0.8)
    X_tr, X_refine = X_labeled[:split], X_labeled[split:]
    y_tr, y_refine = y_labeled[:split], y_labeled[split:]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    params = {
        "objective": "binary" if task == "binary" else "regression",
        "metric": "binary_logloss" if task == "binary" else "l1",
        "verbose": -1, "n_jobs": -1, "num_leaves": 63, "learning_rate": 0.05,
    }
    if use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = True

    m = lgb.train(params, dtrain, num_boost_round=500)
    preds = m.predict(X_refine)

    # Only replace labels where model is very confident
    y_refined = y_refine.copy()
    if task == "binary":
        confident_mask = (preds > threshold) | (preds < (1 - threshold))
        y_refined[confident_mask] = (preds[confident_mask] > 0.5).astype(float)
        n_replaced = confident_mask.sum()
    else:
        # For regression, blend prediction with original (noise reduction)
        blend_weight = 0.3
        y_refined = (1 - blend_weight) * y_refine + blend_weight * preds
        n_replaced = len(preds)

    # Return same-sized dataset (no duplication)
    X_out = np.vstack([X_tr, X_refine])
    y_out = np.concatenate([y_tr, y_refined])
    log_step(f"Refined {n_replaced}/{len(y_refine)} labels", 2)

    return X_out, y_out


# ══════════════════════════════════════════════
# Part 16: Noise Injection
# ══════════════════════════════════════════════
def add_training_noise(X, noise_std=0.01):
    rng = np.random.RandomState(42)
    noise = rng.normal(0, noise_std, X.shape)
    mask = X != 0
    X_noisy = X.copy()
    X_noisy[mask] += noise[mask] * np.abs(X[mask])
    return X_noisy


# ══════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════
def main():
    global _GLOBAL_START
    _GLOBAL_START = time.time()

    print("=" * 70)
    print("  KEIBA-PREDICTOR GOD-LEVEL TRAINING PIPELINE")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["model_dir"], exist_ok=True)

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

    # FIX: Sort by date to ensure shift(1) never uses future data
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

    # Free original dataframe memory
    del df
    gc.collect()
    log_step(f"Memory freed. DataFrame shape: {df_train.shape}")

    # Build complete feature list
    # Additional leak columns: race results or intermediate values not available at prediction time
    GOD_LEAK_COLS = {
        "race_class_proxy",   # intermediate (could mislead — keep class_change instead)
        "race_prize_level",   # from prize_money (race result)
        "prize_class_ratio",  # from prize_money (race result)
        "_is_win", "_is_top3",  # target variables
    }
    # Also exclude any remaining _tmp_ columns
    tmp_cols = {c for c in df_train.columns if c.startswith("_tmp_")}
    all_exclude = fe.LEAK_FEATURES | fe.EXCLUDE_COLS | GOD_LEAK_COLS | tmp_cols

    feature_cols = fe.get_feature_columns(df_train)
    feature_set = set(feature_cols)  # O(1) lookup instead of O(n)
    all_numeric = [c for c in df_train.columns
                   if df_train[c].dtype in ["float64", "int64", "int32", "float32"]
                   and c not in all_exclude]
    for c in all_numeric:
        if c not in feature_set:
            feature_cols.append(c)
            feature_set.add(c)

    # Final safety check: remove any column that's in all_exclude
    feature_cols = [c for c in feature_cols if c not in all_exclude]
    log_step(f"Total features: {len(feature_cols)}")
    log_step(f"Excluded {len(all_exclude)} leak/ID columns")

    # ── Step 6: Data Preparation ──
    print("\n" + "=" * 70)
    print("  [6/12] DATA PREPARATION")
    print("=" * 70)

    X = df_train[feature_cols].values.copy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y_position = df_train["finish_position"].values
    y_win = (df_train["finish_position"] == 1).astype(int).values
    y_top3 = (df_train["finish_position"] <= 3).astype(int).values

    race_ids = df_train["race_id"].values
    race_dates = df_train["race_date"].values if "race_date" in df_train.columns else None

    sample_weights = compute_time_decay_weights(df_train, CONFIG["time_decay_half_life_days"])
    log_step(f"Sample weights range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

    # Save stats before freeing DataFrame
    _n_train_rows = len(df_train)

    # Free DataFrame memory (X, y arrays are already extracted)
    del df_train
    gc.collect()
    log_step("DataFrame freed from memory")

    tscv = TimeSeriesSplit(n_splits=CONFIG["n_cv_folds"])
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-1]
    X_train, X_val = X[train_idx], X[val_idx]
    w_train = sample_weights[train_idx]

    log_step(f"Train: {len(X_train):,} / Val: {len(X_val):,}")

    # ── Step 7: Adversarial Validation ──
    print("\n" + "=" * 70)
    print("  [7/12] ADVERSARIAL VALIDATION + DEBIASING")
    print("=" * 70)

    adv_auc, adv_drift, adv_feat_names, adv_imp = adversarial_validation(
        X_train, X_val, feature_cols)
    adv_feature_weights = adversarial_debiasing(
        feature_cols, adv_imp, adv_feat_names, threshold_pct=90)

    # NOTE: Adversarial debiasing weights are saved but NOT applied to X globally.
    # Reason: Permanently scaling X reduces signal for Boruta/Optuna feature selection.
    # Instead, the weights are logged for analysis. The actual distribution shift is
    # handled by: (1) excluding year/month from features, (2) Boruta removing unstable
    # features, (3) PurgedTimeSeriesSplit with embargo in stacking.
    if adv_feature_weights is not None and (adv_feature_weights < 1.0).any():
        n_downweighted = int((adv_feature_weights < 1.0).sum())
        log_step(f"Adversarial debiasing identified {n_downweighted} drifted features (logged, not applied to X)")

    # ── Step 8: Hybrid Feature Selection ──
    print("\n" + "=" * 70)
    print("  [8/12] HYBRID FEATURE SELECTION")
    print("=" * 70)

    # FIX: Use most RECENT training data for Boruta (not random — respects temporal order)
    sample_size = min(200000, len(X_train))
    sample_idx = np.arange(len(X_train) - sample_size, len(X_train))

    boruta_selected, boruta_mask = boruta_selection(
        X_train[sample_idx], y_top3[train_idx][sample_idx],
        feature_cols, max_iter=CONFIG["boruta_max_iter"],
        min_features=CONFIG["boruta_min_features"]
    )

    # FIX: Use ONLY training data for hybrid feature selection (no validation leakage)
    hybrid_selected, hybrid_mask = hybrid_feature_selection(
        X[train_idx], y_top3[train_idx], feature_cols, boruta_mask,
        n_folds=min(5, CONFIG["n_cv_folds"])
    )

    X_selected = X[:, hybrid_mask]
    X_train_sel = X_selected[train_idx]
    X_val_sel = X_selected[val_idx]
    selected_features = hybrid_selected

    log_step(f"Final features: {len(selected_features)} / {len(feature_cols)}")

    with open(f"{CONFIG['output_dir']}/feature_selection.json", "w") as f:
        json.dump({
            "boruta_selected": boruta_selected,
            "hybrid_selected": hybrid_selected,
            "total_features": len(feature_cols),
            "selected_features": len(selected_features),
        }, f, ensure_ascii=False, indent=2)

    # ── Step 9: Optuna Optimization ──
    print("\n" + "=" * 70)
    print("  [9/12] OPTUNA OPTIMIZATION")
    print("=" * 70)

    tasks = [
        ("position", "regression", y_position),
        ("top3", "binary", y_top3),
        ("win", "binary", y_win),
    ]

    all_results = {}
    all_best_params = {}

    for task_name, task_type, y_target in tasks:
        print(f"\n  --- {task_name} ({task_type}) ---")
        y_tr = y_target[train_idx]
        y_va = y_target[val_idx]

        log_step(f"Optuna: {CONFIG['optuna_trials']} trials with pruning...")
        best_lgb_params = optimize_with_pruning(
            X_train_sel, y_tr, X_val_sel, y_va,
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

    # ── Step 10: Pseudo Labeling + GOD Stacking ──
    print("\n" + "=" * 70)
    print("  [10/12] PSEUDO LABELING + GOD STACKING")
    print("=" * 70)

    for task_name, task_type, y_target in tasks:
        log_step(f"=== {task_name} ===")
        result = all_results[task_name]

        # FIX: Use ONLY training data for pseudo labeling (no validation leakage)
        X_train_selected = X_selected[train_idx]
        y_train_target = y_target[train_idx]
        X_aug, y_aug = X_train_selected.copy(), y_train_target.copy()
        if task_type == "binary":
            for _ in range(CONFIG["pseudo_label_rounds"]):
                X_aug, y_aug = pseudo_label_round(
                    X_aug, y_aug, task_type, use_gpu=use_gpu,
                    threshold=CONFIG["pseudo_label_threshold"],
                )

        X_noisy = add_training_noise(X_aug, CONFIG["noise_std"])

        # FIX: Self-training preserves data size, so dates/ids/weights always match
        aug_dates = race_dates[train_idx] if race_dates is not None else None
        aug_race_ids = race_ids[train_idx] if race_ids is not None else None
        aug_weights = sample_weights[train_idx]

        stack_result = train_god_stacking(
            X_noisy, y_aug, task_type, aug_race_ids, aug_dates,
            result["models_config"], n_folds=CONFIG["n_cv_folds"],
            use_gpu=use_gpu, sample_weights=aug_weights,
        )

        bma_weights = bayesian_model_averaging(stack_result["oof_preds"], y_aug, task_type)
        result["stacking"] = stack_result
        result["bma_weights"] = bma_weights

        # Free augmented data
        del X_aug, y_aug, X_noisy
        gc.collect()

    # ── Step 11: LambdaRank ──
    print("\n" + "=" * 70)
    print("  [11/12] LAMBDARANK TRAINING")
    print("=" * 70)

    lambdarank_model = None
    if race_ids is not None:
        train_race_ids = race_ids[train_idx]
        val_race_ids = race_ids[val_idx]

        _, tr_starts = np.unique(train_race_ids, return_index=True)
        tr_counts = pd.Series(train_race_ids).groupby(train_race_ids).transform("count").values
        tr_group_sizes = tr_counts[np.sort(tr_starts)].tolist()

        _, va_starts = np.unique(val_race_ids, return_index=True)
        va_counts = pd.Series(val_race_ids).groupby(val_race_ids).transform("count").values
        va_group_sizes = va_counts[np.sort(va_starts)].tolist()

        best_params = all_best_params.get("position", {})
        lambdarank_model = train_lambdarank(
            X_train_sel, y_position[train_idx], tr_group_sizes,
            X_val_sel, y_position[val_idx], va_group_sizes,
            best_params, use_gpu=use_gpu
        )

    # ── Step 12: Calibration + Save ──
    print("\n" + "=" * 70)
    print("  [12/12] CALIBRATION + MODEL SAVING")
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

        # Use all-model OOF mean for calibration since the server model now uses
        # the full GOD ensemble (all model types + stacking), not just LGB.
        # nanmean across model types handles missing model columns gracefully.
        oof_mean_all = np.nanmean(oof, axis=1)
        all_valid_mask = ~np.isnan(oof_mean_all)

        # Use training-only y values (matching oof_preds which is from training data)
        y_train_win = y_win[train_idx]
        y_train_top3 = y_top3[train_idx]
        y_cal = y_train_win if task_name == "win" else y_train_top3
        if len(all_valid_mask) > len(y_cal):
            all_valid_mask = all_valid_mask[:len(y_cal)]
            oof_valid = oof_mean_all[:len(y_cal)][all_valid_mask]
            y_valid = y_cal[all_valid_mask]
        else:
            oof_valid = oof_mean_all[all_valid_mask]
            y_valid = y_cal[all_valid_mask]
        oof_mean = oof_valid  # All-model mean predictions for calibration

        try:
            calibration_models[task_name] = isotonic_calibration(y_valid, oof_mean)
            log_step(f"Isotonic calibration fitted: {task_name}")
        except Exception as e:
            log_step(f"Isotonic calibration failed for {task_name}: {e}")

        try:
            ts = TemperatureScaler()
            ts.fit(y_valid, oof_mean)
            temperature_scalers[task_name] = ts
            log_step(f"Temperature scaling: {task_name} T={ts.temperature:.3f}")
        except Exception as e:
            log_step(f"Temperature scaling failed for {task_name}: {e}")

    # ── Final held-out evaluation on val_idx ──
    log_step("Final held-out evaluation on validation set...")
    holdout_results = {}
    for task_name, task_type, y_target in tasks:
        result = all_results[task_name]
        stack = result["stacking"]
        lgb_models = stack["fold_models"].get("lightgbm", [])
        if not lgb_models:
            continue

        # Ensemble prediction: average across all fold models
        preds_list = []
        for m in lgb_models:
            try:
                preds_list.append(m.predict(X_val_sel))
            except Exception:
                pass
        if preds_list:
            ensemble_pred = np.mean(preds_list, axis=0)
            y_val_true = y_target[val_idx]
            if task_type == "regression":
                from sklearn.metrics import mean_absolute_error
                score = mean_absolute_error(y_val_true, ensemble_pred)
                log_step(f"  {task_name} holdout MAE: {score:.4f}", 1)
            else:
                from sklearn.metrics import roc_auc_score
                try:
                    score = roc_auc_score(y_val_true, ensemble_pred)
                    log_step(f"  {task_name} holdout AUC: {score:.4f}", 1)
                except ValueError:
                    score = 0.5
            holdout_results[task_name] = score

    # Server-compatible model
    log_step("Building server-compatible model...")

    server_models = {}
    eval_results = {}
    feat_importances = {}

    for task_name, result in all_results.items():
        stack = result["stacking"]
        fold_models = stack["fold_models"]  # dict: {model_name: [fold_models]}
        meta_model = stack["meta_model"]     # LGB Booster or None
        l3_model = stack["l3_model"]         # Ridge/LogReg or None

        # Need at least one model type with fold models
        has_any_models = any(len(v) > 0 for v in fold_models.values())
        if not has_any_models:
            continue

        # Calibration and temperature scaling for binary tasks
        cal = calibration_models.get(task_name) if task_name in ["win", "top3"] else None
        ts = temperature_scalers.get(task_name) if task_name in ["win", "top3"] else None

        # BMA weights
        bma_w = result.get("bma_weights", None)
        if isinstance(bma_w, np.ndarray):
            bma_w = bma_w.tolist() if len(bma_w) > 0 else None
        elif isinstance(bma_w, list) and len(bma_w) == 0:
            bma_w = None

        # Get ordered model names from models_config
        models_config_names = [name for name, _ in result["models_config"]]

        task_type = result["task_type"]

        # Build full GOD ensemble model (all model types + stacking + calibration)
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

        # Count models per type for logging
        model_counts = {k: len(v) for k, v in fold_models.items() if len(v) > 0}
        log_step(f"  {task_name}: GOD ensemble built - {model_counts}, "
                 f"meta={'yes' if meta_model else 'no'}, "
                 f"l3={'yes' if l3_model else 'no'}, "
                 f"cal={'yes' if cal else 'no'}, "
                 f"ts={'yes' if ts else 'no'}")

        scores = stack["fold_scores"]
        metric = "mae" if task_type == "regression" else "auc"
        eval_results[task_name] = {
            f"mean_{metric}": float(np.mean(scores)) if scores else 0,
            f"std_{metric}": float(np.std(scores)) if scores else 0,
            "holdout": float(holdout_results.get(task_name, 0)),
        }

        # Average feature importance across all LGB folds
        try:
            all_imp = server_models[task_name].feature_importance(importance_type="gain")
            if all_imp is not None:
                feat_importances[task_name] = pd.DataFrame({
                    "feature": selected_features, "importance": all_imp,
                }).sort_values("importance", ascending=False)
        except Exception:
            pass

    # Identify GOD-specific features (not in base FeatureEngine)
    base_features = set(fe.get_feature_columns(pd.DataFrame(columns=feature_cols)))
    god_only_features = [f for f in selected_features if f not in base_features]

    save_data = {
        "version": "3.0-god",
        "models": server_models,
        "feature_columns": selected_features,
        "evaluation_results": eval_results,
        "feature_importance": {k: v.to_dict() for k, v in feat_importances.items()},
        "saved_at": datetime.now().isoformat(),
        "n_features": len(selected_features),
        "trained_on": "local_god_full",
        "model_type": "god_ensemble",
        # Server integration: list of GOD-specific features that need special computation
        # If server doesn't compute these, they'll be 0-filled (degraded accuracy)
        "god_features": god_only_features,
        "god_features_count": len(god_only_features),
        # LambdaRank model for ranking optimization (optional, used by server if available)
        "lambdarank_model": lambdarank_model,
    }
    if god_only_features:
        log_step(f"  {len(god_only_features)} GOD-specific features in model")
        log_step(f"   Server must compute these for full accuracy: {god_only_features[:5]}...", 1)

    model_path = f"{CONFIG['model_dir']}/keiba_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)
    log_step(f"Server-compatible GOD model: {model_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{CONFIG['model_dir']}/keiba_model_v{timestamp}.pkl"
    with open(backup_path, "wb") as f:
        pickle.dump(save_data, f)
    log_step(f"Backup: {backup_path}")

    # Full GOD model (legacy format kept for backward compatibility)
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

    god_path = f"{CONFIG['model_dir']}/keiba_model_god.pkl"
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

        for task_name in ["win", "top3", "position"]:
            if task_name in server_models and server_models[task_name]:
                # Use first LGB fold model for SHAP (representative)
                sm = server_models[task_name]
                lgb_folds = sm.fold_models.get("lightgbm", []) if hasattr(sm, "fold_models") else []
                if not lgb_folds:
                    continue
                shap_model = lgb_folds[0]
                explainer = shap.TreeExplainer(shap_model)
                sample_n = min(2000, len(X_val_sel))
                sample = pd.DataFrame(X_val_sel[:sample_n], columns=selected_features)
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
            plt.title(f"Feature Importance - {task_name} (GOD)")
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
        "pipeline": "god",
        "data_rows": int(_n_train_rows),
        "n_races": int(n_races),
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
            k: {"temperature": float(temperature_scalers[k].temperature)}
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
    print("  GOD-LEVEL TRAINING COMPLETE")
    print("=" * 70)

    for name, scores in eval_results.items():
        print(f"\n  [{name}]")
        for k, v in scores.items():
            print(f"    {k}: {v:.4f}")

    print(f"\n  Features: {len(feature_cols)} -> Boruta {len(boruta_selected)} "
          f"-> Hybrid {len(selected_features)}")
    print(f"  Adversarial AUC: {adv_auc:.4f}")
    print(f"  LambdaRank: {'Trained' if lambdarank_model else 'Skipped'}")
    print(f"  Runtime: {hours}h {minutes}m")

    print(f"\n  Output:")
    print(f"    {CONFIG['model_dir']}/keiba_model.pkl         (server-compatible)")
    print(f"    {CONFIG['model_dir']}/keiba_model_god.pkl     (full GOD model)")
    print(f"    {CONFIG['output_dir']}/training_report.json")
    print(f"    {CONFIG['output_dir']}/feature_selection.json")
    print(f"    {CONFIG['output_dir']}/shap_*.png")
    print(f"    {CONFIG['output_dir']}/importance_*.png")

    if god_only_features:
        print(f"\n  ⚠️  GOD-specific features: {len(god_only_features)}")
        print(f"     Server needs GOD feature computation for full accuracy.")
        print(f"     Without it, {len(god_only_features)} features will be 0-filled.")

    print(f"\n  Upload to server:")
    print(f"    scp {CONFIG['model_dir']}/keiba_model.pkl user@server:/opt/keiba-predictor/models/")

    return all_results, eval_results


if __name__ == "__main__":
    main()
