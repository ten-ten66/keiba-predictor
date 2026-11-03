"""
競馬予想AI - リトレイン用データ準備 (prepare_retrain.py)
=====================================================
bloodline.csv と _all_races.csv をマージし、血統統計を計算して
_all_races_with_blood.csv を出力する。

使い方:
  python prepare_retrain.py
  python prepare_retrain.py --copy-static   # static/ にもコピー

出力:
  data/_all_races_with_blood.csv
"""

import os
import sys
import shutil
import time
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

RACES_CSV = DATA_DIR / "_all_races.csv"
BLOOD_CSV = DATA_DIR / "bloodline.csv"
OUTPUT_CSV = DATA_DIR / "_all_races_with_blood.csv"


def load_data():
    """Load race data and bloodline data."""
    logger.info("Loading _all_races.csv ...")
    df = pd.read_csv(RACES_CSV, low_memory=False)
    logger.info(f"  Races: {len(df):,} rows, {df['race_id'].nunique():,} races, "
                f"{df['horse_id'].nunique():,} unique horses")

    logger.info("Loading bloodline.csv ...")
    bl = pd.read_csv(BLOOD_CSV)
    logger.info(f"  Bloodline: {len(bl):,} horses")
    return df, bl


def merge_bloodline(df, bl):
    """Merge bloodline data onto race data by horse_id."""
    # Ensure horse_id types match
    df["horse_id"] = df["horse_id"].astype(str)
    bl["horse_id"] = bl["horse_id"].astype(str)

    # Only keep core bloodline columns to avoid conflicts
    blood_cols = ["horse_id", "sire", "dam", "sire_sire", "dam_sire"]
    available = [c for c in blood_cols if c in bl.columns]
    bl_slim = bl[available].drop_duplicates(subset=["horse_id"])

    # Drop existing bloodline columns from df if any
    for c in available:
        if c in df.columns and c != "horse_id":
            df.drop(columns=[c], inplace=True)

    merged = df.merge(bl_slim, on="horse_id", how="left")

    has_sire = merged["sire"].notna().sum() if "sire" in merged.columns else 0
    coverage = has_sire / len(merged) * 100
    logger.info(f"  Merged: {has_sire:,}/{len(merged):,} rows have bloodline ({coverage:.1f}%)")

    # Coverage by unique horse
    total_horses = merged["horse_id"].nunique()
    horses_with_blood = merged.loc[merged["sire"].notna(), "horse_id"].nunique() if "sire" in merged.columns else 0
    logger.info(f"  Unique horses with bloodline: {horses_with_blood:,}/{total_horses:,} "
                f"({horses_with_blood/total_horses*100:.1f}%)")

    return merged



# ---------------------------------------------------------------------------
# 血統統計カラム定義（get_bloodline_stat_columns() で参照可能）
# ---------------------------------------------------------------------------
#
# 基本統計 (sire / dam 各々):
#   {parent}_win_rate        : 勝率（過去レースの1着率）
#   {parent}_top3_rate       : 複勝率（3着以内率）
#   {parent}_avg_pos         : 平均着順
#   {parent}_n_runners       : 出走頭数（サンプル数）
#   {parent}_std_pos         : 着順の標準偏差
#
# 距離カテゴリ別 (sire / dam 各々):
#   {parent}_dist_win_rate   : 距離カテゴリ別勝率
#   {parent}_dist_avg_pos    : 距離カテゴリ別平均着順
#   （距離カテゴリ: sprint=~1400m, mile=~1800m, mid=~2200m, long=2200m+）
#
# 馬場タイプ別 (sire / dam 各々):
#   {parent}_surface_win_rate : 馬場タイプ別勝率 (turf/dirt/jump)
#   {parent}_surface_avg_pos  : 馬場タイプ別平均着順
#
# 馬場状態別 (sire / dam 各々):
#   {parent}_cond_win_rate   : 馬場状態別勝率 (good/slightly_heavy/heavy/bad)
#
# 母父 (dam_sire):
#   dam_sire_win_rate        : 母父の勝率
#   dam_sire_top3_rate       : 母父の複勝率
#   dam_sire_avg_pos         : 母父の平均着順
#
# 父父 (sire_sire):
#   sire_sire_win_rate       : 父父の勝率
#   sire_sire_avg_pos        : 父父の平均着順
#
# 交互作用:
#   sire_dist_n              : 父×距離カテゴリの出走サンプル数
# ---------------------------------------------------------------------------

BLOODLINE_STAT_COLUMNS = [
    # sire overall
    "sire_win_rate", "sire_top3_rate", "sire_avg_pos", "sire_n_runners", "sire_std_pos",
    # sire by distance
    "sire_dist_win_rate", "sire_dist_avg_pos",
    # sire by surface
    "sire_surface_win_rate", "sire_surface_avg_pos",
    # sire by condition
    "sire_cond_win_rate",
    # dam overall
    "dam_win_rate", "dam_top3_rate", "dam_avg_pos", "dam_n_runners", "dam_std_pos",
    # dam by distance
    "dam_dist_win_rate", "dam_dist_avg_pos",
    # dam by surface
    "dam_surface_win_rate", "dam_surface_avg_pos",
    # dam by condition
    "dam_cond_win_rate",
    # dam_sire (broodmare sire)
    "dam_sire_win_rate", "dam_sire_top3_rate", "dam_sire_avg_pos",
    # sire_sire (paternal grandsire)
    "sire_sire_win_rate", "sire_sire_avg_pos",
    # interaction
    "sire_dist_n",
]


def get_bloodline_stat_columns(available_cols=None):
    """compute_sire_dam_stats() が出力する血統統計カラムの一覧を返す。

    Parameters
    ----------
    available_cols : list or None
        対象DataFrameのカラムリスト。指定時はそのカラムに存在するものだけ返す。

    Returns
    -------
    list of str
    """
    if available_cols is None:
        return list(BLOODLINE_STAT_COLUMNS)
    return [c for c in BLOODLINE_STAT_COLUMNS if c in available_cols]


def compute_sire_dam_stats(df):
    """Compute sire/dam historical statistics (win rate, avg position, etc.)
    grouped by distance category and surface type.

    Generated columns are documented in BLOODLINE_STAT_COLUMNS above.

    IMPORTANT: Uses shift(1) + expanding() to avoid future data leakage.
    Data must be sorted by date before calling.
    """
    logger.info("Sorting by race_date for leak-free statistics ...")
    if "race_date" in df.columns:
        df = df.sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)
    else:
        df = df.sort_values(["race_id", "horse_number"]).reset_index(drop=True)

    # Need numeric finish_position
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")

    # Distance categories for granular bloodline stats
    if "distance" in df.columns:
        df["_dist_cat"] = pd.cut(
            pd.to_numeric(df["distance"], errors="coerce"),
            bins=[0, 1400, 1800, 2200, 9999],
            labels=["sprint", "mile", "mid", "long"]
        ).astype(str).fillna("unknown")

    # Course type encoding
    if "course_type" in df.columns:
        df["_surface"] = df["course_type"].map(
            {"芝": "turf", "ダート": "dirt", "障害": "jump"}
        ).fillna("other")

    # Track condition encoding
    if "track_condition" in df.columns:
        df["_condition"] = df["track_condition"].map(
            {"良": "good", "稍重": "slightly_heavy", "重": "heavy", "不良": "bad"}
        ).fillna("unknown")

    stats_computed = 0

    for parent in ["sire", "dam"]:
        if parent not in df.columns:
            continue
        valid_mask = df[parent].notna() & (df[parent] != "")
        if valid_mask.sum() == 0:
            continue

        logger.info(f"  Computing {parent} statistics ...")

        # Overall stats
        g = df.groupby(parent)["finish_position"]
        df[f"{parent}_win_rate"] = g.transform(
            lambda x: (x.shift(1) == 1).expanding().mean())
        df[f"{parent}_top3_rate"] = g.transform(
            lambda x: (x.shift(1) <= 3).expanding().mean())
        df[f"{parent}_avg_pos"] = g.transform(
            lambda x: x.shift(1).expanding().mean())
        df[f"{parent}_n_runners"] = g.transform(
            lambda x: x.shift(1).expanding().count())
        df[f"{parent}_std_pos"] = g.transform(
            lambda x: x.shift(1).expanding().std())
        stats_computed += 5

        # By distance category
        if "_dist_cat" in df.columns:
            gd = df.groupby([parent, "_dist_cat"])["finish_position"]
            df[f"{parent}_dist_win_rate"] = gd.transform(
                lambda x: (x.shift(1) == 1).expanding().mean())
            df[f"{parent}_dist_avg_pos"] = gd.transform(
                lambda x: x.shift(1).expanding().mean())
            stats_computed += 2

        # By surface type
        if "_surface" in df.columns:
            gs = df.groupby([parent, "_surface"])["finish_position"]
            df[f"{parent}_surface_win_rate"] = gs.transform(
                lambda x: (x.shift(1) == 1).expanding().mean())
            df[f"{parent}_surface_avg_pos"] = gs.transform(
                lambda x: x.shift(1).expanding().mean())
            stats_computed += 2

        # By track condition
        if "_condition" in df.columns:
            gc = df.groupby([parent, "_condition"])["finish_position"]
            df[f"{parent}_cond_win_rate"] = gc.transform(
                lambda x: (x.shift(1) == 1).expanding().mean())
            stats_computed += 1

    # dam_sire stats (broodmare sire)
    if "dam_sire" in df.columns:
        valid_mask = df["dam_sire"].notna() & (df["dam_sire"] != "")
        if valid_mask.sum() > 0:
            logger.info("  Computing dam_sire (broodmare sire) statistics ...")
            g = df.groupby("dam_sire")["finish_position"]
            df["dam_sire_win_rate"] = g.transform(
                lambda x: (x.shift(1) == 1).expanding().mean())
            df["dam_sire_top3_rate"] = g.transform(
                lambda x: (x.shift(1) <= 3).expanding().mean())
            df["dam_sire_avg_pos"] = g.transform(
                lambda x: x.shift(1).expanding().mean())
            stats_computed += 3

    # sire_sire stats (paternal grandsire)
    if "sire_sire" in df.columns:
        valid_mask = df["sire_sire"].notna() & (df["sire_sire"] != "")
        if valid_mask.sum() > 0:
            logger.info("  Computing sire_sire (paternal grandsire) statistics ...")
            g = df.groupby("sire_sire")["finish_position"]
            df["sire_sire_win_rate"] = g.transform(
                lambda x: (x.shift(1) == 1).expanding().mean())
            df["sire_sire_avg_pos"] = g.transform(
                lambda x: x.shift(1).expanding().mean())
            stats_computed += 2

    # Sire x Distance interaction (cross-stat)
    if "sire" in df.columns and "_dist_cat" in df.columns:
        logger.info("  Computing sire x distance interaction ...")
        df["sire_dist_key"] = df["sire"].astype(str) + "_" + df["_dist_cat"].astype(str)
        g = df.groupby("sire_dist_key")["finish_position"]
        df["sire_dist_n"] = g.transform(lambda x: x.shift(1).expanding().count())
        df.drop(columns=["sire_dist_key"], inplace=True)
        stats_computed += 1

    # Clean up temp columns
    for tmp_col in ["_dist_cat", "_surface", "_condition"]:
        if tmp_col in df.columns:
            df.drop(columns=[tmp_col], inplace=True)

    logger.info(f"  Computed {stats_computed} bloodline stat columns")
    return df


def report_coverage(df):
    """Print coverage statistics."""
    logger.info("=" * 60)
    logger.info("COVERAGE REPORT")
    logger.info("=" * 60)

    total_rows = len(df)
    total_horses = df["horse_id"].nunique()

    for col in ["sire", "dam", "sire_sire", "dam_sire"]:
        if col in df.columns:
            n = df[col].notna().sum()
            pct = n / total_rows * 100
            logger.info(f"  {col:12s}: {n:>10,} / {total_rows:,} rows ({pct:.1f}%)")

    # Recent data coverage (2020+)
    if "race_date" in df.columns:
        df["_year"] = pd.to_datetime(df["race_date"], errors="coerce").dt.year
        for year_start in [2020, 2023]:
            recent = df[df["_year"] >= year_start]
            if len(recent) > 0:
                has = recent["sire"].notna().sum() if "sire" in recent.columns else 0
                pct = has / len(recent) * 100
                horses = recent["horse_id"].nunique()
                horses_with = recent.loc[recent["sire"].notna(), "horse_id"].nunique() if "sire" in recent.columns else 0
                logger.info(f"  {year_start}+ rows:  {has:,}/{len(recent):,} ({pct:.1f}%) "
                            f"| horses: {horses_with:,}/{horses:,}")
        df.drop(columns=["_year"], inplace=True)

    # Bloodline stat columns
    blood_stat_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["sire_", "dam_", "dam_sire_", "sire_sire_"]
    ) and c not in ["sire_sire"]]
    logger.info(f"  Bloodline stat columns: {len(blood_stat_cols)}")
    for col in sorted(blood_stat_cols)[:10]:
        non_null = df[col].notna().sum()
        logger.info(f"    {col}: {non_null:,} non-null ({non_null/total_rows*100:.1f}%)")
    if len(blood_stat_cols) > 10:
        logger.info(f"    ... and {len(blood_stat_cols) - 10} more")


def main():
    parser = argparse.ArgumentParser(description="Prepare retrain data with bloodline")
    parser.add_argument("--copy-static", action="store_true",
                        help="Copy output to static/ for download")
    parser.add_argument("--no-stats", action="store_true",
                        help="Skip computing sire/dam statistics (just merge)")
    args = parser.parse_args()

    t0 = time.time()

    # Step 1: Load
    df, bl = load_data()

    # Step 2: Merge bloodline
    logger.info("=" * 60)
    logger.info("MERGING BLOODLINE DATA")
    logger.info("=" * 60)
    df = merge_bloodline(df, bl)

    # Step 3: Compute sire/dam stats
    if not args.no_stats:
        logger.info("=" * 60)
        logger.info("COMPUTING BLOODLINE STATISTICS")
        logger.info("=" * 60)
        df = compute_sire_dam_stats(df)

    # Step 4: Report
    report_coverage(df)

    # Step 5: Save
    logger.info("=" * 60)
    logger.info("SAVING")
    logger.info("=" * 60)
    df.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
    logger.info(f"  Saved: {OUTPUT_CSV} ({size_mb:.1f} MB, {len(df):,} rows, {len(df.columns)} columns)")

    # Step 6: Copy to static
    if args.copy_static:
        STATIC_DIR.mkdir(exist_ok=True)
        dst = STATIC_DIR / "_all_races_with_blood.csv"
        shutil.copy2(OUTPUT_CSV, dst)
        logger.info(f"  Copied to: {dst}")

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.0f}s")
    logger.info(f"For retraining, download _all_races_with_blood.csv and run:")
    logger.info(f"  python train_god.py  (it auto-detects bloodline columns)")


if __name__ == "__main__":
    main()
