"""
Speed Figure v2
===============
生タイムを標準化スピード指数に変換する。
数値が高いほど速い = 優秀。

方法:
1. 距離・馬場種別ごとの基準タイムを算出
2. 馬場状態（良/稍重/重/不良）による補正
3. 競馬場ごとの固有速度差を補正
4. 0-130 スケールに変換（100 = 平均、130 = 卓越）
"""

import os
import re
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# --- Venue par adjustment (seconds faster/slower than average) ---
# Positive = track is inherently faster → add bonus to figure
VENUE_ADJUSTMENT = {
    "東京": 0.3,     # Tokyo: slightly fast (wide, flat course)
    "中山": -0.2,    # Nakayama: uphill finish, slightly slow
    "阪神": 0.1,     # Hanshin: generally fair
    "京都": 0.4,     # Kyoto: fast, especially downhill backstretch
    "中京": -0.1,    # Chukyo: average-slow
    "小倉": 0.2,     # Kokura: small track, tends fast
    "新潟": 0.5,     # Niigata: straight course is very fast
    "福島": -0.3,    # Fukushima: small, hilly
    "函館": -0.4,    # Hakodate: small, slow
    "札幌": -0.3,    # Sapporo: slightly slow
}

# --- Track condition adjustment (how many seconds slower vs 良) ---
CONDITION_PENALTY = {
    "良": 0.0,
    "稍重": 0.5,
    "重": 1.2,
    "不良": 2.0,
}


def parse_finish_time(time_str):
    """
    Parse finish time string to seconds.
    Formats: "1:23.4" or "56.7" (seconds only for short races).

    Returns
    -------
    float or None
    """
    if pd.isna(time_str) or time_str is None:
        return None
    time_str = str(time_str).strip()
    if not time_str:
        return None

    try:
        if ":" in time_str:
            parts = time_str.split(":")
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(time_str)
    except (ValueError, IndexError):
        return None


def compute_base_times(df):
    """
    Compute median finish time for each (distance, course_type) combo
    on 良馬場 (good track) only. These serve as par times.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: distance, course_type, track_condition, finish_time_sec

    Returns
    -------
    dict
        {(distance, course_type): median_time_sec}
    """
    good_track = df[df["track_condition"] == "良"].copy()
    if good_track.empty:
        good_track = df.copy()

    base_times = {}
    for (dist, ctype), group in good_track.groupby(["distance", "course_type"]):
        valid = group["finish_time_sec"].dropna()
        if len(valid) >= 5:
            base_times[(dist, ctype)] = valid.median()

    return base_times


def compute_speed_figure(finish_time_sec, distance, course_type, track_condition, venue,
                         base_times=None):
    """
    Convert raw time to standardized speed figure.
    Higher = faster = better.

    Parameters
    ----------
    finish_time_sec : float
        Finish time in seconds
    distance : int
        Race distance in meters
    course_type : str
        "芝" or "ダート"
    track_condition : str
        "良", "稍重", "重", or "不良"
    venue : str
        Competition venue name
    base_times : dict, optional
        Pre-computed base times. If None, uses a simple formula.

    Returns
    -------
    float or None
        Speed figure on 0-130 scale (100 = average). None if cannot compute.
    """
    if finish_time_sec is None or finish_time_sec <= 0:
        return None
    if distance is None or distance <= 0:
        return None

    # 1. Get base (par) time for this distance/surface
    if base_times and (distance, course_type) in base_times:
        par_time = base_times[(distance, course_type)]
    else:
        # Fallback formula: estimate par time from distance
        # Approximate: 芝 ~11.8 sec/furlong, ダート ~12.3 sec/furlong
        # 1 furlong = 201.168m
        furlongs = distance / 201.168
        if course_type == "ダート":
            par_time = furlongs * 12.3
        else:
            par_time = furlongs * 11.8

    # 2. Adjust for track condition (wet tracks are slower, so add back the penalty)
    condition_adj = CONDITION_PENALTY.get(track_condition, 0.0)

    # 3. Adjust for venue (some tracks are inherently faster)
    venue_adj = VENUE_ADJUSTMENT.get(venue, 0.0)

    # 4. Compute adjusted time difference from par
    # adjusted_time = actual_time - condition_penalty + venue_bonus
    adjusted_time = finish_time_sec - condition_adj + venue_adj

    # Time difference from par (negative = faster than par = good)
    time_diff = adjusted_time - par_time

    # 5. Convert to speed figure scale
    # Scale: 1 second faster = approximately 5 points above 100
    # This scaling factor depends on distance (longer races have more variance)
    scaling = 5.0 * (1200.0 / max(distance, 1000))  # shorter races get more weight per second

    speed_figure = 100.0 - (time_diff * scaling)

    # Clamp to reasonable range
    speed_figure = max(30.0, min(140.0, speed_figure))

    return round(speed_figure, 1)


def compute_all_speed_figures(df=None, save=True):
    """
    Compute speed figures for all historical races.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Race history dataframe. If None, loads from _all_races.csv.
    save : bool
        Whether to save results to data/speed_figures.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with race_id, horse_id, horse_name, speed_figure columns.
    """
    if df is None:
        csv_path = os.path.join(DATA_DIR, "_all_races.csv")
        if not os.path.exists(csv_path):
            logger.error(f"Race data not found: {csv_path}")
            return pd.DataFrame()
        df = pd.read_csv(csv_path, dtype={"horse_id": str, "race_id": str})
        logger.info(f"Loaded {len(df)} records from {csv_path}")

    # Parse finish times
    df["finish_time_sec"] = df["finish_time"].apply(parse_finish_time)
    valid_count = df["finish_time_sec"].notna().sum()
    logger.info(f"Parsed finish times: {valid_count}/{len(df)} valid")

    # Compute base (par) times from good-track data
    base_times = compute_base_times(df)
    logger.info(f"Computed base times for {len(base_times)} (distance, surface) combos")

    # Compute speed figure for each row
    speed_figures = []
    for _, row in df.iterrows():
        sf = compute_speed_figure(
            finish_time_sec=row.get("finish_time_sec"),
            distance=row.get("distance"),
            course_type=row.get("course_type", "芝"),
            track_condition=row.get("track_condition", "良"),
            venue=row.get("venue", ""),
            base_times=base_times,
        )
        speed_figures.append(sf)

    df["speed_figure"] = speed_figures

    # Summary stats
    valid_sf = df["speed_figure"].dropna()
    if len(valid_sf) > 0:
        logger.info(
            f"Speed figures computed: {len(valid_sf)} valid. "
            f"Mean={valid_sf.mean():.1f}, Std={valid_sf.std():.1f}, "
            f"Min={valid_sf.min():.1f}, Max={valid_sf.max():.1f}"
        )

    # Select output columns
    output_cols = ["race_id", "horse_id", "horse_name", "race_date", "venue",
                   "distance", "course_type", "track_condition",
                   "finish_position", "finish_time", "finish_time_sec",
                   "speed_figure"]
    existing_cols = [c for c in output_cols if c in df.columns]
    result = df[existing_cols].copy()

    if save:
        out_path = os.path.join(DATA_DIR, "speed_figures.csv")
        result.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved speed figures to {out_path}")

    return result


def get_horse_speed_figures(horse_id, df=None):
    """
    Get speed figure history for a specific horse.

    Parameters
    ----------
    horse_id : str
        Horse ID to look up.
    df : pd.DataFrame, optional
        Pre-loaded speed figures dataframe. If None, loads from CSV.

    Returns
    -------
    list of dict
    """
    horse_id = str(horse_id)

    if df is None:
        csv_path = os.path.join(DATA_DIR, "speed_figures.csv")
        if not os.path.exists(csv_path):
            return []
        df = pd.read_csv(csv_path, dtype={"horse_id": str, "race_id": str})

    horse_data = df[df["horse_id"] == horse_id].copy()
    if horse_data.empty:
        return []

    horse_data = horse_data.sort_values("race_date", ascending=False)
    records = []
    for _, row in horse_data.iterrows():
        records.append({
            "race_id": row.get("race_id"),
            "race_date": row.get("race_date"),
            "venue": row.get("venue"),
            "distance": int(row["distance"]) if pd.notna(row.get("distance")) else None,
            "course_type": row.get("course_type"),
            "track_condition": row.get("track_condition"),
            "finish_position": int(row["finish_position"]) if pd.notna(row.get("finish_position")) else None,
            "finish_time": row.get("finish_time"),
            "speed_figure": round(float(row["speed_figure"]), 1) if pd.notna(row.get("speed_figure")) else None,
        })

    return records


def get_horse_best_figure(horse_id, df=None, n_recent=5):
    """Get best and average of recent N speed figures for a horse."""
    records = get_horse_speed_figures(horse_id, df)
    if not records:
        return {"best": None, "avg_recent": None, "count": 0}

    figures = [r["speed_figure"] for r in records if r["speed_figure"] is not None]
    if not figures:
        return {"best": None, "avg_recent": None, "count": 0}

    recent = figures[:n_recent]
    return {
        "best": max(figures),
        "avg_recent": round(np.mean(recent), 1),
        "count": len(figures),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Computing speed figures for all historical races...")
    result = compute_all_speed_figures(save=True)
    if not result.empty:
        print(f"\nDone. {len(result)} records processed.")
        valid = result["speed_figure"].dropna()
        print(f"Valid speed figures: {len(valid)}")
        print(f"Mean: {valid.mean():.1f}, Std: {valid.std():.1f}")
        print(f"Range: {valid.min():.1f} - {valid.max():.1f}")
    else:
        print("No data processed.")
