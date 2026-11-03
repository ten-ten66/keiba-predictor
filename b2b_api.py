"""
B2B Data API
============
Professional-grade REST API for data consumers: journalists, breeders, bookmakers.

Authentication: API key via X-API-Key header (separate from JWT user auth).
Rate limiting: per-key, configurable per tier.
Usage tracking: all requests logged to data/b2b_usage.jsonl.

Endpoints:
  GET  /b2b/v1/races?date=YYYYMMDD          Race list with metadata
  GET  /b2b/v1/predictions/{race_id}        AI predictions with probabilities
  GET  /b2b/v1/horse/{horse_id}/profile     Complete horse profile
  GET  /b2b/v1/odds/{race_id}               Current/historical odds
  GET  /b2b/v1/stats/jockey/{jockey_id}     Jockey statistics
  GET  /b2b/v1/stats/trainer/{trainer_id}   Trainer statistics
  GET  /b2b/v1/speed-figures?date=YYYYMMDD  Speed figures for a date
  GET  /b2b/v1/value-bets?date=YYYYMMDD     Value bet analysis

Admin:
  POST   /api/admin/api-keys/create         Generate new API key
  GET    /api/admin/api-keys                List all keys with usage stats
  DELETE /api/admin/api-keys/{key_id}       Revoke a key
"""

import os
import sys
import json
import time
import secrets
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import APIRouter, HTTPException, Header, Query, Depends
    from pydantic import BaseModel
except ImportError:
    raise ImportError("pip install fastapi pydantic")

logger = logging.getLogger(__name__)

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
API_KEYS_FILE = DATA_DIR / "api_keys.json"
USAGE_LOG_FILE = DATA_DIR / "b2b_usage.jsonl"

# ============================================================
# API Key Management
# ============================================================

_api_keys_cache: Dict[str, dict] = {}
_api_keys_loaded_at: float = 0


def _load_api_keys() -> Dict[str, dict]:
    """Load API keys from data/api_keys.json with simple caching."""
    global _api_keys_cache, _api_keys_loaded_at
    try:
        mtime = API_KEYS_FILE.stat().st_mtime if API_KEYS_FILE.exists() else 0
    except OSError:
        mtime = 0

    if mtime > _api_keys_loaded_at or not _api_keys_cache:
        if API_KEYS_FILE.exists():
            try:
                with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
                    _api_keys_cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                _api_keys_cache = {}
        else:
            _api_keys_cache = {}
        _api_keys_loaded_at = time.time()
    return _api_keys_cache


def _save_api_keys(keys: Dict[str, dict]):
    """Persist API keys to disk."""
    global _api_keys_cache, _api_keys_loaded_at
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = str(API_KEYS_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(keys, f, ensure_ascii=False, indent=2)
    os.replace(tmp, str(API_KEYS_FILE))
    _api_keys_cache = keys
    _api_keys_loaded_at = time.time()


def generate_api_key() -> str:
    """Generate a new random API key."""
    return "kb_" + secrets.token_hex(24)


# ============================================================
# Rate Limiter (per API key)
# ============================================================

_rate_buckets: Dict[str, list] = {}  # key -> [timestamps]


def _check_rate_limit(api_key: str, limit: int = 100) -> bool:
    """Return True if request is within rate limit (per minute)."""
    now = time.time()
    bucket = _rate_buckets.setdefault(api_key, [])
    bucket[:] = [t for t in bucket if now - t < 60]
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True


# ============================================================
# Usage Logging
# ============================================================

def _log_usage(api_key: str, key_info: dict, endpoint: str, params: dict):
    """Append one line to b2b_usage.jsonl."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "api_key": api_key[:12] + "...",
            "client_name": key_info.get("name", "unknown"),
            "tier": key_info.get("tier", "basic"),
            "endpoint": endpoint,
            "params": params,
        }
        with open(USAGE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Failed to log B2B usage: %s", e)


# ============================================================
# API Key Auth Dependency
# ============================================================

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """Verify the X-API-Key header and enforce rate limits."""
    keys = _load_api_keys()
    if x_api_key not in keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    key_info = keys[x_api_key]

    # Check if key is active
    if key_info.get("revoked", False):
        raise HTTPException(status_code=401, detail="API key has been revoked")

    # Rate limit
    rate_limit = key_info.get("rate_limit", 100)
    if not _check_rate_limit(x_api_key, rate_limit):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({rate_limit} requests/minute)"
        )

    # Update daily request counter
    today = date.today().isoformat()
    if key_info.get("_counter_date") != today:
        key_info["requests_today"] = 0
        key_info["_counter_date"] = today
    key_info["requests_today"] = key_info.get("requests_today", 0) + 1
    key_info["last_used"] = datetime.now().isoformat()
    _save_api_keys(keys)

    return {"api_key": x_api_key, **key_info}


# ============================================================
# Data Helpers
# ============================================================

def _load_race_csv_for_date(date_str: str):
    """Load race CSV data for a given YYYYMMDD date string, returning a DataFrame."""
    import pandas as pd
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        raise HTTPException(400, "Invalid date format. Use YYYYMMDD.")

    year = dt.year
    csv_path = DATA_DIR / f"races_{year}.csv"
    if not csv_path.exists():
        csv_path = DATA_DIR / "_all_races.csv"
    if not csv_path.exists():
        return None, dt

    df = pd.read_csv(csv_path, dtype={"race_id": str, "horse_id": str,
                                       "jockey_id": str, "trainer_id": str})
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    target = dt.strftime("%Y-%m-%d")
    day_df = df[df["race_date"].dt.strftime("%Y-%m-%d") == target]
    return day_df, dt


def _load_all_races_df():
    """Load _all_races.csv as a DataFrame."""
    import pandas as pd
    path = DATA_DIR / "_all_races.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"race_id": str, "horse_id": str,
                                     "jockey_id": str, "trainer_id": str})


def _load_prediction_cache(race_id: str) -> Optional[dict]:
    """Load cached prediction for a race_id."""
    cache_dir = DATA_DIR / "prediction_cache"
    if not cache_dir.exists():
        return None
    for f in cache_dir.iterdir():
        if f.suffix == ".json":
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if data.get("race_id") == race_id:
                    return data
            except (json.JSONDecodeError, IOError):
                continue
    return None


# ============================================================
# B2B Router
# ============================================================

b2b_router = APIRouter(prefix="/b2b/v1", tags=["B2B Data API"])


@b2b_router.get("/races")
async def b2b_races(
    date: str = Query(..., description="Date in YYYYMMDD format"),
    key_info: dict = Depends(verify_api_key),
):
    """Race list with full metadata for a given date."""
    _log_usage(key_info["api_key"], key_info, "/b2b/v1/races", {"date": date})
    import pandas as pd

    day_df, dt = _load_race_csv_for_date(date)
    if day_df is None or day_df.empty:
        # Try upcoming_cache
        cache_dir = DATA_DIR / "upcoming_cache"
        races = []
        if cache_dir.exists():
            for f in sorted(cache_dir.iterdir()):
                if f.suffix == ".json" and date in f.name:
                    try:
                        with open(f, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        races.append({
                            "race_id": data.get("race_id", f.stem),
                            "race_name": data.get("race_name", ""),
                            "venue": data.get("venue", ""),
                            "race_number": data.get("race_number", 0),
                            "distance": data.get("distance", 0),
                            "course_type": data.get("course_type", ""),
                            "n_runners": len(data.get("horses", [])),
                        })
                    except (json.JSONDecodeError, IOError):
                        continue
        return {"status": "ok", "date": date, "count": len(races), "races": races}

    # Group by race_id
    races = []
    for race_id, group in day_df.groupby("race_id"):
        first = group.iloc[0]
        races.append({
            "race_id": race_id,
            "race_name": str(first.get("race_name", "")),
            "venue": str(first.get("venue", "")),
            "race_number": int(first.get("race_number", 0)) if pd.notna(first.get("race_number")) else 0,
            "distance": int(first.get("distance", 0)) if pd.notna(first.get("distance")) else 0,
            "course_type": str(first.get("course_type", "")),
            "track_condition": str(first.get("track_condition", "")),
            "weather": str(first.get("weather", "")),
            "n_runners": len(group),
            "grade": str(first.get("grade", "")) if "grade" in group.columns else "",
        })

    return {"status": "ok", "date": date, "count": len(races), "races": races}


@b2b_router.get("/predictions/{race_id}")
async def b2b_predictions(
    race_id: str,
    key_info: dict = Depends(verify_api_key),
):
    """AI predictions with probabilities for a race."""
    _log_usage(key_info["api_key"], key_info, "/b2b/v1/predictions", {"race_id": race_id})

    cached = _load_prediction_cache(race_id)
    if cached:
        horses = cached.get("horses", cached.get("predictions", []))
        return {
            "status": "ok",
            "race_id": race_id,
            "race_name": cached.get("race_name", ""),
            "model_version": cached.get("model_version", "god_ensemble_v1"),
            "predicted_at": cached.get("predicted_at", cached.get("timestamp", "")),
            "predictions": [
                {
                    "horse_number": h.get("horse_number", h.get("number", 0)),
                    "horse_name": h.get("horse_name", h.get("name", "")),
                    "prob_win": h.get("prob_win", h.get("win_probability", 0)),
                    "prob_top3": h.get("prob_top3", h.get("top3_probability", 0)),
                    "composite_score": h.get("composite_score", h.get("score", 0)),
                    "predicted_rank": h.get("predicted_rank", h.get("rank", 0)),
                }
                for h in horses
            ],
        }

    return {"status": "ok", "race_id": race_id, "predictions": [], "message": "No cached prediction available"}


@b2b_router.get("/horse/{horse_id}/profile")
async def b2b_horse_profile(
    horse_id: str,
    key_info: dict = Depends(verify_api_key),
):
    """Complete horse profile: career stats, bloodline, speed figures."""
    _log_usage(key_info["api_key"], key_info, "/b2b/v1/horse/profile", {"horse_id": horse_id})
    import pandas as pd

    df = _load_all_races_df()
    profile = {"horse_id": horse_id, "career": {}, "bloodline": {}, "speed_figures": {}}

    if df is not None:
        horse_df = df[df["horse_id"] == horse_id]
        if not horse_df.empty:
            first = horse_df.iloc[-1]
            profile["horse_name"] = str(first.get("horse_name", ""))
            profile["sex"] = str(first.get("sex", ""))
            profile["age"] = int(first.get("age", 0)) if pd.notna(first.get("age")) else 0

            # Career stats
            horse_df_clean = horse_df[horse_df["finish_position"].apply(
                lambda x: str(x).replace(".", "").isdigit()
            )].copy()
            if not horse_df_clean.empty:
                horse_df_clean["finish_position"] = horse_df_clean["finish_position"].astype(int)
                total = len(horse_df_clean)
                wins = int((horse_df_clean["finish_position"] == 1).sum())
                top3 = int((horse_df_clean["finish_position"] <= 3).sum())
                profile["career"] = {
                    "starts": total,
                    "wins": wins,
                    "top3": top3,
                    "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                    "top3_rate": round(top3 / total * 100, 1) if total > 0 else 0,
                    "avg_finish": round(float(horse_df_clean["finish_position"].mean()), 1),
                }

    # Bloodline data
    blood_csv = DATA_DIR / "bloodline.csv"
    if blood_csv.exists():
        try:
            blood_df = pd.read_csv(blood_csv, dtype={"horse_id": str})
            hb = blood_df[blood_df["horse_id"] == horse_id]
            if not hb.empty:
                row = hb.iloc[0]
                for col in ["sire", "dam", "sire_of_dam", "sire_sire", "dam_sire"]:
                    if col in row and pd.notna(row[col]):
                        profile["bloodline"][col] = str(row[col])
        except Exception:
            pass

    # Speed figures
    try:
        from speed_figures import get_horse_speed_figures, get_horse_best_figure
        records = get_horse_speed_figures(horse_id)
        best = get_horse_best_figure(horse_id)
        profile["speed_figures"] = {
            "best": best,
            "recent": records[:10] if records else [],
        }
    except Exception as e:
        logger.debug("Speed figures unavailable for %s: %s", horse_id, e)

    return {"status": "ok", **profile}


@b2b_router.get("/odds/{race_id}")
async def b2b_odds(
    race_id: str,
    key_info: dict = Depends(verify_api_key),
):
    """Current/historical odds for a race."""
    _log_usage(key_info["api_key"], key_info, "/b2b/v1/odds", {"race_id": race_id})
    import pandas as pd

    # Check prediction cache for live odds
    cached = _load_prediction_cache(race_id)
    if cached:
        horses = cached.get("horses", cached.get("predictions", []))
        odds_data = [
            {
                "horse_number": h.get("horse_number", h.get("number", 0)),
                "horse_name": h.get("horse_name", h.get("name", "")),
                "odds": h.get("odds", 0),
                "popularity": h.get("popularity", 0),
            }
            for h in horses
            if h.get("odds", 0) > 0
        ]
        return {
            "status": "ok",
            "race_id": race_id,
            "source": "prediction_cache",
            "timestamp": cached.get("predicted_at", cached.get("timestamp", "")),
            "odds": odds_data,
        }

    # Fallback: historical from CSV
    df = _load_all_races_df()
    if df is not None:
        race_df = df[df["race_id"] == race_id]
        if not race_df.empty:
            odds_data = []
            for _, row in race_df.iterrows():
                if pd.notna(row.get("odds")) and float(row.get("odds", 0)) > 0:
                    odds_data.append({
                        "horse_number": int(row.get("horse_number", 0)),
                        "horse_name": str(row.get("horse_name", "")),
                        "odds": float(row.get("odds", 0)),
                        "popularity": int(row.get("popularity", 0)) if pd.notna(row.get("popularity")) else 0,
                        "finish_position": str(row.get("finish_position", "")),
                    })
            return {
                "status": "ok",
                "race_id": race_id,
                "source": "historical",
                "odds": sorted(odds_data, key=lambda x: x.get("odds", 999)),
            }

    return {"status": "ok", "race_id": race_id, "odds": [], "message": "No odds data found"}


@b2b_router.get("/stats/jockey/{jockey_id}")
async def b2b_jockey_stats(
    jockey_id: str,
    key_info: dict = Depends(verify_api_key),
):
    """Jockey statistics: win rate, ROI, venue breakdowns."""
    _log_usage(key_info["api_key"], key_info, "/b2b/v1/stats/jockey", {"jockey_id": jockey_id})
    import pandas as pd

    df = _load_all_races_df()
    if df is None:
        raise HTTPException(404, "Race data not available")

    jdf = df[df["jockey_id"] == jockey_id]
    if jdf.empty:
        raise HTTPException(404, f"No data for jockey {jockey_id}")

    jockey_name = str(jdf.iloc[-1].get("jockey", ""))

    jdf_clean = jdf[jdf["finish_position"].apply(
        lambda x: str(x).replace(".", "").isdigit()
    )].copy()
    jdf_clean["finish_position"] = jdf_clean["finish_position"].astype(int)

    total = len(jdf_clean)
    wins = int((jdf_clean["finish_position"] == 1).sum())
    top3 = int((jdf_clean["finish_position"] <= 3).sum())

    # ROI calculation
    roi = 0
    if "odds" in jdf_clean.columns and total > 0:
        winners = jdf_clean[jdf_clean["finish_position"] == 1]
        total_return = float(winners["odds"].sum()) * 100
        total_wagered = total * 100
        roi = round((total_return / total_wagered - 1) * 100, 1) if total_wagered > 0 else 0

    # By venue breakdown
    venue_stats = {}
    if "venue" in jdf_clean.columns:
        for venue, vgroup in jdf_clean.groupby("venue"):
            v_total = len(vgroup)
            v_wins = int((vgroup["finish_position"] == 1).sum())
            venue_stats[str(venue)] = {
                "starts": v_total,
                "wins": v_wins,
                "win_rate": round(v_wins / v_total * 100, 1) if v_total > 0 else 0,
            }

    # By surface
    surface_stats = {}
    if "course_type" in jdf_clean.columns:
        for surface, sgroup in jdf_clean.groupby("course_type"):
            s_total = len(sgroup)
            s_wins = int((sgroup["finish_position"] == 1).sum())
            surface_stats[str(surface)] = {
                "starts": s_total,
                "wins": s_wins,
                "win_rate": round(s_wins / s_total * 100, 1) if s_total > 0 else 0,
            }

    return {
        "status": "ok",
        "jockey_id": jockey_id,
        "jockey_name": jockey_name,
        "overall": {
            "starts": total,
            "wins": wins,
            "top3": top3,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "top3_rate": round(top3 / total * 100, 1) if total > 0 else 0,
            "roi_pct": roi,
        },
        "by_venue": venue_stats,
        "by_surface": surface_stats,
    }


@b2b_router.get("/stats/trainer/{trainer_id}")
async def b2b_trainer_stats(
    trainer_id: str,
    key_info: dict = Depends(verify_api_key),
):
    """Trainer statistics: win rate, ROI, venue breakdowns."""
    _log_usage(key_info["api_key"], key_info, "/b2b/v1/stats/trainer", {"trainer_id": trainer_id})
    import pandas as pd

    df = _load_all_races_df()
    if df is None:
        raise HTTPException(404, "Race data not available")

    tdf = df[df["trainer_id"] == trainer_id]
    if tdf.empty:
        raise HTTPException(404, f"No data for trainer {trainer_id}")

    trainer_name = str(tdf.iloc[-1].get("trainer", ""))

    tdf_clean = tdf[tdf["finish_position"].apply(
        lambda x: str(x).replace(".", "").isdigit()
    )].copy()
    tdf_clean["finish_position"] = tdf_clean["finish_position"].astype(int)

    total = len(tdf_clean)
    wins = int((tdf_clean["finish_position"] == 1).sum())
    top3 = int((tdf_clean["finish_position"] <= 3).sum())

    roi = 0
    if "odds" in tdf_clean.columns and total > 0:
        winners = tdf_clean[tdf_clean["finish_position"] == 1]
        total_return = float(winners["odds"].sum()) * 100
        total_wagered = total * 100
        roi = round((total_return / total_wagered - 1) * 100, 1) if total_wagered > 0 else 0

    # By venue
    venue_stats = {}
    if "venue" in tdf_clean.columns:
        for venue, vgroup in tdf_clean.groupby("venue"):
            v_total = len(vgroup)
            v_wins = int((vgroup["finish_position"] == 1).sum())
            venue_stats[str(venue)] = {
                "starts": v_total,
                "wins": v_wins,
                "win_rate": round(v_wins / v_total * 100, 1) if v_total > 0 else 0,
            }

    # By surface
    surface_stats = {}
    if "course_type" in tdf_clean.columns:
        for surface, sgroup in tdf_clean.groupby("course_type"):
            s_total = len(sgroup)
            s_wins = int((sgroup["finish_position"] == 1).sum())
            surface_stats[str(surface)] = {
                "starts": s_total,
                "wins": s_wins,
                "win_rate": round(s_wins / s_total * 100, 1) if s_total > 0 else 0,
            }

    return {
        "status": "ok",
        "trainer_id": trainer_id,
        "trainer_name": trainer_name,
        "overall": {
            "starts": total,
            "wins": wins,
            "top3": top3,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "top3_rate": round(top3 / total * 100, 1) if total > 0 else 0,
            "roi_pct": roi,
        },
        "by_venue": venue_stats,
        "by_surface": surface_stats,
    }


@b2b_router.get("/speed-figures")
async def b2b_speed_figures(
    date: str = Query(..., description="Date in YYYYMMDD format"),
    key_info: dict = Depends(verify_api_key),
):
    """Speed figures for all races on a date."""
    _log_usage(key_info["api_key"], key_info, "/b2b/v1/speed-figures", {"date": date})
    import pandas as pd

    day_df, dt = _load_race_csv_for_date(date)
    if day_df is None or day_df.empty:
        return {"status": "ok", "date": date, "races": [], "message": "No data for this date"}

    try:
        from speed_figures import compute_speed_figure, parse_finish_time
    except ImportError:
        raise HTTPException(503, "Speed figures module not available")

    races = {}
    for _, row in day_df.iterrows():
        race_id = str(row.get("race_id", ""))
        if race_id not in races:
            races[race_id] = {
                "race_id": race_id,
                "race_name": str(row.get("race_name", "")),
                "venue": str(row.get("venue", "")),
                "distance": int(row.get("distance", 0)) if pd.notna(row.get("distance")) else 0,
                "course_type": str(row.get("course_type", "")),
                "horses": [],
            }

        finish_time = row.get("finish_time", "")
        figure = None
        if pd.notna(finish_time) and finish_time:
            try:
                time_sec = parse_finish_time(str(finish_time))
                if time_sec and time_sec > 0:
                    figure = compute_speed_figure(
                        time_sec,
                        int(row.get("distance", 1600)),
                        str(row.get("course_type", "芝")),
                        str(row.get("track_condition", "良")),
                        str(row.get("venue", "")),
                    )
            except Exception:
                pass

        races[race_id]["horses"].append({
            "horse_number": int(row.get("horse_number", 0)) if pd.notna(row.get("horse_number")) else 0,
            "horse_name": str(row.get("horse_name", "")),
            "finish_position": str(row.get("finish_position", "")),
            "speed_figure": round(figure, 1) if figure is not None else None,
        })

    return {
        "status": "ok",
        "date": date,
        "count": len(races),
        "races": list(races.values()),
    }


@b2b_router.get("/value-bets")
async def b2b_value_bets(
    date: str = Query(..., description="Date in YYYYMMDD format"),
    threshold: float = Query(1.3, description="Minimum EV threshold"),
    key_info: dict = Depends(verify_api_key),
):
    """Value bet analysis for a date."""
    _log_usage(key_info["api_key"], key_info, "/b2b/v1/value-bets",
               {"date": date, "threshold": threshold})

    try:
        from value_alerts import find_value_bets
        bets = find_value_bets(date, ev_threshold=threshold)
    except ImportError:
        bets = []
    except Exception as e:
        logger.warning("Value bet analysis failed: %s", e)
        bets = []

    return {
        "status": "ok",
        "date": date,
        "ev_threshold": threshold,
        "count": len(bets),
        "value_bets": bets,
    }


# ============================================================
# Admin Endpoints (API Key Management)
# ============================================================

b2b_admin_router = APIRouter(tags=["B2B Admin"])


class CreateAPIKeyRequest(BaseModel):
    name: str
    tier: str = "basic"
    rate_limit: int = 100


@b2b_admin_router.post("/api/admin/api-keys/create")
async def admin_create_api_key(req: CreateAPIKeyRequest):
    """Generate a new B2B API key."""
    keys = _load_api_keys()
    new_key = generate_api_key()

    keys[new_key] = {
        "name": req.name,
        "tier": req.tier,
        "rate_limit": req.rate_limit,
        "created": date.today().isoformat(),
        "requests_today": 0,
        "total_requests": 0,
        "revoked": False,
        "last_used": None,
    }
    _save_api_keys(keys)

    return {
        "status": "ok",
        "api_key": new_key,
        "name": req.name,
        "tier": req.tier,
        "rate_limit": req.rate_limit,
        "message": "Store this key securely. It will not be shown again in full.",
    }


@b2b_admin_router.get("/api/admin/api-keys")
async def admin_list_api_keys():
    """List all API keys with usage statistics."""
    keys = _load_api_keys()

    # Count total requests from usage log
    usage_counts: Dict[str, int] = {}
    if USAGE_LOG_FILE.exists():
        try:
            with open(USAGE_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        client = entry.get("client_name", "unknown")
                        usage_counts[client] = usage_counts.get(client, 0) + 1
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass

    result = []
    for key, info in keys.items():
        masked_key = key[:8] + "..." + key[-4:]
        result.append({
            "key_id": masked_key,
            "name": info.get("name", ""),
            "tier": info.get("tier", "basic"),
            "rate_limit": info.get("rate_limit", 100),
            "created": info.get("created", ""),
            "requests_today": info.get("requests_today", 0),
            "total_logged_requests": usage_counts.get(info.get("name", ""), 0),
            "revoked": info.get("revoked", False),
            "last_used": info.get("last_used"),
        })

    return {"status": "ok", "count": len(result), "keys": result}


@b2b_admin_router.delete("/api/admin/api-keys/{key_id}")
async def admin_revoke_api_key(key_id: str):
    """Revoke an API key by its masked key_id (first8...last4) or full key."""
    keys = _load_api_keys()

    # Try exact match first
    if key_id in keys:
        keys[key_id]["revoked"] = True
        _save_api_keys(keys)
        return {"status": "ok", "message": f"Key revoked"}

    # Try masked match
    for full_key, info in keys.items():
        masked = full_key[:8] + "..." + full_key[-4:]
        if masked == key_id:
            info["revoked"] = True
            _save_api_keys(keys)
            return {"status": "ok", "message": f"Key for '{info.get('name', '')}' revoked"}

    raise HTTPException(404, "API key not found")
