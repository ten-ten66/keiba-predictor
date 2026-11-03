"""
International Race Support
===========================
Framework for supporting non-JRA race data, starting with
Hong Kong Jockey Club (HKJC).

This module provides:
1. Data structures for international race cards
2. Mapping functions to convert foreign formats to JRA-compatible features
3. Mock data for development/testing
4. API endpoint for querying international races

Actual scraping of HKJC (or other) sites can be added later once
the data pipeline is validated.

Supported regions:
  hk  - Hong Kong Jockey Club (HKJC)
  (future: sg - Singapore, au - Australia, uae - UAE)
"""

import os
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
INTL_CACHE_DIR = DATA_DIR / "international_cache"


# ============================================================
# Data Models
# ============================================================

@dataclass
class InternationalHorse:
    """Horse entry in an international race."""
    horse_number: int
    horse_name: str
    horse_name_en: str = ""
    jockey: str = ""
    jockey_id: str = ""
    trainer: str = ""
    trainer_id: str = ""
    weight: float = 0      # impost (carried weight in lbs/kg)
    horse_weight: float = 0  # horse body weight
    age: int = 0
    sex: str = ""
    barrier: int = 0       # barrier/gate (= frame_number)
    odds: float = 0
    rating: int = 0        # official rating (HKJC uses a rating system)
    last_5: str = ""       # last 5 finish positions e.g. "12231"
    country_of_origin: str = ""


@dataclass
class InternationalRace:
    """International race card."""
    race_id: str
    region: str           # "hk", "sg", etc.
    venue: str
    race_name: str
    race_number: int = 0
    race_date: str = ""
    post_time: str = ""   # local time "HH:MM"
    distance: int = 0     # meters
    surface: str = ""     # "turf", "dirt", "all_weather"
    track_condition: str = ""
    race_class: str = ""  # HKJC class: "Class 1", "Group 1", etc.
    prize_money: str = ""
    n_runners: int = 0
    horses: List[InternationalHorse] = field(default_factory=list)


# ============================================================
# Region Configuration
# ============================================================

REGION_CONFIG = {
    "hk": {
        "name": "Hong Kong Jockey Club",
        "short": "HKJC",
        "timezone": "Asia/Hong_Kong",
        "currency": "HKD",
        "base_url": "https://racing.hkjc.com/racing/information/english/Racing/",
        "surfaces": ["turf", "all_weather"],
        "venues": ["Sha Tin", "Happy Valley"],
        "weight_unit": "lbs",
        "distance_unit": "meters",
    },
    "sg": {
        "name": "Singapore Turf Club",
        "short": "STC",
        "timezone": "Asia/Singapore",
        "currency": "SGD",
        "base_url": "",
        "surfaces": ["turf", "polytrack"],
        "venues": ["Kranji"],
        "weight_unit": "kg",
        "distance_unit": "meters",
    },
}


# ============================================================
# Surface & Condition Mapping
# ============================================================

# Map international surface types to JRA course_type encoding
SURFACE_TO_JRA = {
    "turf": "芝",
    "dirt": "ダート",
    "all_weather": "ダート",   # AW treated as dirt for model purposes
    "polytrack": "ダート",
    "synthetic": "ダート",
}

# Map international track conditions to JRA track_condition
CONDITION_TO_JRA = {
    # HKJC conditions
    "good": "良",
    "good to firm": "良",
    "good to yielding": "稍重",
    "yielding": "重",
    "soft": "重",
    "heavy": "不良",
    "wet fast": "稍重",
    "wet slow": "重",
    # Generic
    "fast": "良",
    "firm": "良",
    "standard": "良",
    "slow": "重",
    "sloppy": "不良",
}

# Jockeys who ride in both JRA and HKJC (dual-licensed or visiting)
DUAL_LICENSED_JOCKEYS = {
    # HKJC jockey name -> JRA jockey_id (if known)
    "J Moreira": None,       # Joao Moreira - rides in both
    "Z Purton": None,        # Zac Purton
    "R Moore": None,         # Ryan Moore - international visitor
    "C Demuro": None,        # Christophe Demuro
    "M Demuro": None,        # Mirco Demuro - now JRA-based
    "C Lemaire": None,       # Christophe Lemaire - JRA-based
    "D Lane": None,          # Damian Lane
}

# Trainers with operations in multiple jurisdictions
DUAL_LICENSED_TRAINERS = {
    "A Cruz": None,          # Tony Cruz
    "J Size": None,          # John Size
    "F Lor": None,           # Frankie Lor
}


# ============================================================
# Feature Mapping
# ============================================================

def map_hkjc_to_jra_features(race: InternationalRace) -> Dict[str, Any]:
    """
    Map an HKJC (or other international) race to JRA model feature format.

    Returns a dictionary of mapped features suitable for passing to the
    prediction model.
    """
    # Surface mapping
    jra_surface = SURFACE_TO_JRA.get(race.surface.lower(), "芝")

    # Condition mapping
    jra_condition = CONDITION_TO_JRA.get(race.track_condition.lower(), "良")

    # HKJC weights are in lbs; JRA uses kg
    weight_factor = 1.0
    region_cfg = REGION_CONFIG.get(race.region, {})
    if region_cfg.get("weight_unit") == "lbs":
        weight_factor = 0.4536  # lbs to kg

    mapped_horses = []
    for h in race.horses:
        mapped = {
            "horse_number": h.horse_number,
            "horse_name": h.horse_name_en or h.horse_name,
            "frame_number": h.barrier or h.horse_number,
            "jockey": h.jockey,
            "jockey_id": h.jockey_id or "",
            "trainer": h.trainer,
            "trainer_id": h.trainer_id or "",
            "weight": round(h.weight * weight_factor, 1) if weight_factor != 1.0 else h.weight,
            "horse_weight": h.horse_weight,
            "age": h.age,
            "sex": _map_sex(h.sex),
            "odds": h.odds,
            "popularity": 0,  # computed from odds ranking
        }

        # Check dual-licensed jockey mapping
        if h.jockey in DUAL_LICENSED_JOCKEYS and DUAL_LICENSED_JOCKEYS[h.jockey]:
            mapped["jockey_id"] = DUAL_LICENSED_JOCKEYS[h.jockey]

        mapped_horses.append(mapped)

    # Compute popularity from odds
    sorted_by_odds = sorted(mapped_horses, key=lambda x: x["odds"] if x["odds"] > 0 else 9999)
    for rank, horse in enumerate(sorted_by_odds, 1):
        horse["popularity"] = rank

    return {
        "race_id": race.race_id,
        "race_name": race.race_name,
        "venue": race.venue,
        "distance": race.distance,
        "course_type": jra_surface,
        "track_condition": jra_condition,
        "n_runners": len(mapped_horses),
        "horses": mapped_horses,
        "region": race.region,
        "original_surface": race.surface,
        "original_condition": race.track_condition,
    }


def _map_sex(sex_str: str) -> str:
    """Map international sex codes to JRA convention."""
    sex_map = {
        "c": "牡",    # colt
        "h": "牡",    # horse (male, older)
        "g": "セ",    # gelding
        "f": "牝",    # filly
        "m": "牝",    # mare
        "r": "牡",    # ridgling
        "colt": "牡",
        "horse": "牡",
        "gelding": "セ",
        "filly": "牝",
        "mare": "牝",
        "male": "牡",
        "female": "牝",
    }
    return sex_map.get(sex_str.lower().strip(), "牡")


# ============================================================
# HKJC Data (Mock / Framework)
# ============================================================

def scrape_hkjc_races(date_str: str) -> List[InternationalRace]:
    """
    Scrape upcoming HKJC race cards for a given date.

    HKJC publishes race cards at:
    https://racing.hkjc.com/racing/information/english/Racing/LocalResults.aspx

    For now, returns mock data to validate the pipeline. Real scraping
    can be implemented by:
    1. Fetching the race card HTML
    2. Parsing horse entries, odds, etc.
    3. Building InternationalRace objects

    Args:
        date_str: Date in YYYYMMDD format

    Returns:
        List of InternationalRace objects
    """
    # Check if we have cached data
    cached = _load_cached_international(date_str, "hk")
    if cached:
        return cached

    # For now, return empty list (framework ready for real scraping)
    logger.info("HKJC scraping not yet implemented. Returning empty for date %s", date_str)
    return []


def get_mock_hkjc_race(date_str: str = None) -> InternationalRace:
    """
    Generate a mock HKJC race for testing the pipeline.

    Returns a realistic-looking race at Sha Tin with typical HKJC fields.
    """
    if date_str is None:
        date_str = date.today().strftime("%Y%m%d")

    race = InternationalRace(
        race_id=f"hk_{date_str}_01",
        region="hk",
        venue="Sha Tin",
        race_name="Sha Tin Trophy (Group 2)",
        race_number=8,
        race_date=date_str,
        post_time="15:40",
        distance=1600,
        surface="turf",
        track_condition="good",
        race_class="Group 2",
        prize_money="HKD 5,200,000",
        n_runners=12,
        horses=[
            InternationalHorse(
                horse_number=1, horse_name="Golden Sixty",
                horse_name_en="Golden Sixty",
                jockey="V Ho", trainer="F Lor",
                weight=126, horse_weight=1100,
                age=8, sex="gelding", barrier=3,
                odds=2.5, rating=130,
            ),
            InternationalHorse(
                horse_number=2, horse_name="Romantic Warrior",
                horse_name_en="Romantic Warrior",
                jockey="J Moreira", trainer="D Whyte",
                weight=126, horse_weight=1080,
                age=6, sex="gelding", barrier=7,
                odds=3.2, rating=128,
            ),
            InternationalHorse(
                horse_number=3, horse_name="Voyage Bubble",
                horse_name_en="Voyage Bubble",
                jockey="Z Purton", trainer="J Size",
                weight=126, horse_weight=1060,
                age=5, sex="colt", barrier=1,
                odds=4.8, rating=122,
            ),
            InternationalHorse(
                horse_number=4, horse_name="Lucky Sweynesse",
                horse_name_en="Lucky Sweynesse",
                jockey="A Hamelin", trainer="A Cruz",
                weight=126, horse_weight=1090,
                age=6, sex="gelding", barrier=9,
                odds=8.0, rating=118,
            ),
            InternationalHorse(
                horse_number=5, horse_name="California Spangle",
                horse_name_en="California Spangle",
                jockey="K Teetan", trainer="T Cruz",
                weight=126, horse_weight=1070,
                age=6, sex="gelding", barrier=5,
                odds=12.0, rating=115,
            ),
        ],
    )
    return race


# ============================================================
# Cache Management
# ============================================================

def _load_cached_international(date_str: str, region: str) -> Optional[List[InternationalRace]]:
    """Load cached international race data."""
    cache_file = INTL_CACHE_DIR / f"{region}_{date_str}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        races = []
        for rd in data:
            horses = [InternationalHorse(**h) for h in rd.pop("horses", [])]
            race = InternationalRace(**rd, horses=horses)
            races.append(race)
        return races
    except Exception as e:
        logger.warning("Failed to load international cache: %s", e)
        return None


def save_international_cache(races: List[InternationalRace], date_str: str, region: str):
    """Save international race data to cache."""
    INTL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = INTL_CACHE_DIR / f"{region}_{date_str}.json"
    data = [asdict(r) for r in races]
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d international races to %s", len(races), cache_file)


# ============================================================
# API Router
# ============================================================

try:
    from fastapi import APIRouter, HTTPException, Query
    international_router = APIRouter(tags=["International Races"])

    @international_router.get("/api/international/races")
    async def api_international_races(
        region: str = Query("hk", description="Region code: hk, sg"),
        date: str = Query(..., description="Date in YYYYMMDD format"),
        mock: bool = Query(False, description="Return mock data for testing"),
    ):
        """
        Get international race cards for a region and date.

        Supported regions:
        - hk: Hong Kong Jockey Club
        - sg: Singapore Turf Club (coming soon)
        """
        if region not in REGION_CONFIG:
            raise HTTPException(400, f"Unsupported region: {region}. Supported: {list(REGION_CONFIG.keys())}")

        region_info = REGION_CONFIG[region]

        if mock:
            # Return mock data for testing
            mock_race = get_mock_hkjc_race(date)
            mapped = map_hkjc_to_jra_features(mock_race)
            return {
                "status": "ok",
                "region": region,
                "region_name": region_info["name"],
                "date": date,
                "is_mock": True,
                "count": 1,
                "races": [
                    {
                        "original": asdict(mock_race),
                        "jra_mapped": mapped,
                    }
                ],
            }

        # Real data
        if region == "hk":
            races = scrape_hkjc_races(date)
        else:
            races = []

        if not races:
            return {
                "status": "ok",
                "region": region,
                "region_name": region_info["name"],
                "date": date,
                "count": 0,
                "races": [],
                "message": f"No race data available for {region_info['name']} on {date}. "
                           f"Scraping not yet implemented for this region.",
            }

        result_races = []
        for race in races:
            mapped = map_hkjc_to_jra_features(race)
            result_races.append({
                "original": asdict(race),
                "jra_mapped": mapped,
            })

        return {
            "status": "ok",
            "region": region,
            "region_name": region_info["name"],
            "date": date,
            "count": len(result_races),
            "races": result_races,
        }

except ImportError:
    international_router = None
    logger.warning("FastAPI not available; international_router disabled")


# ============================================================
# CLI / Testing
# ============================================================

if __name__ == "__main__":
    import pprint

    print("=" * 60)
    print("International Race Support - Test")
    print("=" * 60)

    # Test mock HKJC race
    mock = get_mock_hkjc_race("20260320")
    print(f"\nMock HKJC Race: {mock.race_name}")
    print(f"  Venue: {mock.venue}, Distance: {mock.distance}m, Surface: {mock.surface}")
    print(f"  Horses: {mock.n_runners}")
    for h in mock.horses:
        print(f"    #{h.horse_number} {h.horse_name_en} (J: {h.jockey}) odds={h.odds}")

    # Test mapping
    print("\nJRA Feature Mapping:")
    mapped = map_hkjc_to_jra_features(mock)
    print(f"  Surface: {mock.surface} -> {mapped['course_type']}")
    print(f"  Condition: {mock.track_condition} -> {mapped['track_condition']}")
    print(f"  Distance: {mapped['distance']}m")
    print(f"  Mapped horses:")
    for h in mapped["horses"]:
        print(f"    #{h['horse_number']} {h['horse_name']} "
              f"wt={h['weight']}kg pop={h['popularity']} sex={h['sex']}")

    print("\nDone.")
