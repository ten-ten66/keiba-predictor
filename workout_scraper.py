"""
Training/Workout Data Scraper (追い切りデータ)
================================================
Scrapes pre-race workout (oikiri) data from netkeiba.

Workout data includes training times, course conditions, and evaluations
that can be highly predictive of race performance.

Usage:
    from workout_scraper import WorkoutScraper
    scraper = WorkoutScraper()
    data = scraper.scrape_workouts("202506030811")  # race_id
    # or fetch cached
    data = scraper.get_workouts("202506030811")
"""

import os
import json
import time
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WORKOUT_CACHE_DIR = os.path.join(DATA_DIR, "workout_cache")
os.makedirs(WORKOUT_CACHE_DIR, exist_ok=True)

# Try importing scraping libraries
try:
    import requests
    from bs4 import BeautifulSoup
    _SCRAPING_AVAILABLE = True
except ImportError:
    _SCRAPING_AVAILABLE = False
    logger.warning("requests/beautifulsoup4 not installed. Workout scraping disabled.")


class WorkoutScraper:
    """Scrapes and caches pre-race workout data from netkeiba."""

    OIKIRI_URL = "https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "ja,en;q=0.9",
    }
    REQUEST_DELAY = 2.0  # seconds between requests

    def __init__(self, cache_dir: str = WORKOUT_CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._last_request_time = 0.0

    def _cache_path(self, race_id: str) -> str:
        return os.path.join(self.cache_dir, f"workout_{race_id}.json")

    def _load_cache(self, race_id: str) -> Optional[Dict]:
        path = self._cache_path(race_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Cache valid for 24 hours
                cached_at = data.get("cached_at", "")
                if cached_at:
                    cached_dt = datetime.fromisoformat(cached_at)
                    if (datetime.now() - cached_dt).total_seconds() < 86400:
                        return data
            except Exception:
                pass
        return None

    def _save_cache(self, race_id: str, data: Dict):
        path = self._cache_path(race_id)
        data["cached_at"] = datetime.now().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _throttle(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def scrape_workouts(self, race_id: str) -> Optional[Dict]:
        """Scrape pre-race workout times from netkeiba.

        Args:
            race_id: JRA race identifier (e.g., "202506030811")

        Returns:
            dict with race_id, horses list (each with workout entries),
            or None if scraping fails or is unavailable.
        """
        if not _SCRAPING_AVAILABLE:
            logger.warning("Scraping libraries not available")
            return None

        # Check cache first
        cached = self._load_cache(race_id)
        if cached:
            return cached

        url = self.OIKIRI_URL.format(race_id=race_id)
        self._throttle()

        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            resp.encoding = "euc-jp"

            if resp.status_code != 200:
                logger.warning(f"Workout page returned {resp.status_code} for {race_id}")
                return None

            return self._parse_workout_page(race_id, resp.text)

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch workout data for {race_id}: {e}")
            return None

    def _parse_workout_page(self, race_id: str, html: str) -> Optional[Dict]:
        """Parse the oikiri HTML page to extract workout data.

        The netkeiba oikiri page typically has a table with columns:
        - Horse number / name
        - Date
        - Training venue/course (e.g., 美浦W, 栗東CW)
        - Distance
        - Time (overall and splits)
        - Evaluation (A/B/C or descriptive)

        Note: Page structure may change. This parser attempts common patterns
        and falls back gracefully.
        """
        soup = BeautifulSoup(html, "html.parser")

        horses = []

        # Strategy 1: Look for workout table rows
        # Common patterns: table.OikiriTable, div.OikiriData
        tables = soup.find_all("table")

        workout_table = None
        for table in tables:
            # Look for tables that have workout-like content
            text = table.get_text()
            if any(kw in text for kw in ["追い切り", "調教", "タイム", "oikiri"]):
                workout_table = table
                break

        if workout_table is None:
            # Try finding by class patterns
            for cls_pattern in ["OikiriTable", "RaceOikiri", "Training"]:
                workout_table = soup.find("table", class_=re.compile(cls_pattern, re.I))
                if workout_table:
                    break

        if workout_table is None:
            # Try div-based layout
            divs = soup.find_all("div", class_=re.compile(r"Horse|Oikiri|Training", re.I))
            if divs:
                for div in divs:
                    horse_data = self._parse_workout_div(div)
                    if horse_data:
                        horses.append(horse_data)

        if workout_table is not None:
            rows = workout_table.find_all("tr")
            for row in rows[1:]:  # Skip header
                cells = row.find_all(["td", "th"])
                if len(cells) < 3:
                    continue

                horse_data = self._parse_workout_row(cells)
                if horse_data:
                    horses.append(horse_data)

        if not horses:
            # Return a minimal structure indicating no data parsed
            # (page structure may have changed)
            result = {
                "race_id": race_id,
                "horses": [],
                "parse_status": "no_data_found",
                "note": "Workout page structure may have changed. Manual review needed.",
            }
            self._save_cache(race_id, result)
            return result

        result = {
            "race_id": race_id,
            "horses": horses,
            "parse_status": "ok",
            "n_horses": len(horses),
        }
        self._save_cache(race_id, result)
        return result

    def _parse_workout_row(self, cells) -> Optional[Dict]:
        """Parse a single row from a workout table."""
        try:
            texts = [c.get_text(strip=True) for c in cells]

            # Try to extract horse number from first cell
            horse_number = None
            horse_name = ""
            for i, t in enumerate(texts[:3]):
                if t.isdigit():
                    horse_number = int(t)
                elif len(t) > 1 and not t.isdigit():
                    horse_name = t
                    break

            if horse_number is None and horse_name == "":
                return None

            # Try to find time patterns (e.g., "1:12.5", "37.2")
            times = []
            for t in texts:
                time_matches = re.findall(r"(\d{1,2}:\d{2}\.\d|\d{2}\.\d)", t)
                times.extend(time_matches)

            # Try to find evaluation (A/B/C or Japanese evaluation)
            evaluation = ""
            eval_patterns = ["抜群", "良好", "一杯", "馬なり", "強め", "末強め",
                             "仕掛け", "楽走", "追う", "叩く"]
            for t in texts:
                for ep in eval_patterns:
                    if ep in t:
                        evaluation = ep
                        break
                if not evaluation and re.match(r"^[A-C][+-]?$", t):
                    evaluation = t

            # Training venue
            venue = ""
            venue_patterns = ["美浦", "栗東", "北海道", "函館", "札幌",
                              "CW", "坂路", "ウッド", "ポリ", "ダート", "芝"]
            for t in texts:
                for vp in venue_patterns:
                    if vp in t:
                        venue = t
                        break
                if venue:
                    break

            return {
                "horse_number": horse_number,
                "horse_name": horse_name,
                "training_venue": venue,
                "times": times,
                "evaluation": evaluation,
                "raw": " | ".join(texts[:8]),
            }
        except Exception:
            return None

    def _parse_workout_div(self, div) -> Optional[Dict]:
        """Parse workout data from a div-based layout."""
        try:
            text = div.get_text(strip=True)
            if len(text) < 5:
                return None

            horse_number = None
            num_match = re.search(r"(\d{1,2})番", text)
            if num_match:
                horse_number = int(num_match.group(1))

            times = re.findall(r"(\d{1,2}:\d{2}\.\d|\d{2}\.\d)", text)

            return {
                "horse_number": horse_number,
                "times": times,
                "raw": text[:200],
            }
        except Exception:
            return None

    def get_workouts(self, race_id: str) -> Optional[Dict]:
        """Get workout data, preferring cache, falling back to scraping."""
        cached = self._load_cache(race_id)
        if cached:
            return cached
        return self.scrape_workouts(race_id)

    def get_workout_score(self, race_id: str, horse_number: int) -> float:
        """Get a normalized workout score for a horse (0.0 to 1.0).

        Higher = better workout performance relative to field.
        Returns 0.5 (neutral) if no data available.
        """
        data = self.get_workouts(race_id)
        if not data or not data.get("horses"):
            return 0.5

        horses = data["horses"]
        target = None
        all_times = []

        for h in horses:
            if h.get("horse_number") == horse_number:
                target = h

            # Collect last 3F times for comparison
            for t in h.get("times", []):
                try:
                    # Parse time like "37.2" (last 3F)
                    val = float(t) if ":" not in t else None
                    if val and 30 < val < 50:
                        all_times.append(val)
                except (ValueError, TypeError):
                    pass

        if target is None:
            return 0.5

        # Compare target's times to field average
        target_times = []
        for t in target.get("times", []):
            try:
                val = float(t) if ":" not in t else None
                if val and 30 < val < 50:
                    target_times.append(val)
            except (ValueError, TypeError):
                pass

        if not target_times or not all_times:
            # Use evaluation-based scoring
            eval_str = target.get("evaluation", "")
            eval_scores = {
                "抜群": 0.9, "A": 0.85, "A+": 0.9,
                "良好": 0.7, "B+": 0.7, "B": 0.6,
                "強め": 0.65, "末強め": 0.65,
                "馬なり": 0.55, "楽走": 0.5,
                "一杯": 0.4, "C": 0.35,
            }
            for key, score in eval_scores.items():
                if key in eval_str:
                    return score
            return 0.5

        # Score based on relative time (lower = faster = better)
        avg_time = sum(all_times) / len(all_times) if all_times else 37.0
        best_target = min(target_times)
        # Normalize: 0.5 = average, each 0.5s faster = +0.1
        score = 0.5 + (avg_time - best_target) * 0.2
        return max(0.0, min(1.0, round(score, 3)))

    def enrich_predictions(self, race_id: str,
                           predictions: List[Dict]) -> List[Dict]:
        """Add workout scores to prediction results."""
        data = self.get_workouts(race_id)
        if not data or not data.get("horses"):
            return predictions

        for pred in predictions:
            hn = pred.get("horse_number")
            if hn is None:
                continue
            score = self.get_workout_score(race_id, hn)
            pred["workout_score"] = score
            if score != 0.5:
                # Small probability adjustment based on workout
                adj = (score - 0.5) * 0.03  # max ~1.5% adjustment
                if "prob_win" in pred:
                    pred["prob_win"] = max(0.001, pred["prob_win"] + adj)

        return predictions


if __name__ == "__main__":
    print("Workout Scraper")
    print("=" * 50)

    if not _SCRAPING_AVAILABLE:
        print("Install required packages: pip install requests beautifulsoup4")
        exit(1)

    scraper = WorkoutScraper()

    # Example: try scraping a recent race
    test_race_id = "202506030811"
    print(f"\nAttempting to scrape workouts for race {test_race_id}...")
    result = scraper.scrape_workouts(test_race_id)

    if result:
        print(f"Status: {result.get('parse_status')}")
        print(f"Horses found: {len(result.get('horses', []))}")
        for h in result.get("horses", [])[:3]:
            print(f"  #{h.get('horse_number', '?')}: {h.get('horse_name', 'N/A')} "
                  f"- times: {h.get('times', [])} eval: {h.get('evaluation', 'N/A')}")
    else:
        print("No workout data retrieved (page may not exist for this race)")
