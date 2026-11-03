"""
競馬予想AI - 馬場状態リアルタイム更新 & 再予測
================================================
レース当日の馬場状態（良/稍重/重/不良）変化を検知し、
キャッシュ済み予測を再実行する。

雨で良→重に変わるとダートレースの順位が大きく変動するため、
土日の朝〜午後にかけて30分ごとに馬場状態をチェックし、
変化があれば該当レースの予測を自動で再計算する。

Functions:
    get_current_conditions(date_str)   - 当日の馬場状態を取得
    has_condition_changed(date_str)    - キャッシュ予測との差分を検出
    repredict_changed_races(date_str)  - 変化したレースを再予測
"""

import os
import sys
import json
import re
import time
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB = True
except ImportError:
    HAS_WEB = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

# ============================================================
# 定数
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

VENUE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

VALID_CONDITIONS = {"良", "稍重", "重", "不良"}
VALID_WEATHER = {"晴", "曇", "小雨", "雨", "小雪", "雪"}

DATA_DIR = Path(os.environ.get("KEIBA_DATA_DIR", "data"))
PREDICTION_CACHE_DIR = Path(__file__).parent / "data" / "prediction_cache"
CONDITION_CACHE_DIR = Path(__file__).parent / "data" / "track_conditions"

REQUEST_DELAY = 2.5  # seconds between requests

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Lazy-init HTTP session."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(HEADERS)
    return _session


def _fetch_page(url: str, encoding: str = "utf-8") -> Optional[BeautifulSoup]:
    """Fetch a page with rate limiting."""
    if not HAS_WEB:
        logger.error("requests/bs4 not available")
        return None
    time.sleep(REQUEST_DELAY)
    try:
        resp = _get_session().get(url, timeout=30)
        resp.encoding = encoding
        if resp.status_code != 200:
            logger.warning(f"HTTP {resp.status_code}: {url}")
            return None
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        logger.error(f"Request error: {url} -> {e}")
        return None


# ============================================================
# 1. 馬場状態取得
# ============================================================

def _scrape_race_list_conditions(date_str: str) -> Dict[str, Dict]:
    """
    race_list_sub.html からその日の開催場 & レースIDを取得し、
    各場の代表レース(1R)の shutuba ページから馬場状態を読み取る。

    Returns:
        {
            "東京": {"turf": "良", "dirt": "重", "weather": "雨"},
            "中山": {"turf": "稍重", "dirt": "稍重", "weather": "曇"},
            ...
        }
    """
    # Step 1: レース一覧からその日の開催場を特定
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    soup = _fetch_page(url)
    if not soup:
        logger.warning(f"Failed to fetch race list for {date_str}")
        return {}

    # race_id -> venue_code のマッピングを収集
    venue_race_ids: Dict[str, List[str]] = {}
    for link in soup.find_all("a", href=True):
        m = re.search(r"race_id=(\d{12})", link["href"])
        if not m:
            continue
        race_id = m.group(1)
        venue_code = race_id[4:6]
        venue_name = VENUE_CODES.get(venue_code, "")
        if not venue_name:
            continue
        venue_race_ids.setdefault(venue_name, []).append(race_id)

    if not venue_race_ids:
        logger.info(f"No venues found for {date_str}")
        return {}

    # Step 2: 各場の代表レースから馬場状態を取得
    # 最初のレースと最後のレースをチェック（芝/ダート両方を拾うため）
    conditions: Dict[str, Dict] = {}

    for venue_name, race_ids in venue_race_ids.items():
        sorted_ids = sorted(set(race_ids))
        # 複数レースをサンプリングして芝/ダートの両方を取得する
        sample_ids = []
        if sorted_ids:
            sample_ids.append(sorted_ids[0])  # 1R
        if len(sorted_ids) > 5:
            sample_ids.append(sorted_ids[5])  # 6R前後
        if len(sorted_ids) > 1:
            sample_ids.append(sorted_ids[-1])  # 最終R

        venue_cond = {"turf": "良", "dirt": "良", "weather": "晴"}
        turf_found = False
        dirt_found = False

        for rid in sample_ids:
            if turf_found and dirt_found:
                break
            cond = _scrape_shutuba_condition(rid)
            if not cond:
                continue

            weather = cond.get("weather", "")
            if weather and weather in VALID_WEATHER:
                venue_cond["weather"] = weather

            track_cond = cond.get("track_condition", "")
            course_type = cond.get("course_type", "")

            if track_cond in VALID_CONDITIONS:
                if "芝" in course_type and not turf_found:
                    venue_cond["turf"] = track_cond
                    turf_found = True
                elif "ダ" in course_type and not dirt_found:
                    venue_cond["dirt"] = track_cond
                    dirt_found = True

        conditions[venue_name] = venue_cond

    return conditions


def _scrape_shutuba_condition(race_id: str) -> Optional[Dict]:
    """
    shutuba.html から1レースの馬場状態・天候・コースタイプを取得。

    Returns:
        {"weather": "雨", "track_condition": "重", "course_type": "ダート"}
        or None
    """
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    soup = _fetch_page(url, encoding="euc-jp")
    if not soup:
        return None

    result = {"weather": "", "track_condition": "", "course_type": ""}

    # コース情報 (RaceData01)
    course_tag = soup.find("div", class_="RaceData01") or soup.find("div", class_="racedata")
    full_text = ""
    if course_tag:
        full_text = course_tag.get_text()

    # RaceData02 にも馬場状態が載ることがある
    race_data2 = soup.find("div", class_="RaceData02") or soup.find("div", class_="race_otherdata")
    if race_data2:
        full_text += " " + race_data2.get_text()

    # コースタイプ
    if "芝" in full_text and "ダ" not in full_text:
        result["course_type"] = "芝"
    elif "ダ" in full_text:
        result["course_type"] = "ダート"

    # 天候
    weather_match = re.search(r"天候\s*[:：]\s*(\S+)", full_text)
    if weather_match:
        result["weather"] = weather_match.group(1)

    # 馬場状態: "芝 : 良" or "ダート : 重" 形式
    track_match = re.search(r"(?:芝|ダート)\s*[:：]\s*(\S+)", full_text)
    if track_match:
        result["track_condition"] = track_match.group(1)

    return result


def get_current_conditions(date_str: str) -> Dict[str, Dict]:
    """
    指定日の全開催場の現在の馬場状態を取得する。

    Args:
        date_str: "YYYYMMDD" 形式の日付文字列

    Returns:
        {
            "東京": {"turf": "良", "dirt": "重", "weather": "雨"},
            "中山": {"turf": "稍重", "dirt": "稍重", "weather": "曇"},
        }
    """
    logger.info(f"Fetching track conditions for {date_str}")
    conditions = _scrape_race_list_conditions(date_str)

    if conditions:
        # キャッシュに保存
        CONDITION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CONDITION_CACHE_DIR / f"{date_str}.json"
        cache_data = {
            "date": date_str,
            "conditions": conditions,
            "fetched_at": datetime.now().isoformat(),
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Track conditions cached: {list(conditions.keys())}")

    return conditions


def get_cached_conditions(date_str: str) -> Optional[Dict]:
    """キャッシュ済みの馬場状態を読み込む。"""
    cache_file = CONDITION_CACHE_DIR / f"{date_str}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("conditions", {})
    except (json.JSONDecodeError, IOError):
        return None


# ============================================================
# 2. 変化検知
# ============================================================

def has_condition_changed(date_str: str) -> List[Dict]:
    """
    現在の馬場状態とキャッシュ済み予測の馬場状態を比較し、
    変化があったレースのリストを返す。

    Returns:
        [
            {
                "race_id": "202606020701",
                "venue": "中山",
                "course_type": "ダート",
                "cached_condition": "良",
                "current_condition": "重",
                "cache_file": "20260321_202606020701.json",
            },
            ...
        ]
    """
    # 現在の馬場状態を取得
    current = get_current_conditions(date_str)
    if not current:
        logger.info("No current conditions available")
        return []

    # キャッシュ済み予測をスキャン
    if not PREDICTION_CACHE_DIR.exists():
        return []

    changed_races = []

    for cache_file in sorted(PREDICTION_CACHE_DIR.glob(f"{date_str}_*.json")):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        race_info = pred_data.get("race_info", {})
        race_id = race_info.get("race_id", "")
        venue = race_info.get("venue", "")
        course_type = race_info.get("course_type", "")
        cached_condition = race_info.get("track_condition", "良")

        if venue not in current:
            continue

        venue_conditions = current[venue]

        # 芝/ダートに応じた馬場状態を取得
        if "芝" in course_type:
            current_condition = venue_conditions.get("turf", "良")
        elif "ダ" in course_type:
            current_condition = venue_conditions.get("dirt", "良")
        else:
            continue

        if cached_condition != current_condition:
            changed_races.append({
                "race_id": race_id,
                "venue": venue,
                "course_type": course_type,
                "cached_condition": cached_condition,
                "current_condition": current_condition,
                "weather": venue_conditions.get("weather", ""),
                "cache_file": cache_file.name,
            })

    if changed_races:
        logger.info(
            f"Track condition changes detected: {len(changed_races)} races "
            f"({', '.join(r['race_id'] for r in changed_races[:5])})"
        )
    else:
        logger.info(f"No track condition changes for {date_str}")

    return changed_races


# ============================================================
# 3. 再予測
# ============================================================

def repredict_changed_races(
    date_str: str,
    model=None,
    history_df=None,
    scraper=None,
    model_info: Optional[dict] = None,
) -> Dict:
    """
    馬場状態が変化したレースの予測を再実行し、キャッシュを更新する。

    Args:
        date_str: "YYYYMMDD" 形式
        model: RacePredictionModel (None なら内部でロード)
        history_df: 過去データ DataFrame
        scraper: ShutubaScraper インスタンス
        model_info: モデル情報 dict

    Returns:
        {
            "date": "20260321",
            "changed_races": [...],
            "repredicted": 3,
            "failed": 0,
            "skipped": 0,
            "condition_changes": [...],
        }
    """
    changed = has_condition_changed(date_str)

    result = {
        "date": date_str,
        "condition_changes": changed,
        "repredicted": 0,
        "failed": 0,
        "skipped": 0,
    }

    if not changed:
        return result

    # Lazy-load dependencies
    if model is None:
        try:
            from model_trainer import RacePredictionModel
            model = RacePredictionModel(model_dir="models")
            model.load("keiba_model")
        except Exception as e:
            logger.error(f"Failed to load model for re-prediction: {e}")
            result["error"] = str(e)
            return result

    if scraper is None:
        try:
            from shutuba_scraper import ShutubaScraper
            scraper = ShutubaScraper(delay=2.0)
        except Exception as e:
            logger.error(f"Failed to create scraper: {e}")
            result["error"] = str(e)
            return result

    if history_df is None:
        all_races_path = DATA_DIR / "_all_races.csv"
        if all_races_path.exists():
            try:
                import pandas as pd
                history_df = pd.read_csv(str(all_races_path))
            except Exception:
                pass

    if model_info is None:
        model_info = {}

    # Process each changed race
    for change in changed:
        race_id = change["race_id"]
        cache_file = PREDICTION_CACHE_DIR / change["cache_file"]

        try:
            # Re-scrape shutuba (gets latest track condition from the page)
            shutuba_df = scraper.scrape_shutuba(race_id)
            if shutuba_df is None or shutuba_df.empty:
                logger.warning(f"Empty shutuba for {race_id}, skipping re-prediction")
                result["skipped"] += 1
                continue

            # Update track condition in the DataFrame to match real conditions
            new_condition = change["current_condition"]
            if "track_condition" in shutuba_df.columns:
                shutuba_df["track_condition"] = new_condition
            # Also update the numeric encoding
            condition_map = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
            if "track_condition_code" in shutuba_df.columns:
                shutuba_df["track_condition_code"] = condition_map.get(new_condition, 0)

            # Enrich with history
            if history_df is not None:
                try:
                    shutuba_df = scraper.enrich_with_history(shutuba_df, history_df)
                except Exception:
                    pass

            # Re-predict
            import pandas as pd
            predictions = model.predict(shutuba_df)

            # Build response (reuse precache logic)
            from precache_predictions import _build_prediction_response
            new_result = _build_prediction_response(shutuba_df, predictions, model_info)

            # Mark as re-predicted due to condition change
            new_result["repredicted_at"] = datetime.now().isoformat()
            new_result["condition_change"] = {
                "from": change["cached_condition"],
                "to": change["current_condition"],
                "course_type": change["course_type"],
                "weather": change.get("weather", ""),
            }

            # Update race_info with new condition
            new_result["race_info"]["track_condition"] = new_condition
            if change.get("weather"):
                new_result["race_info"]["weather"] = change["weather"]

            # Write updated cache file
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(new_result, f, ensure_ascii=False, indent=2, default=str)

            logger.info(
                f"Re-predicted {race_id}: {change['course_type']} "
                f"{change['cached_condition']} -> {change['current_condition']} "
                f"({new_result['total_horses']} horses)"
            )
            result["repredicted"] += 1

        except Exception as e:
            logger.error(f"Re-prediction failed for {race_id}: {e}")
            result["failed"] += 1

    logger.info(
        f"Re-prediction complete for {date_str}: "
        f"{result['repredicted']} updated, "
        f"{result['failed']} failed, "
        f"{result['skipped']} skipped"
    )

    return result


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="馬場状態チェック & 再予測")
    parser.add_argument("--date", type=str, default=None,
                        help="日付 YYYYMMDD (省略時は今日)")
    parser.add_argument("--check", action="store_true",
                        help="馬場状態の変化を確認するのみ（再予測しない）")
    parser.add_argument("--conditions", action="store_true",
                        help="現在の馬場状態を表示するのみ")
    args = parser.parse_args()

    target_date = args.date or date.today().strftime("%Y%m%d")

    if args.conditions:
        conds = get_current_conditions(target_date)
        if conds:
            print(f"\n--- Track Conditions for {target_date} ---")
            for venue, data in sorted(conds.items()):
                print(f"  {venue}: turf={data['turf']} dirt={data['dirt']} weather={data['weather']}")
        else:
            print("No conditions found.")
    elif args.check:
        changed = has_condition_changed(target_date)
        if changed:
            print(f"\n--- Changes Detected ({len(changed)} races) ---")
            for c in changed:
                print(f"  {c['race_id']} ({c['venue']} {c['course_type']}): "
                      f"{c['cached_condition']} -> {c['current_condition']}")
        else:
            print("No changes detected.")
    else:
        result = repredict_changed_races(target_date)
        print(f"\nResult: {json.dumps(result, ensure_ascii=False, indent=2)}")
