"""
Value Bet Alert System
======================
Identifies high-EV betting opportunities by combining cached predictions
with live odds through the overlay model, and sends push notifications
30-60 minutes before post time.

Usage:
    from value_alerts import find_value_bets, send_value_alerts

    # Find value bets for a date
    bets = find_value_bets("20260321", ev_threshold=1.3)

    # Send notifications for value bets
    result = send_value_alerts("20260321", ev_threshold=1.3)
"""

import os
import json
import logging
import re
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
PREDICTION_CACHE_DIR = DATA_DIR / "prediction_cache"
UPCOMING_CACHE_DIR = DATA_DIR / "upcoming_cache"
MODEL_DIR = BASE_DIR / "models"


def _load_overlay_model():
    """Load the trained odds overlay model."""
    overlay_path = MODEL_DIR / "odds_overlay.pkl"
    if not overlay_path.exists():
        logger.warning("Overlay model not found at %s", overlay_path)
        return None
    try:
        from odds_overlay import OddsOverlayModel
        return OddsOverlayModel.load(str(overlay_path))
    except Exception as e:
        logger.error("Failed to load overlay model: %s", e)
        return None


def _load_cached_predictions(date_str: str) -> List[Dict]:
    """Load all cached predictions for a date from prediction_cache/."""
    if not PREDICTION_CACHE_DIR.exists():
        return []

    results = []
    for cache_file in sorted(PREDICTION_CACHE_DIR.glob(f"{date_str}_*.json")):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            results.append(cached)
        except Exception as e:
            logger.warning("Failed to read cache %s: %s", cache_file, e)
            continue
    return results


def _load_upcoming_race_info(date_str: str) -> Dict[str, Dict]:
    """Load upcoming race info (venue, race_number, race_name, post_time)."""
    race_info_map = {}
    upcoming_file = UPCOMING_CACHE_DIR / f"{date_str}.json"
    if upcoming_file.exists():
        try:
            with open(upcoming_file, "r", encoding="utf-8") as f:
                upcoming = json.load(f)
            for race in upcoming.get("races", []):
                rid = race.get("race_id", "")
                if rid:
                    race_info_map[rid] = race
        except Exception:
            pass
    return race_info_map


def _extract_race_number(race_id: str) -> int:
    """Extract race number from race_id (last 2 digits)."""
    if len(race_id) >= 2:
        try:
            return int(race_id[-2:])
        except ValueError:
            pass
    return 0


def find_value_bets(date_str: str, ev_threshold: float = 1.3) -> List[Dict]:
    """
    Find value bets from cached predictions + live odds via overlay model.

    Args:
        date_str: YYYYMMDD date string.
        ev_threshold: Minimum expected value (default 1.3 = STRONG_VALUE).

    Returns:
        List of value bet dicts sorted by EV descending, each containing:
        - race_id, venue, race_number, race_name
        - horse_number, horse_name
        - odds, ev, calibrated_prob, edge, kelly_fraction
        - bet_type (STRONG_VALUE / VALUE / etc.)
    """
    cached_races = _load_cached_predictions(date_str)
    if not cached_races:
        logger.info("No cached predictions found for %s", date_str)
        return []

    overlay_model = _load_overlay_model()
    if overlay_model is None:
        logger.warning("Overlay model unavailable; falling back to simple EV")
        return _find_value_bets_simple(cached_races, date_str, ev_threshold)

    upcoming_info = _load_upcoming_race_info(date_str)

    value_bets = []

    for race_cache in cached_races:
        predictions = race_cache.get("predictions", [])
        if not predictions:
            continue

        race_info = race_cache.get("race_info", {})
        race_id = race_cache.get("race_id", race_info.get("race_id", ""))

        # Merge upcoming info for venue/race_number
        uc = upcoming_info.get(race_id, {})
        venue = race_info.get("venue", "") or uc.get("venue", "")
        race_number = int(race_info.get("race_number", 0)) or int(uc.get("race_number", 0))
        if race_number == 0:
            race_number = _extract_race_number(race_id)

        raw_name = race_info.get("race_name", "") or uc.get("race_name", "")
        race_name = re.sub(r'^\d+R', '', raw_name).strip()
        race_name = re.sub(r'\d{1,2}:\d{2}', '', race_name).strip()
        race_name = re.sub(r'[ダ芝]?\d+m\d*頭?$', '', race_name).strip()
        if not race_name:
            race_name = raw_name

        # Build base predictions list for overlay model
        # Only process horses that have odds > 0
        horses_with_odds = [h for h in predictions if float(h.get("odds", 0)) > 0]
        if not horses_with_odds:
            continue

        # Build odds dict and base predictions for overlay
        odds_dict = {}
        base_predictions = []
        for h in horses_with_odds:
            hn = int(h.get("horse_number", 0))
            odds_val = float(h.get("odds", 0))
            odds_dict[hn] = odds_val
            base_predictions.append({
                "horse_number": hn,
                "horse_name": h.get("horse_name", ""),
                "prob_win": float(h.get("prob_win", h.get("win_prob", 0))),
                "prob_top3": float(h.get("prob_top3", h.get("top3_prob", 0))),
                "composite_score": float(h.get("composite_score", h.get("score", 0))),
                "popularity": int(h.get("popularity", 0)),
            })

        try:
            overlay_results = overlay_model.predict_from_base_predictions(
                base_predictions, odds_dict
            )
        except Exception as e:
            logger.warning("Overlay prediction failed for %s: %s", race_id, e)
            # Fall back to simple EV for this race
            overlay_results = _simple_ev_for_race(horses_with_odds)

        for result in overlay_results:
            if result["ev"] >= ev_threshold:
                # Determine recommended bet type in Japanese
                ev = result["ev"]
                if ev >= 1.5:
                    bet_recommendation = "単勝 強く推奨"
                elif ev >= 1.3:
                    bet_recommendation = "単勝 推奨"
                elif ev >= 1.15:
                    bet_recommendation = "単勝 検討"
                else:
                    bet_recommendation = "複勝 検討"

                value_bets.append({
                    "race_id": race_id,
                    "venue": venue,
                    "race_number": race_number,
                    "race_name": race_name,
                    "horse_number": result["horse_number"],
                    "horse_name": result["horse_name"],
                    "odds": result["odds"],
                    "ev": result["ev"],
                    "calibrated_prob": result.get("calibrated_prob", 0),
                    "model_prob": result.get("model_prob", 0),
                    "market_prob": result.get("market_prob", 0),
                    "edge": result.get("edge", 0),
                    "kelly_fraction": result.get("kelly_fraction", 0),
                    "bet_type": result.get("bet_type", "VALUE"),
                    "bet_recommendation": bet_recommendation,
                })

    # Sort by EV descending
    value_bets.sort(key=lambda x: x["ev"], reverse=True)

    logger.info(
        "Found %d value bets (EV >= %.2f) for %s across %d races",
        len(value_bets), ev_threshold, date_str, len(cached_races),
    )
    return value_bets


def _simple_ev_for_race(horses: List[Dict]) -> List[Dict]:
    """Fallback: compute simple EV without overlay model."""
    results = []
    for h in horses:
        odds = float(h.get("odds", 0))
        prob_win = float(h.get("prob_win", h.get("win_prob", 0)))
        if odds <= 0 or prob_win <= 0:
            continue
        ev = prob_win * odds
        implied_prob = 1.0 / odds

        if ev > 1.3:
            bet_type = "STRONG_VALUE"
        elif ev > 1.15:
            bet_type = "VALUE"
        elif ev > 1.0:
            bet_type = "MARGINAL"
        else:
            bet_type = "PASS"

        results.append({
            "horse_number": int(h.get("horse_number", 0)),
            "horse_name": h.get("horse_name", ""),
            "odds": round(odds, 1),
            "popularity": int(h.get("popularity", 0)),
            "model_prob": round(prob_win, 4),
            "market_prob": round(implied_prob, 4),
            "calibrated_prob": round(prob_win, 4),
            "edge": round(prob_win - implied_prob, 4),
            "ev": round(ev, 3),
            "kelly_fraction": 0,
            "bet_type": bet_type,
            "is_value_bet": ev > 1.15,
        })
    return sorted(results, key=lambda x: x["ev"], reverse=True)


def _find_value_bets_simple(
    cached_races: List[Dict], date_str: str, ev_threshold: float
) -> List[Dict]:
    """Fallback when overlay model is unavailable: use simple prob * odds."""
    upcoming_info = _load_upcoming_race_info(date_str)
    value_bets = []

    for race_cache in cached_races:
        predictions = race_cache.get("predictions", [])
        if not predictions:
            continue

        race_info = race_cache.get("race_info", {})
        race_id = race_cache.get("race_id", race_info.get("race_id", ""))
        uc = upcoming_info.get(race_id, {})

        venue = race_info.get("venue", "") or uc.get("venue", "")
        race_number = int(race_info.get("race_number", 0)) or int(uc.get("race_number", 0))
        if race_number == 0:
            race_number = _extract_race_number(race_id)

        raw_name = race_info.get("race_name", "") or uc.get("race_name", "")
        race_name = re.sub(r'^\d+R', '', raw_name).strip()
        race_name = re.sub(r'\d{1,2}:\d{2}', '', race_name).strip()
        race_name = re.sub(r'[ダ芝]?\d+m\d*頭?$', '', race_name).strip()
        if not race_name:
            race_name = raw_name

        for h in predictions:
            odds = float(h.get("odds", 0))
            prob_win = float(h.get("prob_win", h.get("win_prob", 0)))
            if odds <= 0 or prob_win <= 0:
                continue
            ev = prob_win * odds
            if ev < ev_threshold:
                continue

            if ev >= 1.5:
                bet_recommendation = "単勝 強く推奨"
            elif ev >= 1.3:
                bet_recommendation = "単勝 推奨"
            elif ev >= 1.15:
                bet_recommendation = "単勝 検討"
            else:
                bet_recommendation = "複勝 検討"

            value_bets.append({
                "race_id": race_id,
                "venue": venue,
                "race_number": race_number,
                "race_name": race_name,
                "horse_number": int(h.get("horse_number", 0)),
                "horse_name": h.get("horse_name", ""),
                "odds": round(odds, 1),
                "ev": round(ev, 3),
                "calibrated_prob": round(prob_win, 4),
                "model_prob": round(prob_win, 4),
                "market_prob": round(1.0 / odds, 4),
                "edge": round(prob_win - 1.0 / odds, 4),
                "kelly_fraction": 0,
                "bet_type": "STRONG_VALUE" if ev > 1.3 else "VALUE",
                "bet_recommendation": bet_recommendation,
            })

    value_bets.sort(key=lambda x: x["ev"], reverse=True)
    logger.info(
        "Found %d value bets (simple EV >= %.2f) for %s",
        len(value_bets), ev_threshold, date_str,
    )
    return value_bets


def send_value_alerts(
    date_str: Optional[str] = None,
    ev_threshold: float = 1.3,
) -> Dict:
    """
    Send LINE/Discord notifications for value bets to all subscribed users.

    Uses the existing notifier infrastructure (notify_user with event_type='value_bet').

    Args:
        date_str: YYYYMMDD date string. Defaults to today.
        ev_threshold: Minimum expected value for alerts.

    Returns:
        Dict with sent count and details.
    """
    from notifier import (
        _load_all_settings, get_user_settings, should_notify,
        send_line, send_discord,
    )

    if date_str is None:
        date_str = date.today().strftime("%Y%m%d")

    value_bets = find_value_bets(date_str, ev_threshold)
    if not value_bets:
        logger.info("send_value_alerts: no value bets found for %s", date_str)
        return {"sent": 0, "value_bets": 0, "message": "バリューベットなし"}

    # Format date label
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
        weekday_map = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
        weekday_str = weekday_map.get(dt.weekday(), "")
        date_label = f"{dt.month}/{dt.day}({weekday_str})"
    except ValueError:
        date_label = date_str

    # Build notification message
    lines = [
        f"\n🏇 バリューベット発見！ {date_label}",
        "",
    ]

    for bet in value_bets[:8]:  # Max 8 bets per notification
        venue = bet["venue"]
        rnum = bet["race_number"]
        rname = bet["race_name"]
        race_label = f"{venue}{rnum}R" if rnum else venue
        if rname:
            race_label = f"{rname} ({race_label})"

        lines.append(f"📍 {race_label}")
        lines.append(
            f"  ◎ {bet['horse_number']}番 {bet['horse_name']}"
        )
        lines.append(
            f"  オッズ: {bet['odds']}倍 → EV: {bet['ev']:.2f}"
        )
        lines.append(
            f"  推奨: {bet['bet_recommendation']}"
        )
        lines.append("")

    if len(value_bets) > 8:
        lines.append(f"他 {len(value_bets) - 8} 件のバリューベットあり")
        lines.append("")

    lines.append(f"詳細: {os.environ.get('KEIBA_PUBLIC_URL', 'http://localhost:8000')}/keiba/#predict")
    lines.append("⚠️ 投資は自己責任でお願いします")

    message = "\n".join(lines)

    # Send to all users with notify_value_bet enabled
    all_settings = _load_all_settings()
    sent_count = 0
    results = {}

    for username, settings in all_settings.items():
        if not settings.get("notify_value_bet", True):
            continue

        user_results = {"line": False, "discord": False}

        # Per-user filtering: check min_ev setting
        user_min_ev = settings.get("min_ev", 1.2)
        user_bets = [b for b in value_bets if b["ev"] >= user_min_ev]
        if not user_bets:
            continue

        # Per-user venue filter
        allowed_venues = settings.get("venues", [])
        if allowed_venues:
            user_bets = [b for b in user_bets if b["venue"] in allowed_venues]
        if not user_bets:
            continue

        line_token = settings.get("line_token", "").strip()
        discord_url = settings.get("discord_webhook", "").strip()

        if line_token:
            user_results["line"] = send_line(line_token, message)
            if user_results["line"]:
                sent_count += 1

        if discord_url:
            user_results["discord"] = send_discord(discord_url, message)
            if user_results["discord"]:
                sent_count += 1

        results[username] = user_results

    logger.info(
        "Value alerts sent: %d channels, %d value bets found",
        sent_count, len(value_bets),
    )
    return {
        "sent": sent_count,
        "value_bets": len(value_bets),
        "results": results,
        "message": message,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Value Bet Alerts")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date in YYYYMMDD format (default: today)",
    )
    parser.add_argument(
        "--threshold", type=float, default=1.3,
        help="EV threshold (default: 1.3)",
    )
    parser.add_argument(
        "--send", action="store_true",
        help="Actually send notifications (otherwise just print)",
    )
    args = parser.parse_args()

    target_date = args.date or date.today().strftime("%Y%m%d")

    print(f"Searching for value bets: date={target_date}, threshold={args.threshold}")
    bets = find_value_bets(target_date, args.threshold)

    if not bets:
        print("No value bets found.")
    else:
        print(f"\nFound {len(bets)} value bet(s):\n")
        for b in bets:
            print(
                f"  {b['venue']}{b['race_number']}R {b['race_name']}  "
                f"{b['horse_number']}番 {b['horse_name']}  "
                f"オッズ {b['odds']}倍  EV={b['ev']:.2f}  "
                f"({b['bet_type']})"
            )

    if args.send:
        print("\nSending notifications...")
        result = send_value_alerts(target_date, args.threshold)
        print(f"Sent to {result['sent']} channel(s)")
    elif bets:
        print("\nUse --send to actually send notifications.")
