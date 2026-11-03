#!/usr/bin/env python3
"""
Pre-cache predictions for upcoming races.
==========================================
Scrapes shutuba tables for upcoming weekend races and runs model predictions,
saving results to data/prediction_cache/{date}_{race_id}.json.

Can be run standalone or imported and called from api_server.py.
"""

import sys
import os
import json
import time
import logging
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import Optional

# Ensure project root is importable
sys.path.insert(0, "/opt/keiba-predictor")
os.chdir("/opt/keiba-predictor")

import pandas as pd
import numpy as np

from shutuba_scraper import ShutubaScraper
from model_trainer import RacePredictionModel
from prediction_ledger import record_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/prediction_cache")
CACHE_TTL_SECONDS = 21600  # 6 hours

DATA_DIR = Path("data")
MODEL_DIR = Path("models")


def _get_upcoming_dates() -> list:
    """Return YYYYMMDD strings for the next Saturday and Sunday."""
    today = date.today()
    dates = []
    for i in range(7):
        d = today + timedelta(days=i)
        if d.weekday() in (5, 6):  # Sat=5, Sun=6
            dates.append(d.strftime("%Y%m%d"))
    return dates


def _normalize_predictions_in_race(horses):
    """Normalize probabilities relative to the race field."""
    if not horses:
        return horses

    # Replace NaN/Inf with 0
    def _safe_float(v):
        try:
            f = float(v)
            if f != f or f == float('inf') or f == float('-inf'):
                return 0.0
            return max(0.0, f)
        except (TypeError, ValueError):
            return 0.0

    win_probs = [_safe_float(h.get('prob_win', 0)) for h in horses]
    top3_probs = [_safe_float(h.get('prob_top3', 0)) for h in horses]

    n = len(horses)

    win_sum = sum(win_probs)
    if win_sum > 0:
        for i, h in enumerate(horses):
            h['prob_win'] = round(win_probs[i] / win_sum, 4)
            h['win_prob'] = h['prob_win']

    top3_sum = sum(top3_probs)
    if top3_sum > 0:
        target_sum = min(3.0, n * 0.99)
        for i, h in enumerate(horses):
            h['prob_top3'] = round(min(0.99, top3_probs[i] / top3_sum * target_sum), 4)
            h['top3_prob'] = h['prob_top3']

    for h in horses:
        w = h['prob_win']
        t = h['prob_top3']
        pos = h.get('pred_position', n / 2)
        pos_score = max(0, 1 - (pos - 1) / max(n - 1, 1))
        h['composite_score'] = round(w * 0.4 + t * 0.35 + pos_score * 0.25, 4)
        h['score'] = h['composite_score']

    for h in horses:
        odds = h.get('odds', 0)
        if odds > 0:
            h['expected_value'] = round(h['prob_win'] * odds, 4)

        ev = h.get('expected_value', 0)
        prob_win = h['prob_win']
        prob_top3 = h['prob_top3']

        if odds == 0:
            # No odds available - use probability-based recommendation
            if prob_win > 0.25:
                rec = '注目（本命）'
            elif prob_top3 > 0.5:
                rec = '複勝候補'
            elif prob_top3 > 0.3:
                rec = '連下候補'
            else:
                rec = '-'
        elif ev > 1.5:
            rec = '強く推奨'
        elif ev > 1.2:
            rec = '推奨'
        elif prob_top3 > 0.3:
            rec = '複勝検討'
        else:
            rec = '見送り'
        h['recommended_bet'] = rec
        h['recommendation'] = rec

        # Recalculate bet_types based on normalized probabilities
        bet_types = []
        if ev > 1.5:
            bet_types.extend(['単勝', '馬単1着固定'])
        elif ev > 1.2:
            bet_types.extend(['単勝', '複勝'])
        if prob_top3 > 0.5:
            bet_types.append('複勝')
        if prob_top3 > 0.4:
            bet_types.append('ワイド軸')
        h['bet_types'] = list(dict.fromkeys(bet_types))  # dedupe

    horses.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
    for i, h in enumerate(horses):
        h['predicted_rank'] = i + 1

    return horses


def _build_prediction_response(
    shutuba_df: pd.DataFrame,
    predictions: pd.DataFrame,
    model_info: dict,
) -> dict:
    """
    Build the prediction response dict in the EXACT same format as
    api_server.py _run_prediction_on_shutuba().
    """
    horses = []
    for _, row in predictions.iterrows():
        h_num = int(row["horse_number"])
        prob_win = float(row.get("prob_win", 0))
        prob_top3 = float(row.get("prob_top3", 0))
        composite = float(row.get("composite_score", 0))

        # Get odds from shutuba data
        odds_row = shutuba_df[shutuba_df["horse_number"] == h_num]
        odds_val = float(odds_row.iloc[0].get("odds", 0)) if not odds_row.empty else 0

        # Recommended bet logic (mirrors api_server.py exactly)
        expected_value = prob_win * odds_val if odds_val > 0 else 0
        if odds_val == 0:
            if prob_win > 0.25:
                recommended_bet = "注目（本命）"
            elif prob_top3 > 0.5:
                recommended_bet = "複勝候補"
            elif prob_top3 > 0.3:
                recommended_bet = "連下候補"
            else:
                recommended_bet = "-"
        elif expected_value > 1.5:
            recommended_bet = "強く推奨"
        elif expected_value > 1.2:
            recommended_bet = "推奨"
        elif prob_top3 > 0.5:
            recommended_bet = "複勝検討"
        else:
            recommended_bet = "見送り"

        bet_types = []
        if expected_value > 1.5:
            bet_types.extend(["単勝", "馬単1着固定"])
        elif expected_value > 1.2:
            bet_types.extend(["単勝", "複勝"])
        if prob_top3 > 0.5:
            bet_types.append("複勝")
        if prob_top3 > 0.4:
            bet_types.append("ワイド軸")

        pred_pos = round(float(row.get("pred_position", 0)), 2)

        horses.append({
            "horse_number": h_num,
            "horse_name": row.get("horse_name", ""),
            "predicted_rank": int(row.get("predicted_rank", 0)),
            "composite_score": round(composite, 4),
            "prob_win": round(prob_win, 4),
            "prob_top3": round(prob_top3, 4),
            "pred_position": pred_pos,
            "odds": round(odds_val, 1),
            "expected_value": round(expected_value, 3),
            "recommended_bet": recommended_bet,
            "bet_types": list(dict.fromkeys(bet_types)),
            # Frontend sorting aliases
            "score": round(composite, 4),
            "win_prob": round(prob_win, 4),
            "top3_prob": round(prob_top3, 4),
            "predicted_position": pred_pos,
            "recommendation": recommended_bet,
        })

    horses = _normalize_predictions_in_race(horses)

    # Race info
    first = shutuba_df.iloc[0]
    race_info = {
        "race_id": str(first.get("race_id", "")),
        "race_name": str(first.get("race_name", "")),
        "venue": str(first.get("venue", "")),
        "course": f"{first.get('course_type', '')}{int(first.get('distance', 0))}m",
        "distance": int(first.get("distance", 0)),
        "course_type": str(first.get("course_type", "")),
        "weather": str(first.get("weather", "")),
        "track_condition": str(first.get("track_condition", "")),
        "race_date": str(first.get("race_date", "")),
    }

    return {
        "race_id": race_info["race_id"],
        "race_info": race_info,
        "predictions": horses,
        "total_horses": len(horses),
        "data_source": "shutuba",
        "model_info": model_info,
        "cached_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_precache(
    model: Optional[RacePredictionModel] = None,
    history_df: Optional[pd.DataFrame] = None,
    scraper: Optional[ShutubaScraper] = None,
    model_info: Optional[dict] = None,
) -> dict:
    """
    Run the pre-cache process for upcoming weekend races.

    Args:
        model: Pre-loaded model (loaded from disk if None)
        history_df: Past race data for enrichment (loaded from disk if None)
        scraper: ShutubaScraper instance (created if None)
        model_info: Model info dict for response (empty dict if None)

    Returns:
        Summary dict with counts of cached/skipped/failed races.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load model if not provided
    if model is None:
        model = RacePredictionModel(model_dir=str(MODEL_DIR))
        model.load("keiba_model")
        logger.info("Model loaded from disk")

    # Load history if not provided
    if history_df is None:
        all_races_path = DATA_DIR / "_all_races.csv"
        if all_races_path.exists():
            history_df = pd.read_csv(str(all_races_path))
            logger.info(f"History loaded: {len(history_df)} rows")

    if scraper is None:
        scraper = ShutubaScraper(delay=2.0)

    if model_info is None:
        model_info = {}

    dates = _get_upcoming_dates()
    if not dates:
        logger.info("No upcoming weekend dates found within next 7 days")
        return {"dates": [], "cached": 0, "skipped": 0, "failed": 0}

    logger.info(f"Pre-caching predictions for dates: {dates}")

    stats = {"dates": dates, "cached": 0, "skipped": 0, "failed": 0}

    for ds in dates:
        try:
            races = scraper.get_upcoming_races(ds)
        except Exception as e:
            logger.warning(f"Failed to get race list for {ds}: {e}")
            continue

        if not races:
            logger.info(f"{ds}: no races found")
            continue

        logger.info(f"{ds}: {len(races)} races found")

        for race in races:
            race_id = race.get("race_id", "")
            if not race_id:
                continue

            cache_file = CACHE_DIR / f"{ds}_{race_id}.json"

            # Check existing cache freshness
            if cache_file.exists():
                age = time.time() - cache_file.stat().st_mtime
                if age < CACHE_TTL_SECONDS:
                    logger.info(f"  Skip {race_id} (cached {age/3600:.1f}h ago)")
                    stats["skipped"] += 1
                    continue

            try:
                shutuba_df = scraper.scrape_shutuba(race_id)
                if shutuba_df is None or shutuba_df.empty:
                    logger.warning(f"  Empty shutuba for {race_id}")
                    stats["failed"] += 1
                    continue

                # 本パイプラインで特徴量化 (api_server の predict と同一経路。
                # 失敗時のみ従来の簡易 enrich_with_history にフォールバック)
                featured_target = None
                if history_df is not None:
                    try:
                        from prediction_context import featurize_with_full_context
                        featured_target = featurize_with_full_context(shutuba_df, history_df)
                    except Exception as feat_err:
                        logger.warning(f"  full-pipeline featurization failed for {race_id}: {feat_err}")

                if featured_target is not None:
                    predictions = model.predict(featured_target)
                else:
                    if history_df is not None:
                        shutuba_df = scraper.enrich_with_history(shutuba_df, history_df)
                    predictions = model.predict(shutuba_df)

                # Build response
                result = _build_prediction_response(shutuba_df, predictions, model_info)

                # Write cache file
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, default=str)

                # Record to prediction ledger for transparency
                try:
                    record_prediction(race_id, result.get("predictions", []), source="precache")
                except Exception as ledger_err:
                    logger.warning(f"  Ledger record failed for {race_id}: {ledger_err}")

                logger.info(f"  Cached {race_id}: {result['total_horses']} horses")
                stats["cached"] += 1

            except Exception as e:
                logger.warning(f"  Error predicting {race_id}: {e}")
                stats["failed"] += 1
                continue

    logger.info(
        f"Pre-cache complete: {stats['cached']} cached, "
        f"{stats['skipped']} skipped, {stats['failed']} failed"
    )
    return stats


def main():
    """CLI entry point."""
    logger.info("Starting prediction pre-cache...")
    try:
        stats = run_precache()
        logger.info(f"Done. Results: {stats}")
    except Exception as e:
        logger.error(f"Pre-cache failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
