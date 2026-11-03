"""
競馬予想AI - レース結果自動収集
================================
レース終了後に結果を取得し、予測キャッシュと比較して的中率を算出する。

使い方:
  python result_collector.py --date 20260321
  python result_collector.py  # 今日の結果を収集

スケジュール:
  土日の 18:00, 21:00 に自動実行（api_server.py のスケジューラから呼出）
"""

import os
import json
import logging
import re
import time
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("KEIBA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")))
RESULTS_DIR = DATA_DIR / "results"
PREDICTION_CACHE_DIR = DATA_DIR / "prediction_cache"
UPCOMING_CACHE_DIR = DATA_DIR / "upcoming_cache"


def _get_race_ids_for_date(date_str: str) -> List[str]:
    """指定日のレースIDリストを prediction_cache と upcoming_cache から取得"""
    race_ids = []

    # upcoming_cache からレースIDを取得
    cache_file = UPCOMING_CACHE_DIR / f"{date_str}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            for race in cached.get("races", []):
                rid = race.get("race_id", "")
                if rid and rid not in race_ids:
                    race_ids.append(rid)
        except Exception as e:
            logger.warning(f"upcoming_cache読み込みエラー: {e}")

    # prediction_cache からもレースIDを収集
    if PREDICTION_CACHE_DIR.exists():
        for cf in PREDICTION_CACHE_DIR.glob(f"{date_str}_*.json"):
            try:
                rid = cf.stem.split("_", 1)[1]
                if rid and rid not in race_ids:
                    race_ids.append(rid)
            except (IndexError, ValueError):
                continue

    return race_ids


def _load_prediction_cache(date_str: str, race_id: str) -> Optional[Dict]:
    """指定レースの予測キャッシュを読み込み"""
    cache_file = PREDICTION_CACHE_DIR / f"{date_str}_{race_id}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"予測キャッシュ読み込みエラー {cache_file.name}: {e}")
        return None


def _scrape_race_result(race_id: str) -> Optional[Dict]:
    """race.netkeiba.com から1レースの結果を取得"""
    try:
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        time.sleep(2.0)
        resp = requests.get(url, headers=headers, timeout=30)
        resp.encoding = "euc-jp"
        if resp.status_code != 200:
            logger.warning(f"HTTP {resp.status_code}: {url}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", id="All_Result_Table")
        if not table:
            return None

        rows = table.find_all("tr")[1:]  # ヘッダーをスキップ
        if not rows:
            return None

        horses = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 8:
                continue
            try:
                finish_text = cols[0].get_text(strip=True)
                finish_pos = int(finish_text) if finish_text.isdigit() else 0
                horse_number = int(cols[2].get_text(strip=True)) if cols[2].get_text(strip=True).isdigit() else 0
                horse_name = cols[3].get_text(strip=True)
                if horse_number > 0 and horse_name:
                    horses.append({
                        "finish_position": finish_pos,
                        "horse_number": horse_number,
                        "horse_name": horse_name,
                    })
            except (ValueError, IndexError):
                continue

        if not horses:
            return None

        # レース名を取得
        race_name = ""
        name_tag = soup.find("div", class_="RaceName")
        if name_tag:
            race_name = name_tag.get_text(strip=True)

        return {
            "race_info": {"race_id": race_id, "race_name": race_name},
            "horses": horses,
        }
    except Exception as e:
        logger.error(f"結果スクレイピングエラー ({race_id}): {e}")
        return None


def collect_results_for_date(date_str: str) -> Dict:
    """
    指定日のレース結果を収集し、予測と比較して的中率を算出する。

    Args:
        date_str: YYYYMMDD 形式の日付

    Returns:
        結果データ (dict)
    """
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    logger.info(f"結果収集開始: {formatted_date}")

    race_ids = _get_race_ids_for_date(date_str)
    if not race_ids:
        logger.info(f"レースIDが見つかりません: {date_str}")
        return {
            "date": formatted_date,
            "races_analyzed": 0,
            "results": [],
            "summary": {
                "tansho_hit_rate": 0,
                "fukusho_hit_rate": 0,
                "our_top3_accuracy": 0,
            },
        }

    results = []
    tansho_hits = 0
    fukusho_hits = 0
    top3_in_top3_total = 0
    races_with_prediction = 0

    for race_id in race_ids:
        logger.info(f"  結果取得中: {race_id}")

        # 実際のレース結果をスクレイピング
        race_result = _scrape_race_result(race_id)
        if not race_result or not race_result.get("horses"):
            logger.warning(f"  結果取得失敗: {race_id}")
            continue

        race_info = race_result.get("race_info", {})
        horses = race_result["horses"]

        # 着順でソートして実際の結果を構築
        sorted_horses = sorted(
            [h for h in horses if h.get("finish_position", 0) > 0],
            key=lambda h: h["finish_position"],
        )

        actual_results = []
        for h in sorted_horses[:5]:  # 上位5着まで記録
            actual_results.append({
                "position": h["finish_position"],
                "horse_number": h["horse_number"],
                "horse_name": h["horse_name"],
            })

        if not actual_results:
            continue

        # 予測キャッシュと比較
        prediction_cache = _load_prediction_cache(date_str, race_id)

        entry = {
            "race_id": race_id,
            "race_name": race_info.get("race_name", ""),
            "actual_results": actual_results,
            "our_prediction": None,
            "tansho_hit": False,
            "fukusho_hit": False,
            "top3_in_our_top3": 0,
        }

        if prediction_cache and prediction_cache.get("predictions"):
            predictions = prediction_cache["predictions"]
            # predicted_rank でソート
            predictions_sorted = sorted(
                predictions,
                key=lambda p: p.get("predicted_rank", 999),
            )

            if predictions_sorted:
                top_pick = predictions_sorted[0]
                entry["our_prediction"] = {
                    "top_pick_number": top_pick.get("horse_number", 0),
                    "top_pick_name": top_pick.get("horse_name", ""),
                    "predicted_rank": top_pick.get("predicted_rank", 1),
                }

                races_with_prediction += 1
                winner_number = actual_results[0]["horse_number"]
                top3_numbers = [r["horse_number"] for r in actual_results[:3]]

                # 単勝的中: 予測1位が実際の1着
                if top_pick.get("horse_number") == winner_number:
                    entry["tansho_hit"] = True
                    tansho_hits += 1

                # 複勝的中: 予測1位が実際の3着以内
                if top_pick.get("horse_number") in top3_numbers:
                    entry["fukusho_hit"] = True
                    fukusho_hits += 1

                # 我々のTOP3が実際のTOP3にいくつ入っているか
                our_top3_numbers = [
                    p.get("horse_number") for p in predictions_sorted[:3]
                ]
                overlap = len(set(our_top3_numbers) & set(top3_numbers))
                entry["top3_in_our_top3"] = overlap
                top3_in_top3_total += overlap

        results.append(entry)

    # サマリー算出
    total_analyzed = len(results)
    summary = {
        "tansho_hit_rate": round(tansho_hits / races_with_prediction, 4) if races_with_prediction > 0 else 0,
        "fukusho_hit_rate": round(fukusho_hits / races_with_prediction, 4) if races_with_prediction > 0 else 0,
        "our_top3_accuracy": round(
            top3_in_top3_total / (races_with_prediction * 3), 4
        ) if races_with_prediction > 0 else 0,
    }

    output = {
        "date": formatted_date,
        "races_analyzed": total_analyzed,
        "races_with_prediction": races_with_prediction,
        "results": results,
        "summary": summary,
        "collected_at": datetime.now().isoformat(),
    }

    # ファイルに保存
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"{date_str}_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(
        f"結果収集完了: {total_analyzed}レース分析, "
        f"単勝的中率={summary['tansho_hit_rate']:.1%}, "
        f"複勝的中率={summary['fukusho_hit_rate']:.1%}, "
        f"TOP3精度={summary['our_top3_accuracy']:.1%}"
    )
    logger.info(f"保存先: {output_file}")

    return output


def load_results(date_str: str) -> Optional[Dict]:
    """保存済みの結果データを読み込み"""
    results_file = RESULTS_DIR / f"{date_str}_results.json"
    if not results_file.exists():
        return None
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"結果データ読み込みエラー: {e}")
        return None


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="レース結果自動収集")
    parser.add_argument("--date", type=str, default=None, help="日付 YYYYMMDD (省略時は今日)")
    args = parser.parse_args()

    if args.date is None:
        target_date = date.today().strftime("%Y%m%d")
    else:
        target_date = args.date

    result = collect_results_for_date(target_date)
    print(json.dumps(result.get("summary", {}), ensure_ascii=False, indent=2))
