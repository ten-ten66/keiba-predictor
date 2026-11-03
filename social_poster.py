"""
競馬予想AI - Twitter/X 自動投稿
================================
AI予測と結果を自動的にTwitter/Xに投稿する。

機能:
  - 予測ツイートのフォーマット
  - 結果ツイートのフォーマット（的中/不的中）
  - Twitter API v2 経由での投稿
  - 日次ベストピック自動投稿
  - 結果収集後の自動投稿（土日18:30）

設定:
  data/twitter_config.json に認証情報を保存
"""

import os
import sys
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("KEIBA_DATA_DIR",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
TWITTER_CONFIG_FILE = os.path.join(DATA_DIR, "twitter_config.json")
PREDICTION_CACHE_DIR = os.path.join(DATA_DIR, "prediction_cache")


# ============================================================
# 設定管理
# ============================================================

def load_twitter_config() -> dict:
    """Twitter API 認証設定を読み込む"""
    if os.path.exists(TWITTER_CONFIG_FILE):
        with open(TWITTER_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "api_key": "",
        "api_secret": "",
        "access_token": "",
        "access_secret": "",
        "bearer_token": "",
        "enabled": False,
        "auto_post_predictions": True,
        "auto_post_results": True,
    }


def save_twitter_config(config: dict):
    """Twitter API 認証設定を保存する"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TWITTER_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info("Twitter設定を保存しました")


# ============================================================
# Twitter API v2 投稿
# ============================================================

def _get_oauth1_session(config: dict):
    """OAuth 1.0a セッションを取得"""
    try:
        from requests_oauthlib import OAuth1Session
    except ImportError:
        logger.error("requests_oauthlib が未インストール: pip install requests-oauthlib")
        return None

    api_key = config.get("api_key", "")
    api_secret = config.get("api_secret", "")
    access_token = config.get("access_token", "")
    access_secret = config.get("access_secret", "")

    if not all([api_key, api_secret, access_token, access_secret]):
        logger.error("Twitter API 認証情報が不完全です")
        return None

    return OAuth1Session(
        api_key,
        client_secret=api_secret,
        resource_owner_key=access_token,
        resource_owner_secret=access_secret,
    )


def post_to_twitter(message: str, api_key=None, api_secret=None,
                     access_token=None, access_secret=None) -> dict:
    """
    Post a tweet using Twitter API v2.

    Args:
        message: ツイート本文（280文字以内）
        api_key, api_secret, access_token, access_secret:
            指定しない場合は twitter_config.json から読み込む

    Returns:
        {"status": "ok", "tweet_id": "..."} or {"status": "error", "message": "..."}
    """
    config = load_twitter_config()

    if api_key:
        config["api_key"] = api_key
    if api_secret:
        config["api_secret"] = api_secret
    if access_token:
        config["access_token"] = access_token
    if access_secret:
        config["access_secret"] = access_secret

    if not config.get("enabled", False):
        logger.warning("Twitter投稿が無効です（enabled=false）")
        return {"status": "disabled", "message": "Twitter投稿が無効です。設定で有効にしてください。"}

    session = _get_oauth1_session(config)
    if session is None:
        return {"status": "error", "message": "OAuth認証セッションの作成に失敗しました"}

    url = "https://api.twitter.com/2/tweets"
    payload = {"text": message[:280]}

    try:
        response = session.post(url, json=payload)
        if response.status_code == 201:
            data = response.json()
            tweet_id = data.get("data", {}).get("id", "unknown")
            logger.info(f"ツイート投稿成功: {tweet_id}")
            return {"status": "ok", "tweet_id": tweet_id, "text": message[:280]}
        else:
            error_msg = f"Twitter API エラー: {response.status_code} {response.text}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    except Exception as e:
        error_msg = f"Twitter投稿例外: {e}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}


# ============================================================
# ツイートフォーマッター
# ============================================================

def format_prediction_tweet(race_info: dict, top3_picks: list) -> str:
    """
    予測をツイート用にフォーマットする。

    Args:
        race_info: {"venue": "東京", "race_name": "天皇賞(春)", "race_number": 11}
        top3_picks: [{"horse_number": 5, "horse_name": "...", "prob_win": 0.23}, ...]

    Returns:
        280文字以内のツイート文字列
    """
    venue = race_info.get("venue", "")
    race_name = race_info.get("race_name", "")
    race_num = race_info.get("race_number", "")

    header = f"\U0001f3c7 {venue}"
    if race_num:
        header += f"{race_num}R"
    if race_name:
        header += f" {race_name}"

    tweet = f"{header}\n\n"
    marks = ["\u25ce", "\u25cb", "\u25b2"]
    for i, h in enumerate(top3_picks[:3]):
        hn = h.get("horse_number", "?")
        name = h.get("horse_name", "???")
        prob = h.get("prob_win", 0) * 100
        tweet += f"{marks[i]} {hn} {name} "
        tweet += f"(\u52dd\u7387{prob:.1f}%)\n"

    tweet += f"\n#\u7af6\u99acAI #\u7af6\u99ac\u4e88\u60f3"
    if venue:
        tweet += f" #{venue}"

    return tweet[:280]


def format_result_tweet(race_name: str, predicted_top3: str,
                        actual_result: str, hit: bool) -> str:
    """
    結果をツイート用にフォーマットする。

    Args:
        race_name: レース名
        predicted_top3: "5ダービーカップ, 3サンライズ, ..."
        actual_result: "5番, 3番, 8番"
        hit: 的中したか

    Returns:
        280文字以内のツイート文字列
    """
    emoji = "\U0001f3af" if hit else "\u274c"
    tweet = f"{emoji} {race_name} \u7d50\u679c\n\n"
    tweet += f"AI\u4e88\u6e2c: {predicted_top3}\n"
    tweet += f"\u5b9f\u969b: {actual_result}\n"
    if hit:
        tweet += f"\n\u7684\u4e2d\uff01 \U0001f389\n"
    tweet += f"\n#\u7af6\u99acAI\u4e88\u6e2c #\u7684\u4e2d\u5831\u544a"
    return tweet[:280]


def format_daily_summary_tweet(date_str: str, total_races: int,
                                hits: int, roi: float) -> str:
    """
    日次サマリーツイートを生成する。

    Args:
        date_str: "2026-03-22"
        total_races: 予測対象レース数
        hits: 的中数
        roi: ROI(%)

    Returns:
        280文字以内のツイート文字列
    """
    hit_rate = (hits / total_races * 100) if total_races > 0 else 0
    emoji = "\U0001f389" if roi > 0 else "\U0001f4ca"

    tweet = f"{emoji} {date_str} AI\u4e88\u6e2c\u30b5\u30de\u30ea\u30fc\n\n"
    tweet += f"\u4e88\u6e2c\u30ec\u30fc\u30b9\u6570: {total_races}\n"
    tweet += f"\u7684\u4e2d: {hits}/{total_races} ({hit_rate:.0f}%)\n"
    tweet += f"ROI: {roi:+.1f}%\n"
    tweet += f"\n#\u7af6\u99acAI #AI\u4e88\u6e2c\u7d50\u679c"
    return tweet[:280]


# ============================================================
# 自動投稿ロジック
# ============================================================

def _load_prediction_cache(date_str: str) -> list:
    """指定日の予測キャッシュを読み込む"""
    results = []
    cache_dir = PREDICTION_CACHE_DIR
    if not os.path.isdir(cache_dir):
        return results

    prefix = date_str.replace("-", "")
    for fname in os.listdir(cache_dir):
        if fname.startswith(prefix) and fname.endswith(".json"):
            fpath = os.path.join(cache_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    results.append(data)
            except Exception:
                continue
    return results


def auto_post_daily_picks(date_str: str = None) -> list:
    """
    当日のベストピックをツイートする。

    Args:
        date_str: "YYYYMMDD" 形式。省略時は今日。

    Returns:
        投稿結果のリスト
    """
    if date_str is None:
        date_str = date.today().strftime("%Y%m%d")

    predictions = _load_prediction_cache(date_str)
    if not predictions:
        logger.info(f"予測キャッシュが見つかりません: {date_str}")
        return []

    config = load_twitter_config()
    if not config.get("auto_post_predictions", True):
        logger.info("予測自動投稿が無効です")
        return []

    results = []
    for pred in predictions:
        horses = pred.get("horses", pred.get("predictions", []))
        if not horses:
            continue

        # top3をソート
        sorted_horses = sorted(horses,
                               key=lambda h: h.get("predicted_rank", h.get("composite_score", 0)),
                               reverse=False)
        top3 = sorted_horses[:3]

        race_info = {
            "venue": pred.get("venue", ""),
            "race_name": pred.get("race_name", ""),
            "race_number": pred.get("race_number", ""),
        }

        tweet_text = format_prediction_tweet(race_info, top3)
        result = post_to_twitter(tweet_text)
        results.append({
            "race": race_info.get("race_name", ""),
            "tweet": tweet_text,
            "result": result,
        })

    logger.info(f"予測ツイート投稿: {len(results)}件")
    return results


def auto_post_results(date_str: str = None) -> list:
    """
    結果収集後に的中/不的中をツイートする。
    土日18:30のスケジュールで呼ばれることを想定。

    Args:
        date_str: "YYYY-MM-DD" 形式。省略時は今日。

    Returns:
        投稿結果のリスト
    """
    if date_str is None:
        date_str = date.today().isoformat()

    config = load_twitter_config()
    if not config.get("auto_post_results", True):
        logger.info("結果自動投稿が無効です")
        return []

    # 結果ファイルを読み込み
    results_file = os.path.join(DATA_DIR, f"results_{date_str}.json")
    if not os.path.exists(results_file):
        logger.info(f"結果ファイルが見つかりません: {results_file}")
        return []

    with open(results_file, "r", encoding="utf-8") as f:
        actual_results = json.load(f)

    # 予測キャッシュを読み込み
    cache_date = date_str.replace("-", "")
    predictions = _load_prediction_cache(cache_date)
    if not predictions:
        logger.info("予測キャッシュなし")
        return []

    posted = []
    total_races = 0
    hits = 0

    for pred in predictions:
        race_id = pred.get("race_id", "")
        if race_id not in actual_results:
            continue

        total_races += 1
        actual = actual_results[race_id]
        finish = actual.get("finish_order", [])
        if not finish:
            continue

        # 実際の上位3頭
        actual_top3 = sorted(finish, key=lambda x: x.get("position", 99))[:3]
        actual_str = ", ".join(
            f"{h.get('horse_number', '?')}{h.get('horse_name', '')}"
            for h in actual_top3
        )

        # AI予測の上位3頭
        horses = pred.get("horses", pred.get("predictions", []))
        sorted_pred = sorted(horses,
                             key=lambda h: h.get("predicted_rank", 99))[:3]
        pred_str = ", ".join(
            f"{h.get('horse_number', '?')}{h.get('horse_name', '')}"
            for h in sorted_pred
        )

        # 的中判定: AI本命が3着以内
        ai_top_number = sorted_pred[0].get("horse_number") if sorted_pred else None
        actual_top3_numbers = [h.get("horse_number") for h in actual_top3]
        hit = ai_top_number in actual_top3_numbers if ai_top_number else False

        if hit:
            hits += 1

        race_name = pred.get("race_name", actual.get("race_name", f"レース{race_id}"))
        tweet_text = format_result_tweet(race_name, pred_str, actual_str, hit)
        result = post_to_twitter(tweet_text)
        posted.append({
            "race": race_name,
            "hit": hit,
            "tweet": tweet_text,
            "result": result,
        })

    # 日次サマリーも投稿
    if total_races > 0:
        summary_tweet = format_daily_summary_tweet(
            date_str, total_races, hits, 0.0  # ROI計算は別途
        )
        summary_result = post_to_twitter(summary_tweet)
        posted.append({
            "race": "daily_summary",
            "tweet": summary_tweet,
            "result": summary_result,
        })

    logger.info(f"結果ツイート投稿: {len(posted)}件 (的中: {hits}/{total_races})")
    return posted


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Twitter/X Auto-Poster")
    parser.add_argument("--post-picks", action="store_true", help="当日のベストピックを投稿")
    parser.add_argument("--post-results", action="store_true", help="結果を投稿")
    parser.add_argument("--date", type=str, default=None, help="対象日 (YYYYMMDD)")
    parser.add_argument("--test", action="store_true", help="テストツイートを投稿")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.test:
        result = post_to_twitter("\U0001f3c7 \u7af6\u99acAI\u4e88\u6e2c\u30c6\u30b9\u30c8\u6295\u7a3f\u3067\u3059\u3002\n\n#\u7af6\u99acAI #\u30c6\u30b9\u30c8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.post_picks:
        results = auto_post_daily_picks(args.date)
        print(f"投稿結果: {len(results)}件")
    elif args.post_results:
        results = auto_post_results(args.date)
        print(f"投稿結果: {len(results)}件")
    else:
        parser.print_help()
