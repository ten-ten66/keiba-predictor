#!/usr/bin/env python3
"""
競馬予想AI - 日次キャッシュ用スクレイパー
==========================================
金・土・日の 06:00 JST にcronで実行し、当日＋翌日（2日分）の
レース一覧をJSON キャッシュファイルに保存する。

  - 金曜 06:00 → 土曜レース用
  - 土曜 06:00 → 日曜レース用
  - 日曜 06:00 → 月曜（祝日）レース用

APIサーバーは /api/races/upcoming でキャッシュを優先参照するため、
ユーザーへの応答が高速化される。

使い方:
  python daily_scraper.py              # 今後2日分をキャッシュ
  python daily_scraper.py --days 14    # 今後14日分をキャッシュ
  python daily_scraper.py --date 20260320  # 特定日のみキャッシュ

cron設定例 (金・土・日 JST 06:00):
  0 6 * * 5,6,0 /opt/keiba-predictor/venv/bin/python /opt/keiba-predictor/daily_scraper.py >> /opt/keiba-predictor/logs/daily_scraper.log 2>&1
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from shutuba_scraper import ShutubaScraper

# ログ設定
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "daily_scraper.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# キャッシュディレクトリ
CACHE_DIR = PROJECT_ROOT / "data" / "upcoming_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 古いキャッシュの保持日数（これより古いファイルは削除）
CACHE_RETENTION_DAYS = 14


def scrape_and_cache(date_str: str, scraper: ShutubaScraper) -> int:
    """
    指定日のレース一覧をスクレイピングしてJSONキャッシュに保存する。

    Args:
        date_str: YYYYMMDD形式の日付
        scraper: ShutubaScraper インスタンス

    Returns:
        キャッシュしたレース数
    """
    cache_file = CACHE_DIR / f"{date_str}.json"

    logger.info(f"スクレイピング開始: {date_str}")
    races = scraper.get_upcoming_races(date_str)

    if not races:
        logger.info(f"  {date_str}: レースなし（非開催日の可能性）")
        # 非開催日でも空のキャッシュを作成（再スクレイピングを防ぐ）
        cache_data = {
            "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
            "races": [],
            "total_races": 0,
            "scraped_at": datetime.now().isoformat(),
        }
    else:
        cache_data = {
            "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
            "races": races,
            "total_races": len(races),
            "scraped_at": datetime.now().isoformat(),
        }
        logger.info(f"  {date_str}: {len(races)} レース検出")

    # JSON保存
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

    logger.info(f"  キャッシュ保存: {cache_file}")
    return len(races)


def cleanup_old_cache():
    """古いキャッシュファイルを削除する"""
    cutoff = datetime.now() - timedelta(days=CACHE_RETENTION_DAYS)
    cutoff_str = cutoff.strftime("%Y%m%d")
    removed = 0

    for cache_file in CACHE_DIR.glob("*.json"):
        # ファイル名からYYYYMMDDを取得
        stem = cache_file.stem
        if len(stem) == 8 and stem.isdigit() and stem < cutoff_str:
            cache_file.unlink()
            removed += 1

    if removed:
        logger.info(f"古いキャッシュ {removed} 件削除（{CACHE_RETENTION_DAYS}日以前）")


def main():
    parser = argparse.ArgumentParser(description="日次レースキャッシュスクレイパー")
    parser.add_argument(
        "--days", type=int, default=2,
        help="今日から何日先までキャッシュするか（デフォルト: 2）"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="特定日のみキャッシュ (YYYYMMDD)"
    )
    parser.add_argument(
        "--delay", type=float, default=2.5,
        help="リクエスト間隔（秒、デフォルト: 2.5）"
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help="古いキャッシュの削除をスキップ"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("日次レースキャッシュ更新開始")
    logger.info("=" * 60)

    scraper = ShutubaScraper(delay=args.delay)

    if args.date:
        # 特定日のみ
        dates = [args.date]
    else:
        # 今日から N 日分
        today = datetime.now()
        dates = [(today + timedelta(days=i)).strftime("%Y%m%d") for i in range(args.days)]

    total_races = 0
    for date_str in dates:
        try:
            count = scrape_and_cache(date_str, scraper)
            total_races += count
        except Exception as e:
            logger.error(f"  {date_str} のスクレイピングでエラー: {e}")

    # 古いキャッシュの掃除
    if not args.no_cleanup:
        cleanup_old_cache()

    logger.info(f"完了: {len(dates)} 日分処理、合計 {total_races} レースをキャッシュ")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
