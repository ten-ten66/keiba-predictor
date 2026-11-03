"""
競馬予想AI - netkeiba.comスクレイパー
======================================
過去レースデータを自動取得してCSV保存する。

使い方:
  # 2020年〜2025年のデータを取得
  python scraper.py --start 2020 --end 2025

  # 特定の年だけ
  python scraper.py --start 2024 --end 2024

  # レース結果ページから直接取得
  python scraper.py --race-id 202406050811

注意:
  - netkeiba.comへの過度なアクセスは避ける（1リクエスト/2秒）
  - 利用規約を確認の上、個人利用の範囲でお願いします
  - robots.txtを尊重してください
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class NetkeibaScaper:
    """netkeiba.comからレースデータを取得"""

    BASE_URL = "https://db.netkeiba.com"
    RACE_URL = "https://race.netkeiba.com"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    }

    # 場コード（JRA 10場）
    VENUE_CODES = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
        "05": "東京", "06": "中山", "07": "中京", "08": "京都",
        "09": "阪神", "10": "小倉",
    }

    def __init__(self, data_dir: str = "data", delay: float = 2.0):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.delay = delay  # リクエスト間隔（秒）
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._request_count = 0
        self._error_count = 0

    def _get(self, url: str, encoding: str = "euc-jp") -> Optional[BeautifulSoup]:
        """HTTPリクエスト（レート制限付き）"""
        time.sleep(self.delay + random.uniform(0, 0.5))
        self._request_count += 1

        try:
            resp = self.session.get(url, timeout=30)
            resp.encoding = encoding
            if resp.status_code != 200:
                logger.warning(f"HTTP {resp.status_code}: {url}")
                self._error_count += 1
                return None
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            logger.error(f"リクエストエラー: {e}")
            self._error_count += 1
            return None

    def get_race_ids_by_year(self, year: int) -> List[str]:
        """指定年のレースID一覧を取得"""
        logger.info(f"📅 {year}年のレースID取得中...")
        race_ids = []

        # 各場の各開催を巡回
        for venue_code in self.VENUE_CODES.keys():
            for kai in range(1, 7):  # 最大6開催
                for day in range(1, 13):  # 最大12日
                    for race_num in range(1, 13):  # 1R〜12R
                        race_id = f"{year}{venue_code}{kai:02d}{day:02d}{race_num:02d}"
                        race_ids.append(race_id)

        logger.info(f"   候補レースID: {len(race_ids)} 件（実在しないIDを含む）")
        return race_ids

    def get_race_list_from_calendar(self, year: int, month: int) -> List[str]:
        """開催カレンダーからレースID一覧を取得（効率的）"""
        url = f"{self.RACE_URL}/top/calendar.html?year={year}&month={month}"
        soup = self._get(url, encoding="utf-8")
        if not soup:
            return []

        race_ids = []
        # カレンダーからレース日のリンクを取得
        for link in soup.find_all("a", href=True):
            href = link["href"]
            # /race/list/YYYYMMDD/ 形式
            match = re.search(r"/race/list/(\d{8})/", href)
            if match:
                date_str = match.group(1)
                # その日のレース一覧を取得
                day_ids = self._get_race_ids_from_date(date_str)
                race_ids.extend(day_ids)

        logger.info(f"   {year}/{month:02d}: {len(race_ids)} レース")
        return race_ids

    def _get_race_ids_from_date(self, date_str: str) -> List[str]:
        """指定日のレースID一覧を取得"""
        url = f"{self.RACE_URL}/race/list/{date_str}/"
        soup = self._get(url, encoding="utf-8")
        if not soup:
            return []

        race_ids = []
        for link in soup.find_all("a", href=True):
            match = re.search(r"/race/(\d{12})/", link["href"])
            if match:
                race_ids.append(match.group(1))

        return list(set(race_ids))

    def scrape_race_result(self, race_id: str) -> Optional[Dict]:
        """1レースの結果を取得"""
        url = f"{self.BASE_URL}/race/{race_id}/"
        soup = self._get(url)
        if not soup:
            return None

        # レース情報ヘッダー
        race_info = self._parse_race_info(soup, race_id)
        if not race_info:
            return None

        # 出走馬テーブル
        table = soup.find("table", class_="race_table_01")
        if not table:
            return None

        horses = []
        rows = table.find_all("tr")[1:]  # ヘッダー行をスキップ

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 15:
                continue

            horse_data = self._parse_horse_row(cols, race_info)
            if horse_data:
                horses.append(horse_data)

        if not horses:
            return None

        return {"race_info": race_info, "horses": horses}

    def _parse_race_info(self, soup: BeautifulSoup, race_id: str) -> Optional[Dict]:
        """レース情報を解析"""
        try:
            # レース名
            race_name_tag = soup.find("h1", class_="racedata_title") or soup.find("h1")
            race_name = race_name_tag.get_text(strip=True) if race_name_tag else ""

            # コース情報 (例: "芝右 1600m / 天候 : 晴 / 芝 : 良")
            race_data = soup.find("div", class_="racedata")
            course_info = ""
            if race_data:
                spans = race_data.find_all("span")
                if spans:
                    course_info = spans[0].get_text(strip=True)

            # コースタイプ
            course_type = "芝" if "芝" in course_info else ("ダート" if "ダ" in course_info else "障害")

            # 距離
            dist_match = re.search(r"(\d{3,4})m", course_info)
            distance = int(dist_match.group(1)) if dist_match else 0

            # 天候・馬場
            weather = "不明"
            track_condition = "不明"
            if race_data:
                text = race_data.get_text()
                weather_match = re.search(r"天候\s*:\s*(\S+)", text)
                track_match = re.search(r"(?:芝|ダート)\s*:\s*(\S+)", text)
                if weather_match:
                    weather = weather_match.group(1)
                if track_match:
                    track_condition = track_match.group(1)

            # 日付
            date_tag = soup.find("div", class_="race_otherdata")
            race_date = ""
            if date_tag:
                date_match = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", date_tag.get_text())
                if date_match:
                    race_date = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"

            # 場名
            venue_code = race_id[4:6]
            venue = self.VENUE_CODES.get(venue_code, "不明")

            return {
                "race_id": race_id,
                "race_name": race_name,
                "race_date": race_date,
                "venue": venue,
                "course_type": course_type,
                "distance": distance,
                "weather": weather,
                "track_condition": track_condition,
            }
        except Exception as e:
            logger.error(f"レース情報解析エラー ({race_id}): {e}")
            return None

    def _parse_horse_row(self, cols, race_info: Dict) -> Optional[Dict]:
        """出走馬1行を解析"""
        try:
            # 着順
            finish_pos_text = cols[0].get_text(strip=True)
            try:
                finish_position = int(finish_pos_text)
            except ValueError:
                finish_position = 0  # 中止・失格等

            # 枠番・馬番
            frame_number = int(cols[1].get_text(strip=True)) if cols[1].get_text(strip=True).isdigit() else 0
            horse_number = int(cols[2].get_text(strip=True)) if cols[2].get_text(strip=True).isdigit() else 0

            # 馬名・ID
            horse_link = cols[3].find("a")
            horse_name = horse_link.get_text(strip=True) if horse_link else cols[3].get_text(strip=True)
            horse_id_match = re.search(r"/horse/(\w+)/", horse_link["href"]) if horse_link and horse_link.get("href") else None
            horse_id = horse_id_match.group(1) if horse_id_match else ""

            # 性齢（例: "牡3"）
            sex_age = cols[4].get_text(strip=True)
            sex = sex_age[0] if sex_age else ""
            try:
                age = int(sex_age[1:])
            except (ValueError, IndexError):
                age = 0

            # 斤量
            try:
                weight = float(cols[5].get_text(strip=True))
            except ValueError:
                weight = 0

            # 騎手
            jockey_link = cols[6].find("a")
            jockey_name = jockey_link.get_text(strip=True) if jockey_link else cols[6].get_text(strip=True)
            jockey_id_match = re.search(r"/jockey/(\w+)/", jockey_link["href"]) if jockey_link and jockey_link.get("href") else None
            jockey_id = jockey_id_match.group(1) if jockey_id_match else ""

            # タイム
            finish_time = cols[7].get_text(strip=True) if len(cols) > 7 else ""

            # 着差
            # margin = cols[8].get_text(strip=True) if len(cols) > 8 else ""

            # 通過順
            passing_order = cols[10].get_text(strip=True) if len(cols) > 10 else ""

            # 上り3F
            last_3f_text = cols[11].get_text(strip=True) if len(cols) > 11 else ""
            try:
                last_3f = float(last_3f_text)
            except ValueError:
                last_3f = 0

            # オッズ
            odds_text = cols[12].get_text(strip=True) if len(cols) > 12 else ""
            try:
                odds = float(odds_text)
            except ValueError:
                odds = 0

            # 人気
            pop_text = cols[13].get_text(strip=True) if len(cols) > 13 else ""
            try:
                popularity = int(pop_text)
            except ValueError:
                popularity = 0

            # 馬体重（例: "480(+4)"）
            weight_text = cols[14].get_text(strip=True) if len(cols) > 14 else ""
            horse_weight = 0
            weight_diff = 0
            w_match = re.match(r"(\d+)\(([+-]?\d+)\)", weight_text)
            if w_match:
                horse_weight = int(w_match.group(1))
                weight_diff = int(w_match.group(2))

            # 調教師
            trainer_link = cols[18].find("a") if len(cols) > 18 else None
            trainer_name = trainer_link.get_text(strip=True) if trainer_link else ""
            trainer_id_match = re.search(r"/trainer/(\w+)/", trainer_link["href"]) if trainer_link and trainer_link.get("href") else None
            trainer_id = trainer_id_match.group(1) if trainer_id_match else ""

            # 賞金
            prize_text = cols[20].get_text(strip=True).replace(",", "") if len(cols) > 20 else "0"
            try:
                prize_money = float(prize_text)
            except ValueError:
                prize_money = 0

            return {
                **race_info,
                "horse_id": horse_id,
                "horse_name": horse_name,
                "age": age,
                "sex": sex,
                "weight": weight,
                "horse_weight": horse_weight,
                "weight_diff": weight_diff,
                "jockey_id": jockey_id,
                "jockey_name": jockey_name,
                "trainer_id": trainer_id,
                "trainer_name": trainer_name,
                "frame_number": frame_number,
                "horse_number": horse_number,
                "odds": odds,
                "popularity": popularity,
                "finish_position": finish_position,
                "finish_time": finish_time,
                "prize_money": prize_money,
                "passing_order": passing_order,
                "last_3f": last_3f,
            }
        except Exception as e:
            logger.debug(f"馬データ解析エラー: {e}")
            return None

    def scrape_year(self, year: int, save_interval: int = 50) -> pd.DataFrame:
        """1年分のレースデータを取得"""
        logger.info(f"{'='*60}")
        logger.info(f"🏇 {year}年のレースデータ取得開始")
        logger.info(f"{'='*60}")

        all_records = []
        race_count = 0
        csv_path = self.data_dir / f"races_{year}.csv"

        # 月ごとにカレンダーからレースIDを取得
        for month in range(1, 13):
            logger.info(f"\n📅 {year}年{month}月...")
            race_ids = self.get_race_list_from_calendar(year, month)

            for race_id in race_ids:
                result = self.scrape_race_result(race_id)
                if result and result["horses"]:
                    all_records.extend(result["horses"])
                    race_count += 1

                    if race_count % 10 == 0:
                        logger.info(f"   進捗: {race_count} レース / {len(all_records)} レコード")

                    # 定期保存
                    if race_count % save_interval == 0:
                        df = pd.DataFrame(all_records)
                        df.to_csv(csv_path, index=False, encoding="utf-8")
                        logger.info(f"   💾 中間保存: {csv_path} ({len(df)} レコード)")

                # エラーが多い場合は待機
                if self._error_count > 10:
                    logger.warning("⚠️ エラー多発。30秒待機...")
                    time.sleep(30)
                    self._error_count = 0

        # 最終保存
        df = pd.DataFrame(all_records)
        if not df.empty:
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"\n✅ {year}年完了: {race_count} レース / {len(df)} レコード")
            logger.info(f"   💾 保存: {csv_path}")
        else:
            logger.warning(f"⚠️ {year}年: データ取得なし")

        return df

    def scrape_range(self, start_year: int, end_year: int) -> pd.DataFrame:
        """複数年のデータを取得"""
        all_dfs = []
        for year in range(start_year, end_year + 1):
            df = self.scrape_year(year)
            if not df.empty:
                all_dfs.append(df)

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined_path = self.data_dir / f"races_{start_year}_{end_year}.csv"
            combined.to_csv(combined_path, index=False, encoding="utf-8")
            logger.info(f"\n🏆 全期間完了: {len(combined)} レコード → {combined_path}")
            return combined

        return pd.DataFrame()

    def get_stats(self) -> Dict:
        """取得統計"""
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": f"{self._error_count / max(self._request_count, 1) * 100:.1f}%",
        }


class RaceIdGenerator:
    """netkeiba.comのレースID体系からIDを生成"""

    @staticmethod
    def generate_ids(year: int, venue_codes: List[str] = None) -> List[str]:
        """
        レースID体系: YYYYVVKKDDNN
        YYYY: 年
        VV: 場コード (01-10)
        KK: 開催回 (01-05)
        DD: 日目 (01-12)
        NN: レース番号 (01-12)
        """
        if venue_codes is None:
            venue_codes = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

        ids = []
        for venue in venue_codes:
            for kai in range(1, 6):
                for day in range(1, 9):
                    for race in range(1, 13):
                        ids.append(f"{year}{venue}{kai:02d}{day:02d}{race:02d}")
        return ids


def main():
    parser = argparse.ArgumentParser(description="netkeiba.com スクレイパー")
    parser.add_argument("--start", type=int, default=2023, help="開始年")
    parser.add_argument("--end", type=int, default=2025, help="終了年")
    parser.add_argument("--race-id", type=str, help="特定レースIDを取得")
    parser.add_argument("--data-dir", type=str, default="data", help="保存先ディレクトリ")
    parser.add_argument("--delay", type=float, default=2.0, help="リクエスト間隔（秒）")

    args = parser.parse_args()
    scraper = NetkeibaScaper(data_dir=args.data_dir, delay=args.delay)

    if args.race_id:
        result = scraper.scrape_race_result(args.race_id)
        if result:
            df = pd.DataFrame(result["horses"])
            print(df.to_string())
            csv_path = f"{args.data_dir}/race_{args.race_id}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n💾 保存: {csv_path}")
    else:
        scraper.scrape_range(args.start, args.end)

    stats = scraper.get_stats()
    logger.info(f"\n📊 取得統計: {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
