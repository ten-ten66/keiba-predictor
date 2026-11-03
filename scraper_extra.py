"""
競馬予想AI - 追加データスクレイパー
====================================
既存レースデータに対して追加情報をスクレイピング。

取得対象:
  1. 血統データ（父・母父）         : 馬の個別ページ
  2. 調教タイム                     : 調教ページ
  3. 払い戻し詳細（単勝〜三連単）    : レース結果ページ
  4. 予想印（◎○▲△）               : netkeiba予想ページ
  5. 朝一オッズ                     : オッズページ（レース当日のみ）

使い方:
  # 血統データ取得（既存CSVの全馬）
  python scraper_extra.py --bloodline

  # 払い戻しデータ取得（2024-2025年）
  python scraper_extra.py --payouts --start 2024 --end 2025

  # 予想印取得（直近のレース）
  python scraper_extra.py --marks --race-ids 202506050811,202506050812

  # 全部（血統 + 払戻）
  python scraper_extra.py --all --start 2024 --end 2025
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
import argparse
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class ExtraScraper:
    """追加データスクレイパー"""

    BASE_URL = "https://db.netkeiba.com"
    RACE_URL = "https://race.netkeiba.com"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    }

    def __init__(self, data_dir: str = "data", delay: float = 2.0):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._count = 0

    def _get(self, url: str, encoding: str = "euc-jp") -> Optional[BeautifulSoup]:
        """レート制限付きGET"""
        time.sleep(self.delay + random.uniform(0, 0.5))
        self._count += 1
        try:
            resp = self.session.get(url, timeout=30)
            resp.encoding = encoding
            if resp.status_code != 200:
                return None
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            logger.warning(f"リクエストエラー: {e}")
            return None

    # ================================================================
    # 1. 血統データ（父・母父）
    # ================================================================

    def scrape_bloodline(self, horse_ids: List[str], save_path: str = None) -> pd.DataFrame:
        """馬の血統情報を取得（父・母父・母）"""
        save_path = save_path or str(self.data_dir / "bloodline.csv")

        # 既存データ読み込み
        existing = {}
        if os.path.exists(save_path):
            df_existing = pd.read_csv(save_path)
            existing = dict(zip(df_existing["horse_id"], df_existing.to_dict("records")))
            logger.info(f"既存血統データ: {len(existing)} 頭")

        todo_ids = [h for h in horse_ids if h not in existing and h]
        logger.info(f"血統取得対象: {len(todo_ids)} 頭 (既存: {len(existing)})")

        records = list(existing.values())
        count = 0

        for horse_id in todo_ids:
            data = self._scrape_horse_page(horse_id)
            if data:
                records.append(data)
                count += 1

            if count % 50 == 0 and count > 0:
                logger.info(f"  血統進捗: {count}/{len(todo_ids)}")
                pd.DataFrame(records).to_csv(save_path, index=False)

            if count % 500 == 0 and count > 0:
                # セッションリフレッシュ
                self.session = requests.Session()
                self.session.headers.update(self.HEADERS)

        df = pd.DataFrame(records)
        if not df.empty:
            df.to_csv(save_path, index=False)
            logger.info(f"血統データ保存: {save_path} ({len(df)} 頭)")
        return df

    def _scrape_horse_page(self, horse_id: str, retries: int = 2) -> Optional[Dict]:
        """馬の血統ページから血統情報を取得"""
        url = f"{self.BASE_URL}/horse/ped/{horse_id}/"
        for attempt in range(retries + 1):
            soup = self._get(url)
            if soup and soup.find("table", class_="blood_table"):
                break
            if attempt < retries:
                logger.debug(f"リトライ ({horse_id}) attempt={attempt+1}")
                time.sleep(3)
        if not soup:
            return None

        try:
            result = {"horse_id": horse_id}

            # 血統テーブル (class="blood_table") - 5代血統表
            # rowspan=16: 父(1st), 母(2nd)
            # rowspan=8: 父父(1st), 父母(2nd), 母父(3rd), 母母(4th)
            blood_table = soup.find("table", class_="blood_table")
            if blood_table:
                rs16_names = []  # sire, dam
                rs8_names = []   # sire_sire, sire_dam, dam_sire, dam_dam
                for td in blood_table.find_all("td"):
                    rs = td.get("rowspan", "1")
                    a = td.find("a")
                    name = a.get_text(strip=True) if a else ""
                    if rs == "16" and name:
                        rs16_names.append(name)
                    elif rs == "8" and name:
                        rs8_names.append(name)
                result["sire"] = rs16_names[0] if len(rs16_names) > 0 else ""
                result["dam"] = rs16_names[1] if len(rs16_names) > 1 else ""
                result["sire_sire"] = rs8_names[0] if len(rs8_names) > 0 else ""
                result["dam_sire"] = rs8_names[2] if len(rs8_names) > 2 else ""

            # 生年月日・性別・毛色等（プロフィール）
            prof_table = soup.find("table", class_="db_prof_table")
            if prof_table:
                for row in prof_table.find_all("tr"):
                    th = row.find("th")
                    td = row.find("td")
                    if not th or not td:
                        continue
                    label = th.get_text(strip=True)
                    value = td.get_text(strip=True)

                    if "生年月日" in label:
                        result["birth_date"] = value
                    elif "調教師" in label:
                        result["trainer_name_full"] = value
                    elif "馬主" in label:
                        result["owner"] = value
                    elif "生産者" in label:
                        result["breeder"] = value

            return result if "sire" in result or "dam_sire" in result else None

        except Exception as e:
            logger.debug(f"血統解析エラー ({horse_id}): {e}")
            return None

    # ================================================================
    # 2. 払い戻しデータ
    # ================================================================

    def scrape_payouts(self, race_ids: List[str], save_path: str = None) -> pd.DataFrame:
        """レース結果ページから払い戻し情報を取得"""
        save_path = save_path or str(self.data_dir / "payouts.csv")

        existing = set()
        records = []
        if os.path.exists(save_path):
            df_existing = pd.read_csv(save_path)
            existing = set(df_existing["race_id"].unique())
            records = df_existing.to_dict("records")
            logger.info(f"既存払戻データ: {len(existing)} レース")

        todo = [r for r in race_ids if r not in existing]
        logger.info(f"払戻取得対象: {len(todo)} レース")

        count = 0
        for race_id in todo:
            data = self._scrape_payout_page(race_id)
            if data:
                records.extend(data)
                count += 1

            if count % 50 == 0 and count > 0:
                logger.info(f"  払戻進捗: {count}/{len(todo)}")
                pd.DataFrame(records).to_csv(save_path, index=False)

        df = pd.DataFrame(records)
        if not df.empty:
            df.to_csv(save_path, index=False)
            logger.info(f"払戻データ保存: {save_path} ({len(df)} 件)")
        return df

    def _scrape_payout_page(self, race_id: str) -> Optional[List[Dict]]:
        """レース結果ページの払い戻しテーブルを解析"""
        url = f"{self.BASE_URL}/race/{race_id}/"
        soup = self._get(url)
        if not soup:
            return None

        try:
            results = []

            # 払い戻しテーブル (class="pay_table_01")
            pay_tables = soup.find_all("table", class_="pay_table_01")
            if not pay_tables:
                # 別パターン
                pay_tables = soup.find_all("table", {"summary": re.compile("払戻")})

            for table in pay_tables:
                rows = table.find_all("tr")
                for row in rows:
                    th = row.find("th")
                    tds = row.find_all("td")
                    if not th or len(tds) < 2:
                        continue

                    bet_type = th.get_text(strip=True)
                    # <br>区切りで複数結果（複勝・ワイド等）
                    combos = [s.strip() for s in tds[0].get_text(separator="\n").split("\n") if s.strip()]
                    payouts_raw = [s.strip() for s in tds[1].get_text(separator="\n").split("\n") if s.strip()]
                    pops_raw = [s.strip() for s in tds[2].get_text(separator="\n").split("\n") if s.strip()] if len(tds) > 2 else []

                    for i, combo in enumerate(combos):
                        payout_str = payouts_raw[i] if i < len(payouts_raw) else "0"
                        try:
                            payout = int(payout_str.replace(",", "").replace("円", ""))
                        except ValueError:
                            payout = 0
                        pop = int(pops_raw[i]) if i < len(pops_raw) and pops_raw[i].isdigit() else 0

                        results.append({
                            "race_id": race_id,
                            "bet_type": bet_type,
                            "combination": combo,
                            "payout": payout,
                            "popularity": pop,
                        })

            return results if results else None

        except Exception as e:
            logger.debug(f"払戻解析エラー ({race_id}): {e}")
            return None

    # ================================================================
    # 3. 調教タイム
    # ================================================================

    def scrape_training(self, race_ids: List[str], save_path: str = None) -> pd.DataFrame:
        """調教タイムを取得"""
        save_path = save_path or str(self.data_dir / "training.csv")

        existing = set()
        records = []
        if os.path.exists(save_path):
            df_existing = pd.read_csv(save_path)
            existing = set(df_existing["race_id"].unique())
            records = df_existing.to_dict("records")

        todo = [r for r in race_ids if r not in existing]
        logger.info(f"調教データ取得対象: {len(todo)} レース")

        count = 0
        for race_id in todo:
            data = self._scrape_training_page(race_id)
            if data:
                records.extend(data)
                count += 1

            if count % 50 == 0 and count > 0:
                logger.info(f"  調教進捗: {count}/{len(todo)}")
                pd.DataFrame(records).to_csv(save_path, index=False)

        df = pd.DataFrame(records)
        if not df.empty:
            df.to_csv(save_path, index=False)
            logger.info(f"調教データ保存: {save_path} ({len(df)} 件)")
        return df

    def _scrape_training_page(self, race_id: str) -> Optional[List[Dict]]:
        """調教ページを解析（OikiriTable形式）"""
        url = f"{self.RACE_URL}/race/oikiri.html?race_id={race_id}"
        soup = self._get(url, encoding="euc-jp")
        if not soup:
            return None

        try:
            results = []
            table = soup.find("table", class_="OikiriTable")
            if not table:
                return None

            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if len(cols) < 5:
                    continue

                # cols[0]=枠番, cols[1]=馬番, cols[2]=印, cols[3]=馬名, cols[4]=コメント, cols[5]=評価
                try:
                    horse_number = int(cols[1].get_text(strip=True))
                except (ValueError, IndexError):
                    continue

                horse_name = ""
                name_tag = cols[3].find("a") or cols[3]
                horse_name = name_tag.get_text(strip=True).replace("前走", "")

                comment = cols[4].get_text(strip=True) if len(cols) > 4 else ""
                evaluation = cols[5].get_text(strip=True) if len(cols) > 5 else ""

                # 評価をスコア化 (A=5, B=4, C=3, D=2, E=1)
                eval_score = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}.get(evaluation, 0)

                results.append({
                    "race_id": race_id,
                    "horse_number": horse_number,
                    "horse_name": horse_name,
                    "training_comment": comment,
                    "training_evaluation": evaluation,
                    "training_eval_score": eval_score,
                })

            return results if results else None

        except Exception as e:
            logger.debug(f"調教解析エラー ({race_id}): {e}")
            return None

    # ================================================================
    # 4. 予想印（◎○▲△）
    # ================================================================

    def scrape_marks(self, race_ids: List[str], save_path: str = None) -> pd.DataFrame:
        """netkeiba予想印を取得"""
        save_path = save_path or str(self.data_dir / "marks.csv")

        existing = set()
        records = []
        if os.path.exists(save_path):
            df_existing = pd.read_csv(save_path)
            existing = set(df_existing["race_id"].unique())
            records = df_existing.to_dict("records")

        todo = [r for r in race_ids if r not in existing]
        logger.info(f"予想印取得対象: {len(todo)} レース")

        count = 0
        for race_id in todo:
            data = self._scrape_marks_page(race_id)
            if data:
                records.extend(data)
                count += 1

            if count % 50 == 0 and count > 0:
                logger.info(f"  予想印進捗: {count}/{len(todo)}")
                pd.DataFrame(records).to_csv(save_path, index=False)

        df = pd.DataFrame(records)
        if not df.empty:
            df.to_csv(save_path, index=False)
            logger.info(f"予想印データ保存: {save_path} ({len(df)} 件)")
        return df

    def _scrape_marks_page(self, race_id: str) -> Optional[List[Dict]]:
        """予想印ページを解析"""
        url = f"{self.RACE_URL}/race/yoso.html?race_id={race_id}"
        soup = self._get(url, encoding="utf-8")
        if not soup:
            return None

        try:
            results = []

            # 印テーブル
            tables = soup.find_all("table")
            for table in tables:
                rows = table.find_all("tr")
                if len(rows) < 2:
                    continue

                # ヘッダーから記者名を取得
                header = rows[0]
                ths = header.find_all("th")
                reporters = [th.get_text(strip=True) for th in ths[1:]]  # 0番目は馬番

                for row in rows[1:]:
                    cols = row.find_all("td")
                    if len(cols) < 2:
                        continue

                    horse_number = 0
                    num_text = cols[0].get_text(strip=True)
                    try:
                        horse_number = int(num_text)
                    except ValueError:
                        continue

                    # 各記者の印
                    marks = {}
                    mark_count = {"◎": 0, "○": 0, "▲": 0, "△": 0, "☆": 0}
                    for i, col in enumerate(cols[1:]):
                        mark = col.get_text(strip=True)
                        if mark in mark_count:
                            mark_count[mark] += 1
                        reporter = reporters[i] if i < len(reporters) else f"記者{i+1}"
                        marks[reporter] = mark

                    # 印のスコア化
                    mark_score = (
                        mark_count["◎"] * 10
                        + mark_count["○"] * 8
                        + mark_count["▲"] * 6
                        + mark_count["△"] * 4
                        + mark_count["☆"] * 3
                    )

                    results.append({
                        "race_id": race_id,
                        "horse_number": horse_number,
                        "mark_honmei": mark_count["◎"],
                        "mark_taikou": mark_count["○"],
                        "mark_tanana": mark_count["▲"],
                        "mark_renka": mark_count["△"],
                        "mark_hoshi": mark_count["☆"],
                        "mark_score": mark_score,
                        "mark_total_reporters": len(reporters),
                    })

            return results if results else None

        except Exception as e:
            logger.debug(f"予想印解析エラー ({race_id}): {e}")
            return None

    # ================================================================
    # 5. 朝一オッズ
    # ================================================================

    def scrape_morning_odds(self, race_ids: List[str], save_path: str = None) -> pd.DataFrame:
        """朝一オッズを取得（レース当日朝のみ有効）"""
        save_path = save_path or str(self.data_dir / "morning_odds.csv")

        records = []
        for race_id in race_ids:
            data = self._scrape_odds_page(race_id)
            if data:
                records.extend(data)

        df = pd.DataFrame(records)
        if not df.empty:
            df.to_csv(save_path, index=False)
            logger.info(f"朝一オッズ保存: {save_path} ({len(df)} 件)")
        return df

    def _scrape_odds_page(self, race_id: str) -> Optional[List[Dict]]:
        """単勝オッズページを解析"""
        url = f"{self.RACE_URL}/odds/index.html?race_id={race_id}&type=b1"
        soup = self._get(url, encoding="utf-8")
        if not soup:
            return None

        try:
            results = []
            # オッズテーブル
            table = soup.find("table", id="odds_tan_block") or soup.find("table", class_="race_table")
            if not table:
                return None

            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if len(cols) < 3:
                    continue

                horse_number = 0
                odds = 0.0

                num_text = cols[0].get_text(strip=True)
                try:
                    horse_number = int(num_text)
                except ValueError:
                    continue

                odds_text = cols[-1].get_text(strip=True) if cols else ""
                try:
                    odds = float(odds_text)
                except ValueError:
                    pass

                results.append({
                    "race_id": race_id,
                    "horse_number": horse_number,
                    "morning_odds": odds,
                    "scraped_at": datetime.now().isoformat(),
                })

            return results if results else None

        except Exception as e:
            logger.debug(f"オッズ解析エラー ({race_id}): {e}")
            return None


# ================================================================
# ユーティリティ: 既存CSVからID抽出
# ================================================================

def get_horse_ids_from_csv(data_dir: str = "data", years: range = None) -> List[str]:
    """既存CSVからユニークな馬IDを抽出"""
    # _all_races.csv を優先（個別CSVが壊れている場合のフォールバック）
    all_path = f"{data_dir}/_all_races.csv"
    if os.path.exists(all_path):
        try:
            df = pd.read_csv(all_path, usecols=["horse_id", "race_date"])
            if years is not None:
                df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
                df = df[df["race_date"].dt.year.isin(list(years))]
            ids = df["horse_id"].dropna().unique()
            logger.info(f"_all_races.csv から馬ID {len(ids):,} 件抽出")
            return sorted(ids)
        except Exception as e:
            logger.warning(f"_all_races.csv 読み込み失敗: {e}")

    ids = set()
    if years is None:
        years = range(2015, 2026)
    for year in years:
        path = f"{data_dir}/races_{year}.csv"
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, usecols=["horse_id"])
                ids.update(df["horse_id"].dropna().unique())
            except Exception as e:
                logger.warning(f"{path} 読み込みスキップ: {e}")
    return sorted(ids)


def get_race_ids_from_csv(data_dir: str = "data", years: range = None) -> List[str]:
    """既存CSVからユニークなレースIDを抽出"""
    # _all_races.csv を優先
    all_path = f"{data_dir}/_all_races.csv"
    if os.path.exists(all_path):
        try:
            df = pd.read_csv(all_path, usecols=["race_id", "race_date"])
            if years is not None:
                df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
                df = df[df["race_date"].dt.year.isin(list(years))]
            ids = df["race_id"].dropna().unique()
            logger.info(f"_all_races.csv からレースID {len(ids):,} 件抽出")
            return sorted([str(int(i)) for i in ids])
        except Exception as e:
            logger.warning(f"_all_races.csv 読み込み失敗: {e}")

    ids = set()
    if years is None:
        years = range(2024, 2026)
    for year in years:
        path = f"{data_dir}/races_{year}.csv"
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, usecols=["race_id"])
                ids.update(df["race_id"].dropna().unique())
            except Exception as e:
                logger.warning(f"{path} 読み込みスキップ: {e}")
    return sorted([str(int(i)) for i in ids])


# ================================================================
# メイン
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="追加データスクレイパー")
    parser.add_argument("--bloodline", action="store_true", help="血統データ取得")
    parser.add_argument("--payouts", action="store_true", help="払い戻しデータ取得")
    parser.add_argument("--training", action="store_true", help="調教タイム取得")
    parser.add_argument("--marks", action="store_true", help="予想印取得")
    parser.add_argument("--morning-odds", action="store_true", help="朝一オッズ取得")
    parser.add_argument("--all", action="store_true", help="全データ取得")
    parser.add_argument("--start", type=int, default=2024, help="開始年")
    parser.add_argument("--end", type=int, default=2025, help="終了年")
    parser.add_argument("--race-ids", type=str, help="カンマ区切りのレースID")
    parser.add_argument("--delay", type=float, default=2.0, help="リクエスト間隔(秒)")

    args = parser.parse_args()
    scraper = ExtraScraper(delay=args.delay)
    years = range(args.start, args.end + 1)

    if args.race_ids:
        race_ids = args.race_ids.split(",")
    else:
        race_ids = get_race_ids_from_csv(years=years)

    if args.bloodline or args.all:
        logger.info("=" * 50)
        logger.info("血統データ取得")
        logger.info("=" * 50)
        horse_ids = get_horse_ids_from_csv(years=years)
        scraper.scrape_bloodline(horse_ids)

    if args.payouts or args.all:
        logger.info("=" * 50)
        logger.info("払い戻しデータ取得")
        logger.info("=" * 50)
        scraper.scrape_payouts(race_ids)

    if args.training or args.all:
        logger.info("=" * 50)
        logger.info("調教タイム取得")
        logger.info("=" * 50)
        scraper.scrape_training(race_ids)

    if args.marks or args.all:
        logger.info("=" * 50)
        logger.info("予想印取得")
        logger.info("=" * 50)
        scraper.scrape_marks(race_ids)

    if args.morning_odds:
        logger.info("=" * 50)
        logger.info("朝一オッズ取得")
        logger.info("=" * 50)
        scraper.scrape_morning_odds(race_ids)

    logger.info(f"\n完了！ リクエスト数: {scraper._count}")


if __name__ == "__main__":
    main()
