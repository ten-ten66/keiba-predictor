#!/usr/bin/env python3
"""
競馬予想AI - 大規模スクレイパー (20年以上対応)
================================================
netkeiba.comから2001年〜2025年の全レースデータを取得する。

特徴:
  - 中断再開: 進捗をJSONに保存。途中で止めても続きから再開
  - エラー耐性: 自動リトライ・指数バックオフ・エラー上限管理
  - 進捗通知: Discord Webhook / コンソール / ログファイル
  - レート制限: 1リクエスト2秒（+ランダム0〜1秒）
  - メモリ効率: 年ごとにCSV保存。メモリに全件保持しない
  - デーモン対応: nohup / screen / systemd で長期稼働可能

規模想定:
  25年 × 約3,400レース = 約85,000レース / 約1,100,000レコード
  リクエスト数: 約100,000〜150,000（カレンダー + レース結果）
  所要時間: 約3〜7日（ネットワーク状況による）

使い方:
  # 全期間取得（中断再開対応）
  python scraper_v2.py

  # 年を指定
  python scraper_v2.py --start 2001 --end 2025

  # ドライラン（実際にはアクセスしない）
  python scraper_v2.py --dry-run

  # バックグラウンド実行
  nohup python scraper_v2.py > /dev/null 2>&1 &

  # screenで実行（推奨）
  screen -S keiba
  python scraper_v2.py
  # Ctrl+A, D でデタッチ
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re
import os
import sys
import json
import csv
import logging
import traceback
import signal
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import argparse
import random

# ============================================================
# 設定
# ============================================================

CONFIG = {
    # ネットワーク
    "base_delay": 2.0,          # 基本待機秒
    "random_delay": 1.0,        # ランダム追加秒（0〜この値）
    "timeout": 30,              # HTTPタイムアウト
    "max_retries": 3,           # 1リクエストの最大リトライ
    "backoff_base": 5,          # リトライ時の基本待機
    "backoff_max": 120,         # バックオフ最大待機
    "error_pause": 60,          # 連続エラー時の長めの待機
    "error_threshold": 20,      # この回数連続エラーで長期待機
    "long_pause": 300,          # 長期待機（秒）

    # データ
    "data_dir": "data",
    "progress_file": "data/scrape_progress.json",
    "log_file": "data/scraper.log",
    "save_interval": 30,        # N レースごとにCSV保存
    "stats_interval": 100,      # N レースごとに統計表示

    # 対象
    "start_year": 2001,
    "end_year": 2025,

    # 通知
    "discord_webhook": os.environ.get("KEIBA_DISCORD_WEBHOOK", ""),
    "notify_interval": 500,     # N レースごとにDiscord通知

    # JRA場コード
    "venue_codes": {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
        "05": "東京", "06": "中山", "07": "中京", "08": "京都",
        "09": "阪神", "10": "小倉",
    },
}


# ============================================================
# ロガー設定
# ============================================================

def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("keiba_scraper")
    logger.setLevel(logging.INFO)

    # ファイルハンドラ
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # コンソールハンドラ
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    return logger


# ============================================================
# 進捗管理
# ============================================================

class ProgressTracker:
    """中断再開用の進捗管理"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = {
            "completed_race_ids": [],
            "completed_years": [],
            "completed_dates": [],
            "failed_race_ids": [],
            "stats": {
                "total_requests": 0,
                "total_races": 0,
                "total_records": 0,
                "total_errors": 0,
                "started_at": "",
                "last_updated": "",
            },
        }
        self._completed_set: Set[str] = set()
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    saved = json.load(f)
                    self.data.update(saved)
                self._completed_set = set(self.data["completed_race_ids"])
                logger.info(f"📂 進捗読み込み: {len(self._completed_set)} レース完了済み")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"⚠️ 進捗ファイル破損。バックアップ作成して初期化: {e}")
                backup = self.filepath + ".bak"
                try:
                    os.rename(self.filepath, backup)
                except OSError:
                    pass
                self._completed_set = set()

    def save(self):
        self.data["stats"]["last_updated"] = datetime.now().isoformat()
        self.data["completed_race_ids"] = list(self._completed_set)
        # アトミック書き込み: 一時ファイルに書いてからリネーム（書込中クラッシュ対策）
        tmp_path = self.filepath + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(self.data, f, ensure_ascii=False)
            os.replace(tmp_path, self.filepath)
        except OSError as e:
            logger.warning(f"⚠️ 進捗保存失敗: {e}")
            # tmpファイルが残ったら消す
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def is_done(self, race_id: str) -> bool:
        return race_id in self._completed_set

    def mark_done(self, race_id: str):
        self._completed_set.add(race_id)
        self.data["stats"]["total_races"] += 1

    def mark_date_done(self, date_str: str):
        if date_str not in self.data["completed_dates"]:
            self.data["completed_dates"].append(date_str)

    def is_date_done(self, date_str: str) -> bool:
        return date_str in self.data["completed_dates"]

    def mark_year_done(self, year: int):
        if year not in self.data["completed_years"]:
            self.data["completed_years"].append(year)

    def is_year_done(self, year: int) -> bool:
        return year in self.data["completed_years"]

    def mark_failed(self, race_id: str):
        if race_id not in self.data["failed_race_ids"]:
            self.data["failed_race_ids"].append(race_id)
        self.data["stats"]["total_errors"] += 1

    def add_records(self, n: int):
        self.data["stats"]["total_records"] += n

    def add_request(self):
        self.data["stats"]["total_requests"] += 1

    @property
    def stats(self):
        return self.data["stats"]


# ============================================================
# Discord通知
# ============================================================

def notify_discord(message: str, webhook_url: str = None):
    url = webhook_url or CONFIG["discord_webhook"]
    if not url:
        return
    try:
        requests.post(url, json={
            "content": message,
            "username": "🏇 KEIBA Scraper",
        }, timeout=10)
    except Exception:
        pass


# ============================================================
# HTTPクライアント
# ============================================================

class HttpClient:
    """レート制限・リトライ付きHTTPクライアント"""

    def __init__(self):
        self._create_session()
        self._consecutive_errors = 0
        self._request_count = 0

    def _create_session(self):
        """セッション作成/リセット"""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })

    def get(self, url: str, encoding: str = "euc-jp") -> Optional[BeautifulSoup]:
        """レート制限 + リトライ付きGETリクエスト"""
        # 1000リクエストごとにセッションリセット（クッキー蓄積対策）
        self._request_count += 1
        if self._request_count % 1000 == 0:
            self._create_session()
            logger.info("  🔄 セッションリセット")

        progress.add_request()

        for attempt in range(CONFIG["max_retries"]):
            # レート制限（リトライ時は初回のdelayをスキップしない）
            delay = CONFIG["base_delay"] + random.uniform(0, CONFIG["random_delay"])
            time.sleep(delay)

            try:
                resp = self.session.get(url, timeout=CONFIG["timeout"])

                # エンコーディング: 指定値→レスポンスヘッダ→utf-8の順で試行
                resp.encoding = encoding
                text = resp.text
                # euc-jpで文字化けする場合はutf-8にフォールバック
                if encoding == "euc-jp" and "\ufffd" in text[:500]:
                    resp.encoding = "utf-8"
                    text = resp.text

                if resp.status_code == 200:
                    self._consecutive_errors = 0
                    soup = BeautifulSoup(text, "html.parser")
                    # 空ページ検出（404だがステータス200を返すページ対策）
                    if not soup.find("table") and not soup.find("div", class_="race_table_01"):
                        # レース結果ページなのにテーブルがない→存在しないレース
                        if "race" in url and "list" not in url and "calendar" not in url:
                            return None
                    return soup

                elif resp.status_code == 404:
                    # 存在しないページ → リトライ不要
                    return None

                elif resp.status_code == 429:
                    # レート制限 → 長めに待機
                    wait = min(CONFIG["backoff_max"], CONFIG["backoff_base"] * (2 ** attempt) + random.uniform(0, 10))
                    logger.warning(f"⚠️ 429 Rate Limited. {wait:.0f}秒待機...")
                    time.sleep(wait)

                elif resp.status_code >= 500:
                    # サーバーエラー → リトライ
                    wait = CONFIG["backoff_base"] * (2 ** attempt)
                    logger.warning(f"⚠️ {resp.status_code} Server Error. {wait:.0f}秒後にリトライ ({attempt+1}/{CONFIG['max_retries']})")
                    time.sleep(wait)

                else:
                    logger.warning(f"⚠️ HTTP {resp.status_code}: {url}")
                    return None

            except requests.Timeout:
                wait = CONFIG["backoff_base"] * (2 ** attempt)
                logger.warning(f"⏱️ タイムアウト. {wait:.0f}秒後にリトライ ({attempt+1}/{CONFIG['max_retries']})")
                time.sleep(wait)

            except requests.ConnectionError:
                wait = CONFIG["backoff_base"] * (2 ** (attempt + 1))
                logger.warning(f"🔌 接続エラー. {wait:.0f}秒後にリトライ ({attempt+1}/{CONFIG['max_retries']})")
                time.sleep(wait)

            except Exception as e:
                logger.error(f"❌ 予期せぬエラー: {e}")
                time.sleep(CONFIG["backoff_base"])

        # 全リトライ失敗
        self._consecutive_errors += 1
        if self._consecutive_errors >= CONFIG["error_threshold"]:
            logger.warning(f"🚨 連続{self._consecutive_errors}回エラー. {CONFIG['long_pause']}秒待機...")
            notify_discord(f"🚨 連続{self._consecutive_errors}回エラー発生。{CONFIG['long_pause']}秒待機中。")
            time.sleep(CONFIG["long_pause"])
            self._consecutive_errors = 0

        return None


# ============================================================
# レース結果パーサー
# ============================================================

def parse_race_result(soup: BeautifulSoup, race_id: str) -> Optional[Dict]:
    """レース結果ページを解析"""
    try:
        # レース名（複数のh1がある場合、中身のあるものを使う）
        race_name = ""
        for h1 in soup.find_all("h1"):
            text = h1.get_text(strip=True)
            if text:
                race_name = text
                break
        if not race_name:
            return None

        # コース情報（div.mainrace_data or div.racedata）
        race_data = (
            soup.find("div", class_="mainrace_data")
            or soup.find("div", class_="racedata")
            or soup.find("div", class_="data_intro")
        )
        course_info = ""
        if race_data:
            course_info = race_data.get_text(strip=True)
        else:
            # ページ全体からコース情報を探す
            full_text = soup.get_text()
            m = re.search(r"(芝|ダ(?:ート)?|障)\s*(\d{3,4})m", full_text)
            if m:
                course_info = m.group(0)

        course_type = "芝" if "芝" in course_info else ("ダート" if "ダ" in course_info else "障害")
        dist_match = re.search(r"(\d{3,4})m", course_info)
        distance = int(dist_match.group(1)) if dist_match else 0

        # 天候・馬場（race_data or ページ全体から）
        weather = "不明"
        track_condition = "不明"
        search_text = race_data.get_text() if race_data else soup.get_text()
        wm = re.search(r"天候\s*[:：]\s*(\S+)", search_text)
        tm = re.search(r"(?:芝|ダート)\s*[:：]\s*(\S+)", search_text)
        if wm:
            weather = wm.group(1).strip("/").strip()
        if tm:
            track_condition = tm.group(1).strip("/").strip()

        # 日付（ページ全体から YYYY/M/D or YYYY年M月D日 を探す）
        race_date = ""
        full_text = soup.get_text()
        dm = re.search(r"(\d{4})[/年](\d{1,2})[/月](\d{1,2})", full_text)
        if dm:
            race_date = f"{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}"

        # 場名
        venue_code = race_id[4:6]
        venue = CONFIG["venue_codes"].get(venue_code, "不明")

        race_info = {
            "race_id": race_id,
            "race_name": race_name,
            "race_date": race_date,
            "venue": venue,
            "course_type": course_type,
            "distance": distance,
            "weather": weather,
            "track_condition": track_condition,
        }

        # 出走馬テーブル
        table = soup.find("table", class_="race_table_01")
        if not table:
            return None

        horses = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) < 13:
                continue

            horse = parse_horse_row(cols, race_info)
            if horse:
                horses.append(horse)

        if not horses:
            return None

        return {"race_info": race_info, "horses": horses}

    except Exception as e:
        logger.debug(f"パースエラー ({race_id}): {e}")
        return None


def parse_horse_row(cols, race_info: Dict) -> Optional[Dict]:
    """出走馬1行を解析"""
    try:
        # 着順
        pos_text = cols[0].get_text(strip=True)
        try:
            finish_position = int(pos_text)
        except ValueError:
            finish_position = 0

        frame_number = safe_int(cols[1].get_text(strip=True))
        horse_number = safe_int(cols[2].get_text(strip=True))

        # 馬名・ID
        horse_link = cols[3].find("a")
        horse_name = horse_link.get_text(strip=True) if horse_link else cols[3].get_text(strip=True)
        horse_id = ""
        if horse_link and horse_link.get("href"):
            m = re.search(r"/horse/(\w+)/", horse_link["href"])
            if m:
                horse_id = m.group(1)

        # 性齢
        sex_age = cols[4].get_text(strip=True)
        sex = sex_age[0] if sex_age else ""
        age = safe_int(sex_age[1:]) if len(sex_age) > 1 else 0

        # 斤量
        weight = safe_float(cols[5].get_text(strip=True))

        # 騎手
        jockey_link = cols[6].find("a")
        jockey_name = jockey_link.get_text(strip=True) if jockey_link else cols[6].get_text(strip=True)
        jockey_id = ""
        if jockey_link and jockey_link.get("href"):
            # href例: /jockey/result/recent/01032 or /jockey/01032/
            m = re.search(r"/jockey/(?:result/recent/)?(\d+)", jockey_link["href"])
            if m:
                jockey_id = m.group(1)

        # タイム
        finish_time = cols[7].get_text(strip=True) if len(cols) > 7 else ""

        # 通過順
        passing_order = cols[10].get_text(strip=True) if len(cols) > 10 else ""

        # 上り3F
        last_3f = safe_float(cols[11].get_text(strip=True)) if len(cols) > 11 else 0

        # オッズ
        odds = safe_float(cols[12].get_text(strip=True)) if len(cols) > 12 else 0

        # 人気
        popularity = safe_int(cols[13].get_text(strip=True)) if len(cols) > 13 else 0

        # 馬体重
        weight_text = cols[14].get_text(strip=True) if len(cols) > 14 else ""
        horse_weight = 0
        weight_diff = 0
        wm = re.match(r"(\d+)\(([+-]?\d+)\)", weight_text)
        if wm:
            horse_weight = int(wm.group(1))
            weight_diff = int(wm.group(2))

        # 調教師
        trainer_name = ""
        trainer_id = ""
        if len(cols) > 18:
            trainer_link = cols[18].find("a")
            if trainer_link:
                trainer_name = trainer_link.get_text(strip=True)
                if trainer_link.get("href"):
                    # href例: /trainer/result/recent/01234 or /trainer/01234/
                    m = re.search(r"/trainer/(?:result/recent/)?(\d+)", trainer_link["href"])
                    if m:
                        trainer_id = m.group(1)

        # 賞金
        prize = 0
        if len(cols) > 20:
            prize_text = cols[20].get_text(strip=True).replace(",", "").replace("円", "")
            prize = safe_float(prize_text)

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
            "prize_money": prize,
            "passing_order": passing_order,
            "last_3f": last_3f,
        }

    except Exception:
        return None


def safe_int(s: str) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


def safe_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


# ============================================================
# レースID取得（カレンダー方式）
# ============================================================

def get_race_dates(client: HttpClient, year: int, month: int) -> List[str]:
    """開催日一覧を取得（db.netkeiba.com ベース。2001年〜全対応）"""
    # db.netkeiba.com/race/list/YYYYMM/ → 日付ごとのレースIDリンクが取れる
    ym = f"{year}{month:02d}"
    url = f"https://db.netkeiba.com/race/list/{ym}/"
    soup = client.get(url, encoding="euc-jp")
    if not soup:
        return []

    dates = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        # /race/YYYYVVKKDDNN/ → 先頭4桁が年、日付は別途
        m = re.search(r"/race/(\d{12})/", href)
        if m:
            race_id = m.group(1)
            # レースIDから直接日付は取れないので、ページ内の日付リンクを探す
        # /race/list/YYYYMMDD/ パターン
        m2 = re.search(r"/race/list/(\d{8})/", href)
        if m2:
            dates.add(m2.group(1))

    # 日付リンクが見つからない場合、レースIDからdb.netkeiba.comの日付別ページを探す
    if not dates:
        # YYYYMM ページにレースIDが直接ある場合、日付ごとにグループ化されてる
        # テーブルやdivから日付テキストを探す
        for text_node in soup.find_all(string=re.compile(r"\d{4}/\d{1,2}/\d{1,2}")):
            dm = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", text_node)
            if dm:
                d = f"{dm.group(1)}{int(dm.group(2)):02d}{int(dm.group(3)):02d}"
                if d.startswith(ym):
                    dates.add(d)

    # それでも見つからなければ、1日〜31日を総当たりで試す
    if not dates:
        import calendar
        _, last_day = calendar.monthrange(year, month)
        for day in range(1, last_day + 1):
            dates.add(f"{ym}{day:02d}")

    # 該当月のみフィルタ
    dates = {d for d in dates if d.startswith(ym)}

    return sorted(dates)


def get_race_ids_from_date(client: HttpClient, date_str: str) -> List[str]:
    """指定日のレースID一覧を取得（db.netkeiba.com を使用）"""
    url = f"https://db.netkeiba.com/race/list/{date_str}/"
    soup = client.get(url, encoding="euc-jp")
    if not soup:
        return []

    ids = set()
    for link in soup.find_all("a", href=True):
        m = re.search(r"/race/(\d{12})/", link["href"])
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def get_race_ids_bruteforce(year: int) -> List[str]:
    """
    ブルートフォースでレースIDを生成（カレンダーが取れない古い年用）

    ID体系: YYYYVVKKDDNN
      YYYY=年, VV=場コード, KK=開催回, DD=日, NN=レース番号
    """
    ids = []
    for venue_code in CONFIG["venue_codes"].keys():
        for kai in range(1, 7):
            for day in range(1, 13):
                for race_num in range(1, 13):
                    ids.append(f"{year}{venue_code}{kai:02d}{day:02d}{race_num:02d}")
    return ids


# ============================================================
# メインスクレイパー
# ============================================================

class MassiveScraper:
    """大規模長期稼働スクレイパー"""

    def __init__(self, data_dir: str = "data", dry_run: bool = False):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.client = HttpClient()
        self.dry_run = dry_run
        self._running = True
        self._year_records: List[Dict] = []

        # シグナルハンドラ（Ctrl+C で安全に停止）
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info("\n🛑 停止シグナル受信。現在のデータを保存して終了します...")
        self._running = False

    def run(self, start_year: int, end_year: int):
        """メイン実行"""
        if not progress.stats["started_at"]:
            progress.data["stats"]["started_at"] = datetime.now().isoformat()

        total_start = time.time()
        logger.info("=" * 60)
        logger.info(f"🏇 大規模スクレイピング開始: {start_year}〜{end_year}")
        logger.info(f"   完了済み: {progress.stats['total_races']} レース")
        logger.info("=" * 60)

        notify_discord(
            f"🏇 **スクレイピング開始**\n"
            f"期間: {start_year}〜{end_year}\n"
            f"完了済み: {progress.stats['total_races']} レース"
        )

        for year in range(start_year, end_year + 1):
            if not self._running:
                break

            if progress.is_year_done(year):
                logger.info(f"⏭️ {year}年: スキップ（完了済み）")
                continue

            self._scrape_year(year)

            if self._running:
                progress.mark_year_done(year)
                progress.save()

        elapsed = time.time() - total_start
        hours = elapsed / 3600

        summary = (
            f"🏆 **スクレイピング完了**\n"
            f"レース: {progress.stats['total_races']:,}\n"
            f"レコード: {progress.stats['total_records']:,}\n"
            f"リクエスト: {progress.stats['total_requests']:,}\n"
            f"エラー: {progress.stats['total_errors']:,}\n"
            f"所要時間: {hours:.1f}時間"
        )
        logger.info(summary)
        notify_discord(summary)

    def _scrape_year(self, year: int):
        """1年分を取得"""
        logger.info(f"\n{'='*50}")
        logger.info(f"📅 {year}年 開始")
        logger.info(f"{'='*50}")

        self._year_records = []
        year_race_count = 0
        csv_path = self.data_dir / f"races_{year}.csv"

        # 既存CSVがあれば読み込んで追記モード
        existing_ids = set()
        if csv_path.exists():
            try:
                existing = pd.read_csv(csv_path)
                existing_ids = set(existing["race_id"].unique())
                self._year_records = existing.to_dict("records")
                logger.info(f"   既存CSV: {len(existing_ids)} レース / {len(self._year_records)} レコード")
            except Exception:
                pass

        # カレンダーからレースIDを取得
        for month in range(1, 13):
            if not self._running:
                break

            logger.info(f"\n  📆 {year}年{month}月...")

            # カレンダー方式（db.netkeiba.com ベース。全年対応）
            dates = get_race_dates(self.client, year, month)

            if not dates:
                logger.info(f"    開催日なし。スキップ")
                continue

            for date_str in dates:
                if not self._running:
                    break

                if progress.is_date_done(date_str):
                    continue

                race_ids = get_race_ids_from_date(self.client, date_str)
                logger.info(f"    {date_str}: {len(race_ids)} レース")

                for race_id in race_ids:
                    if not self._running:
                        break
                    if progress.is_done(race_id) or race_id in existing_ids:
                        continue

                    n = self._scrape_single_race(race_id)
                    if n > 0:
                        year_race_count += 1

                    # 定期保存
                    if year_race_count > 0 and year_race_count % CONFIG["save_interval"] == 0:
                        self._save_year_csv(csv_path)

                    # 統計表示
                    if progress.stats["total_races"] % CONFIG["stats_interval"] == 0 and progress.stats["total_races"] > 0:
                        self._print_stats()

                    # Discord通知
                    if progress.stats["total_races"] % CONFIG["notify_interval"] == 0 and progress.stats["total_races"] > 0:
                        self._notify_progress()

                progress.mark_date_done(date_str)

        # 年の最終保存
        self._save_year_csv(csv_path)
        logger.info(f"  ✅ {year}年完了: {year_race_count} レース / {len(self._year_records)} レコード")

    def _scrape_single_race(self, race_id: str) -> int:
        """1レースを取得。レコード数を返す"""
        if self.dry_run:
            logger.info(f"      [DRY] {race_id}")
            return 0

        url = f"https://db.netkeiba.com/race/{race_id}/"
        soup = self.client.get(url)
        if not soup:
            progress.mark_failed(race_id)
            return 0

        result = parse_race_result(soup, race_id)
        if not result or not result["horses"]:
            return 0

        n_horses = len(result["horses"])
        self._year_records.extend(result["horses"])
        progress.mark_done(race_id)
        progress.add_records(n_horses)

        return n_horses

    def _save_year_csv(self, csv_path: Path):
        """年のCSVを保存"""
        if not self._year_records:
            return
        df = pd.DataFrame(self._year_records)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        progress.save()
        logger.info(f"    💾 保存: {csv_path.name} ({len(df)} レコード)")

    def _print_stats(self):
        s = progress.stats
        elapsed = (datetime.now() - datetime.fromisoformat(s["started_at"])).total_seconds() / 3600
        rate = s["total_races"] / max(elapsed, 0.01)
        logger.info(
            f"  📊 進捗: {s['total_races']:,}R / {s['total_records']:,}レコード / "
            f"{s['total_requests']:,}req / エラー{s['total_errors']} / "
            f"{elapsed:.1f}h ({rate:.0f} R/h)"
        )

    def _notify_progress(self):
        s = progress.stats
        elapsed = (datetime.now() - datetime.fromisoformat(s["started_at"])).total_seconds() / 3600
        rate = s["total_races"] / max(elapsed, 0.01)
        notify_discord(
            f"📊 **進捗報告**\n"
            f"レース: {s['total_races']:,}\n"
            f"レコード: {s['total_records']:,}\n"
            f"エラー: {s['total_errors']}\n"
            f"経過: {elapsed:.1f}時間 ({rate:.0f} R/h)"
        )


# ============================================================
# メイン
# ============================================================

logger = setup_logger(CONFIG["log_file"])
progress = ProgressTracker(CONFIG["progress_file"])


def main():
    parser = argparse.ArgumentParser(description="競馬データ大規模スクレイパー")
    parser.add_argument("--start", type=int, default=CONFIG["start_year"], help="開始年")
    parser.add_argument("--end", type=int, default=CONFIG["end_year"], help="終了年")
    parser.add_argument("--data-dir", type=str, default=CONFIG["data_dir"], help="保存先")
    parser.add_argument("--dry-run", action="store_true", help="ドライラン")
    parser.add_argument("--delay", type=float, default=None, help="リクエスト間隔(秒)")
    args = parser.parse_args()

    if args.delay is not None:
        CONFIG["base_delay"] = args.delay

    scraper = MassiveScraper(data_dir=args.data_dir, dry_run=args.dry_run)
    scraper.run(args.start, args.end)


if __name__ == "__main__":
    main()
