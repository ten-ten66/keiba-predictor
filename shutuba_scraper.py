"""
競馬予想AI - 出馬表（shutuba）スクレイパー
============================================
未来レースの出馬表をnetkeiba.comから取得し、
予測モデルに投入可能なDataFrameを生成する。

使い方:
  from shutuba_scraper import ShutubaScraper

  scraper = ShutubaScraper()

  # 特定レースの出馬表を取得
  df = scraper.scrape_shutuba("202406050811")

  # 指定日の全レース一覧を取得
  races = scraper.get_upcoming_races("20240608")
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re
import logging
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ShutubaScraper:
    """netkeiba.comから出馬表（未来レース）のデータを取得"""

    RACE_URL = "https://race.netkeiba.com"
    DB_URL = "https://db.netkeiba.com"
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

    TRACK_CONDITION_MAP = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
    WEATHER_MAP = {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "小雪": 4, "雪": 5}
    COURSE_TYPE_MAP = {"芝": 0, "ダート": 1, "障害": 2}

    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def _get(self, url: str, encoding: str = "utf-8") -> Optional[BeautifulSoup]:
        """HTTPリクエスト（レート制限付き）"""
        time.sleep(self.delay + random.uniform(0, 0.5))
        try:
            resp = self.session.get(url, timeout=30)
            resp.encoding = encoding
            if resp.status_code != 200:
                logger.warning(f"HTTP {resp.status_code}: {url}")
                return None
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            logger.error(f"リクエストエラー: {e}")
            return None

    # ------------------------------------------------------------------
    # Public: 指定日のレース一覧取得
    # ------------------------------------------------------------------

    def get_upcoming_races(self, date_str: str) -> List[Dict]:
        """
        指定日（YYYYMMDD）のレース一覧を取得する。

        Returns:
            [
                {
                    "race_id": "202406050811",
                    "race_name": "...",
                    "venue": "東京",
                    "race_number": 11,
                    "start_time": "15:40",
                    "course_info": "芝2400m",
                },
                ...
            ]
        """
        # race_list.htmlはJS動的読み込みのため、race_list_sub.htmlを使う
        url = f"{self.RACE_URL}/top/race_list_sub.html?kaisai_date={date_str}"
        soup = self._get(url, encoding="utf-8")
        if not soup:
            logger.warning(f"レース一覧取得失敗: {date_str}")
            return []

        races = []

        # race_list_sub.htmlからレースリンクを探す
        for link in soup.find_all("a", href=True):
            href = link["href"]
            match = re.search(r"race_id=(\d{12})", href)
            if not match:
                continue

            race_id = match.group(1)
            # Avoid duplicates
            if any(r["race_id"] == race_id for r in races):
                continue

            race_number = int(race_id[-2:]) if race_id[-2:].isdigit() else 0
            venue_code = race_id[4:6]
            venue = self.VENUE_CODES.get(venue_code, "不明")

            # Extract race name from link text
            race_name = link.get_text(strip=True) or ""

            # Try to find start time from parent
            start_time = ""
            parent = link.find_parent("li") or link.find_parent("div")
            if parent:
                time_match = re.search(r"(\d{1,2}:\d{2})", parent.get_text())
                if time_match:
                    start_time = time_match.group(1)

            # Try to find course info (e.g. 芝2400m)
            course_info = ""
            if parent:
                course_match = re.search(r"(芝|ダート|障害)\D*(\d{3,5})\s*m", parent.get_text())
                if course_match:
                    course_info = f"{course_match.group(1)}{course_match.group(2)}m"

            # Extract grade/class from race name and surrounding text
            grade = self._parse_grade(race_name, parent)

            races.append({
                "race_id": race_id,
                "race_name": race_name,
                "venue": venue,
                "race_number": race_number,
                "start_time": start_time,
                "course_info": course_info,
                "grade": grade,
                "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
            })

        # Deduplicate and sort
        seen = set()
        unique_races = []
        for r in races:
            if r["race_id"] not in seen:
                seen.add(r["race_id"])
                unique_races.append(r)

        unique_races.sort(key=lambda r: (r["venue"], r["race_number"]))
        logger.info(f"{date_str}: {len(unique_races)} レース検出")
        return unique_races

    # ------------------------------------------------------------------
    # Internal: レースグレード判定
    # ------------------------------------------------------------------

    def _parse_grade(self, race_name: str, parent_el) -> str:
        """
        レース名やHTML要素からグレード/クラスを判定する。

        Returns:
            "G1", "G2", "G3", "L" (Listed), "OP" (Open),
            "3勝クラス", "2勝クラス", "1勝クラス", "未勝利", "新馬",
            or "その他"
        """
        # Combine race name and parent text for broader matching
        text = race_name or ""
        if parent_el:
            text += " " + parent_el.get_text(strip=False)

        # Also check for grade icons/spans (netkeiba uses Icon_GradeType classes)
        grade_from_icon = ""
        if parent_el:
            icon = parent_el.find("span", class_=re.compile(r"Icon_GradeType|Grade"))
            if icon:
                icon_text = icon.get_text(strip=True)
                if icon_text:
                    grade_from_icon = icon_text

        # Priority-based matching
        # G1 (also matches GI, GⅠ, (G1), (GI))
        if re.search(r"(?:G[1IⅠ]|GI(?!I|V)|（G[1IⅠ]）|\(G[1IⅠ]\))", text) or grade_from_icon in ("G1", "GI", "GⅠ"):
            return "G1"

        # G2 (also matches GII, GⅡ)
        if re.search(r"(?:G[2Ⅱ]|GII(?!I)|\(G[2Ⅱ]\)|（G[2Ⅱ]）)", text) or grade_from_icon in ("G2", "GII", "GⅡ"):
            return "G2"

        # G3 (also matches GIII, GⅢ)
        if re.search(r"(?:G[3Ⅲ]|GIII|\(G[3Ⅲ]\)|（G[3Ⅲ]）)", text) or grade_from_icon in ("G3", "GIII", "GⅢ"):
            return "G3"

        # Listed (L)
        if re.search(r"(?:\(L\)|（L）|リステッド|Listed)", text) or grade_from_icon == "L":
            return "L"

        # Open class (OP, オープン)
        if re.search(r"(?:オープン|OP(?:EN)?(?!クラス)|(?:^|\s)OP(?:\s|$))", text, re.IGNORECASE) or grade_from_icon == "OP":
            return "OP"

        # Condition races (条件戦)
        if re.search(r"3勝クラス|1600万下", text):
            return "3勝クラス"
        if re.search(r"2勝クラス|1000万下", text):
            return "2勝クラス"
        if re.search(r"1勝クラス|500万下", text):
            return "1勝クラス"
        if re.search(r"未勝利", text):
            return "未勝利"
        if re.search(r"新馬", text):
            return "新馬"

        return "その他"

    # ------------------------------------------------------------------
    # Public: 出馬表スクレイピング
    # ------------------------------------------------------------------

    def scrape_shutuba(self, race_id: str) -> Optional[pd.DataFrame]:
        """
        出馬表ページをスクレイピングし、予測モデル用のDataFrameを返す。

        Args:
            race_id: 12桁のレースID (例: "202406050811")

        Returns:
            DataFrame with columns matching model input, or None on failure
        """
        url = f"{self.RACE_URL}/race/shutuba.html?race_id={race_id}"
        soup = self._get(url, encoding="euc-jp")
        if not soup:
            logger.error(f"出馬表取得失敗: {race_id}")
            return None

        # -- レース情報ヘッダー --
        race_info = self._parse_shutuba_race_info(soup, race_id)
        if not race_info:
            logger.error(f"レース情報解析失敗: {race_id}")
            return None

        # -- 出走馬テーブル --
        horses = self._parse_shutuba_table(soup, race_info)
        if not horses:
            logger.warning(f"出走馬データなし: {race_id}")
            return None

        df = pd.DataFrame(horses)

        # -- 特徴量の計算 --
        df = self._compute_features_for_prediction(df)

        logger.info(
            f"出馬表取得完了: {race_id} "
            f"({race_info.get('venue', '?')}{race_info.get('race_number', '?')}R "
            f"{race_info.get('course_type', '?')}{race_info.get('distance', '?')}m) "
            f"- {len(df)}頭"
        )
        return df

    # ------------------------------------------------------------------
    # Internal: レース情報パース
    # ------------------------------------------------------------------

    def _parse_shutuba_race_info(self, soup: BeautifulSoup, race_id: str) -> Optional[Dict]:
        """出馬表ページからレース情報を抽出"""
        try:
            # レース名
            race_name = ""
            name_tag = (
                soup.find("h1", class_="RaceName")
                or soup.find("div", class_="RaceName")
                or soup.find("h1", class_="racedata_title")
                or soup.find("h1")
            )
            if name_tag:
                race_name = name_tag.get_text(strip=True)

            # コース情報 (例: "芝右 2400m")
            course_info_text = ""
            course_tag = (
                soup.find("div", class_="RaceData01")
                or soup.find("div", class_="racedata")
            )
            if course_tag:
                course_info_text = course_tag.get_text(strip=True)

            # コースタイプ
            if "芝" in course_info_text and "ダ" not in course_info_text:
                course_type = "芝"
            elif "ダ" in course_info_text:
                course_type = "ダート"
            else:
                course_type = "障害" if "障" in course_info_text else "芝"

            # 距離
            dist_match = re.search(r"(\d{3,4})m", course_info_text)
            distance = int(dist_match.group(1)) if dist_match else 0

            # 天候・馬場状態
            weather = "晴"  # デフォルト（未来レースでは確定前）
            track_condition = "良"  # デフォルト

            race_data2 = soup.find("div", class_="RaceData02") or soup.find("div", class_="race_otherdata")
            full_text = ""
            if race_data2:
                full_text = race_data2.get_text()
            if course_tag:
                full_text += " " + course_tag.get_text()

            weather_match = re.search(r"天候\s*[:：]\s*(\S+)", full_text)
            if weather_match:
                weather = weather_match.group(1)

            track_match = re.search(r"(?:芝|ダート)\s*[:：]\s*(\S+)", full_text)
            if track_match:
                track_condition = track_match.group(1)

            # 日付
            race_date = ""
            # race_id体系: YYYYPPDDRRCC -> 年は先頭4桁
            year = race_id[:4]
            # 日付はページ内テキストから探す
            date_match = re.search(r"(\d{4})[年/](\d{1,2})[月/](\d{1,2})", full_text)
            if date_match:
                race_date = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
            else:
                # race_id からは正確な日付が分からないので今日の日付を入れる
                race_date = datetime.now().strftime("%Y-%m-%d")

            # 場名
            venue_code = race_id[4:6]
            venue = self.VENUE_CODES.get(venue_code, "不明")

            # レース番号
            race_number = int(race_id[-2:]) if race_id[-2:].isdigit() else 0

            return {
                "race_id": race_id,
                "race_name": race_name,
                "race_date": race_date,
                "venue": venue,
                "course_type": course_type,
                "distance": distance,
                "weather": weather,
                "track_condition": track_condition,
                "race_number": race_number,
            }
        except Exception as e:
            logger.error(f"レース情報解析エラー ({race_id}): {e}")
            return None

    # ------------------------------------------------------------------
    # Internal: 出馬表テーブルパース
    # ------------------------------------------------------------------

    def _parse_shutuba_table(self, soup: BeautifulSoup, race_info: Dict) -> List[Dict]:
        """出馬表テーブルから出走馬データを抽出"""
        # shutuba.html の出走馬テーブルを探す
        table = (
            soup.find("table", class_="Shutuba_Table")
            or soup.find("table", class_="RaceTable01")
            or soup.find("table", class_="race_table_01")
            or soup.find("table", id="shutuba_table")
        )

        if not table:
            # フォールバック: テーブルを探して馬名が含まれているものを選ぶ
            tables = soup.find_all("table")
            for t in tables:
                text = t.get_text()
                if "馬名" in text or "枠" in text:
                    table = t
                    break

        if not table:
            logger.warning("出馬表テーブルが見つかりません")
            return []

        horses = []
        rows = table.find_all("tr")

        # ヘッダー行をスキップ（最初の1-2行はヘッダー）
        data_rows = []
        for row in rows:
            tds = row.find_all("td")
            if len(tds) >= 4:  # データ行は少なくとも4列
                data_rows.append(row)

        for row in data_rows:
            horse = self._parse_shutuba_row(row, race_info)
            if horse:
                horses.append(horse)

        return horses

    def _parse_shutuba_row(self, row, race_info: Dict) -> Optional[Dict]:
        """出馬表の1行（1頭分）を解析"""
        try:
            cols = row.find_all("td")
            if len(cols) < 4:
                return None

            # 枠番（1列目）- 確定前は空の場合あり
            frame_text = cols[0].get_text(strip=True)
            frame_number = int(frame_text) if frame_text.isdigit() else 0

            # 馬番（2列目）- 確定前は空の場合あり
            horse_num_text = cols[1].get_text(strip=True)
            horse_number = int(horse_num_text) if horse_num_text.isdigit() else 0

            # 枠番・馬番が未確定の場合、trのidから馬番を推測 (id="tr_15" -> 15)
            if horse_number == 0:
                tr_id = row.get("id", "")
                id_match = re.search(r"tr_(\d+)", tr_id)
                if id_match:
                    horse_number = int(id_match.group(1))

            # 馬名が取れるかで有効行を判断（馬番0でも馬名があればOK）

            # 馬名・馬ID（3列目付近 - リンクから取得）
            horse_name = ""
            horse_id = ""
            for col in cols[2:6]:
                link = col.find("a", href=re.compile(r"/horse/"))
                if link:
                    horse_name = link.get_text(strip=True)
                    id_match = re.search(r"/horse/(\w+)", link["href"])
                    if id_match:
                        horse_id = id_match.group(1)
                    break

            if not horse_name:
                # Fallback: テキストから馬名を推測
                for col in cols[2:6]:
                    text = col.get_text(strip=True)
                    if len(text) >= 2 and not text.isdigit():
                        horse_name = text
                        break

            # 性齢（例: "牡3"）
            sex = ""
            age = 0
            for col in cols:
                text = col.get_text(strip=True)
                sex_age_match = re.match(r"^([牡牝セ])(\d+)$", text)
                if sex_age_match:
                    sex = sex_age_match.group(1)
                    age = int(sex_age_match.group(2))
                    break

            # 斤量
            weight = 0.0
            for col in cols:
                text = col.get_text(strip=True)
                try:
                    val = float(text)
                    if 48 <= val <= 62:  # 斤量の妥当な範囲
                        weight = val
                        break
                except ValueError:
                    pass

            # 騎手名・ID
            jockey_name = ""
            jockey_id = ""
            for col in cols:
                link = col.find("a", href=re.compile(r"/jockey/"))
                if link:
                    jockey_name = link.get_text(strip=True)
                    jid_match = re.search(r"/jockey/(\w+)", link["href"])
                    if jid_match:
                        jockey_id = jid_match.group(1)
                    break

            # 調教師名・ID
            trainer_name = ""
            trainer_id = ""
            for col in cols:
                link = col.find("a", href=re.compile(r"/trainer/"))
                if link:
                    trainer_name = link.get_text(strip=True)
                    tid_match = re.search(r"/trainer/(\w+)", link["href"])
                    if tid_match:
                        trainer_id = tid_match.group(1)
                    break

            # 馬体重（出馬表では前走体重が表示されることがある、当日は空の場合あり）
            horse_weight = 0
            weight_diff = 0
            for col in cols:
                text = col.get_text(strip=True)
                w_match = re.match(r"(\d{3,4})\(([+-]?\d+)\)", text)
                if w_match:
                    horse_weight = int(w_match.group(1))
                    weight_diff = int(w_match.group(2))
                    break
                # 体重のみ（変動なし）
                w_match2 = re.match(r"^(\d{3,4})$", text)
                if w_match2:
                    val = int(w_match2.group(1))
                    if 350 <= val <= 600:  # 馬体重の妥当な範囲
                        horse_weight = val
                        break

            # オッズ（前日オッズまたは当日オッズ）
            odds = 0.0
            for col in cols:
                # オッズは "3.5" のような小数
                odds_tag = col.find("span", class_=re.compile(r"Odds|odds"))
                if odds_tag:
                    try:
                        odds = float(odds_tag.get_text(strip=True))
                    except ValueError:
                        pass
                    break

            if odds == 0:
                # オッズ列(class=Popular)から取得を試みる
                for col in cols:
                    col_classes = col.get("class", [])
                    if any("Popular" in c for c in col_classes) and not any("Ninki" in c for c in col_classes):
                        text = col.get_text(strip=True)
                        if text and text != "---.-" and "**" not in text:
                            try:
                                val = float(text)
                                if 1.0 < val < 9999:
                                    odds = val
                            except ValueError:
                                pass
                        break
                # NOTE: Do NOT use generic float fallback - it picks up weight (57.0) as odds

            # 人気
            popularity = 0
            # class="Popular_Ninki" の列から取得
            for col in cols:
                col_classes = col.get("class", [])
                if any("Ninki" in c for c in col_classes):
                    text = col.get_text(strip=True)
                    if text and "**" not in text:
                        try:
                            popularity = int(text)
                        except ValueError:
                            pass
                    break
            if popularity == 0:
                pop_tag = row.find("span", class_=re.compile(r"Pop|pop|Ninki|ninki"))
                if pop_tag:
                    try:
                        popularity = int(pop_tag.get_text(strip=True))
                    except ValueError:
                        pass

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
                # 未来レースなので結果データは空
                "finish_position": None,
                "finish_time": None,
                "prize_money": 0,
                "passing_order": None,
                "last_3f": None,
            }
        except Exception as e:
            logger.debug(f"出馬表行解析エラー: {e}")
            return None

    # ------------------------------------------------------------------
    # Internal: 予測用特徴量の計算
    # ------------------------------------------------------------------

    def _compute_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        出馬表DataFrameに対して、モデルが必要とする35特徴量を計算する。

        モデルが期待する特徴量:
        - 基本: distance, age, weight, horse_weight, weight_diff, frame_number,
                horse_number, odds, popularity
        - エンコード: track_condition_enc, weather_enc, course_type_enc, sex_enc
        - 時系列: days_since_last_race
        - レース内相対: odds_rank, odds_ratio, weight_vs_avg
        - 斤量分析: weight_change, weight_vs_self_avg, horse_weight_change_rate,
                     big_weight_change
        - 季節: age_x_season
        - レースレベル: race_avg_odds, race_min_odds, odds_vs_race_avg, n_runners
        - Elo: horse_elo, horse_glicko, horse_glicko_rd, jockey_elo, trainer_elo,
               horse_elo_rank, jockey_elo_rank, combined_elo, combined_elo_rank

        出馬表のみでは過去データが不足するため、Elo等の履歴依存特徴量は
        デフォルト値（初期レーティング等）で埋める。
        """
        # --- カテゴリエンコード ---
        df["track_condition_enc"] = df["track_condition"].map(self.TRACK_CONDITION_MAP).fillna(0).astype(int)
        df["weather_enc"] = df["weather"].map(self.WEATHER_MAP).fillna(0).astype(int)
        df["course_type_enc"] = df["course_type"].map(self.COURSE_TYPE_MAP).fillna(0).astype(int)
        df["sex_enc"] = df["sex"].map({"牡": 0, "牝": 1, "セ": 2}).fillna(-1).astype(int)

        # --- 日付処理 ---
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
        if df["race_date"].notna().any():
            month = df["race_date"].dt.month.fillna(1).astype(int)
        else:
            month = pd.Series([datetime.now().month] * len(df))

        # --- 数値型変換（安全に） ---
        for col in ["distance", "weight", "horse_weight", "weight_diff",
                     "frame_number", "horse_number", "odds", "popularity", "age"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # --- 出走頭数 ---
        df["n_runners"] = df.groupby("race_id")["horse_number"].transform("count")

        # --- レース内相対指標（オッズ） ---
        if df["odds"].sum() > 0:
            df["odds_rank"] = df.groupby("race_id")["odds"].rank()
            min_odds = df.groupby("race_id")["odds"].transform("min").clip(lower=0.1)
            df["odds_ratio"] = df["odds"] / min_odds
            df["race_avg_odds"] = df.groupby("race_id")["odds"].transform("mean")
            df["race_min_odds"] = df.groupby("race_id")["odds"].transform("min")
            df["odds_vs_race_avg"] = df["odds"] / df["race_avg_odds"].clip(lower=1)
        else:
            # オッズが全部0の場合（未確定）→ 均等に
            n = len(df)
            df["odds_rank"] = range(1, n + 1)
            df["odds_ratio"] = 1.0
            df["race_avg_odds"] = 10.0
            df["race_min_odds"] = 1.0
            df["odds_vs_race_avg"] = 1.0

        # --- 体重の相対指標 ---
        if df["horse_weight"].sum() > 0:
            df["weight_vs_avg"] = df.groupby("race_id")["horse_weight"].transform(
                lambda x: x - x.mean()
            )
        else:
            df["weight_vs_avg"] = 0.0

        # --- 斤量分析（過去データなし → デフォルト値） ---
        df["weight_change"] = 0.0  # 前走との斤量差（不明）
        df["weight_vs_self_avg"] = 0.0  # 自身平均との差（不明）
        df["horse_weight_change_rate"] = 0.0  # 馬体重変化率（不明）
        df["big_weight_change"] = (df["weight_diff"].abs() >= 10).astype(int)

        # --- 季節×馬齢 ---
        season_enc = (month - 1) // 3  # 0=冬, 1=春, 2=夏, 3=秋
        df["age_x_season"] = df["age"] * season_enc

        # --- 前走からの間隔（出馬表のみでは不明 → 中4週=28日をデフォルト） ---
        df["days_since_last_race"] = 28.0

        # --- Eloレーティング（過去データなし → 初期値） ---
        # 初期レーティング: 1500
        ELO_INITIAL = 1500.0
        GLICKO_INITIAL_RD = 350.0

        df["horse_elo"] = ELO_INITIAL
        df["horse_glicko"] = ELO_INITIAL
        df["horse_glicko_rd"] = GLICKO_INITIAL_RD
        df["jockey_elo"] = ELO_INITIAL
        df["trainer_elo"] = ELO_INITIAL

        # Eloランク（レース内順位 → 全員同値なら全員1位）
        df["horse_elo_rank"] = 1.0
        df["jockey_elo_rank"] = 1.0
        df["combined_elo"] = ELO_INITIAL
        df["combined_elo_rank"] = 1.0

        # --- 人気が0の場合はオッズランクから推定 ---
        if "popularity" in df.columns:
            df["popularity"] = df["popularity"].astype(float)
            mask = df["popularity"] == 0
            if mask.any() and df["odds"].sum() > 0:
                df.loc[mask, "popularity"] = df.loc[mask, "odds_rank"]
            df["popularity"] = df["popularity"].astype(int)

        return df

    # ------------------------------------------------------------------
    # Public: 過去データと結合してElo等を推定する拡張版
    # ------------------------------------------------------------------

    def enrich_with_history(
        self, shutuba_df: pd.DataFrame, history_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        出馬表DataFrameに対して、過去データがあればElo等を計算し直す。
        history_df が None の場合はそのまま返す。

        Args:
            shutuba_df: scrape_shutuba()の戻り値
            history_df: 過去レースの全データ（race_id, horse_id, jockey_id, etc.）

        Returns:
            enriched DataFrame
        """
        if history_df is None or history_df.empty:
            return shutuba_df

        try:
            # 出走馬のIDリスト
            horse_ids = shutuba_df["horse_id"].unique().tolist()
            jockey_ids = shutuba_df["jockey_id"].unique().tolist()
            trainer_ids = shutuba_df["trainer_id"].unique().tolist()

            # 過去データから該当馬/騎手/調教師のレーティングを推定
            # 簡易版: 過去の平均着順からEloを推定
            for _, row in shutuba_df.iterrows():
                idx = row.name
                hid = row["horse_id"]
                jid = row["jockey_id"]
                tid = row["trainer_id"]

                if hid and "horse_id" in history_df.columns:
                    h_hist = history_df[history_df["horse_id"] == hid]
                    if not h_hist.empty and "finish_position" in h_hist.columns:
                        avg_pos = h_hist["finish_position"].mean()
                        # 簡易Elo推定: 平均着順が低いほど高レーティング
                        shutuba_df.at[idx, "horse_elo"] = 1500 + (8 - avg_pos) * 30
                        shutuba_df.at[idx, "horse_glicko"] = 1500 + (8 - avg_pos) * 30
                        # RDを走った回数で下げる
                        n_races = len(h_hist)
                        shutuba_df.at[idx, "horse_glicko_rd"] = max(50, 350 - n_races * 10)
                        # 前走からの間隔
                        if "race_date" in h_hist.columns:
                            last_date = pd.to_datetime(h_hist["race_date"], errors="coerce").max()
                            if pd.notna(last_date) and pd.notna(row["race_date"]):
                                days = (pd.Timestamp(row["race_date"]) - last_date).days
                                if days > 0:
                                    shutuba_df.at[idx, "days_since_last_race"] = float(days)
                        # 斤量変化
                        if "weight" in h_hist.columns:
                            prev_weight = h_hist["weight"].iloc[-1]
                            if prev_weight > 0 and row["weight"] > 0:
                                shutuba_df.at[idx, "weight_change"] = row["weight"] - prev_weight
                        # 馬体重変化率
                        if "horse_weight" in h_hist.columns:
                            prev_hw = h_hist["horse_weight"].iloc[-1]
                            if prev_hw > 0 and row["horse_weight"] > 0:
                                shutuba_df.at[idx, "horse_weight_change_rate"] = (
                                    (row["horse_weight"] - prev_hw) / prev_hw
                                )

                if jid and "jockey_id" in history_df.columns:
                    j_hist = history_df[history_df["jockey_id"] == jid]
                    if not j_hist.empty and "finish_position" in j_hist.columns:
                        avg_pos = j_hist["finish_position"].mean()
                        shutuba_df.at[idx, "jockey_elo"] = 1500 + (8 - avg_pos) * 20

                if tid and "trainer_id" in history_df.columns:
                    t_hist = history_df[history_df["trainer_id"] == tid]
                    if not t_hist.empty and "finish_position" in t_hist.columns:
                        avg_pos = t_hist["finish_position"].mean()
                        shutuba_df.at[idx, "trainer_elo"] = 1500 + (8 - avg_pos) * 15

            # Eloランク再計算
            shutuba_df["horse_elo_rank"] = shutuba_df.groupby("race_id")["horse_elo"].rank(
                ascending=False
            )
            shutuba_df["jockey_elo_rank"] = shutuba_df.groupby("race_id")["jockey_elo"].rank(
                ascending=False
            )
            shutuba_df["combined_elo"] = (
                shutuba_df["horse_elo"] * 0.6
                + shutuba_df["jockey_elo"] * 0.25
                + shutuba_df["trainer_elo"] * 0.15
            )
            shutuba_df["combined_elo_rank"] = shutuba_df.groupby("race_id")["combined_elo"].rank(
                ascending=False
            )

            # === GOD特徴量の完全計算 ===
            for _, row in shutuba_df.iterrows():
                idx = row.name
                hid = row["horse_id"]
                jid = row["jockey_id"]
                tid = row["trainer_id"]

                if hid and "horse_id" in history_df.columns:
                    h_hist = history_df[history_df["horse_id"] == hid].copy()
                    if not h_hist.empty and "finish_position" in h_hist.columns:
                        fp = h_hist["finish_position"].dropna()
                        n = len(fp)

                        # weight_vs_self_avg
                        if "weight" in h_hist.columns:
                            avg_w = h_hist["weight"].mean()
                            if avg_w > 0:
                                shutuba_df.at[idx, "weight_vs_self_avg"] = row["weight"] - avg_w

                        # horse_avg_pos_3/5, horse_win_rate_3/5, horse_top3_rate_5
                        last3 = fp.tail(3)
                        last5 = fp.tail(5)
                        last10 = fp.tail(10)
                        shutuba_df.at[idx, "horse_avg_pos_3"] = last3.mean() if len(last3) > 0 else 8.0
                        shutuba_df.at[idx, "horse_win_rate_3"] = (last3 == 1).mean() if len(last3) > 0 else 0.0
                        shutuba_df.at[idx, "horse_win_rate_5"] = (last5 == 1).mean() if len(last5) > 0 else 0.0
                        shutuba_df.at[idx, "horse_top3_rate_5"] = (last5 <= 3).mean() if len(last5) > 0 else 0.0
                        shutuba_df.at[idx, "horse_total_races"] = float(n)

                        # horse_avg_prize
                        if "prize" in h_hist.columns:
                            shutuba_df.at[idx, "horse_avg_prize_3"] = h_hist["prize"].tail(3).mean()
                            shutuba_df.at[idx, "horse_avg_prize_5"] = h_hist["prize"].tail(5).mean()
                        else:
                            shutuba_df.at[idx, "horse_avg_prize_3"] = 0.0
                            shutuba_df.at[idx, "horse_avg_prize_5"] = 0.0

                        # horse_avg_last3f
                        if "last3f" in h_hist.columns:
                            l3f = h_hist["last3f"].dropna()
                            shutuba_df.at[idx, "horse_avg_last3f_3"] = l3f.tail(3).mean() if len(l3f) > 0 else 0.0
                            shutuba_df.at[idx, "horse_avg_last3f_10"] = l3f.tail(10).mean() if len(l3f) > 0 else 0.0
                        else:
                            shutuba_df.at[idx, "horse_avg_last3f_3"] = 0.0
                            shutuba_df.at[idx, "horse_avg_last3f_10"] = 0.0

                        # prev_position, position_trend_3
                        shutuba_df.at[idx, "prev_position"] = fp.iloc[-1] if n > 0 else 8.0
                        if len(last3) >= 2:
                            shutuba_df.at[idx, "position_trend_3"] = last3.iloc[0] - last3.iloc[-1]
                        else:
                            shutuba_df.at[idx, "position_trend_3"] = 0.0

                        # horse_norm_pos_avg
                        shutuba_df.at[idx, "horse_norm_pos_avg_3"] = last3.mean() if len(last3) > 0 else 8.0
                        shutuba_df.at[idx, "horse_norm_pos_avg_5"] = last5.mean() if len(last5) > 0 else 8.0

                        # Speed index (simplified: based on finish_position inverse)
                        if "time" in h_hist.columns and "distance" in h_hist.columns:
                            t = h_hist[["time", "distance"]].dropna()
                            if len(t) > 0:
                                spd = t["distance"] / t["time"].replace(0, np.nan) * 60
                                shutuba_df.at[idx, "avg_speed_index_3"] = spd.tail(3).mean()
                                shutuba_df.at[idx, "max_speed_index_5"] = spd.tail(5).max()

                        # Course/venue specific stats
                        r_dist = row.get("distance", 0)
                        r_venue = row.get("venue", "")
                        if "distance" in h_hist.columns and r_dist > 0:
                            dist_hist = h_hist[abs(h_hist["distance"] - r_dist) <= 200]
                            if len(dist_hist) > 0:
                                shutuba_df.at[idx, "horse_course_avg_pos"] = dist_hist["finish_position"].mean()
                                shutuba_df.at[idx, "horse_course_top3_rate"] = (dist_hist["finish_position"] <= 3).mean()
                                shutuba_df.at[idx, "horse_dist_combo_avg"] = dist_hist["finish_position"].mean()
                        if "venue" in h_hist.columns and r_venue:
                            v_hist = h_hist[h_hist["venue"] == r_venue]
                            if len(v_hist) > 0:
                                shutuba_df.at[idx, "horse_venue_avg_pos"] = v_hist["finish_position"].mean()

                        # Pace features
                        if "corner3" in h_hist.columns and "finish_position" in h_hist.columns:
                            h_hist["_pos_change"] = h_hist["corner3"] - h_hist["finish_position"]
                            shutuba_df.at[idx, "horse_avg_pos_change"] = h_hist["_pos_change"].tail(5).mean()
                            shutuba_df.at[idx, "horse_avg_early_pace"] = h_hist["corner3"].tail(5).mean()
                            shutuba_df.at[idx, "horse_pace_consistency"] = h_hist["_pos_change"].tail(5).std()

                # Jockey stats (recent 30 races)
                if jid and "jockey_id" in history_df.columns:
                    j_hist = history_df[history_df["jockey_id"] == jid]
                    if not j_hist.empty and "finish_position" in j_hist.columns:
                        last30 = j_hist.tail(30)
                        shutuba_df.at[idx, "jockey_win_rate_30"] = (last30["finish_position"] == 1).mean()
                        r_venue = row.get("venue", "")
                        if "venue" in j_hist.columns and r_venue:
                            jv = j_hist[j_hist["venue"] == r_venue]
                            if len(jv) > 0:
                                shutuba_df.at[idx, "jockey_venue_avg_pos"] = jv["finish_position"].mean()

                # Trainer stats
                if tid and "trainer_id" in history_df.columns:
                    t_hist = history_df[history_df["trainer_id"] == tid]
                    if not t_hist.empty and "finish_position" in t_hist.columns:
                        last30 = t_hist.tail(30)
                        shutuba_df.at[idx, "trainer_avg_pos_30"] = last30["finish_position"].mean()

            # === Race-level features ===
            for rid in shutuba_df["race_id"].unique():
                mask = shutuba_df["race_id"] == rid
                race = shutuba_df[mask]
                n_runners = len(race)
                shutuba_df.loc[mask, "n_runners"] = float(n_runners)

                # odds-based features (only if odds available)
                if "odds" in race.columns and race["odds"].sum() > 0:
                    shutuba_df.loc[mask, "odds_rank"] = race["odds"].rank()
                    avg_odds = race["odds"].mean()
                    min_odds = race["odds"].min()
                    shutuba_df.loc[mask, "race_avg_odds"] = avg_odds
                    shutuba_df.loc[mask, "race_min_odds"] = min_odds
                    shutuba_df.loc[mask, "odds_vs_race_avg"] = race["odds"] / avg_odds if avg_odds > 0 else 1.0
                    shutuba_df.loc[mask, "odds_ratio"] = race["odds"] / min_odds if min_odds > 0 else 1.0
                    shutuba_df.loc[mask, "odds_rank_pct"] = race["odds"].rank(pct=True)
                    mean_o = race["odds"].mean()
                    std_o = race["odds"].std()
                    shutuba_df.loc[mask, "odds_zscore"] = (race["odds"] - mean_o) / std_o if std_o > 0 else 0

                # weight-based features
                if "weight" in race.columns:
                    shutuba_df.loc[mask, "weight_vs_avg"] = race["weight"] - race["weight"].mean()
                    shutuba_df.loc[mask, "weight_rank_pct"] = race["weight"].rank(pct=True)

                # horse_weight features
                if "horse_weight" in race.columns:
                    shutuba_df.loc[mask, "hw_rank_pct"] = race["horse_weight"].rank(pct=True)
                    mean_hw = race["horse_weight"].mean()
                    std_hw = race["horse_weight"].std()
                    shutuba_df.loc[mask, "hw_zscore"] = (race["horse_weight"] - mean_hw) / std_hw if std_hw > 0 else 0

                # age features
                if "age" in race.columns:
                    shutuba_df.loc[mask, "age_rank_pct"] = race["age"].rank(pct=True)
                    shutuba_df.loc[mask, "age_vs_race_mean"] = race["age"] - race["age"].mean()

            # === Interaction features ===
            if "horse_weight" in shutuba_df.columns and "distance" in shutuba_df.columns:
                hw = shutuba_df["horse_weight"].replace(0, np.nan)
                dist = shutuba_df["distance"].replace(0, np.nan)
                shutuba_df["hw_x_dist"] = hw * dist
                shutuba_df["hw_per_dist"] = hw / dist

            if "age" in shutuba_df.columns and "distance" in shutuba_df.columns:
                shutuba_df["age_x_dist"] = shutuba_df["age"] * shutuba_df["distance"]

            if "weight" in shutuba_df.columns and "horse_weight" in shutuba_df.columns:
                hw = shutuba_df["horse_weight"].replace(0, np.nan)
                shutuba_df["weight_x_hw"] = shutuba_df["weight"] * hw
                shutuba_df["weight_ratio_hw"] = shutuba_df["weight"] / hw

            if "odds" in shutuba_df.columns and "popularity" in shutuba_df.columns:
                shutuba_df["odds_x_pop"] = shutuba_df["odds"] * shutuba_df["popularity"]

            # rest_x_prev_pos
            if "days_since_last_race" in shutuba_df.columns and "prev_position" in shutuba_df.columns:
                shutuba_df["rest_x_prev_pos"] = shutuba_df["days_since_last_race"] * shutuba_df["prev_position"]

            # age_x_season
            if "age" in shutuba_df.columns and "month" in shutuba_df.columns:
                shutuba_df["age_x_season"] = shutuba_df["age"] * shutuba_df["month"]

            # prize_rank_in_race (use horse_avg_prize_5 as proxy)
            if "horse_avg_prize_5" in shutuba_df.columns:
                for rid in shutuba_df["race_id"].unique():
                    mask = shutuba_df["race_id"] == rid
                    shutuba_df.loc[mask, "prize_rank_in_race"] = shutuba_df.loc[mask, "horse_avg_prize_5"].rank(ascending=False)

            # big_weight_change
            if "horse_weight_change_rate" in shutuba_df.columns:
                shutuba_df["big_weight_change"] = (shutuba_df["horse_weight_change_rate"].abs() > 0.03).astype(float)

            # class features (defaults if not available)
            for col in ["class_change", "class_momentum_3", "sire_n_runners", "dam_n_runners"]:
                if col not in shutuba_df.columns:
                    shutuba_df[col] = 0.0

            # te_trainer_id_win (use trainer win rate as proxy)
            if "te_trainer_id_win" not in shutuba_df.columns and "trainer_avg_pos_30" in shutuba_df.columns:
                shutuba_df["te_trainer_id_win"] = 1.0 / shutuba_df["trainer_avg_pos_30"].replace(0, 10)

            logger.info(f"GOD特徴量計算完了: {len(shutuba_df.columns)} columns")

        except Exception as e:
            logger.warning(f"過去データ結合中にエラー: {e}")
            import traceback
            traceback.print_exc()

        return shutuba_df


# ------------------------------------------------------------------
# CLI テスト用
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="出馬表スクレイパー")
    parser.add_argument("--race-id", type=str, help="レースID (例: 202406050811)")
    parser.add_argument("--date", type=str, help="日付 (YYYYMMDD, 例: 20240608)")
    parser.add_argument("--delay", type=float, default=2.0, help="リクエスト間隔（秒）")

    args = parser.parse_args()
    scraper = ShutubaScraper(delay=args.delay)

    if args.race_id:
        df = scraper.scrape_shutuba(args.race_id)
        if df is not None:
            print(df[["horse_number", "horse_name", "jockey_name", "odds", "popularity"]].to_string())
            print(f"\n全カラム: {list(df.columns)}")
        else:
            print("データ取得失敗")

    elif args.date:
        races = scraper.get_upcoming_races(args.date)
        for r in races:
            print(f"  {r['race_id']} | {r['venue']}{r['race_number']:2d}R | {r.get('race_name', '')}")
    else:
        # デフォルト: 今日の日付でテスト
        today = datetime.now().strftime("%Y%m%d")
        print(f"今日 ({today}) のレース一覧:")
        races = scraper.get_upcoming_races(today)
        if races:
            for r in races:
                print(f"  {r['race_id']} | {r['venue']}{r['race_number']:2d}R | {r.get('race_name', '')}")
        else:
            print("  レースなし（開催日ではない可能性）")
