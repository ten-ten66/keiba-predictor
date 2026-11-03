"""
競馬予想AI - リアルタイムデータ取得ジョブ
==========================================
当日のレース情報・オッズ・結果を定期取得する。

スケジュール:
  06:00  当日の出走表を取得
  09:00〜15:30  5分ごとにオッズ更新
  12:00〜17:00  レース終了後に結果を取得
  17:30  日次まとめ→Discord通知

起動方法:
  # 単体実行
  python realtime_job.py

  # API サーバーと一緒に起動（推奨）
  python realtime_job.py --daemon
"""

import os
import sys
import json
import time
import re
import logging
import threading
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB = True
except ImportError:
    HAS_WEB = False

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 設定
# ============================================================

DATA_DIR = os.environ.get("KEIBA_DATA_DIR", "data")
DISCORD_WEBHOOK = os.environ.get("KEIBA_DISCORD_WEBHOOK", "")
API_BASE = os.environ.get("KEIBA_API_BASE", "http://localhost:8000")
DELAY = 2.5  # リクエスト間隔

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, "realtime.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("realtime")


# ============================================================
# HTTP ヘルパー
# ============================================================

session = requests.Session()
session.headers.update(HEADERS)


def fetch_page(url: str, encoding: str = "utf-8") -> Optional[BeautifulSoup]:
    time.sleep(DELAY)
    try:
        resp = session.get(url, timeout=20)
        resp.encoding = encoding
        if resp.status_code == 200:
            return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        log.error(f"取得エラー: {url} → {e}")
    return None


def notify(msg: str):
    log.info(msg)
    if DISCORD_WEBHOOK:
        try:
            requests.post(DISCORD_WEBHOOK, json={"content": msg, "username": "🏇 KEIBA AI"}, timeout=10)
        except Exception:
            pass


def push_to_api(endpoint: str, data: dict) -> bool:
    """APIサーバーにデータを送信"""
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


# ============================================================
# 1. 当日の出走表取得
# ============================================================

def fetch_today_races() -> List[Dict]:
    """当日の出走表を取得"""
    today = date.today()
    date_str = today.strftime("%Y%m%d")
    log.info(f"📋 出走表取得: {date_str}")

    url = f"https://race.netkeiba.com/race/list/{date_str}/"
    soup = fetch_page(url)
    if not soup:
        log.warning("出走表ページの取得失敗")
        return []

    # レースID一覧を取得
    race_ids = set()
    for link in soup.find_all("a", href=True):
        m = re.search(r"/race/(\d{12})/", link["href"])
        if m:
            race_ids.add(m.group(1))

    race_ids = sorted(race_ids)
    log.info(f"  {len(race_ids)} レース発見")

    # 各レースの出走表を取得
    races = []
    for race_id in race_ids:
        race_data = fetch_shutuba(race_id)
        if race_data:
            races.append(race_data)

    # JSON保存
    output_path = os.path.join(DATA_DIR, f"today_{today.isoformat()}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"date": today.isoformat(), "races": races, "updated_at": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)

    log.info(f"  ✅ {len(races)} レース保存: {output_path}")
    notify(f"📋 **出走表取得完了**\n{today.isoformat()} / {len(races)} レース")

    return races


def fetch_shutuba(race_id: str) -> Optional[Dict]:
    """1レースの出走表を取得"""
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    soup = fetch_page(url)
    if not soup:
        return None

    # レース名
    race_name = ""
    name_tag = soup.find("div", class_="RaceName")
    if name_tag:
        race_name = name_tag.get_text(strip=True)

    # コース情報
    data_tag = soup.find("div", class_="RaceData01")
    course_info = data_tag.get_text(strip=True) if data_tag else ""

    # 発走時刻
    post_time = ""
    time_match = re.search(r"(\d{1,2}:\d{2})", course_info)
    if time_match:
        post_time = time_match.group(1)

    # 出走馬
    horses = []
    table = soup.find("table", class_="Shutuba_Table") or soup.find("table", id="shutuba_table")
    if table:
        for row in table.find_all("tr", class_="HorseList"):
            cols = row.find_all("td")
            if len(cols) < 8:
                continue
            try:
                horse = {
                    "frame_number": int(cols[0].get_text(strip=True) or "0"),
                    "horse_number": int(cols[1].get_text(strip=True) or "0"),
                    "horse_name": cols[3].get_text(strip=True),
                    "sex_age": cols[4].get_text(strip=True),
                    "weight": cols[5].get_text(strip=True),
                    "jockey_name": cols[6].get_text(strip=True),
                    "trainer_name": cols[7].get_text(strip=True) if len(cols) > 7 else "",
                    "odds": 0,
                    "popularity": 0,
                }
                horses.append(horse)
            except (ValueError, IndexError):
                continue

    if not horses:
        return None

    return {
        "race_id": race_id,
        "race_name": race_name,
        "course_info": course_info,
        "post_time": post_time,
        "horses": horses,
    }


# ============================================================
# 2. リアルタイムオッズ取得
# ============================================================

def fetch_all_odds():
    """当日の全レースのオッズを更新"""
    today = date.today()
    today_file = os.path.join(DATA_DIR, f"today_{today.isoformat()}.json")

    if not os.path.exists(today_file):
        log.info("⏭️ 出走表なし。オッズ取得スキップ")
        return

    with open(today_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    races = data.get("races", [])
    updated_count = 0

    for race in races:
        race_id = race["race_id"]
        odds = fetch_race_odds(race_id)
        if odds:
            # 出走表にオッズを反映
            for horse in race.get("horses", []):
                hn = horse.get("horse_number", 0)
                if hn in odds:
                    horse["odds"] = odds[hn]

            # オッズキャッシュ（APIサーバーと共有するファイル）
            odds_path = os.path.join(DATA_DIR, f"odds_{race_id}.json")
            with open(odds_path, "w", encoding="utf-8") as f:
                json.dump({
                    "race_id": race_id,
                    "odds": odds,
                    "updated_at": datetime.now().isoformat(),
                }, f, ensure_ascii=False)

            updated_count += 1

    # 出走表ファイルを更新
    data["updated_at"] = datetime.now().isoformat()
    with open(today_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log.info(f"💹 オッズ更新: {updated_count}/{len(races)} レース")


def fetch_race_odds(race_id: str) -> Dict[int, float]:
    """1レースの単勝オッズを取得"""
    url = f"https://race.netkeiba.com/odds/index.html?type=b1&race_id={race_id}"
    soup = fetch_page(url)
    if not soup:
        return {}

    odds = {}
    # オッズテーブルを解析
    for row in soup.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 4:
            continue
        try:
            horse_num = int(cols[1].get_text(strip=True))
            odds_text = cols[3].get_text(strip=True).replace(",", "")
            odds_val = float(odds_text)
            if odds_val > 0:
                odds[horse_num] = odds_val
        except (ValueError, IndexError):
            continue

    return odds


# ============================================================
# 3. レース結果取得
# ============================================================

def fetch_completed_results():
    """終了したレースの結果を取得"""
    today = date.today()
    today_file = os.path.join(DATA_DIR, f"today_{today.isoformat()}.json")

    if not os.path.exists(today_file):
        return

    with open(today_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    races = data.get("races", [])
    results_file = os.path.join(DATA_DIR, f"results_{today.isoformat()}.json")

    # 既取得の結果を読み込み
    existing_results = {}
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)

    new_count = 0
    for race in races:
        race_id = race["race_id"]
        if race_id in existing_results:
            continue

        result = fetch_race_result(race_id)
        if result:
            existing_results[race_id] = result
            new_count += 1

            # APIに結果送信
            push_to_api("/api/results", {
                "race_id": race_id,
                "race_name": result.get("race_name", ""),
                "race_date": today.isoformat(),
                "venue": result.get("venue", ""),
                "course_type": result.get("course_type", ""),
                "distance": result.get("distance", 0),
                "track_condition": result.get("track_condition", ""),
                "finish_order": result.get("finish_order", []),
                "payouts": result.get("payouts", {}),
            })

    if new_count > 0:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)
        log.info(f"🏁 結果取得: +{new_count}レース (計{len(existing_results)})")


def fetch_race_result(race_id: str) -> Optional[Dict]:
    """1レースの結果を取得"""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    soup = fetch_page(url, encoding="euc-jp")
    if not soup:
        return None

    table = soup.find("table", class_="race_table_01")
    if not table:
        return None

    finish_order = []
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) < 14:
            continue
        try:
            pos = int(cols[0].get_text(strip=True))
            horse_link = cols[3].find("a")
            horse_name = horse_link.get_text(strip=True) if horse_link else ""
            odds = float(cols[12].get_text(strip=True) or "0")

            finish_order.append({
                "position": pos,
                "horse_number": int(cols[2].get_text(strip=True) or "0"),
                "horse_name": horse_name,
                "frame_number": int(cols[1].get_text(strip=True) or "0"),
                "odds": odds,
                "popularity": int(cols[13].get_text(strip=True) or "0"),
            })
        except (ValueError, IndexError):
            continue

    if not finish_order:
        return None

    # 払い戻し情報
    payouts = {}
    pay_table = soup.find("table", class_="pay_table_01")
    if pay_table:
        for row in pay_table.find_all("tr"):
            cols = row.find_all("td")
            th = row.find("th")
            if not th or len(cols) < 2:
                continue
            bet_type = th.get_text(strip=True)
            combo_text = cols[0].get_text(strip=True)
            payout_text = cols[1].get_text(strip=True).replace(",", "").replace("円", "")
            try:
                payout = int(payout_text)
                combo = [int(n) for n in re.findall(r"\d+", combo_text)]
                if bet_type not in payouts:
                    payouts[bet_type] = []
                payouts[bet_type].append({"combination": combo, "payout": payout})
            except ValueError:
                continue

    # レース情報
    race_data = soup.find("div", class_="racedata")
    course_info = race_data.get_text(strip=True) if race_data else ""
    course_type = "芝" if "芝" in course_info else "ダート"
    dist_match = re.search(r"(\d{3,4})m", course_info)
    distance = int(dist_match.group(1)) if dist_match else 0

    weather = "不明"
    track_condition = "不明"
    if race_data:
        text = race_data.get_text()
        wm = re.search(r"天候\s*:\s*(\S+)", text)
        tm = re.search(r"(?:芝|ダート)\s*:\s*(\S+)", text)
        if wm:
            weather = wm.group(1)
        if tm:
            track_condition = tm.group(1)

    race_name = ""
    name_tag = soup.find("h1", class_="racedata_title") or soup.find("h1")
    if name_tag:
        race_name = name_tag.get_text(strip=True)

    venue_code = race_id[4:6]
    venue_map = {"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}

    return {
        "race_id": race_id,
        "race_name": race_name,
        "venue": venue_map.get(venue_code, ""),
        "course_type": course_type,
        "distance": distance,
        "track_condition": track_condition,
        "weather": weather,
        "finish_order": finish_order,
        "payouts": payouts,
    }


# ============================================================
# 4. 日次まとめ通知
# ============================================================

def daily_summary():
    """日次まとめ"""
    today = date.today()
    results_file = os.path.join(DATA_DIR, f"results_{today.isoformat()}.json")

    if not os.path.exists(results_file):
        notify("📊 本日のレース結果なし")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    lines = [f"📊 **{today.isoformat()} 日次まとめ**", f"取得レース: {len(results)}", ""]

    for race_id, result in list(results.items())[:10]:
        top3 = result.get("finish_order", [])[:3]
        race_name = result.get("race_name", race_id)
        lines.append(f"**{race_name}**")
        for h in top3:
            medal = ["🥇", "🥈", "🥉"][h["position"] - 1] if h["position"] <= 3 else ""
            lines.append(f"  {medal} {h['position']}着 {h['horse_number']}番 {h['horse_name']} ({h.get('odds', '-')}倍)")
        lines.append("")

    notify("\n".join(lines))


# ============================================================
# スケジューラー
# ============================================================

def setup_schedule():
    """スケジュール設定"""
    if not HAS_SCHEDULE:
        log.warning("⚠️ scheduleパッケージ未インストール。pip install schedule")
        return
    # 朝: 出走表取得
    schedule.every().day.at("06:00").do(fetch_today_races)
    schedule.every().day.at("07:00").do(fetch_today_races)  # バックアップ

    # 午前〜午後: オッズ更新（5分ごと）
    for hour in range(9, 16):
        for minute in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
            schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(fetch_all_odds)

    # 午後: 結果取得（10分ごと）
    for hour in range(12, 18):
        for minute in [0, 10, 20, 30, 40, 50]:
            schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(fetch_completed_results)

    # 夕方: 日次まとめ
    schedule.every().day.at("17:30").do(daily_summary)

    # 自動学習パイプライン
    schedule.every().day.at("18:00").do(_run_daily_learn)    # 日次: 結果CSV追記 & 再学習チェック
    schedule.every().monday.at("03:00").do(_run_weekly_learn)  # 週次: 強制再学習

    # Twitter/X 結果自動投稿（土日18:30）
    schedule.every().saturday.at("18:30").do(_auto_tweet_results)
    schedule.every().sunday.at("18:30").do(_auto_tweet_results)

    log.info("⏰ スケジュール設定完了")
    log.info("  06:00/07:00  出走表取得")
    log.info("  09:00〜15:55 オッズ更新 (5分毎)")
    log.info("  12:00〜17:50 結果取得 (10分毎)")
    log.info("  17:30        日次まとめ")
    log.info("  18:00        自動学習パイプライン")
    log.info("  月曜03:00    週次強制再学習")
    log.info("  土日18:30    Twitter結果自動投稿")


def _run_daily_learn():
    """日次自動学習パイプライン"""
    try:
        from auto_learn import daily_pipeline
        daily_pipeline()
    except Exception as e:
        log.error(f"自動学習エラー: {e}")


def _run_weekly_learn():
    """週次強制再学習"""
    try:
        from auto_learn import weekly_retrain
        weekly_retrain()
    except Exception as e:
        log.error(f"週次再学習エラー: {e}")


def _auto_tweet_results():
    """結果収集後にTwitter/Xへ自動投稿（土日18:30）"""
    try:
        from social_poster import auto_post_results
        today_str = date.today().isoformat()
        results = auto_post_results(today_str)
        if results:
            log.info(f"🐦 Twitter結果投稿: {len(results)}件")
        else:
            log.info("🐦 Twitter結果投稿: 対象なし")
    except ImportError:
        log.warning("social_poster モジュールが読み込めません")
    except Exception as e:
        log.error(f"Twitter結果投稿エラー: {e}")


def run_scheduler():
    """スケジューラーのメインループ"""
    if not HAS_SCHEDULE:
        log.error("❌ scheduleパッケージ未インストール。pip install schedule")
        return
    setup_schedule()
    log.info("🚀 リアルタイムジョブ開始")
    notify("🚀 **リアルタイムジョブ開始**")

    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
        except KeyboardInterrupt:
            log.info("🛑 停止")
            break
        except Exception as e:
            log.error(f"スケジューラーエラー: {e}")
            time.sleep(60)


def run_as_thread():
    """バックグラウンドスレッドで起動（APIサーバーと並列実行用）"""
    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()
    log.info("🧵 バックグラウンドスレッドで起動")
    return t


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="リアルタイムデータ取得")
    parser.add_argument("--daemon", action="store_true", help="バックグラウンド起動")
    parser.add_argument("--once", choices=["shutuba", "odds", "results", "summary"], help="1回だけ実行")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.once:
        {"shutuba": fetch_today_races, "odds": fetch_all_odds, "results": fetch_completed_results, "summary": daily_summary}[args.once]()
    elif args.daemon:
        run_as_thread()
        # メインスレッドはAPIサーバー起動を待つ
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        run_scheduler()
