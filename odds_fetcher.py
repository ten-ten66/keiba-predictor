"""
オッズ自動取得モジュール
========================
レース当日（土日）にnetkeiba.comから最新オッズを取得し、
予測キャッシュファイルを更新する。

対応馬券種:
  - 単勝 (tansho / b1)
  - 馬連 (umaren / b4)
  - ワイド (wide / b5)
  - 三連複 (sanrenpuku / b6)

使い方:
  # スタンドアロン実行
  python odds_fetcher.py [--date YYYYMMDD]

  # モジュールとして利用
  from odds_fetcher import fetch_odds, fetch_all_odds, update_predictions_with_odds
"""

import os
import sys
import json
import time
import re
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from itertools import combinations

logger = logging.getLogger(__name__)

# データディレクトリ
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / os.environ.get("KEIBA_DATA_DIR", "data")
UPCOMING_CACHE_DIR = DATA_DIR / "upcoming_cache"
PREDICTION_CACHE_DIR = DATA_DIR / "prediction_cache"

# リクエスト間隔（秒）- netkeiba負荷軽減
REQUEST_DELAY = 2.0

# マルチベットオッズキャッシュ（メモリ内、プロセス存続中有効）
_multi_odds_cache: Dict[str, Dict[str, Any]] = {}
_MULTI_ODDS_CACHE_TTL = 300  # 5分


def _get_headers() -> dict:
    """共通HTTPヘッダーを返す"""
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ja,en;q=0.5',
    }


def _parse_odds_value(text: str) -> Optional[float]:
    """オッズ文字列をfloatにパースする。失敗時はNone。"""
    try:
        cleaned = text.strip().replace(',', '').replace('---', '').replace('--', '')
        if not cleaned or cleaned == '-':
            return None
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def fetch_umaren_odds(race_id: str) -> Dict[Tuple[int, int], float]:
    """
    馬連 (umaren) オッズを取得する。

    URL: https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b4

    Args:
        race_id: 12桁のレースID

    Returns:
        {(horse1, horse2): odds} の辞書。horse1 < horse2。
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("requests / beautifulsoup4 がインストールされていません")
        return {}

    url = f'https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b4'
    odds_dict: Dict[Tuple[int, int], float] = {}

    try:
        r = requests.get(url, headers=_get_headers(), timeout=15)
        r.encoding = 'euc-jp'
        soup = BeautifulSoup(r.text, 'html.parser')

        # パターンA: テーブル行から馬連オッズを取得
        # netkeiba の馬連テーブルは行ごとに軸馬番号があり、
        # 各セルが相手馬とのオッズを示す
        tables = soup.find_all('table', class_=re.compile(r'Odds_Table|odds_table|RaceOdds'))
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue
                # 組み合わせ番号とオッズのペアを探す
                for cell in cells:
                    # data-odds 属性があるセル
                    if cell.get('data-odds'):
                        odds_val = _parse_odds_value(cell['data-odds'])
                        # data-umaban 等で組み合わせ番号を取得
                        umaban = cell.get('data-umaban', '')
                        if odds_val and umaban:
                            parts = re.findall(r'\d+', umaban)
                            if len(parts) == 2:
                                h1, h2 = sorted([int(parts[0]), int(parts[1])])
                                if h1 > 0 and h2 > 0 and h1 != h2:
                                    odds_dict[(h1, h2)] = odds_val

        # パターンB: span.Odds + 組み合わせをテキストから取得
        if not odds_dict:
            # 馬連オッズテーブルの各行を走査
            for row in soup.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 3:
                    try:
                        # 最初のセルに「1-2」のような組み合わせ
                        combo_text = cells[0].get_text(strip=True)
                        match = re.match(r'(\d+)\s*[-ー―]\s*(\d+)', combo_text)
                        if match:
                            h1, h2 = sorted([int(match.group(1)), int(match.group(2))])
                            # オッズを後続セルから取得
                            for cell in cells[1:]:
                                odds_val = _parse_odds_value(cell.get_text(strip=True))
                                if odds_val and odds_val > 1.0:
                                    odds_dict[(h1, h2)] = odds_val
                                    break
                    except (ValueError, TypeError):
                        continue

        # パターンC: 馬連の三角テーブル形式
        # 行の先頭が馬番、各セルが対応する相手馬とのオッズ
        if not odds_dict:
            for table in soup.find_all('table'):
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 2:
                        continue
                    # 先頭セルが馬番か確認
                    first_text = cells[0].get_text(strip=True)
                    if not first_text.isdigit():
                        continue
                    axis_num = int(first_text)
                    if axis_num <= 0:
                        continue
                    # 各セルのオッズを取得（セル番号=相手馬番号）
                    for j, cell in enumerate(cells[1:], 1):
                        odds_val = _parse_odds_value(cell.get_text(strip=True))
                        if odds_val and odds_val > 1.0 and j != axis_num:
                            h1, h2 = sorted([axis_num, j])
                            if h1 != h2:
                                odds_dict[(h1, h2)] = odds_val

        if odds_dict:
            logger.info(f"馬連オッズ取得成功: race_id={race_id}, {len(odds_dict)}組")
        else:
            logger.warning(f"馬連オッズ取得失敗: race_id={race_id}")

    except Exception as e:
        logger.warning(f"馬連オッズ取得エラー {race_id}: {e}")

    return odds_dict


def fetch_wide_odds(race_id: str) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    ワイド (wide) オッズを取得する。

    URL: https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b5
    ワイドオッズは範囲（最低〜最高）で表示される。

    Args:
        race_id: 12桁のレースID

    Returns:
        {(horse1, horse2): (min_odds, max_odds)} の辞書。horse1 < horse2。
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("requests / beautifulsoup4 がインストールされていません")
        return {}

    url = f'https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b5'
    odds_dict: Dict[Tuple[int, int], Tuple[float, float]] = {}

    try:
        r = requests.get(url, headers=_get_headers(), timeout=15)
        r.encoding = 'euc-jp'
        soup = BeautifulSoup(r.text, 'html.parser')

        # パターンA: data-odds属性を持つ要素
        for el in soup.find_all(attrs={'data-odds': True}):
            try:
                odds_text = el['data-odds']
                umaban = el.get('data-umaban', '')
                if not umaban:
                    continue
                parts = re.findall(r'\d+', umaban)
                if len(parts) != 2:
                    continue
                h1, h2 = sorted([int(parts[0]), int(parts[1])])
                if h1 <= 0 or h2 <= 0 or h1 == h2:
                    continue
                # ワイドは「min-max」形式の場合がある
                odds_parts = re.findall(r'[\d.]+', odds_text)
                if len(odds_parts) >= 2:
                    odds_dict[(h1, h2)] = (float(odds_parts[0]), float(odds_parts[1]))
                elif len(odds_parts) == 1:
                    val = float(odds_parts[0])
                    odds_dict[(h1, h2)] = (val, val)
            except (ValueError, TypeError):
                continue

        # パターンB: テーブル行からワイドオッズを取得
        if not odds_dict:
            for row in soup.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) < 3:
                    continue
                try:
                    combo_text = cells[0].get_text(strip=True)
                    match = re.match(r'(\d+)\s*[-ー―]\s*(\d+)', combo_text)
                    if not match:
                        continue
                    h1, h2 = sorted([int(match.group(1)), int(match.group(2))])
                    # ワイドオッズは「min - max」形式
                    for cell in cells[1:]:
                        text = cell.get_text(strip=True)
                        range_match = re.match(r'([\d,.]+)\s*[-〜~ー―]\s*([\d,.]+)', text)
                        if range_match:
                            min_odds = float(range_match.group(1).replace(',', ''))
                            max_odds = float(range_match.group(2).replace(',', ''))
                            if min_odds > 0 and max_odds > 0:
                                odds_dict[(h1, h2)] = (min_odds, max_odds)
                                break
                        else:
                            odds_val = _parse_odds_value(text)
                            if odds_val and odds_val > 1.0:
                                odds_dict[(h1, h2)] = (odds_val, odds_val)
                                break
                except (ValueError, TypeError):
                    continue

        # パターンC: 三角テーブル形式
        if not odds_dict:
            for table in soup.find_all('table'):
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 2:
                        continue
                    first_text = cells[0].get_text(strip=True)
                    if not first_text.isdigit():
                        continue
                    axis_num = int(first_text)
                    if axis_num <= 0:
                        continue
                    for j, cell in enumerate(cells[1:], 1):
                        text = cell.get_text(strip=True)
                        if not text or text == '-':
                            continue
                        range_match = re.match(r'([\d,.]+)\s*[-〜~ー―]\s*([\d,.]+)', text)
                        if range_match:
                            min_o = float(range_match.group(1).replace(',', ''))
                            max_o = float(range_match.group(2).replace(',', ''))
                            if min_o > 0 and max_o > 0 and j != axis_num:
                                h1, h2 = sorted([axis_num, j])
                                odds_dict[(h1, h2)] = (min_o, max_o)
                        else:
                            odds_val = _parse_odds_value(text)
                            if odds_val and odds_val > 1.0 and j != axis_num:
                                h1, h2 = sorted([axis_num, j])
                                odds_dict[(h1, h2)] = (odds_val, odds_val)

        if odds_dict:
            logger.info(f"ワイドオッズ取得成功: race_id={race_id}, {len(odds_dict)}組")
        else:
            logger.warning(f"ワイドオッズ取得失敗: race_id={race_id}")

    except Exception as e:
        logger.warning(f"ワイドオッズ取得エラー {race_id}: {e}")

    return odds_dict


def fetch_sanrenpuku_odds(race_id: str) -> Dict[Tuple[int, int, int], float]:
    """
    三連複 (sanrenpuku) オッズを取得する。

    URL: https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b6

    Args:
        race_id: 12桁のレースID

    Returns:
        {(horse1, horse2, horse3): odds} の辞書。horse1 < horse2 < horse3。
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("requests / beautifulsoup4 がインストールされていません")
        return {}

    url = f'https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b6'
    odds_dict: Dict[Tuple[int, int, int], float] = {}

    try:
        r = requests.get(url, headers=_get_headers(), timeout=15)
        r.encoding = 'euc-jp'
        soup = BeautifulSoup(r.text, 'html.parser')

        # パターンA: data-odds属性を持つ要素
        for el in soup.find_all(attrs={'data-odds': True}):
            try:
                odds_val = _parse_odds_value(el['data-odds'])
                umaban = el.get('data-umaban', '')
                if not odds_val or not umaban:
                    continue
                parts = re.findall(r'\d+', umaban)
                if len(parts) != 3:
                    continue
                h1, h2, h3 = sorted([int(parts[0]), int(parts[1]), int(parts[2])])
                if h1 > 0 and h2 > 0 and h3 > 0 and len({h1, h2, h3}) == 3:
                    odds_dict[(h1, h2, h3)] = odds_val
            except (ValueError, TypeError):
                continue

        # パターンB: テーブル行から三連複オッズを取得
        if not odds_dict:
            for row in soup.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) < 2:
                    continue
                try:
                    combo_text = cells[0].get_text(strip=True)
                    # 「1-2-3」形式を検索
                    match = re.match(r'(\d+)\s*[-ー―]\s*(\d+)\s*[-ー―]\s*(\d+)', combo_text)
                    if not match:
                        continue
                    h1, h2, h3 = sorted([int(match.group(1)), int(match.group(2)), int(match.group(3))])
                    if len({h1, h2, h3}) != 3:
                        continue
                    for cell in cells[1:]:
                        odds_val = _parse_odds_value(cell.get_text(strip=True))
                        if odds_val and odds_val > 1.0:
                            odds_dict[(h1, h2, h3)] = odds_val
                            break
                except (ValueError, TypeError):
                    continue

        # パターンC: 三連複は軸2頭固定テーブル形式の場合もある
        if not odds_dict:
            for table in soup.find_all('table'):
                # テーブルヘッダーやクラスから三連複テーブルを特定
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    for cell in cells:
                        text = cell.get_text(strip=True)
                        # 「1-2-3  45.0」のようなパターン
                        match = re.match(r'(\d+)\s*[-ー―]\s*(\d+)\s*[-ー―]\s*(\d+)', text)
                        if match:
                            h1, h2, h3 = sorted([int(match.group(1)), int(match.group(2)), int(match.group(3))])
                            if len({h1, h2, h3}) != 3:
                                continue
                            # 隣のセルからオッズ取得
                            next_sib = cell.find_next_sibling('td')
                            if next_sib:
                                odds_val = _parse_odds_value(next_sib.get_text(strip=True))
                                if odds_val and odds_val > 1.0:
                                    odds_dict[(h1, h2, h3)] = odds_val

        if odds_dict:
            logger.info(f"三連複オッズ取得成功: race_id={race_id}, {len(odds_dict)}組")
        else:
            logger.warning(f"三連複オッズ取得失敗: race_id={race_id}")

    except Exception as e:
        logger.warning(f"三連複オッズ取得エラー {race_id}: {e}")

    return odds_dict


def fetch_all_odds(race_id: str) -> Dict[str, Any]:
    """
    指定レースの全馬券種別オッズをまとめて取得する。

    Args:
        race_id: 12桁のレースID

    Returns:
        {
            "tansho": {1: 5.2, 2: 12.0, ...},
            "umaren": {(1,2): 15.5, (1,3): 22.0, ...},
            "wide": {(1,2): (3.5, 8.2), ...},
            "sanrenpuku": {(1,2,3): 45.0, ...}
        }
    """
    # キャッシュチェック
    cache_key = race_id
    if cache_key in _multi_odds_cache:
        cached = _multi_odds_cache[cache_key]
        age = time.time() - cached.get("_ts", 0)
        if age < _MULTI_ODDS_CACHE_TTL:
            logger.debug(f"マルチオッズキャッシュヒット: {race_id} ({age:.0f}秒前)")
            return {k: v for k, v in cached.items() if not k.startswith("_")}

    result: Dict[str, Any] = {}

    # 単勝
    result["tansho"] = fetch_odds(race_id)
    time.sleep(REQUEST_DELAY)

    # 馬連
    result["umaren"] = fetch_umaren_odds(race_id)
    time.sleep(REQUEST_DELAY)

    # ワイド
    result["wide"] = fetch_wide_odds(race_id)
    time.sleep(REQUEST_DELAY)

    # 三連複
    result["sanrenpuku"] = fetch_sanrenpuku_odds(race_id)

    # キャッシュに保存
    _multi_odds_cache[cache_key] = {**result, "_ts": time.time()}

    types_found = [k for k, v in result.items() if v]
    logger.info(f"全オッズ取得完了: race_id={race_id}, 取得成功={types_found}")

    return result


def _calc_recommended_bets(predictions: List[Dict], multi_odds: Dict[str, Any]) -> Dict[str, Any]:
    """
    予測データとマルチベットオッズからおすすめの買い方を計算する。

    Args:
        predictions: 予測データのリスト（各馬のdict）
        multi_odds: fetch_all_odds() の戻り値

    Returns:
        recommended_bets dict (tansho, umaren, wide, sanrenpuku)
    """
    recommended = {}

    # 馬番 -> 確率のマッピング
    prob_win_map = {}
    prob_top3_map = {}
    for h in predictions:
        h_num = int(h.get("horse_number", 0))
        if h_num > 0:
            prob_win_map[h_num] = float(h.get("prob_win", h.get("win_prob", 0)))
            prob_top3_map[h_num] = float(h.get("prob_top3", h.get("top3_prob", 0)))

    # --- 単勝 ---
    tansho_odds = multi_odds.get("tansho", {})
    best_tansho = None
    best_tansho_ev = 0
    for h_num, odds_val in tansho_odds.items():
        pw = prob_win_map.get(h_num, 0)
        if pw > 0 and odds_val > 0:
            ev = pw * odds_val
            if ev > best_tansho_ev:
                best_tansho_ev = ev
                best_tansho = {
                    "horses": [h_num],
                    "odds": round(odds_val, 1),
                    "ev": round(ev, 3),
                    "confidence": "high" if ev > 1.5 else "medium" if ev > 1.2 else "low",
                }
    if best_tansho:
        recommended["tansho"] = best_tansho

    # --- 馬連 ---
    umaren_odds = multi_odds.get("umaren", {})
    best_umaren = None
    best_umaren_ev = 0
    for combo, odds_val in umaren_odds.items():
        h1, h2 = combo
        pw1 = prob_win_map.get(h1, 0)
        pw2 = prob_win_map.get(h2, 0)
        pt1 = prob_top3_map.get(h1, 0)
        pt2 = prob_top3_map.get(h2, 0)
        # P(h1 & h2 in top2) = P(h1 wins)*P(h2 2nd|h1 wins) + P(h2 wins)*P(h1 2nd|h2 wins)
        # 近似: pw1 * pw2 / (1-pw1) + pw2 * pw1 / (1-pw2) だが、
        # よりシンプルに: pw1*pt2*0.5 + pw2*pt1*0.5 (独立近似+補正)
        p_both_top2 = pw1 * pw2 / max(1 - pw1, 0.01) + pw2 * pw1 / max(1 - pw2, 0.01)
        p_both_top2 = min(p_both_top2, 0.95)  # 上限
        ev = p_both_top2 * odds_val
        if ev > best_umaren_ev:
            best_umaren_ev = ev
            best_umaren = {
                "horses": sorted([h1, h2]),
                "odds": round(odds_val, 1),
                "ev": round(ev, 3),
                "confidence": "high" if ev > 1.5 else "medium" if ev > 1.2 else "low",
            }
    if best_umaren:
        recommended["umaren"] = best_umaren

    # --- ワイド ---
    wide_odds = multi_odds.get("wide", {})
    best_wide = None
    best_wide_ev = 0
    for combo, odds_range in wide_odds.items():
        h1, h2 = combo
        pt1 = prob_top3_map.get(h1, 0)
        pt2 = prob_top3_map.get(h2, 0)
        # P(both in top3) - 相関補正
        p_both_top3 = pt1 * pt2 * 0.85
        min_odds, max_odds = odds_range
        # 期待値は最小オッズで控えめに計算
        ev = p_both_top3 * min_odds
        if ev > best_wide_ev:
            best_wide_ev = ev
            best_wide = {
                "horses": sorted([h1, h2]),
                "odds": [round(min_odds, 1), round(max_odds, 1)],
                "ev": round(ev, 3),
                "confidence": "high" if ev > 1.5 else "medium" if ev > 1.2 else "low",
            }
    if best_wide:
        recommended["wide"] = best_wide

    # --- 三連複 ---
    sanrenpuku_odds = multi_odds.get("sanrenpuku", {})
    best_sanrenpuku = None
    best_sanrenpuku_ev = 0
    for combo, odds_val in sanrenpuku_odds.items():
        h1, h2, h3 = combo
        pt1 = prob_top3_map.get(h1, 0)
        pt2 = prob_top3_map.get(h2, 0)
        pt3 = prob_top3_map.get(h3, 0)
        # P(all 3 in top3) - 近似: 独立確率 * 相関補正
        p_all_top3 = pt1 * pt2 * pt3 * 0.7  # 相関補正
        ev = p_all_top3 * odds_val
        if ev > best_sanrenpuku_ev:
            best_sanrenpuku_ev = ev
            best_sanrenpuku = {
                "horses": sorted([h1, h2, h3]),
                "odds": round(odds_val, 1),
                "ev": round(ev, 3),
                "confidence": "high" if ev > 1.5 else "medium" if ev > 1.2 else "low",
            }
    if best_sanrenpuku:
        recommended["sanrenpuku"] = best_sanrenpuku

    return recommended


def update_cache_with_multi_odds(date_str: Optional[str] = None) -> Dict[str, int]:
    """
    指定日のキャッシュ済み予測ファイルにマルチベットオッズを追加する。

    各予測キャッシュに recommended_bets セクションを追加し、
    実オッズベースの期待値を計算する。

    Args:
        date_str: YYYYMMDD形式の日付。省略時は今日。

    Returns:
        {"updated": 更新件数, "failed": 失敗件数, "skipped": スキップ件数}
    """
    if date_str is None:
        date_str = date.today().strftime("%Y%m%d")

    PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    race_ids = get_race_ids_for_date(date_str)
    if not race_ids:
        logger.info(f"マルチオッズ更新対象なし: {date_str}")
        return {"updated": 0, "failed": 0, "skipped": 0}

    logger.info(f"マルチオッズ取得開始: {date_str}, {len(race_ids)}レース")

    stats = {"updated": 0, "failed": 0, "skipped": 0}

    for i, race_id in enumerate(race_ids):
        try:
            # キャッシュファイルを探す
            cache_files = list(PREDICTION_CACHE_DIR.glob(f"*_{race_id}.json"))
            if not cache_files:
                stats["skipped"] += 1
                continue

            # マルチオッズ取得
            multi_odds = fetch_all_odds(race_id)
            if not any(multi_odds.get(k) for k in ["umaren", "wide", "sanrenpuku"]):
                logger.warning(f"マルチオッズ取得結果なし: {race_id}")
                stats["failed"] += 1
                continue

            # 各キャッシュファイルを更新
            for cache_file in cache_files:
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached = json.load(f)

                    predictions = cached.get("predictions", cached.get("horses", []))

                    # recommended_bets を計算
                    recommended_bets = _calc_recommended_bets(predictions, multi_odds)

                    # マルチオッズデータをシリアライズ可能な形式に変換
                    multi_odds_serializable = {}
                    if multi_odds.get("tansho"):
                        multi_odds_serializable["tansho"] = {
                            str(k): v for k, v in multi_odds["tansho"].items()
                        }
                    if multi_odds.get("umaren"):
                        multi_odds_serializable["umaren"] = {
                            f"{k[0]}-{k[1]}": v for k, v in multi_odds["umaren"].items()
                        }
                    if multi_odds.get("wide"):
                        multi_odds_serializable["wide"] = {
                            f"{k[0]}-{k[1]}": list(v) for k, v in multi_odds["wide"].items()
                        }
                    if multi_odds.get("sanrenpuku"):
                        multi_odds_serializable["sanrenpuku"] = {
                            f"{k[0]}-{k[1]}-{k[2]}": v for k, v in multi_odds["sanrenpuku"].items()
                        }

                    cached["multi_odds"] = multi_odds_serializable
                    cached["recommended_bets"] = recommended_bets
                    cached["multi_odds_updated_at"] = datetime.now().isoformat()

                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(cached, f, ensure_ascii=False, indent=2)

                    logger.info(f"マルチオッズ更新: {cache_file.name}")
                    stats["updated"] += 1

                except Exception as e:
                    logger.error(f"キャッシュ更新エラー {cache_file}: {e}")
                    stats["failed"] += 1

            # レース間の遅延
            if i < len(race_ids) - 1:
                time.sleep(REQUEST_DELAY)

            # 進捗ログ
            if (i + 1) % 5 == 0:
                logger.info(f"マルチオッズ進捗: {i+1}/{len(race_ids)} レース完了")

        except Exception as e:
            logger.error(f"マルチオッズ取得エラー {race_id}: {e}")
            stats["failed"] += 1

    logger.info(f"マルチオッズ更新完了: {stats}")
    return stats


def fetch_odds(race_id: str) -> Dict[int, float]:
    """
    netkeiba.comから指定レースの単勝オッズを取得する。

    Args:
        race_id: 12桁のレースID（例: 202607010311）

    Returns:
        dict: {馬番: オッズ} の辞書。取得失敗時は空dict。
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("requests / beautifulsoup4 がインストールされていません")
        return {}

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ja,en;q=0.5',
    }

    odds_dict: Dict[int, float] = {}

    # 方法1: メインのオッズページ
    url = f'https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b1'
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.encoding = 'euc-jp'
        soup = BeautifulSoup(r.text, 'html.parser')

        # パターンA: Odds_Table クラスのテーブル
        odds_table = soup.find('table', class_='Odds_Table') or soup.find('table', id='odds_tan_table')
        if odds_table:
            rows = odds_table.find_all('tr')
            for row in rows:
                tds = row.find_all('td')
                if len(tds) >= 2:
                    # 馬番とオッズを探す
                    num_td = row.find('td', class_='Num') or row.find('td', class_='Umaban')
                    odds_td = row.find('td', class_='Odds') or row.find('td', class_='Popular')
                    if num_td and odds_td:
                        try:
                            horse_num = int(re.sub(r'\D', '', num_td.get_text(strip=True)))
                            odds_text = odds_td.get_text(strip=True).replace(',', '')
                            odds_val = float(odds_text)
                            if horse_num > 0 and odds_val > 0:
                                odds_dict[horse_num] = odds_val
                        except (ValueError, TypeError):
                            continue

        # パターンB: data-odds属性を持つ要素
        if not odds_dict:
            for el in soup.find_all(attrs={'data-odds': True}):
                try:
                    odds_val = float(el['data-odds'])
                    # 馬番を親要素から探す
                    parent_row = el.find_parent('tr')
                    if parent_row:
                        num_el = parent_row.find(class_='Num') or parent_row.find(class_='Umaban')
                        if num_el:
                            horse_num = int(re.sub(r'\D', '', num_el.get_text(strip=True)))
                            if horse_num > 0 and odds_val > 0:
                                odds_dict[horse_num] = odds_val
                except (ValueError, TypeError):
                    continue

        # パターンC: span.Odds でオッズを取得
        if not odds_dict:
            odds_spans = soup.find_all('span', class_='Odds')
            for idx, span in enumerate(odds_spans, 1):
                try:
                    odds_val = float(span.get_text(strip=True).replace(',', ''))
                    if odds_val > 0:
                        odds_dict[idx] = odds_val
                except (ValueError, TypeError):
                    continue

        if odds_dict:
            logger.info(f"オッズ取得成功 (HTML): race_id={race_id}, {len(odds_dict)}頭")
            return odds_dict

    except Exception as e:
        logger.warning(f"HTMLオッズ取得失敗 {race_id}: {e}")

    # 方法2: API エンドポイント
    time.sleep(REQUEST_DELAY)
    api_url = f'https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1'
    try:
        r = requests.get(api_url, headers=headers, timeout=15)
        data = r.json() if r.status_code == 200 else None
        if data:
            # APIレスポンスのパース（複数形式に対応）
            if isinstance(data, dict):
                # {horse_num: odds} 形式
                if 'data' in data:
                    items = data['data']
                    if isinstance(items, dict):
                        for k, v in items.items():
                            try:
                                horse_num = int(k)
                                odds_val = float(v) if not isinstance(v, dict) else float(v.get('odds', v.get('tanshow', 0)))
                                if horse_num > 0 and odds_val > 0:
                                    odds_dict[horse_num] = odds_val
                            except (ValueError, TypeError):
                                continue
                    elif isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                try:
                                    horse_num = int(item.get('horse_num', item.get('umaban', 0)))
                                    odds_val = float(item.get('odds', item.get('tanshow', 0)))
                                    if horse_num > 0 and odds_val > 0:
                                        odds_dict[horse_num] = odds_val
                                except (ValueError, TypeError):
                                    continue

            if odds_dict:
                logger.info(f"オッズ取得成功 (API): race_id={race_id}, {len(odds_dict)}頭")
                return odds_dict

    except Exception as e:
        logger.warning(f"APIオッズ取得失敗 {race_id}: {e}")

    # 方法3: 旧式URL
    time.sleep(REQUEST_DELAY)
    old_url = f'https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b1&housiki=c0'
    try:
        r = requests.get(old_url, headers=headers, timeout=15)
        r.encoding = 'euc-jp'
        soup = BeautifulSoup(r.text, 'html.parser')

        # 全てのテーブル行を走査
        for row in soup.find_all('tr'):
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3:
                try:
                    # 馬番候補（最初の数字セル）
                    num_text = cells[0].get_text(strip=True)
                    if num_text.isdigit():
                        horse_num = int(num_text)
                        # オッズ候補（数字.数字パターン）
                        for cell in cells[1:]:
                            text = cell.get_text(strip=True).replace(',', '')
                            if re.match(r'^\d+\.\d+$', text):
                                odds_val = float(text)
                                if horse_num > 0 and odds_val > 0:
                                    odds_dict[horse_num] = odds_val
                                break
                except (ValueError, TypeError):
                    continue

        if odds_dict:
            logger.info(f"オッズ取得成功 (旧URL): race_id={race_id}, {len(odds_dict)}頭")

    except Exception as e:
        logger.warning(f"旧URLオッズ取得失敗 {race_id}: {e}")

    if not odds_dict:
        logger.warning(f"オッズ取得失敗（全方法）: race_id={race_id}")

    return odds_dict


def _recalc_expected_value(horse: dict) -> dict:
    """
    馬データの expected_value と recommended_bet を再計算する。

    Args:
        horse: 予測データの馬1頭分のdict

    Returns:
        更新された馬dict
    """
    prob_win = float(horse.get("prob_win", horse.get("win_prob", 0)))
    prob_top3 = float(horse.get("prob_top3", horse.get("top3_prob", 0)))
    odds_val = float(horse.get("odds", 0))

    expected_value = prob_win * odds_val if odds_val > 0 else 0

    # 推奨ベット判定
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

    # 馬券種別
    bet_types = []
    if expected_value > 1.5:
        bet_types.extend(["単勝", "馬単1着固定"])
    elif expected_value > 1.2:
        bet_types.extend(["単勝", "複勝"])
    if prob_top3 > 0.5:
        bet_types.append("複勝")
    if prob_top3 > 0.4:
        bet_types.append("ワイド軸")
    bet_types = list(dict.fromkeys(bet_types))  # dedupe

    horse["expected_value"] = round(expected_value, 3)
    horse["recommended_bet"] = recommended_bet
    horse["recommendation"] = recommended_bet
    horse["bet_types"] = bet_types

    return horse


def update_prediction_cache_with_odds(race_id: str, odds_dict: Dict[int, float]) -> bool:
    """
    予測キャッシュファイルのオッズを更新し、expected_value等を再計算する。

    Args:
        race_id: レースID
        odds_dict: {馬番: オッズ} の辞書

    Returns:
        更新成功ならTrue
    """
    if not odds_dict:
        return False

    PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # キャッシュファイルを探す
    found = False
    for cache_file in PREDICTION_CACHE_DIR.glob(f"*_{race_id}.json"):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)

            predictions = cached.get("predictions", cached.get("horses", []))
            updated_count = 0
            for horse in predictions:
                h_num = int(horse.get("horse_number", 0))
                if h_num in odds_dict:
                    horse["odds"] = round(odds_dict[h_num], 1)
                    _recalc_expected_value(horse)
                    updated_count += 1

            cached["odds_updated_at"] = datetime.now().isoformat()
            cached["odds_source"] = "netkeiba"

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached, f, ensure_ascii=False, indent=2)

            logger.info(f"キャッシュ更新: {cache_file.name}, {updated_count}頭のオッズ更新")
            found = True

        except Exception as e:
            logger.error(f"キャッシュ更新エラー {cache_file}: {e}")

    return found


def get_race_ids_for_date(date_str: str) -> List[str]:
    """
    指定日のレースIDリストを取得する。

    Args:
        date_str: YYYYMMDD形式の日付

    Returns:
        レースIDのリスト
    """
    race_ids = []

    # upcoming_cacheからレースIDを取得
    cache_file = UPCOMING_CACHE_DIR / f"{date_str}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            races = cached.get("races", [])
            for race in races:
                rid = race.get("race_id", "")
                if rid:
                    race_ids.append(rid)
        except Exception as e:
            logger.warning(f"upcoming_cache読み込みエラー: {e}")

    # prediction_cacheからもレースIDを収集
    if PREDICTION_CACHE_DIR.exists():
        for cache_file in PREDICTION_CACHE_DIR.glob(f"{date_str}_*.json"):
            try:
                rid = cache_file.stem.split("_", 1)[1]
                if rid and rid not in race_ids:
                    race_ids.append(rid)
            except (IndexError, ValueError):
                continue

    return race_ids


def fetch_all_odds_for_date(date_str: Optional[str] = None) -> Dict[str, Dict[int, float]]:
    """
    指定日の全レースのオッズを取得し、予測キャッシュを更新する。

    Args:
        date_str: YYYYMMDD形式の日付。省略時は今日。

    Returns:
        {race_id: {馬番: オッズ}} の辞書
    """
    if date_str is None:
        date_str = date.today().strftime("%Y%m%d")

    race_ids = get_race_ids_for_date(date_str)
    if not race_ids:
        logger.info(f"対象レースなし: {date_str}")
        return {}

    logger.info(f"オッズ取得開始: {date_str}, {len(race_ids)}レース")

    all_odds = {}
    success_count = 0

    for i, race_id in enumerate(race_ids):
        if i > 0:
            time.sleep(REQUEST_DELAY)

        odds = fetch_odds(race_id)
        if odds:
            all_odds[race_id] = odds
            # 予測キャッシュも更新
            update_prediction_cache_with_odds(race_id, odds)
            success_count += 1

        # 進捗ログ（10レースごと）
        if (i + 1) % 10 == 0:
            logger.info(f"進捗: {i+1}/{len(race_ids)} レース完了")

    logger.info(f"オッズ取得完了: {success_count}/{len(race_ids)} レース成功")
    return all_odds


def is_race_day(target_date: Optional[date] = None) -> bool:
    """土曜または日曜かどうかを判定する"""
    if target_date is None:
        target_date = date.today()
    return target_date.weekday() in (5, 6)  # Saturday=5, Sunday=6


# ============================================================
# オッズ変動検出
# ============================================================

# 朝一オッズの永続キャッシュ（レースIDをキー、{馬番: オッズ} を値とする辞書）
# プロセス内メモリキャッシュ。別プロセス間の共有が必要なら morning_odds.csv を使う。
_morning_odds_store: Dict[str, Dict[int, float]] = {}


def record_morning_odds(race_id: str, odds_dict: Dict[int, float]) -> None:
    """朝一オッズを記録する。

    Args:
        race_id:   レースID
        odds_dict: {馬番: オッズ} の辞書
    """
    if odds_dict:
        _morning_odds_store[race_id] = dict(odds_dict)
        logger.debug(f"朝一オッズ記録: race_id={race_id}, {len(odds_dict)}頭")

        # CSV にも永続化（data/morning_odds.csv）
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        morning_path = DATA_DIR / "morning_odds.csv"
        rows = [
            {"race_id": race_id, "horse_number": hn, "morning_odds": od,
             "recorded_at": datetime.now().isoformat()}
            for hn, od in odds_dict.items()
        ]
        import csv
        file_exists = morning_path.exists()
        with open(morning_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["race_id", "horse_number", "morning_odds", "recorded_at"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)


def load_morning_odds_from_csv(race_id: str) -> Dict[int, float]:
    """CSV から朝一オッズをロードする（プロセス再起動後の復元用）。

    Args:
        race_id: レースID

    Returns:
        {馬番: オッズ} の辞書。データ無しなら空dict。
    """
    morning_path = DATA_DIR / "morning_odds.csv"
    if not morning_path.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(morning_path, dtype={"race_id": str})
        df = df[df["race_id"] == str(race_id)]
        if df.empty:
            return {}
        return {int(row["horse_number"]): float(row["morning_odds"])
                for _, row in df.iterrows()
                if pd.notna(row["morning_odds"]) and float(row["morning_odds"]) > 0}
    except Exception as e:
        logger.warning(f"朝一オッズCSV読み込みエラー: {e}")
        return {}


def calc_odds_change(
    race_id: str,
    current_odds: Dict[int, float],
    morning_odds: Optional[Dict[int, float]] = None,
    smart_money_threshold: float = 0.30,
) -> Dict[int, Dict[str, Any]]:
    """朝一オッズと直前オッズの差分を計算し、スマートマネーシグナルを生成する。

    Parameters
    ----------
    race_id : str
        レースID
    current_odds : dict
        直前の {馬番: オッズ} 辞書
    morning_odds : dict or None
        朝一オッズ。None の場合は内部キャッシュ → CSV の順で自動補完。
    smart_money_threshold : float
        odds_change_ratio のしきい値。この値以上の下落率を「スマートマネー」と判定。
        デフォルト 0.30 = 30%以上下落。

    Returns
    -------
    dict
        {馬番: {
            "morning_odds"     : float or None,  朝一オッズ
            "current_odds"     : float,           直前オッズ
            "odds_change"      : float or None,   変化量 (current - morning)
            "odds_change_ratio": float or None,   変化率 (change / morning), 負=下落
            "smart_money_signal": int,            1=スマートマネー疑い, 0=通常
        }}

    Notes
    -----
    - 朝一オッズが未記録の馬は odds_change / odds_change_ratio = None となる。
    - smart_money_signal は odds_change_ratio <= -threshold の馬に 1 を立てる。
      例: 朝一10倍 → 直前7倍 → ratio=-0.30 → signal=1 (ちょうどしきい値)
    """
    # 朝一オッズの解決順: 引数 → メモリキャッシュ → CSV
    if morning_odds is None:
        morning_odds = _morning_odds_store.get(race_id)
    if not morning_odds:
        morning_odds = load_morning_odds_from_csv(race_id)
    if not morning_odds:
        logger.debug(f"朝一オッズ未記録: race_id={race_id}")

    result: Dict[int, Dict[str, Any]] = {}
    for horse_num, cur_odds in current_odds.items():
        morn = morning_odds.get(horse_num) if morning_odds else None

        change: Optional[float] = None
        change_ratio: Optional[float] = None
        if morn is not None and morn > 0 and cur_odds > 0:
            change = round(cur_odds - morn, 2)
            change_ratio = round(change / morn, 4)

        # スマートマネーシグナル: 閾値を超えたオッズ下落
        signal = 0
        if change_ratio is not None and change_ratio <= -smart_money_threshold:
            signal = 1

        result[horse_num] = {
            "morning_odds": morn,
            "current_odds": cur_odds,
            "odds_change": change,
            "odds_change_ratio": change_ratio,
            "smart_money_signal": signal,
        }

    # 変化率降順でログ出力（上位3頭）
    if result:
        sorted_horses = sorted(
            [(hn, v["odds_change_ratio"]) for hn, v in result.items()
             if v["odds_change_ratio"] is not None],
            key=lambda x: x[1]
        )
        if sorted_horses:
            top_str = ", ".join(f"{hn}番:{r:+.1%}" for hn, r in sorted_horses[:3])
            signals = sum(v["smart_money_signal"] for v in result.values())
            logger.info(
                f"オッズ変動 race_id={race_id}: "
                f"大変動TOP3=[{top_str}], スマートマネー疑い={signals}頭"
            )

    return result


def add_odds_change_features(
    df: "pd.DataFrame",
    race_id: str,
    current_odds: Dict[int, float],
    morning_odds: Optional[Dict[int, float]] = None,
    smart_money_threshold: float = 0.30,
) -> "pd.DataFrame":
    """calc_odds_change() の結果を DataFrame の特徴量カラムとして付与する。

    追加カラム:
      odds_change_ratio  : 朝一→直前のオッズ変化率（負=下落）。朝一未記録はNaN。
      smart_money_signal : 閾値以上の下落で 1、それ以外 0。

    Parameters
    ----------
    df : pd.DataFrame
        horse_number カラムを持つ DataFrame
    race_id, current_odds, morning_odds, smart_money_threshold :
        calc_odds_change() と同じ

    Returns
    -------
    pd.DataFrame
        odds_change_ratio, smart_money_signal カラムが追加された DataFrame
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        logger.error("pandas が必要です")
        return df

    change_info = calc_odds_change(race_id, current_odds, morning_odds, smart_money_threshold)

    df = df.copy()
    df["odds_change_ratio"] = df["horse_number"].map(
        {hn: v["odds_change_ratio"] for hn, v in change_info.items()}
    )
    df["smart_money_signal"] = df["horse_number"].map(
        {hn: v["smart_money_signal"] for hn, v in change_info.items()}
    ).fillna(0).astype(int)

    return df


# ============================================================
# CLI エントリーポイント
# ============================================================

def main():
    """コマンドライン実行"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="netkeiba オッズ自動取得")
    parser.add_argument("--date", type=str, default=None,
                        help="対象日付 YYYYMMDD（省略時は今日）")
    parser.add_argument("--race-id", type=str, default=None,
                        help="特定レースIDのみ取得")
    parser.add_argument("--force", action="store_true",
                        help="土日以外でも強制実行")
    parser.add_argument("--multi", action="store_true",
                        help="馬連・ワイド・三連複のオッズも取得")
    args = parser.parse_args()

    target_date_str = args.date or date.today().strftime("%Y%m%d")

    # 土日チェック
    if not args.force and not args.race_id:
        try:
            td = datetime.strptime(target_date_str, "%Y%m%d").date()
            if not is_race_day(td):
                print(f"レース開催日ではありません（{td.strftime('%A')}）。--force で強制実行可能。")
                return
        except ValueError:
            pass

    if args.race_id:
        # 特定レースのみ
        if args.multi:
            print(f"レース {args.race_id} の全種別オッズを取得中...")
            all_type_odds = fetch_all_odds(args.race_id)
            for bet_type, odds_data in all_type_odds.items():
                if odds_data:
                    print(f"\n  [{bet_type}] {len(odds_data)}件")
                    items = list(odds_data.items())[:5]
                    for k, v in items:
                        print(f"    {k}: {v}")
                    if len(odds_data) > 5:
                        print(f"    ... 他{len(odds_data)-5}件")
                else:
                    print(f"\n  [{bet_type}] 取得失敗")
        else:
            print(f"レース {args.race_id} のオッズを取得中...")
            odds = fetch_odds(args.race_id)
            if odds:
                print(f"取得成功: {len(odds)}頭")
                for num in sorted(odds.keys()):
                    print(f"  {num}番: {odds[num]}倍")
                update_prediction_cache_with_odds(args.race_id, odds)
            else:
                print("オッズを取得できませんでした")
    else:
        # 全レース
        print(f"日付 {target_date_str} の全レースオッズを取得中...")
        all_odds = fetch_all_odds_for_date(target_date_str)
        print(f"\n完了: {len(all_odds)} レースのオッズを取得")
        for rid, odds in all_odds.items():
            print(f"  {rid}: {len(odds)}頭")

        if args.multi:
            print(f"\nマルチベットオッズも取得中...")
            stats = update_cache_with_multi_odds(target_date_str)
            print(f"マルチオッズ更新: 成功={stats['updated']}, 失敗={stats['failed']}, スキップ={stats['skipped']}")


if __name__ == "__main__":
    main()
