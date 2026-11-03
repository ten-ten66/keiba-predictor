"""
競馬予想AI v2 - 高度分析エンジン
=================================
追加機能:
  1. ケリー基準 — 数学的に最適な賭け金を算出
  2. 騎手×馬の相性分析 — 過去の組み合わせ実績
  3. 妙味検出（バリューベット） — AI確率 vs オッズの乖離
  4. URL指定でレース情報を自動取得
  5. コース適性の詳細分析（左右回り・坂・直線長）
"""

import pandas as pd
import numpy as np
import json
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ==============================================================
# 1. ケリー基準 資金管理
# ==============================================================

class KellyCriterion:
    """
    ケリー基準: 期待値がプラスの賭けに対し、
    資金の何%を賭けるのが最適かを数学的に算出。

    Kelly% = (bp - q) / b
      b = オッズ - 1 (純利益倍率)
      p = 勝つ確率 (AI予測)
      q = 負ける確率 (1 - p)
    """

    def __init__(self, fraction: float = 0.25):
        """
        fraction: ケリー基準の何分のいくつを使うか（0.25 = クォーターケリー推奨）
        フルケリーはリスクが高すぎるため、1/4〜1/2が実用的。
        """
        self.fraction = fraction

    def calculate(self, prob_win: float, odds: float) -> Dict:
        """
        最適賭け金の比率を計算

        Args:
            prob_win: 勝つ確率 (0.0〜1.0)
            odds: オッズ（例: 5.0 = 5倍）

        Returns:
            kelly_pct: 資金に対する最適賭け比率
            bet_amount: 推奨賭け金（bankrollに対する）
            expected_value: 期待値
            edge: エッジ（優位性）
        """
        import math
        # NaN/Inf/範囲外をガード
        if (not isinstance(prob_win, (int, float)) or not isinstance(odds, (int, float))
            or math.isnan(prob_win) or math.isnan(odds)
            or math.isinf(prob_win) or math.isinf(odds)
            or odds <= 1 or prob_win <= 0 or prob_win >= 1):
            return {
                "kelly_full_pct": 0,
                "kelly_adjusted_pct": 0,
                "expected_value": 0,
                "edge_pct": 0,
                "recommendation": "見送り",
            }

        b = odds - 1  # 純利益倍率
        p = prob_win
        q = 1 - p

        # 期待値
        ev = p * b - q
        edge = ev / b if b > 0 else 0

        # ケリー%
        kelly_full = (b * p - q) / b if b > 0 else 0
        kelly_adjusted = max(0, kelly_full * self.fraction)

        # 推奨判定
        if kelly_full <= 0:
            rec = "見送り"
        elif kelly_adjusted < 0.01:
            rec = "少額"
        elif kelly_adjusted < 0.03:
            rec = "通常"
        elif kelly_adjusted < 0.06:
            rec = "強気"
        else:
            rec = "大勝負"

        return {
            "kelly_full_pct": round(kelly_full * 100, 2),
            "kelly_adjusted_pct": round(kelly_adjusted * 100, 2),
            "expected_value": round(ev + 1, 3),  # 1.0基準（1.0超=期待値プラス）
            "edge_pct": round(edge * 100, 2),
            "recommendation": rec,
        }

    def optimal_bets(
        self,
        horses: List[Dict],
        bankroll: int = 10000,
        max_bet_pct: float = 0.10,
    ) -> List[Dict]:
        """
        レース全馬に対して最適賭け金を算出

        Args:
            horses: [{"horse_number": 1, "prob_win": 0.3, "odds": 5.0, ...}, ...]
            bankroll: 資金
            max_bet_pct: 1レースあたりの最大賭け比率

        Returns:
            推奨賭けリスト（期待値プラスの馬のみ）
        """
        results = []
        for h in horses:
            calc = self.calculate(h.get("prob_win", 0), h.get("odds", 0))

            if calc["expected_value"] <= 1.0:
                continue  # 期待値マイナスは除外

            # 賭け金計算
            bet_pct = min(calc["kelly_adjusted_pct"] / 100, max_bet_pct)
            bet_amount = max(100, int(bankroll * bet_pct / 100) * 100)  # 100円単位

            results.append({
                "horse_number": h.get("horse_number"),
                "horse_name": h.get("horse_name", ""),
                "prob_win": h.get("prob_win", 0),
                "odds": h.get("odds", 0),
                "bet_amount": bet_amount,
                **calc,
            })

        return sorted(results, key=lambda x: x["expected_value"], reverse=True)


# ==============================================================
# 2. 騎手×馬 相性分析
# ==============================================================

class JockeyHorseAffinity:
    """騎手と馬の組み合わせ実績から相性スコアを算出"""

    def __init__(self):
        self.combo_stats: Dict[str, Dict] = {}

    def build_from_data(self, df: pd.DataFrame):
        """過去データから騎手×馬の統計を構築"""
        if "jockey_id" not in df.columns or "horse_id" not in df.columns:
            return

        for (jid, hid), group in df.groupby(["jockey_id", "horse_id"]):
            key = f"{jid}_{hid}"
            races = len(group)
            if races < 2:
                continue  # 2回以上のデータが必要

            wins = (group["finish_position"] == 1).sum()
            top3 = (group["finish_position"] <= 3).sum()
            avg_pos = group["finish_position"].mean()

            self.combo_stats[key] = {
                "jockey_id": jid,
                "horse_id": hid,
                "races": races,
                "wins": int(wins),
                "top3": int(top3),
                "win_rate": round(wins / races, 3),
                "top3_rate": round(top3 / races, 3),
                "avg_position": round(avg_pos, 2),
            }

    def get_affinity_score(self, jockey_id: str, horse_id: str) -> Dict:
        """相性スコアを取得（0-100）"""
        key = f"{jockey_id}_{horse_id}"
        stats = self.combo_stats.get(key)

        if not stats:
            return {"score": 50, "label": "データなし", "stats": None}

        # スコア算出: 勝率×40 + 複勝率×30 + 経験×15 + 着順×15
        win_score = min(stats["win_rate"] * 200, 40)  # 勝率20%で満点
        top3_score = min(stats["top3_rate"] * 60, 30)  # 複勝率50%で満点
        exp_score = min(stats["races"] / 10 * 15, 15)  # 10走で満点
        pos_score = max(0, (10 - stats["avg_position"]) / 10 * 15)

        score = round(win_score + top3_score + exp_score + pos_score)

        if score >= 80:
            label = "最高相性"
        elif score >= 65:
            label = "好相性"
        elif score >= 50:
            label = "普通"
        elif score >= 35:
            label = "やや不安"
        else:
            label = "相性悪い"

        return {"score": score, "label": label, "stats": stats}

    def get_jockey_best_horses(self, jockey_id: str, top_n: int = 5) -> List[Dict]:
        """騎手の相性が良い馬TOP-N"""
        matches = []
        for key, stats in self.combo_stats.items():
            if stats["jockey_id"] == jockey_id:
                score = self.get_affinity_score(jockey_id, stats["horse_id"])["score"]
                matches.append({**stats, "score": score})
        return sorted(matches, key=lambda x: x["score"], reverse=True)[:top_n]


# ==============================================================
# 3. 妙味検出（バリューベット）
# ==============================================================

class ValueDetector:
    """
    AI予測確率とオッズを比較し、市場が過小評価している馬を検出。
    「妙味」= AIが思う勝率 > オッズが示す勝率 → 賭ける価値あり。
    """

    @staticmethod
    def detect(horses: List[Dict], threshold: float = 1.15) -> List[Dict]:
        """
        妙味馬を検出

        Args:
            horses: [{"horse_number": 1, "prob_win": 0.2, "prob_top3": 0.5, "odds": 8.0}, ...]
            threshold: 期待値がこの値以上なら「妙味あり」（デフォルト1.15 = 15%のエッジ）

        Returns:
            妙味馬リスト（妙味度の高い順）
        """
        values = []
        for h in horses:
            odds = h.get("odds", 0)
            prob_win = h.get("prob_win", 0)
            prob_top3 = h.get("prob_top3", 0)

            if odds <= 0:
                continue

            # オッズが示す暗黙の確率
            implied_prob = 1 / odds

            # 単勝の期待値
            ev_win = prob_win * odds

            # 複勝の概算期待値（オッズの1/3〜1/4で計算）
            est_fukusho_odds = max(1.1, odds * 0.3)
            ev_top3 = prob_top3 * est_fukusho_odds

            # 妙味度 = AI確率 / 暗黙確率
            value_ratio = prob_win / implied_prob if implied_prob > 0 else 0

            # 乖離アラートレベル
            if ev_win >= 2.0:
                alert = "超妙味"
            elif ev_win >= 1.5:
                alert = "大妙味"
            elif ev_win >= threshold:
                alert = "妙味あり"
            else:
                alert = None

            if alert:
                values.append({
                    "horse_number": h.get("horse_number"),
                    "horse_name": h.get("horse_name", ""),
                    "odds": odds,
                    "prob_win": prob_win,
                    "prob_top3": prob_top3,
                    "implied_prob": round(implied_prob, 4),
                    "ev_win": round(ev_win, 3),
                    "ev_top3": round(ev_top3, 3),
                    "value_ratio": round(value_ratio, 2),
                    "alert": alert,
                })

        return sorted(values, key=lambda x: x["ev_win"], reverse=True)


# ==============================================================
# 4. URL指定レース読み込み
# ==============================================================

class RaceImporter:
    """
    netkeiba.comのURLを指定してレース情報を自動取得。
    出走表（shutuba）と結果（result）の両方に対応。
    """

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    @classmethod
    def from_url(cls, url: str) -> Optional[Dict]:
        """
        URLからレース情報を取得

        対応URL:
          https://race.netkeiba.com/race/shutuba.html?race_id=202406050811
          https://db.netkeiba.com/race/202406050811/
        """
        if not HAS_REQUESTS:
            print("⚠️ requestsライブラリが必要: pip install requests beautifulsoup4")
            return None

        race_id = cls._extract_race_id(url)
        if not race_id:
            print(f"⚠️ レースIDを抽出できません: {url}")
            return None

        print(f"📥 レース取得中: {race_id}")

        if "shutuba" in url:
            return cls._fetch_shutuba(race_id)
        else:
            return cls._fetch_result(race_id)

    @staticmethod
    def _extract_race_id(url: str) -> Optional[str]:
        """URLからレースIDを抽出"""
        # race_id=パラメータ
        match = re.search(r"race_id=(\d{12})", url)
        if match:
            return match.group(1)
        # /race/XXXX/ パス
        match = re.search(r"/race/(\d{12})", url)
        if match:
            return match.group(1)
        return None

    @classmethod
    def _fetch_shutuba(cls, race_id: str) -> Optional[Dict]:
        """出走表を取得"""
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        try:
            resp = requests.get(url, headers=cls.HEADERS, timeout=15)
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"⚠️ 取得エラー: {e}")
            return None

        # レース名
        race_name = ""
        name_tag = soup.find("div", class_="RaceName")
        if name_tag:
            race_name = name_tag.get_text(strip=True)

        # レース情報
        race_data = soup.find("div", class_="RaceData01")
        course_info = race_data.get_text(strip=True) if race_data else ""

        horses = []
        table = soup.find("table", class_="Shutuba_Table") or soup.find("table", id="shutuba_table")
        if table:
            for row in table.find_all("tr", class_="HorseList"):
                cols = row.find_all("td")
                if len(cols) < 10:
                    continue

                horse = {
                    "frame_number": int(cols[0].get_text(strip=True) or 0),
                    "horse_number": int(cols[1].get_text(strip=True) or 0),
                    "horse_name": cols[3].get_text(strip=True),
                    "sex_age": cols[4].get_text(strip=True),
                    "weight": cols[5].get_text(strip=True),
                    "jockey_name": cols[6].get_text(strip=True),
                    "trainer_name": cols[7].get_text(strip=True),
                }

                # オッズ取得（あれば）
                odds_tag = row.find("td", class_="Txt_R")
                if odds_tag:
                    try:
                        horse["odds"] = float(odds_tag.get_text(strip=True))
                    except ValueError:
                        horse["odds"] = 0

                horses.append(horse)

        return {
            "race_id": race_id,
            "race_name": race_name,
            "course_info": course_info,
            "horses": horses,
            "type": "shutuba",
        }

    @classmethod
    def _fetch_result(cls, race_id: str) -> Optional[Dict]:
        """レース結果を取得"""
        url = f"https://db.netkeiba.com/race/{race_id}/"
        try:
            resp = requests.get(url, headers=cls.HEADERS, timeout=15)
            resp.encoding = "euc-jp"
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"⚠️ 取得エラー: {e}")
            return None

        # 結果テーブル
        horses = []
        table = soup.find("table", class_="race_table_01")
        if table:
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if len(cols) < 13:
                    continue

                try:
                    pos = int(cols[0].get_text(strip=True))
                except ValueError:
                    pos = 0

                horse_link = cols[3].find("a")
                horse_name = horse_link.get_text(strip=True) if horse_link else ""

                try:
                    odds = float(cols[12].get_text(strip=True))
                except (ValueError, IndexError):
                    odds = 0

                horses.append({
                    "finish_position": pos,
                    "frame_number": int(cols[1].get_text(strip=True) or 0),
                    "horse_number": int(cols[2].get_text(strip=True) or 0),
                    "horse_name": horse_name,
                    "odds": odds,
                })

        # 払い戻し情報
        payouts = cls._parse_payouts(soup)

        return {
            "race_id": race_id,
            "horses": sorted(horses, key=lambda x: x.get("finish_position", 99)),
            "payouts": payouts,
            "type": "result",
        }

    @staticmethod
    def _parse_payouts(soup: BeautifulSoup) -> Dict:
        """払い戻しテーブルを解析"""
        payouts = {}
        payout_table = soup.find("table", class_="pay_table_01")
        if not payout_table:
            return payouts

        for row in payout_table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            bet_type_tag = row.find("th")
            if not bet_type_tag:
                continue

            bet_type = bet_type_tag.get_text(strip=True)
            combo_text = cols[0].get_text(strip=True)
            payout_text = cols[1].get_text(strip=True).replace(",", "").replace("円", "")

            try:
                payout = int(payout_text)
            except ValueError:
                continue

            # 組み合わせをパース
            combo = []
            for num in re.findall(r"\d+", combo_text):
                combo.append(int(num))

            if bet_type not in payouts:
                payouts[bet_type] = []
            payouts[bet_type].append({"combination": combo, "payout": payout})

        return payouts


# ==============================================================
# 5. 統合予測レポート生成
# ==============================================================

def generate_race_report(
    horses: List[Dict],
    bankroll: int = 10000,
    kelly_fraction: float = 0.25,
) -> Dict:
    """
    1レース分の統合分析レポートを生成。
    ダッシュボードUIが読み込むJSON形式で出力。
    """
    kelly = KellyCriterion(fraction=kelly_fraction)
    value_detector = ValueDetector()

    # 妙味検出
    value_horses = value_detector.detect(horses, threshold=1.10)

    # ケリー基準で賭け金算出
    kelly_bets = kelly.optimal_bets(horses, bankroll=bankroll)

    # AI推奨買い目
    recommendations = []

    # 単勝: 期待値最大の馬
    if kelly_bets:
        best = kelly_bets[0]
        recommendations.append({
            "type": "単勝",
            "selections": [best["horse_number"]],
            "amount": best["bet_amount"],
            "reason": f"EV{best['expected_value']:.2f} / ケリー{best['kelly_adjusted_pct']}%",
            "confidence": best["recommendation"],
        })

    # 複勝: 期待値プラスかつ複勝率50%超の馬
    for h in horses:
        if h.get("prob_top3", 0) > 0.5 and h.get("odds", 0) > 2.0:
            calc = kelly.calculate(h["prob_top3"], h["odds"] * 0.3)
            if calc["expected_value"] > 1.0:
                recommendations.append({
                    "type": "複勝",
                    "selections": [h["horse_number"]],
                    "amount": max(100, int(bankroll * 0.03 / 100) * 100),
                    "reason": f"複勝率{h['prob_top3']*100:.0f}%",
                    "confidence": "通常",
                })

    # ワイド: 上位2頭
    sorted_horses = sorted(horses, key=lambda x: x.get("prob_top3", 0), reverse=True)
    if len(sorted_horses) >= 2:
        h1, h2 = sorted_horses[0], sorted_horses[1]
        recommendations.append({
            "type": "ワイド",
            "selections": sorted([h1["horse_number"], h2["horse_number"]]),
            "amount": max(100, int(bankroll * 0.02 / 100) * 100),
            "reason": f"複勝率{h1['prob_top3']*100:.0f}%×{h2['prob_top3']*100:.0f}%",
            "confidence": "通常",
        })

    return {
        "horses": sorted(horses, key=lambda x: x.get("composite_score", 0), reverse=True),
        "value_alerts": value_horses,
        "kelly_bets": kelly_bets,
        "recommendations": recommendations,
        "bankroll": bankroll,
        "total_recommended_bet": sum(r["amount"] for r in recommendations),
    }


if __name__ == "__main__":
    # テスト
    print("=" * 50)
    print("🏇 高度分析エンジン テスト")
    print("=" * 50)

    test_horses = [
        {"horse_number": 1, "horse_name": "サクラフローラ", "prob_win": 0.31, "prob_top3": 0.72, "odds": 2.8},
        {"horse_number": 2, "horse_name": "メイショウハルカ", "prob_win": 0.22, "prob_top3": 0.63, "odds": 4.5},
        {"horse_number": 3, "horse_name": "アドマイヤルナ", "prob_win": 0.16, "prob_top3": 0.56, "odds": 6.2},
        {"horse_number": 4, "horse_name": "ゴールドティアラ", "prob_win": 0.11, "prob_top3": 0.45, "odds": 8.1},
        {"horse_number": 5, "horse_name": "スターライトノヴァ", "prob_win": 0.08, "prob_top3": 0.33, "odds": 12.5},
        {"horse_number": 6, "horse_name": "ウインフェアリー", "prob_win": 0.05, "prob_top3": 0.29, "odds": 15.0},
        {"horse_number": 7, "horse_name": "ロードクイーン", "prob_win": 0.04, "prob_top3": 0.20, "odds": 22.3},
        {"horse_number": 8, "horse_name": "テイエムプリンス", "prob_win": 0.03, "prob_top3": 0.15, "odds": 35.0},
    ]

    report = generate_race_report(test_horses, bankroll=10000)

    print("\n🎯 妙味アラート:")
    for v in report["value_alerts"]:
        print(f"  [{v['alert']}] {v['horse_number']}番 {v['horse_name']} "
              f"EV={v['ev_win']:.2f} (AI{v['prob_win']*100:.0f}% vs 市場{v['implied_prob']*100:.0f}%)")

    print("\n💰 ケリー基準 推奨:")
    for k in report["kelly_bets"]:
        print(f"  {k['horse_number']}番 {k['horse_name']} → ¥{k['bet_amount']:,} "
              f"({k['recommendation']}) EV={k['expected_value']:.2f}")

    print(f"\n📋 推奨買い目 ({len(report['recommendations'])}点 / 計¥{report['total_recommended_bet']:,}):")
    for r in report["recommendations"]:
        sel = "-".join(str(s) for s in r["selections"])
        print(f"  [{r['type']}] {sel} / ¥{r['amount']:,} ({r['reason']})")
