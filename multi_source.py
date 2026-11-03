"""
競馬予想AI - 外部予想サイト統合エンジン
========================================
複数の予想サイト/データソースから情報を集約し、
AI予測と統合してより精度の高い予想を生成する。

対応ソース:
  1. netkeiba.com — 予想印・コンピ指数
  2. 競馬ラボ — 調教評価・パドック評価
  3. JRA-VAN — 指数・データマイニング
  4. 極ウマ（日刊スポーツ） — 記者予想
  5. 競馬新聞各紙 — ◎○▲△☆の印集計

統合方法:
  各ソースの予想を数値化し、重み付きアンサンブルで統合スコアを算出。
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB = True
except ImportError:
    HAS_WEB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ==============================================================
# 予想印の数値変換
# ==============================================================

MARK_SCORES = {
    "◎": 10, "○": 8, "▲": 6, "△": 4, "☆": 3, "×": 2,
    "注": 3, "穴": 5,
}


@dataclass
class ExternalPrediction:
    """外部予想ソースの1件"""
    source_name: str          # ソース名（例: "netkeiba", "日刊スポーツ"）
    source_type: str          # "index" / "mark" / "rank" / "score"
    race_id: str
    horse_number: int
    horse_name: str = ""
    # データ
    mark: str = ""            # 予想印（◎○▲△☆）
    rank: int = 0             # 順位予想
    score: float = 0.0        # 指数・スコア
    comment: str = ""         # コメント
    fetched_at: str = ""


@dataclass
class AggregatedPrediction:
    """統合後の予想"""
    horse_number: int
    horse_name: str
    # 各ソースのデータ
    sources: List[Dict] = field(default_factory=list)
    # 統合スコア
    consensus_score: float = 0.0       # 総合支持度（0-100）
    consensus_rank: int = 0            # 総合順位
    mark_summary: str = ""             # 印の集計（例: "◎3 ○2 ▲1"）
    avg_rank: float = 0.0              # 平均順位
    disagreement: float = 0.0          # ソース間の意見の割れ具合


class PredictionAggregator:
    """外部予想を集約・統合"""

    # ソースごとの重み（信頼度）
    SOURCE_WEIGHTS = {
        "netkeiba_compi": 1.0,        # コンピ指数
        "netkeiba_marks": 0.8,        # 予想家の印
        "nikkansports": 0.7,          # 日刊スポーツ
        "keibalab": 0.6,              # 競馬ラボ
        "sponichi": 0.6,              # スポニチ
        "sankei": 0.5,                # サンケイスポーツ
        "umanity": 0.7,              # ウマニティ
        "ai_prediction": 1.2,         # 自前AI（最重要）
    }

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.predictions: Dict[str, List[ExternalPrediction]] = {}

    def add_prediction(self, pred: ExternalPrediction):
        """予想を追加"""
        key = f"{pred.race_id}_{pred.horse_number}"
        if key not in self.predictions:
            self.predictions[key] = []
        self.predictions[key].append(pred)

    def add_marks_batch(
        self,
        race_id: str,
        source_name: str,
        marks: Dict[int, str],
    ):
        """印を一括登録（{馬番: 印}）"""
        for horse_num, mark in marks.items():
            pred = ExternalPrediction(
                source_name=source_name,
                source_type="mark",
                race_id=race_id,
                horse_number=horse_num,
                mark=mark,
                score=MARK_SCORES.get(mark, 0),
                fetched_at=datetime.now().isoformat(),
            )
            self.add_prediction(pred)

    def add_rankings_batch(
        self,
        race_id: str,
        source_name: str,
        rankings: List[int],
    ):
        """順位予想を一括登録（[1位馬番, 2位馬番, ...]）"""
        for rank, horse_num in enumerate(rankings, 1):
            pred = ExternalPrediction(
                source_name=source_name,
                source_type="rank",
                race_id=race_id,
                horse_number=horse_num,
                rank=rank,
                score=max(0, 20 - rank * 2),
                fetched_at=datetime.now().isoformat(),
            )
            self.add_prediction(pred)

    def add_scores_batch(
        self,
        race_id: str,
        source_name: str,
        scores: Dict[int, float],
    ):
        """指数を一括登録（{馬番: 指数}）"""
        for horse_num, score in scores.items():
            pred = ExternalPrediction(
                source_name=source_name,
                source_type="index",
                race_id=race_id,
                horse_number=horse_num,
                score=score,
                fetched_at=datetime.now().isoformat(),
            )
            self.add_prediction(pred)

    def aggregate(self, race_id: str, horses: List[Dict] = None) -> List[AggregatedPrediction]:
        """レース単位で全ソースを統合"""
        # レースに関連する全予想を集約
        horse_data: Dict[int, List[ExternalPrediction]] = {}
        for key, preds in self.predictions.items():
            if not key.startswith(race_id):
                continue
            for p in preds:
                if p.horse_number not in horse_data:
                    horse_data[p.horse_number] = []
                horse_data[p.horse_number].append(p)

        # 馬名マッピング
        horse_names = {}
        if horses:
            for h in horses:
                horse_names[h.get("horse_number", 0)] = h.get("horse_name", "")

        results = []
        for horse_num, preds in horse_data.items():
            # ソースごとのデータ整理
            sources = []
            weighted_scores = []
            marks_count = {}
            ranks = []

            for p in preds:
                weight = self.SOURCE_WEIGHTS.get(p.source_name, 0.5)

                source_info = {
                    "name": p.source_name,
                    "type": p.source_type,
                    "mark": p.mark,
                    "rank": p.rank,
                    "score": p.score,
                    "comment": p.comment,
                }
                sources.append(source_info)

                # スコア加重
                if p.score > 0:
                    weighted_scores.append(p.score * weight)

                # 印集計
                if p.mark:
                    marks_count[p.mark] = marks_count.get(p.mark, 0) + 1

                # 順位集計
                if p.rank > 0:
                    ranks.append(p.rank)

            # 統合スコア算出
            if weighted_scores:
                consensus_score = sum(weighted_scores) / len(weighted_scores)
                # 0-100に正規化
                consensus_score = min(100, consensus_score * 5)
            else:
                consensus_score = 0

            # 印サマリー
            mark_parts = []
            for mark in ["◎", "○", "▲", "△", "☆"]:
                cnt = marks_count.get(mark, 0)
                if cnt > 0:
                    mark_parts.append(f"{mark}{cnt}")
            mark_summary = " ".join(mark_parts)

            # 意見の割れ具合
            disagreement = 0
            if ranks:
                import numpy as np
                disagreement = round(float(np.std(ranks)), 2)

            results.append(AggregatedPrediction(
                horse_number=horse_num,
                horse_name=horse_names.get(horse_num, f"馬{horse_num}"),
                sources=sources,
                consensus_score=round(consensus_score, 1),
                mark_summary=mark_summary,
                avg_rank=round(sum(ranks) / len(ranks), 1) if ranks else 0,
                disagreement=disagreement,
            ))

        # スコア順にソート & ランク付与
        results.sort(key=lambda x: x.consensus_score, reverse=True)
        for i, r in enumerate(results):
            r.consensus_rank = i + 1

        return results


# ==============================================================
# 外部サイトスクレイパー
# ==============================================================

class ExternalSiteScraper:
    """各予想サイトからデータを取得"""

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    def __init__(self, delay: float = 2.0):
        self.delay = delay

    def _get(self, url: str, encoding: str = "utf-8") -> Optional[BeautifulSoup]:
        if not HAS_WEB:
            return None
        time.sleep(self.delay)
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            resp.encoding = encoding
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"⚠️ 取得エラー: {e}")
            return None

    def fetch_netkeiba_marks(self, race_id: str) -> Dict[int, str]:
        """netkeiba.com: 予想家の印を集計"""
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        soup = self._get(url)
        if not soup:
            return {}

        marks = {}
        # 予想印テーブルを解析（実際のHTML構造に合わせて調整が必要）
        table = soup.find("table", class_="Shutuba_Table")
        if table:
            for row in table.find_all("tr", class_="HorseList"):
                cols = row.find_all("td")
                if len(cols) < 3:
                    continue
                try:
                    horse_num = int(cols[1].get_text(strip=True))
                except ValueError:
                    continue
                # 印を取得
                mark_td = row.find("td", class_="Mark")
                if mark_td:
                    mark_text = mark_td.get_text(strip=True)
                    if mark_text in MARK_SCORES:
                        marks[horse_num] = mark_text

        return marks

    def fetch_netkeiba_compi(self, race_id: str) -> Dict[int, float]:
        """netkeiba.com: コンピ指数を取得"""
        url = f"https://race.netkeiba.com/race/comp.html?race_id={race_id}"
        soup = self._get(url)
        if not soup:
            return {}

        scores = {}
        table = soup.find("table", id="sorttable1")
        if table:
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if len(cols) < 4:
                    continue
                try:
                    horse_num = int(cols[1].get_text(strip=True))
                    score = float(cols[3].get_text(strip=True))
                    scores[horse_num] = score
                except (ValueError, IndexError):
                    continue

        return scores

    def fetch_nikkansports(self, race_id: str) -> Dict[int, str]:
        """日刊スポーツ: 記者予想の印"""
        # 極ウマのURLパターン
        url = f"https://p.nikkansports.com/goku-uma/race/result/{race_id}"
        soup = self._get(url)
        if not soup:
            return {}

        marks = {}
        # 実際のHTML構造に合わせて調整
        return marks


# ==============================================================
# 統合: AI予測 + 外部予想
# ==============================================================

def merge_ai_and_external(
    ai_predictions: List[Dict],
    external_aggregated: List[AggregatedPrediction],
    ai_weight: float = 0.6,
    external_weight: float = 0.4,
) -> List[Dict]:
    """
    自前AIの予測と外部予想サイトの集約結果をマージ。

    Args:
        ai_predictions: AI予測リスト [{horse_number, prob_win, prob_top3, ...}]
        external_aggregated: 外部予想の統合結果
        ai_weight: AIの重み (デフォルト0.6)
        external_weight: 外部予想の重み (デフォルト0.4)

    Returns:
        統合済みの予測リスト
    """
    # 外部予想をdict化
    ext_map = {a.horse_number: a for a in external_aggregated}

    merged = []
    for ai in ai_predictions:
        hn = ai.get("horse_number")
        ext = ext_map.get(hn)

        horse = {**ai}

        if ext:
            # AIスコアと外部スコアを統合
            ai_score = ai.get("composite_score", ai.get("prob_win", 0) * 100)
            ext_score = ext.consensus_score

            horse["final_score"] = round(
                ai_score * ai_weight + ext_score * external_weight, 2
            )
            horse["consensus_marks"] = ext.mark_summary
            horse["consensus_rank"] = ext.consensus_rank
            horse["source_count"] = len(ext.sources)
            horse["disagreement"] = ext.disagreement
            horse["external_sources"] = ext.sources
        else:
            horse["final_score"] = round(
                ai.get("composite_score", ai.get("prob_win", 0) * 100) * ai_weight, 2
            )
            horse["consensus_marks"] = ""
            horse["consensus_rank"] = 0
            horse["source_count"] = 0
            horse["disagreement"] = 0
            horse["external_sources"] = []

        merged.append(horse)

    # 最終スコア順でソート
    merged.sort(key=lambda x: x["final_score"], reverse=True)
    for i, h in enumerate(merged):
        h["final_rank"] = i + 1

    return merged


# ==============================================================
# 手動入力ヘルパー（スクレイパーが使えない場合）
# ==============================================================

def input_newspaper_marks(race_id: str, source_name: str = "スポーツ新聞") -> Dict[int, str]:
    """
    競馬新聞の印を手動入力するヘルパー。

    使い方:
      marks = input_newspaper_marks("202603150611", "日刊スポーツ")
      # 入力例: 3:◎ 7:○ 5:▲ 2:△ 8:☆
    """
    print(f"📰 {source_name} の印を入力")
    print("  形式: 馬番:印 馬番:印 ... (例: 3:◎ 7:○ 5:▲)")
    print("  印: ◎ ○ ▲ △ ☆ × 注 穴")

    line = input("  > ").strip()
    marks = {}
    for pair in line.split():
        parts = pair.split(":")
        if len(parts) == 2:
            try:
                marks[int(parts[0])] = parts[1]
            except ValueError:
                pass

    print(f"  ✅ {len(marks)} 件登録")
    return marks


# ==============================================================
# テスト
# ==============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("🏇 外部予想統合エンジン テスト")
    print("=" * 50)

    agg = PredictionAggregator()

    race_id = "202603150611"

    # 各紙の印をシミュレート
    agg.add_marks_batch(race_id, "日刊スポーツ", {3: "◎", 7: "○", 5: "▲", 2: "△", 8: "☆"})
    agg.add_marks_batch(race_id, "スポニチ", {3: "◎", 2: "○", 7: "▲", 5: "△", 1: "☆"})
    agg.add_marks_batch(race_id, "サンスポ", {3: "◎", 5: "○", 7: "▲", 2: "△", 4: "☆"})
    agg.add_marks_batch(race_id, "東スポ", {7: "◎", 3: "○", 5: "▲", 8: "△", 2: "☆"})

    # コンピ指数シミュレート
    agg.add_scores_batch(race_id, "netkeiba_compi", {
        3: 90, 7: 78, 5: 72, 2: 68, 8: 55, 1: 48, 4: 42, 6: 35,
    })

    # AI予測
    ai_preds = [
        {"horse_number": 3, "horse_name": "サクラフローラ", "prob_win": 0.31, "composite_score": 89},
        {"horse_number": 7, "horse_name": "メイショウハルカ", "prob_win": 0.22, "composite_score": 81},
        {"horse_number": 5, "horse_name": "アドマイヤルナ", "prob_win": 0.16, "composite_score": 74},
        {"horse_number": 2, "horse_name": "ゴールドティアラ", "prob_win": 0.11, "composite_score": 65},
        {"horse_number": 8, "horse_name": "スターライトノヴァ", "prob_win": 0.08, "composite_score": 55},
        {"horse_number": 1, "horse_name": "ウインフェアリー", "prob_win": 0.05, "composite_score": 48},
        {"horse_number": 4, "horse_name": "ロードクイーン", "prob_win": 0.04, "composite_score": 38},
        {"horse_number": 6, "horse_name": "テイエムプリンス", "prob_win": 0.02, "composite_score": 30},
    ]

    # 外部予想を集約
    horses = [{"horse_number": h["horse_number"], "horse_name": h["horse_name"]} for h in ai_preds]
    aggregated = agg.aggregate(race_id, horses)

    print("\n📊 外部予想集約結果:")
    for a in aggregated:
        print(f"  {a.consensus_rank}位: {a.horse_number}番 {a.horse_name} "
              f"(スコア{a.consensus_score} / 印: {a.mark_summary})")

    # AI + 外部を統合
    merged = merge_ai_and_external(ai_preds, aggregated)

    print("\n🏆 最終統合結果:")
    for h in merged:
        marks = h.get("consensus_marks", "")
        print(f"  {h['final_rank']}位: {h['horse_number']}番 {h['horse_name']} "
              f"(最終{h['final_score']:.1f} / AI{h['composite_score']} / 印: {marks})")
