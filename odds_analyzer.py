"""
競馬予想AI - オッズ分析エンジン
================================
オッズの変動パターンを分析し、内部情報の兆候や妙味を検出。

機能:
  1. オッズ変動率の計算（朝一→最終）
  2. 異常変動アラート（急に売れた馬の検出）
  3. 複勝/単勝オッズ比分析（プロ資金の流入指標）
  4. オッズヒストリー（時系列変動）
  5. 市場効率性分析（オッズと実際の勝率の乖離）
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class OddsAnalyzer:
    """オッズの変動・異常を分析"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.history: Dict[str, List[Dict]] = {}  # race_id -> [{time, odds:{horse_num: odds}}]

    def record_snapshot(self, race_id: str, odds: Dict[int, float]):
        """オッズのスナップショットを記録"""
        if race_id not in self.history:
            self.history[race_id] = []

        self.history[race_id].append({
            "time": datetime.now().isoformat(),
            "odds": odds,
        })

        # ファイルにも保存
        path = self.data_dir / f"odds_history_{race_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history[race_id], f, ensure_ascii=False)

    def load_history(self, race_id: str) -> List[Dict]:
        """保存済みオッズ履歴を読み込み"""
        path = self.data_dir / f"odds_history_{race_id}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.history[race_id] = json.load(f)
        return self.history.get(race_id, [])

    def analyze_movement(self, race_id: str) -> List[Dict]:
        """オッズ変動分析"""
        snapshots = self.history.get(race_id, [])
        if len(snapshots) < 2:
            return []

        first = snapshots[0]["odds"]
        last = snapshots[-1]["odds"]

        results = []
        for horse_num_str, final_odds in last.items():
            horse_num = int(horse_num_str) if isinstance(horse_num_str, str) else horse_num_str
            initial_odds = first.get(str(horse_num), first.get(horse_num, 0))

            if initial_odds <= 0 or final_odds <= 0:
                continue

            change_rate = (final_odds - initial_odds) / initial_odds
            change_pct = change_rate * 100

            # アラート判定
            alert = None
            if change_pct <= -30:
                alert = "急騰（大量買い）"
            elif change_pct <= -15:
                alert = "注意（買い集中）"
            elif change_pct >= 50:
                alert = "人気落ち"

            # 時系列の変動パターン
            trend = []
            for snap in snapshots:
                o = snap["odds"].get(str(horse_num), snap["odds"].get(horse_num, 0))
                trend.append(o)

            # 変動の速度（最後の3スナップショットの変化量）
            recent_trend = "安定"
            if len(trend) >= 3:
                recent_changes = [trend[i] - trend[i-1] for i in range(-2, 0)]
                if all(c < -0.3 for c in recent_changes):
                    recent_trend = "急降下中"
                elif all(c > 0.3 for c in recent_changes):
                    recent_trend = "上昇中"

            results.append({
                "horse_number": horse_num,
                "initial_odds": round(initial_odds, 1),
                "final_odds": round(final_odds, 1),
                "change_pct": round(change_pct, 1),
                "alert": alert,
                "recent_trend": recent_trend,
                "snapshots": len(snapshots),
                "odds_history": trend,
            })

        return sorted(results, key=lambda x: x["change_pct"])

    def detect_anomalies(self, race_id: str, threshold_pct: float = -20) -> List[Dict]:
        """異常オッズ変動を検出"""
        movements = self.analyze_movement(race_id)
        return [m for m in movements if m["change_pct"] <= threshold_pct]

    def market_efficiency(self, df: pd.DataFrame) -> Dict:
        """
        市場効率性分析: オッズから算出される確率と実際の勝率を比較。

        完全効率市場なら implied_prob == actual_win_rate。
        乖離があれば、そこに妙味がある。
        """
        if "odds" not in df.columns or "finish_position" not in df.columns:
            return {}

        # オッズ帯ごとに実際の勝率を集計
        df = df.copy()
        df["implied_prob"] = 1 / df["odds"].clip(lower=1)

        bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
        labels = ["1-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-50%", "50%+"]
        df["prob_bin"] = pd.cut(df["implied_prob"], bins=bins, labels=labels)

        result = {}
        for bin_label, group in df.groupby("prob_bin", observed=True):
            actual_win_rate = (group["finish_position"] == 1).mean()
            implied_avg = group["implied_prob"].mean()
            n = len(group)

            # 効率性比率: actual / implied （1.0 = 完全効率）
            efficiency = actual_win_rate / implied_avg if implied_avg > 0 else 0

            result[str(bin_label)] = {
                "count": n,
                "implied_prob_avg": round(implied_avg * 100, 2),
                "actual_win_rate": round(actual_win_rate * 100, 2),
                "efficiency_ratio": round(efficiency, 3),
                "edge": round((actual_win_rate - implied_avg) * 100, 2),
            }

        return result

    def favorite_longshot_bias(self, df: pd.DataFrame) -> Dict:
        """
        人気薄バイアス分析:
        一般に人気馬は過大評価、穴馬は過小評価される傾向がある。
        この傾向をデータで確認。
        """
        if "popularity" not in df.columns:
            return {}

        result = {}
        for pop, group in df.groupby("popularity"):
            if pop > 18:
                break
            n = len(group)
            if n < 10:
                continue
            win_rate = (group["finish_position"] == 1).mean()
            avg_odds = group["odds"].mean()
            implied = 1 / avg_odds if avg_odds > 0 else 0
            roi = win_rate * avg_odds

            result[int(pop)] = {
                "count": n,
                "win_rate": round(win_rate * 100, 2),
                "avg_odds": round(avg_odds, 1),
                "implied_prob": round(implied * 100, 2),
                "roi_per_100": round(roi * 100, 1),
            }

        return result


if __name__ == "__main__":
    print("=" * 50)
    print("📊 オッズ分析テスト")
    print("=" * 50)

    analyzer = OddsAnalyzer(data_dir="data")

    # オッズスナップショットのシミュレート
    analyzer.record_snapshot("TEST001", {1: 3.5, 2: 5.0, 3: 8.2, 4: 12.0, 5: 25.0})
    analyzer.record_snapshot("TEST001", {1: 3.2, 2: 5.5, 3: 6.8, 4: 11.0, 5: 30.0})
    analyzer.record_snapshot("TEST001", {1: 2.8, 2: 6.0, 3: 5.2, 4: 10.5, 5: 35.0})

    print("\n📈 オッズ変動:")
    movements = analyzer.analyze_movement("TEST001")
    for m in movements:
        alert_str = f" ⚠️ {m['alert']}" if m["alert"] else ""
        print(f"  {m['horse_number']}番: {m['initial_odds']}→{m['final_odds']} ({m['change_pct']:+.1f}%) {m['recent_trend']}{alert_str}")

    print("\n🚨 異常検出:")
    anomalies = analyzer.detect_anomalies("TEST001")
    for a in anomalies:
        print(f"  {a['horse_number']}番: {a['change_pct']:+.1f}% → {a['alert']}")

    # 市場効率性テスト
    from data_processor import SampleDataGenerator, RaceDataProcessor
    raw = SampleDataGenerator().generate(n_races=500)
    proc = RaceDataProcessor()
    proc.raw_data = raw
    df = proc.preprocess()

    print("\n📊 市場効率性:")
    eff = analyzer.market_efficiency(df)
    for k, v in eff.items():
        print(f"  {k}: 暗黙{v['implied_prob_avg']}% → 実際{v['actual_win_rate']}% (効率{v['efficiency_ratio']})")

    print("\n📊 人気薄バイアス:")
    bias = analyzer.favorite_longshot_bias(df)
    for pop, v in list(bias.items())[:8]:
        roi_str = f"{'🟢' if v['roi_per_100'] >= 100 else '🔴'} ROI {v['roi_per_100']}%"
        print(f"  {pop}番人気: 勝率{v['win_rate']}% / 平均{v['avg_odds']}倍 / {roi_str}")
