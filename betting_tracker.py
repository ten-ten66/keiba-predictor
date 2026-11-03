"""
競馬予想AI - 予想記録 & 収支管理システム
==========================================
自分の予想を記録し、レース結果と照合して払い戻し金・収支を自動計算する。

機能:
  1. 予想の登録（馬券種別・買い目・金額）
  2. レース結果の取得 & 照合
  3. 払い戻し金の自動計算
  4. 収支履歴のCSV/JSON出力
  5. 統計ダッシュボード用データ生成

馬券種別:
  単勝 / 複勝 / 枠連 / 馬連 / ワイド / 馬単 / 三連複 / 三連単
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np


class BetType(str, Enum):
    """馬券種別"""
    TANSHO = "単勝"      # 1着を当てる
    FUKUSHO = "複勝"     # 3着以内を当てる
    WAKUREN = "枠連"     # 1-2着の枠番（順不同）
    UMAREN = "馬連"      # 1-2着の馬番（順不同）
    WIDE = "ワイド"      # 3着以内の2頭（順不同）
    UMATAN = "馬単"      # 1-2着の馬番（順序あり）
    SANRENPUKU = "三連複"  # 1-2-3着の馬番（順不同）
    SANRENTAN = "三連単"   # 1-2-3着の馬番（順序あり）


@dataclass
class Bet:
    """1つの馬券購入"""
    id: str                        # 一意ID
    race_id: str                   # レースID
    race_name: str                 # レース名
    race_date: str                 # レース日
    venue: str                     # 開催場
    bet_type: str                  # 馬券種別
    selections: List[int]          # 選択馬番（例: [3, 7] = 馬連3-7）
    amount: int                    # 購入金額
    is_ai_recommended: bool = False  # AI推奨かどうか
    memo: str = ""                 # メモ
    # 結果（レース後に更新）
    is_settled: bool = False       # 結果確定済み
    is_hit: bool = False           # 的中したか
    payout_odds: float = 0.0       # 払い戻し倍率
    payout_amount: float = 0.0     # 払い戻し金額
    profit: float = 0.0            # 損益
    created_at: str = ""           # 登録日時
    settled_at: str = ""           # 確定日時


@dataclass
class RaceResult:
    """レース結果"""
    race_id: str
    race_name: str
    race_date: str
    venue: str
    course_type: str
    distance: int
    track_condition: str
    # 着順結果
    finish_order: List[Dict]  # [{"position": 1, "horse_number": 3, "horse_name": "...", "odds": 5.2}, ...]
    # 払い戻し情報
    payouts: Dict[str, List[Dict]]  # {"単勝": [{"combination": [3], "payout": 520}], ...}


class BettingTracker:
    """予想記録 & 収支管理"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.bets_file = self.data_dir / "bets.json"
        self.results_file = self.data_dir / "race_results.json"
        self.history_file = self.data_dir / "betting_history.csv"

        self.bets: List[Bet] = []
        self.results: Dict[str, RaceResult] = {}

        self._load()

    def _load(self):
        """保存済みデータを読み込み（破損ファイル対応）"""
        if self.bets_file.exists():
            try:
                with open(self.bets_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.bets = [Bet(**b) for b in data]
                print(f"📂 {len(self.bets)} 件の予想を読み込み")
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"⚠️ 予想ファイル破損。バックアップ作成して初期化: {e}")
                self.bets_file.rename(self.bets_file.with_suffix(".json.bak"))
                self.bets = []

        if self.results_file.exists():
            try:
                with open(self.results_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.results = {k: RaceResult(**v) for k, v in data.items()}
                print(f"📂 {len(self.results)} 件の結果を読み込み")
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"⚠️ 結果ファイル破損。バックアップ作成して初期化: {e}")
                self.results_file.rename(self.results_file.with_suffix(".json.bak"))
                self.results = {}

    def _save(self):
        """データを保存"""
        with open(self.bets_file, "w", encoding="utf-8") as f:
            json.dump([asdict(b) for b in self.bets], f, ensure_ascii=False, indent=2)

        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in self.results.items()}, f, ensure_ascii=False, indent=2)

    # ============================================================
    # 予想の登録
    # ============================================================

    # 有効な馬券種別
    VALID_BET_TYPES = {"単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"}

    # 馬券種別ごとの必要selections数
    SELECTION_COUNTS = {"単勝": 1, "複勝": 1, "枠連": 2, "馬連": 2, "ワイド": 2, "馬単": 2, "三連複": 3, "三連単": 3}

    def add_bet(
        self,
        race_id: str,
        race_name: str,
        race_date: str,
        venue: str,
        bet_type: str,
        selections: List[int],
        amount: int = 100,
        is_ai_recommended: bool = False,
        memo: str = "",
    ) -> Bet:
        """予想を登録（バリデーション付き）"""
        # バリデーション
        if bet_type not in self.VALID_BET_TYPES:
            raise ValueError(f"不正な馬券種別: '{bet_type}' (有効: {self.VALID_BET_TYPES})")
        if amount <= 0:
            raise ValueError(f"金額は1以上: {amount}")
        if not selections:
            raise ValueError("買い目（selections）が空です")
        expected = self.SELECTION_COUNTS.get(bet_type, 1)
        if len(selections) != expected:
            raise ValueError(f"{bet_type}はselections {expected}個必要（{len(selections)}個指定）")
        if any(s <= 0 for s in selections):
            raise ValueError(f"馬番/枠番は1以上: {selections}")

        bet_id = f"B{len(self.bets)+1:06d}"
        bet = Bet(
            id=bet_id,
            race_id=race_id,
            race_name=race_name,
            race_date=race_date,
            venue=venue,
            bet_type=bet_type,
            selections=selections,
            amount=amount,
            is_ai_recommended=is_ai_recommended,
            memo=memo,
            created_at=datetime.now().isoformat(),
        )
        self.bets.append(bet)
        self._save()

        sel_str = "-".join(str(s) for s in selections)
        print(f"✅ 予想登録: [{bet_type}] {sel_str} / ¥{amount:,} ({race_name})")
        return bet

    def add_bets_bulk(self, bets_data: List[Dict]) -> List[Bet]:
        """複数の予想を一括登録"""
        results = []
        for b in bets_data:
            bet = self.add_bet(**b)
            results.append(bet)
        return results

    def add_ai_recommendations(
        self,
        race_id: str,
        race_name: str,
        race_date: str,
        venue: str,
        predictions: pd.DataFrame,
        budget: int = 1000,
    ) -> List[Bet]:
        """AI予測結果から自動で馬券を生成"""
        bets = []
        race_preds = predictions[predictions["race_id"] == race_id].sort_values("predicted_rank")

        if len(race_preds) < 3:
            print("⚠️ 予測データが不足しています")
            return bets

        top1 = int(race_preds.iloc[0]["horse_number"])
        top2 = int(race_preds.iloc[1]["horse_number"])
        top3 = int(race_preds.iloc[2]["horse_number"])

        unit = budget // 10  # 予算を10分割

        # 単勝: 予測1位
        bets.append(self.add_bet(
            race_id=race_id, race_name=race_name, race_date=race_date, venue=venue,
            bet_type=BetType.TANSHO, selections=[top1], amount=unit * 2,
            is_ai_recommended=True, memo="AI推奨: 予測1位の単勝",
        ))

        # 複勝: 予測1-3位
        for h in [top1, top2, top3]:
            bets.append(self.add_bet(
                race_id=race_id, race_name=race_name, race_date=race_date, venue=venue,
                bet_type=BetType.FUKUSHO, selections=[h], amount=unit,
                is_ai_recommended=True, memo=f"AI推奨: 複勝",
            ))

        # ワイド: 上位3頭BOX（3点）
        from itertools import combinations
        for combo in combinations([top1, top2, top3], 2):
            bets.append(self.add_bet(
                race_id=race_id, race_name=race_name, race_date=race_date, venue=venue,
                bet_type=BetType.WIDE, selections=sorted(combo), amount=unit,
                is_ai_recommended=True, memo="AI推奨: 上位3頭ワイドBOX",
            ))

        # 馬連: 予測1位-2位
        bets.append(self.add_bet(
            race_id=race_id, race_name=race_name, race_date=race_date, venue=venue,
            bet_type=BetType.UMAREN, selections=sorted([top1, top2]), amount=unit * 2,
            is_ai_recommended=True, memo="AI推奨: 上位2頭馬連",
        ))

        return bets

    # ============================================================
    # レース結果の登録 & 照合
    # ============================================================

    def register_result(
        self,
        race_id: str,
        race_name: str,
        race_date: str,
        venue: str,
        course_type: str,
        distance: int,
        track_condition: str,
        finish_order: List[Dict],
        payouts: Dict[str, List[Dict]],
    ) -> RaceResult:
        """レース結果を登録"""
        # 二重登録チェック
        if race_id in self.results:
            already_settled = any(b.is_settled and b.race_id == race_id for b in self.bets)
            if already_settled:
                print(f"⚠️ {race_name}: 結果は登録済み（照合済み馬券あり）。上書きしますが再照合はしません。")
                self.results[race_id] = RaceResult(
                    race_id=race_id, race_name=race_name, race_date=race_date,
                    venue=venue, course_type=course_type, distance=distance,
                    track_condition=track_condition, finish_order=finish_order, payouts=payouts,
                )
                self._save()
                return self.results[race_id]

        result = RaceResult(
            race_id=race_id,
            race_name=race_name,
            race_date=race_date,
            venue=venue,
            course_type=course_type,
            distance=distance,
            track_condition=track_condition,
            finish_order=finish_order,
            payouts=payouts,
        )
        self.results[race_id] = result
        self._save()

        # 該当レースの予想を自動照合
        settled = self.settle_bets(race_id)
        print(f"✅ 結果登録: {race_name} ({len(settled)} 件の予想を照合)")
        return result

    def register_result_from_scraper(self, race_data: Dict) -> Optional[RaceResult]:
        """スクレイパーの取得結果からレース結果を登録"""
        if not race_data or "horses" not in race_data:
            return None

        race_info = race_data["race_info"]
        horses = sorted(race_data["horses"], key=lambda x: x.get("finish_position", 99))

        finish_order = []
        for h in horses:
            if h.get("finish_position", 0) > 0:
                finish_order.append({
                    "position": h["finish_position"],
                    "horse_number": h["horse_number"],
                    "horse_name": h["horse_name"],
                    "frame_number": h.get("frame_number", 0),
                    "odds": h.get("odds", 0),
                    "popularity": h.get("popularity", 0),
                })

        # 払い戻し計算（推定値: 実際はnetkeiba等から取得）
        payouts = self._estimate_payouts(finish_order)

        return self.register_result(
            race_id=race_info["race_id"],
            race_name=race_info["race_name"],
            race_date=race_info["race_date"],
            venue=race_info.get("venue", ""),
            course_type=race_info.get("course_type", ""),
            distance=race_info.get("distance", 0),
            track_condition=race_info.get("track_condition", ""),
            finish_order=finish_order,
            payouts=payouts,
        )

    def _estimate_payouts(self, finish_order: List[Dict]) -> Dict[str, List[Dict]]:
        """払い戻し金を推定計算（実データが無い場合のフォールバック）"""
        if len(finish_order) < 3:
            return {}

        first = finish_order[0]
        second = finish_order[1]
        third = finish_order[2]

        payouts = {}

        # 単勝
        tansho_payout = int(first.get("odds", 1) * 100)
        payouts["単勝"] = [{"combination": [first["horse_number"]], "payout": tansho_payout}]

        # 複勝（概算: 単勝オッズの約1/3〜1/4）
        fukusho_payouts = []
        for h in [first, second, third]:
            odds = h.get("odds", 1)
            fp = max(110, int(odds * 100 * 0.3))
            fukusho_payouts.append({"combination": [h["horse_number"]], "payout": fp})
        payouts["複勝"] = fukusho_payouts

        # 枠連
        payouts["枠連"] = [{
            "combination": sorted([first.get("frame_number", 0), second.get("frame_number", 0)]),
            "payout": int(tansho_payout * 0.8),
        }]

        # 馬連
        umaren_payout = int(tansho_payout * first.get("odds", 1) * 0.4)
        payouts["馬連"] = [{
            "combination": sorted([first["horse_number"], second["horse_number"]]),
            "payout": max(200, umaren_payout),
        }]

        # ワイド（3通り）
        wide_payouts = []
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            h1, h2 = finish_order[i], finish_order[j]
            wp = max(110, int((h1.get("odds", 1) + h2.get("odds", 1)) * 30))
            wide_payouts.append({
                "combination": sorted([h1["horse_number"], h2["horse_number"]]),
                "payout": wp,
            })
        payouts["ワイド"] = wide_payouts

        # 馬単
        payouts["馬単"] = [{
            "combination": [first["horse_number"], second["horse_number"]],
            "payout": max(300, int(umaren_payout * 2)),
        }]

        # 三連複
        sanrenpuku_payout = max(500, int(tansho_payout * 3))
        payouts["三連複"] = [{
            "combination": sorted([first["horse_number"], second["horse_number"], third["horse_number"]]),
            "payout": sanrenpuku_payout,
        }]

        # 三連単
        payouts["三連単"] = [{
            "combination": [first["horse_number"], second["horse_number"], third["horse_number"]],
            "payout": max(1000, int(sanrenpuku_payout * 6)),
        }]

        return payouts

    def settle_bets(self, race_id: str) -> List[Bet]:
        """指定レースの予想を結果と照合"""
        if race_id not in self.results:
            print(f"⚠️ レース結果が未登録: {race_id}")
            return []

        result = self.results[race_id]
        settled = []

        for bet in self.bets:
            if bet.race_id != race_id or bet.is_settled:
                continue

            is_hit, payout_per_100 = self._check_hit(bet, result)

            bet.is_settled = True
            bet.is_hit = is_hit
            bet.payout_odds = payout_per_100 / 100 if payout_per_100 > 0 else 0
            bet.payout_amount = (bet.amount / 100) * payout_per_100 if is_hit else 0
            bet.profit = bet.payout_amount - bet.amount
            bet.settled_at = datetime.now().isoformat()

            settled.append(bet)

        self._save()
        return settled

    def _check_hit(self, bet: Bet, result: RaceResult) -> Tuple[bool, int]:
        """的中判定 & 払い戻し金額（100円あたり）を返す"""
        payouts = result.payouts.get(bet.bet_type, [])
        finish_order = result.finish_order

        if not finish_order:
            return False, 0

        top3_numbers = [h["horse_number"] for h in finish_order[:3]]
        top2_numbers = [h["horse_number"] for h in finish_order[:2]]

        sel = bet.selections

        if bet.bet_type == BetType.TANSHO:
            # 1着の馬番が一致
            if sel[0] == finish_order[0]["horse_number"]:
                for p in payouts:
                    if sel[0] in p["combination"]:
                        return True, p["payout"]
            return False, 0

        elif bet.bet_type == BetType.FUKUSHO:
            # 3着以内に含まれる
            if sel[0] in top3_numbers:
                for p in payouts:
                    if sel[0] in p["combination"]:
                        return True, p["payout"]
            return False, 0

        elif bet.bet_type == BetType.UMAREN:
            # 1-2着の組み合わせ（順不同）
            if sorted(sel) == sorted(top2_numbers):
                for p in payouts:
                    if sorted(p["combination"]) == sorted(sel):
                        return True, p["payout"]
            return False, 0

        elif bet.bet_type == BetType.WIDE:
            # 3着以内の2頭の組み合わせ（順不同）
            if all(s in top3_numbers for s in sel):
                for p in payouts:
                    if sorted(p["combination"]) == sorted(sel):
                        return True, p["payout"]
            return False, 0

        elif bet.bet_type == BetType.UMATAN:
            # 1-2着の順序あり
            if len(sel) >= 2 and sel[0] == finish_order[0]["horse_number"] and sel[1] == finish_order[1]["horse_number"]:
                for p in payouts:
                    if p["combination"] == sel:
                        return True, p["payout"]
            return False, 0

        elif bet.bet_type == BetType.WAKUREN:
            # 1-2着の枠番（順不同）
            top2_frames = sorted([h.get("frame_number", 0) for h in finish_order[:2]])
            if sorted(sel) == top2_frames:
                for p in payouts:
                    if sorted(p["combination"]) == sorted(sel):
                        return True, p["payout"]
            return False, 0

        elif bet.bet_type == BetType.SANRENPUKU:
            # 1-2-3着の馬番（順不同）
            if sorted(sel) == sorted(top3_numbers):
                for p in payouts:
                    if sorted(p["combination"]) == sorted(sel):
                        return True, p["payout"]
            return False, 0

        elif bet.bet_type == BetType.SANRENTAN:
            # 1-2-3着の馬番（順序あり）
            if sel == top3_numbers:
                for p in payouts:
                    if p["combination"] == sel:
                        return True, p["payout"]
            return False, 0

        return False, 0

    # ============================================================
    # 収支・統計
    # ============================================================

    def get_summary(self, race_date: str = None, ai_only: bool = False) -> Dict:
        """収支サマリーを取得"""
        bets = self.bets

        if race_date:
            bets = [b for b in bets if b.race_date == race_date]
        if ai_only:
            bets = [b for b in bets if b.is_ai_recommended]

        settled = [b for b in bets if b.is_settled]
        unsettled = [b for b in bets if not b.is_settled]

        total_bet = sum(b.amount for b in settled)
        total_payout = sum(b.payout_amount for b in settled)
        total_profit = total_payout - total_bet
        n_hits = sum(1 for b in settled if b.is_hit)
        roi = (total_payout / total_bet * 100) if total_bet > 0 else 0

        # 馬券種別ごとの集計
        by_type = {}
        for b in settled:
            if b.bet_type not in by_type:
                by_type[b.bet_type] = {"count": 0, "hits": 0, "bet": 0, "payout": 0}
            by_type[b.bet_type]["count"] += 1
            by_type[b.bet_type]["hits"] += 1 if b.is_hit else 0
            by_type[b.bet_type]["bet"] += b.amount
            by_type[b.bet_type]["payout"] += b.payout_amount

        for bt in by_type.values():
            bt["hit_rate"] = round(bt["hits"] / bt["count"] * 100, 1) if bt["count"] > 0 else 0
            bt["roi"] = round(bt["payout"] / bt["bet"] * 100, 1) if bt["bet"] > 0 else 0
            bt["profit"] = bt["payout"] - bt["bet"]

        # 日別推移
        daily = {}
        for b in settled:
            d = b.race_date
            if d not in daily:
                daily[d] = {"bet": 0, "payout": 0, "hits": 0, "count": 0}
            daily[d]["bet"] += b.amount
            daily[d]["payout"] += b.payout_amount
            daily[d]["hits"] += 1 if b.is_hit else 0
            daily[d]["count"] += 1

        daily_list = []
        cumulative_profit = 0
        for d in sorted(daily.keys()):
            v = daily[d]
            profit = v["payout"] - v["bet"]
            cumulative_profit += profit
            daily_list.append({
                "date": d,
                "bet": v["bet"],
                "payout": round(v["payout"]),
                "profit": round(profit),
                "cumulative_profit": round(cumulative_profit),
                "hit_rate": round(v["hits"] / v["count"] * 100, 1) if v["count"] > 0 else 0,
                "roi": round(v["payout"] / v["bet"] * 100, 1) if v["bet"] > 0 else 0,
            })

        return {
            "overview": {
                "total_bets": len(settled),
                "total_bet_amount": total_bet,
                "total_payout": round(total_payout),
                "total_profit": round(total_profit),
                "roi_percent": round(roi, 1),
                "hit_count": n_hits,
                "hit_rate": round(n_hits / len(settled) * 100, 1) if settled else 0,
                "unsettled_count": len(unsettled),
                "unsettled_amount": sum(b.amount for b in unsettled),
            },
            "by_bet_type": by_type,
            "daily": daily_list,
        }

    def get_race_detail(self, race_id: str) -> Dict:
        """特定レースの予想 vs 結果の詳細"""
        race_bets = [b for b in self.bets if b.race_id == race_id]
        result = self.results.get(race_id)

        bet_details = []
        for b in race_bets:
            sel_str = "-".join(str(s) for s in b.selections)
            bet_details.append({
                "id": b.id,
                "bet_type": b.bet_type,
                "selections": sel_str,
                "amount": b.amount,
                "is_ai": b.is_ai_recommended,
                "is_settled": b.is_settled,
                "is_hit": b.is_hit,
                "payout_odds": b.payout_odds,
                "payout_amount": round(b.payout_amount),
                "profit": round(b.profit),
                "memo": b.memo,
            })

        result_detail = None
        if result:
            result_detail = {
                "race_name": result.race_name,
                "race_date": result.race_date,
                "venue": result.venue,
                "course": f"{result.course_type}{result.distance}m",
                "track": result.track_condition,
                "finish_order": result.finish_order[:5],  # 上位5着
                "payouts": result.payouts,
            }

        total_bet = sum(b.amount for b in race_bets)
        total_payout = sum(b.payout_amount for b in race_bets if b.is_settled)

        return {
            "race_id": race_id,
            "bets": bet_details,
            "result": result_detail,
            "total_bet": total_bet,
            "total_payout": round(total_payout),
            "profit": round(total_payout - total_bet),
        }

    def export_history_csv(self, output_path: str = None) -> str:
        """収支履歴をCSVエクスポート"""
        if output_path is None:
            output_path = str(self.history_file)

        rows = []
        for b in self.bets:
            rows.append({
                "ID": b.id,
                "日付": b.race_date,
                "レース": b.race_name,
                "開催": b.venue,
                "馬券種別": b.bet_type,
                "買い目": "-".join(str(s) for s in b.selections),
                "購入金額": b.amount,
                "AI推奨": "○" if b.is_ai_recommended else "",
                "確定": "○" if b.is_settled else "",
                "的中": "○" if b.is_hit else ("×" if b.is_settled else ""),
                "払戻倍率": b.payout_odds,
                "払戻金額": round(b.payout_amount),
                "損益": round(b.profit),
                "メモ": b.memo,
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"💾 収支履歴CSV: {output_path} ({len(df)} 件)")
        return output_path

    def export_dashboard_json(self, output_path: str = None) -> str:
        """ダッシュボード用JSONエクスポート"""
        if output_path is None:
            output_path = str(self.data_dir / "dashboard_data.json")

        summary = self.get_summary()
        ai_summary = self.get_summary(ai_only=True)

        # 直近のレース詳細
        recent_races = []
        seen_races = set()
        for b in reversed(self.bets):
            if b.race_id not in seen_races and b.is_settled:
                seen_races.add(b.race_id)
                detail = self.get_race_detail(b.race_id)
                recent_races.append(detail)
                if len(recent_races) >= 20:
                    break

        data = {
            "updated_at": datetime.now().isoformat(),
            "overall": summary,
            "ai_only": ai_summary,
            "recent_races": recent_races,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"💾 ダッシュボードJSON: {output_path}")
        return output_path

    def print_summary(self, ai_only: bool = False):
        """コンソールに収支サマリーを表示"""
        s = self.get_summary(ai_only=ai_only)
        o = s["overview"]

        label = "【AI推奨のみ】" if ai_only else "【全体】"
        print(f"\n{'='*50}")
        print(f"📊 収支サマリー {label}")
        print(f"{'='*50}")
        print(f"  購入数:     {o['total_bets']} 件")
        print(f"  的中数:     {o['hit_count']} 件 ({o['hit_rate']}%)")
        print(f"  投資額:     ¥{o['total_bet_amount']:,}")
        print(f"  払戻額:     ¥{o['total_payout']:,}")
        print(f"  損益:       ¥{o['total_profit']:+,}")
        print(f"  回収率:     {o['roi_percent']}%")

        if o['unsettled_count'] > 0:
            print(f"  未確定:     {o['unsettled_count']} 件 (¥{o['unsettled_amount']:,})")

        print(f"\n  【馬券種別ごと】")
        for bt, v in s["by_bet_type"].items():
            profit_str = f"¥{v['profit']:+,.0f}"
            print(f"    {bt:6s}: {v['count']:3d}件 / 的中{v['hit_rate']:5.1f}% / 回収{v['roi']:6.1f}% / {profit_str}")

    def print_race_detail(self, race_id: str):
        """レース別詳細を表示"""
        detail = self.get_race_detail(race_id)

        print(f"\n{'='*50}")
        if detail["result"]:
            r = detail["result"]
            print(f"🏇 {r['race_name']} ({r['race_date']} {r['venue']})")
            print(f"   {r['course']} / {r['track']}")
            print(f"\n  【レース結果】")
            for h in r["finish_order"]:
                marker = "🥇" if h["position"] == 1 else "🥈" if h["position"] == 2 else "🥉" if h["position"] == 3 else "  "
                print(f"    {marker} {h['position']:2d}着: {h['horse_number']:2d}番 {h['horse_name']} (単勝{h.get('odds', '-')}倍)")

        print(f"\n  【自分の予想】")
        for b in detail["bets"]:
            ai = "🤖" if b["is_ai"] else "  "
            hit = "✅" if b["is_hit"] else "❌" if b["is_settled"] else "⏳"
            print(
                f"    {ai} {hit} [{b['bet_type']:4s}] {b['selections']:10s} "
                f"¥{b['amount']:>6,} → ¥{b['payout_amount']:>8,} "
                f"({b['profit']:+,})"
            )

        print(f"\n  投資: ¥{detail['total_bet']:,} / 払戻: ¥{detail['total_payout']:,} / 損益: ¥{detail['profit']:+,}")


# ============================================================
# テスト用: サンプルデータで動作確認
# ============================================================

def demo():
    """デモ実行"""
    print("🏇" * 20)
    print("  予想記録 & 収支管理デモ")
    print("🏇" * 20)

    tracker = BettingTracker(data_dir="data/tracker_demo")

    # --- レース1: 予想を登録 ---
    race1 = {
        "race_id": "202603150611",
        "race_name": "フラワーカップ (G3)",
        "race_date": "2026-03-15",
        "venue": "中山",
    }

    tracker.add_bet(**race1, bet_type="単勝", selections=[3], amount=500, memo="本命サクラフローラ")
    tracker.add_bet(**race1, bet_type="複勝", selections=[3], amount=300)
    tracker.add_bet(**race1, bet_type="複勝", selections=[7], amount=300, is_ai_recommended=True)
    tracker.add_bet(**race1, bet_type="馬連", selections=[3, 7], amount=500, is_ai_recommended=True)
    tracker.add_bet(**race1, bet_type="ワイド", selections=[3, 7], amount=200, is_ai_recommended=True)
    tracker.add_bet(**race1, bet_type="ワイド", selections=[3, 5], amount=200)
    tracker.add_bet(**race1, bet_type="三連複", selections=[3, 5, 7], amount=300, memo="穴狙い")
    tracker.add_bet(**race1, bet_type="三連単", selections=[3, 7, 5], amount=200, memo="勝負！")

    # --- レース1: 結果を登録 ---
    tracker.register_result(
        race_id="202603150611",
        race_name="フラワーカップ (G3)",
        race_date="2026-03-15",
        venue="中山",
        course_type="芝",
        distance=1800,
        track_condition="良",
        finish_order=[
            {"position": 1, "horse_number": 3, "horse_name": "サクラフローラ", "frame_number": 2, "odds": 2.8, "popularity": 1},
            {"position": 2, "horse_number": 7, "horse_name": "ロードクイーン", "frame_number": 4, "odds": 22.3, "popularity": 7},
            {"position": 3, "horse_number": 5, "horse_name": "スターライトノヴァ", "frame_number": 3, "odds": 12.5, "popularity": 5},
            {"position": 4, "horse_number": 2, "horse_name": "メイショウハルカ", "frame_number": 1, "odds": 4.5, "popularity": 2},
            {"position": 5, "horse_number": 1, "horse_name": "アドマイヤルナ", "frame_number": 1, "odds": 6.2, "popularity": 3},
        ],
        payouts={
            "単勝": [{"combination": [3], "payout": 280}],
            "複勝": [
                {"combination": [3], "payout": 130},
                {"combination": [7], "payout": 580},
                {"combination": [5], "payout": 320},
            ],
            "馬連": [{"combination": [3, 7], "payout": 8920}],
            "ワイド": [
                {"combination": [3, 7], "payout": 2450},
                {"combination": [3, 5], "payout": 980},
                {"combination": [5, 7], "payout": 4560},
            ],
            "馬単": [{"combination": [3, 7], "payout": 15840}],
            "三連複": [{"combination": [3, 5, 7], "payout": 32100}],
            "三連単": [{"combination": [3, 7, 5], "payout": 128400}],
        },
    )

    # --- レース2 ---
    race2 = {
        "race_id": "202603150911",
        "race_name": "阪神大賞典 (G2)",
        "race_date": "2026-03-15",
        "venue": "阪神",
    }
    tracker.add_bet(**race2, bet_type="単勝", selections=[1], amount=1000, is_ai_recommended=True)
    tracker.add_bet(**race2, bet_type="馬連", selections=[1, 3], amount=500)
    tracker.add_bet(**race2, bet_type="三連複", selections=[1, 2, 3], amount=300)

    tracker.register_result(
        race_id="202603150911",
        race_name="阪神大賞典 (G2)",
        race_date="2026-03-15",
        venue="阪神",
        course_type="芝",
        distance=3000,
        track_condition="稍重",
        finish_order=[
            {"position": 1, "horse_number": 1, "horse_name": "ディープインパクトJr", "frame_number": 1, "odds": 1.8, "popularity": 1},
            {"position": 2, "horse_number": 4, "horse_name": "ステイゴールドMk2", "frame_number": 4, "odds": 11.0, "popularity": 4},
            {"position": 3, "horse_number": 2, "horse_name": "キングオブステイヤー", "frame_number": 2, "odds": 5.2, "popularity": 2},
        ],
        payouts={
            "単勝": [{"combination": [1], "payout": 180}],
            "複勝": [
                {"combination": [1], "payout": 110},
                {"combination": [4], "payout": 290},
                {"combination": [2], "payout": 160},
            ],
            "馬連": [{"combination": [1, 4], "payout": 2150}],
            "ワイド": [
                {"combination": [1, 4], "payout": 650},
                {"combination": [1, 2], "payout": 230},
                {"combination": [2, 4], "payout": 1420},
            ],
            "三連複": [{"combination": [1, 2, 4], "payout": 4870}],
            "三連単": [{"combination": [1, 4, 2], "payout": 18350}],
        },
    )

    # --- 結果表示 ---
    tracker.print_race_detail("202603150611")
    tracker.print_race_detail("202603150911")
    tracker.print_summary()
    tracker.print_summary(ai_only=True)

    # --- エクスポート ---
    tracker.export_history_csv()
    tracker.export_dashboard_json()


if __name__ == "__main__":
    demo()
