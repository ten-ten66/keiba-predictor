"""
競馬予想AI - バックテスト基盤
================================
「もし過去にこの戦略で買っていたら？」を高速シミュレーション。

機能:
  1. 任意の期間・戦略でバックテスト
  2. 馬券種別ごとの回収率計算
  3. ドローダウン分析
  4. 月別/距離別/場別の成績内訳
  5. モンテカルロシミュレーション（三連系の組み合わせ期待値）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass, field
from itertools import combinations, permutations
import json


@dataclass
class BacktestResult:
    """バックテスト結果"""
    strategy_name: str
    period: str
    total_races: int
    total_bets: int
    total_bet_amount: int
    total_return: float
    profit: float
    roi_percent: float
    hit_count: int
    hit_rate: float
    max_drawdown: float
    max_drawdown_pct: float
    best_month: str
    worst_month: str
    monthly_results: List[Dict] = field(default_factory=list)
    venue_results: Dict[str, Dict] = field(default_factory=dict)
    distance_results: Dict[str, Dict] = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)


class Backtester:
    """バックテストエンジン"""

    def __init__(self, df: pd.DataFrame, predictions: pd.DataFrame, payouts_path: str = "data/payouts.csv"):
        """
        Args:
            df: 実績データ（race_id, horse_number, finish_position, odds等）
            predictions: 予測データ（race_id, horse_number, predicted_rank, prob_win, prob_top3等）
            payouts_path: 払戻CSVパス（あれば実配当を使用）
        """
        self.df = df.copy()
        self.predictions = predictions.copy()

        # マージ（存在するカラムのみ使用）
        merge_cols = ["race_id", "horse_number", "finish_position"]
        optional_cols = ["odds", "race_date", "venue", "distance", "course_type"]
        for c in optional_cols:
            if c in df.columns:
                merge_cols.append(c)

        self.merged = predictions.merge(
            df[merge_cols].drop_duplicates(),
            on=["race_id", "horse_number"],
            how="inner",
        )

        # デフォルト値
        if "odds" not in self.merged.columns:
            self.merged["odds"] = 5.0
        if "venue" not in self.merged.columns:
            self.merged["venue"] = ""
        if "distance" not in self.merged.columns:
            self.merged["distance"] = 0

        if "race_date" in self.merged.columns:
            self.merged["race_date"] = pd.to_datetime(self.merged["race_date"], errors="coerce", format="mixed")

        # 払戻データ読み込み（実配当をバックテストに使用）
        self.payouts = {}
        import os
        if os.path.exists(payouts_path):
            pay_df = pd.read_csv(payouts_path)
            pay_df["race_id"] = pay_df["race_id"].astype(str)
            for _, row in pay_df.iterrows():
                key = (str(row["race_id"]), row["bet_type"], str(row["combination"]))
                self.payouts[key] = row["payout"]
            print(f"  払戻データ読込: {len(self.payouts):,} 件")

    def run(
        self,
        strategy: str = "top1_win",
        bet_amount: int = 100,
        start_date: str = None,
        end_date: str = None,
        bankroll: int = 100000,
        kelly_fraction: float = 0.25,
    ) -> BacktestResult:
        """バックテスト実行"""
        data = self.merged.copy()

        if start_date:
            data = data[data["race_date"] >= pd.Timestamp(start_date)]
        if end_date:
            data = data[data["race_date"] <= pd.Timestamp(end_date)]

        period = f"{data['race_date'].min().date()} ~ {data['race_date'].max().date()}" if len(data) > 0 else "N/A"

        # 戦略ごとのシミュレーション
        if strategy == "top1_win":
            results = self._strategy_top1_win(data, bet_amount)
        elif strategy == "top3_wide_box":
            results = self._strategy_top3_wide_box(data, bet_amount)
        elif strategy == "value_bet":
            results = self._strategy_value_bet(data, bet_amount)
        elif strategy == "kelly":
            results = self._strategy_kelly(data, bankroll, kelly_fraction)
        elif strategy == "top2_umaren":
            results = self._strategy_top2_umaren(data, bet_amount)
        elif strategy == "top3_sanrenpuku":
            results = self._strategy_top3_sanrenpuku(data, bet_amount)
        elif strategy == "fukusho_tengarashi":
            results = self._strategy_fukusho_tengarashi(data, bet_amount)
        elif strategy == "formation":
            results = self._strategy_formation(data, bet_amount)
        elif strategy == "wide_1_nagashi":
            results = self._strategy_wide_1_nagashi(data, bet_amount)
        else:
            raise ValueError(f"未知の戦略: {strategy}")

        return self._compile_results(strategy, period, results, data)

    def run_all_strategies(self, bet_amount: int = 100, bankroll: int = 100000) -> List[BacktestResult]:
        """全戦略を比較"""
        strategies = ["top1_win", "top3_wide_box", "value_bet", "kelly", "top2_umaren", "top3_sanrenpuku", "fukusho_tengarashi", "formation", "wide_1_nagashi"]
        results = []
        for s in strategies:
            try:
                r = self.run(strategy=s, bet_amount=bet_amount, bankroll=bankroll)
                results.append(r)
            except Exception as e:
                print(f"  ⚠️ {s}: {e}")
        return results

    # ====== 戦略実装 ======

    def _strategy_top1_win(self, data, bet_amount) -> List[Dict]:
        """予測1位に単勝"""
        bets = []
        for race_id, group in data.groupby("race_id"):
            top = group.sort_values("predicted_rank").iloc[0]
            hit = top["finish_position"] == 1
            if hit:
                real_payout = self._lookup_payout(race_id, "単勝", [int(top["horse_number"])])
                if real_payout:
                    ret = bet_amount / 100 * real_payout
                else:
                    ret = bet_amount * top["odds"]
            else:
                ret = 0
            bets.append({"race_id": race_id, "bet": bet_amount, "return": ret, "hit": hit, "date": top.get("race_date"), "venue": top.get("venue", ""), "distance": top.get("distance", 0)})
        return bets

    def _lookup_payout(self, race_id, bet_type, nums) -> Optional[int]:
        """払戻データから実配当を検索"""
        race_id_str = str(race_id)
        # ワイド: "3 - 5", 馬連: "3 - 5", 三連複: "2 - 3 - 5"
        combo_str = " - ".join(str(n) for n in sorted(nums))
        key = (race_id_str, bet_type, combo_str)
        return self.payouts.get(key)

    def _strategy_top3_wide_box(self, data, bet_amount) -> List[Dict]:
        """予測上位3頭のワイドBOX（3点）"""
        bets = []
        for race_id, group in data.groupby("race_id"):
            tops = group.sort_values("predicted_rank").head(3)
            if len(tops) < 3:
                continue
            top_nums = tops["horse_number"].tolist()
            top3_actual = set(group[group["finish_position"] <= 3]["horse_number"].tolist())

            total_bet = 0
            total_ret = 0
            hit = False
            for combo in combinations(top_nums, 2):
                total_bet += bet_amount
                if set(combo).issubset(top3_actual):
                    # 実配当を検索、なければ概算
                    real_payout = self._lookup_payout(race_id, "ワイド", combo)
                    if real_payout:
                        total_ret += bet_amount / 100 * real_payout
                    else:
                        h1_odds = group[group["horse_number"] == combo[0]]["odds"].values[0] if len(group[group["horse_number"] == combo[0]]) > 0 else 1
                        h2_odds = group[group["horse_number"] == combo[1]]["odds"].values[0] if len(group[group["horse_number"] == combo[1]]) > 0 else 1
                        total_ret += bet_amount / 100 * max(110, (h1_odds + h2_odds) * 30)
                    hit = True

            bets.append({"race_id": race_id, "bet": total_bet, "return": total_ret, "hit": hit, "date": tops.iloc[0].get("race_date"), "venue": tops.iloc[0].get("venue", ""), "distance": tops.iloc[0].get("distance", 0)})
        return bets

    def _strategy_value_bet(self, data, bet_amount) -> List[Dict]:
        """期待値1.2倍超の馬のみ単勝"""
        bets = []
        for race_id, group in data.groupby("race_id"):
            for _, row in group.iterrows():
                ev = row.get("prob_win", 0) * row.get("odds", 0)
                if ev > 1.2:
                    hit = row["finish_position"] == 1
                    ret = bet_amount * row["odds"] if hit else 0
                    bets.append({"race_id": race_id, "bet": bet_amount, "return": ret, "hit": hit, "date": row.get("race_date"), "venue": row.get("venue", ""), "distance": row.get("distance", 0)})
        return bets

    def _strategy_kelly(self, data, bankroll, fraction) -> List[Dict]:
        """ケリー基準で賭け金を調整（バンクロール上限あり）"""
        bets = []
        current_bankroll = bankroll
        max_bankroll = bankroll * 1000  # 初期資金の1000倍を上限
        max_bet = bankroll * 10  # 1回の賭け上限
        for race_id, group in data.groupby("race_id"):
            if current_bankroll <= 0:
                break
            top = group.sort_values("predicted_rank").iloc[0]
            prob = top.get("prob_win", 0)
            odds = top.get("odds", 0)
            if odds <= 1 or prob <= 0:
                continue
            b = odds - 1
            q = 1 - prob
            kelly_pct = max(0, (b * prob - q) / b) * fraction
            bet = max(100, min(int(current_bankroll * kelly_pct / 100) * 100, int(current_bankroll * 0.1), max_bet))
            if bet < 100:
                continue
            hit = top["finish_position"] == 1
            ret = bet * odds if hit else 0
            current_bankroll = min(current_bankroll + ret - bet, max_bankroll)
            bets.append({"race_id": race_id, "bet": bet, "return": ret, "hit": hit, "date": top.get("race_date"), "venue": top.get("venue", ""), "distance": top.get("distance", 0), "bankroll": current_bankroll})
        return bets

    def _strategy_top2_umaren(self, data, bet_amount) -> List[Dict]:
        """予測上位2頭の馬連"""
        bets = []
        for race_id, group in data.groupby("race_id"):
            tops = group.sort_values("predicted_rank").head(2)
            if len(tops) < 2:
                continue
            top_nums = set(tops["horse_number"].tolist())
            actual_top2 = set(group[group["finish_position"] <= 2]["horse_number"].tolist())
            hit = top_nums == actual_top2
            if hit:
                real_payout = self._lookup_payout(race_id, "馬連", top_nums)
                if real_payout:
                    ret = bet_amount / 100 * real_payout
                else:
                    o1 = tops.iloc[0]["odds"]
                    o2 = tops.iloc[1]["odds"]
                    ret = bet_amount / 100 * max(200, o1 * o2 * 40)
            else:
                ret = 0
            bets.append({"race_id": race_id, "bet": bet_amount, "return": ret, "hit": hit, "date": tops.iloc[0].get("race_date"), "venue": tops.iloc[0].get("venue", ""), "distance": tops.iloc[0].get("distance", 0)})
        return bets

    def _strategy_top3_sanrenpuku(self, data, bet_amount) -> List[Dict]:
        """予測上位3頭の三連複"""
        bets = []
        for race_id, group in data.groupby("race_id"):
            tops = group.sort_values("predicted_rank").head(3)
            if len(tops) < 3:
                continue
            top_nums = set(tops["horse_number"].tolist())
            actual_top3 = set(group[group["finish_position"] <= 3]["horse_number"].tolist())
            hit = top_nums == actual_top3
            if hit:
                real_payout = self._lookup_payout(race_id, "三連複", top_nums)
                if real_payout:
                    ret = bet_amount / 100 * real_payout
                else:
                    avg_odds = tops["odds"].mean()
                    ret = bet_amount / 100 * max(500, avg_odds ** 2 * 20)
            else:
                ret = 0
            bets.append({"race_id": race_id, "bet": bet_amount, "return": ret, "hit": hit, "date": tops.iloc[0].get("race_date"), "venue": tops.iloc[0].get("venue", ""), "distance": tops.iloc[0].get("distance", 0)})
        return bets

    def _strategy_fukusho_tengarashi(self, data, bet_amount, max_consecutive: int = 5) -> List[Dict]:
        """複勝転がし: 予測1位に複勝で賭け、的中なら配当全額を次レースへ転がす。
        連続的中がmax_consecutiveに達するか、外れたらリセット。"""
        bets = []
        # レースを日付順に並べる
        race_ids = data.groupby("race_id").first().sort_values("race_date").index.tolist() if "race_date" in data.columns else data["race_id"].unique().tolist()

        current_stake = bet_amount
        consecutive_wins = 0
        streak_initial_bet = bet_amount  # 転がし開始時の元賭け金を記録

        for race_id in race_ids:
            group = data[data["race_id"] == race_id]
            if group.empty:
                continue
            top = group.sort_values("predicted_rank").iloc[0]
            hit = top["finish_position"] <= 3  # 複勝 = 3着以内

            if hit:
                # 複勝配当: 実配当を検索、なければオッズ * 0.25 で概算
                real_payout = self._lookup_payout(race_id, "複勝", [int(top["horse_number"])])
                if real_payout:
                    ret = current_stake / 100 * real_payout
                else:
                    place_odds = max(1.1, top["odds"] * 0.25)
                    ret = current_stake * place_odds
                consecutive_wins += 1
                bets.append({
                    "race_id": race_id, "bet": current_stake, "return": ret, "hit": True,
                    "date": top.get("race_date"), "venue": top.get("venue", ""), "distance": top.get("distance", 0),
                })
                if consecutive_wins >= max_consecutive:
                    # 目標連勝に到達、利確してリセット
                    current_stake = bet_amount
                    consecutive_wins = 0
                else:
                    # 配当を次のレースに転がす
                    current_stake = ret
            else:
                # 外れ: 転がし分を全額失う
                bets.append({
                    "race_id": race_id, "bet": current_stake, "return": 0, "hit": False,
                    "date": top.get("race_date"), "venue": top.get("venue", ""), "distance": top.get("distance", 0),
                })
                current_stake = bet_amount
                consecutive_wins = 0
        return bets

    def _strategy_formation(self, data, bet_amount) -> List[Dict]:
        """三連単フォーメーション: 1着に予測1-2位、2着に予測1-4位、3着に予測1-6位。
        同一馬の重複を除く全組み合わせを購入。"""
        bets = []
        for race_id, group in data.groupby("race_id"):
            ranked = group.sort_values("predicted_rank")
            if len(ranked) < 6:
                continue

            first_candidates = ranked.head(2)["horse_number"].tolist()   # 1着候補: 予測1-2位
            second_candidates = ranked.head(4)["horse_number"].tolist()  # 2着候補: 予測1-4位
            third_candidates = ranked.head(6)["horse_number"].tolist()   # 3着候補: 予測1-6位

            # 実際の着順上位3頭の馬番
            actual_top3 = group.nsmallest(3, "finish_position")
            if len(actual_top3) < 3:
                continue
            actual_1st = actual_top3.iloc[0]["horse_number"]
            actual_2nd = actual_top3.iloc[1]["horse_number"]
            actual_3rd = actual_top3.iloc[2]["horse_number"]

            total_bet = 0
            total_ret = 0
            hit = False

            for h1 in first_candidates:
                for h2 in second_candidates:
                    if h2 == h1:
                        continue
                    for h3 in third_candidates:
                        if h3 == h1 or h3 == h2:
                            continue
                        total_bet += bet_amount
                        # この組み合わせが的中か判定
                        if h1 == actual_1st and h2 == actual_2nd and h3 == actual_3rd:
                            real_payout = self._lookup_payout(race_id, "三連単", [int(h1), int(h2), int(h3)])
                            if real_payout is None:
                                # 順番付きキーでも試す: "h1 → h2 → h3"
                                combo_str = f"{int(h1)} → {int(h2)} → {int(h3)}"
                                key = (str(race_id), "三連単", combo_str)
                                real_payout = self.payouts.get(key)
                            if real_payout:
                                total_ret += bet_amount / 100 * real_payout
                            else:
                                # 概算: 上位馬のオッズの積をベースに
                                o1 = group[group["horse_number"] == h1]["odds"].values[0] if len(group[group["horse_number"] == h1]) > 0 else 5
                                o2 = group[group["horse_number"] == h2]["odds"].values[0] if len(group[group["horse_number"] == h2]) > 0 else 5
                                o3 = group[group["horse_number"] == h3]["odds"].values[0] if len(group[group["horse_number"] == h3]) > 0 else 5
                                total_ret += bet_amount / 100 * max(1000, o1 * o2 * o3 * 10)
                            hit = True

            if total_bet > 0:
                bets.append({
                    "race_id": race_id, "bet": total_bet, "return": total_ret, "hit": hit,
                    "date": ranked.iloc[0].get("race_date"), "venue": ranked.iloc[0].get("venue", ""),
                    "distance": ranked.iloc[0].get("distance", 0),
                })
        return bets

    def _strategy_wide_1_nagashi(self, data, bet_amount) -> List[Dict]:
        """ワイド1頭流し: 予測1位を軸に、予測2-5位との4点ワイド"""
        bets = []
        for race_id, group in data.groupby("race_id"):
            ranked = group.sort_values("predicted_rank")
            if len(ranked) < 5:
                continue

            axis_horse = ranked.iloc[0]["horse_number"]  # 軸: 予測1位
            partner_horses = ranked.iloc[1:5]["horse_number"].tolist()  # 相手: 予測2-5位

            actual_top3 = set(group[group["finish_position"] <= 3]["horse_number"].tolist())

            total_bet = 0
            total_ret = 0
            hit = False

            for partner in partner_horses:
                total_bet += bet_amount
                combo = (axis_horse, partner)
                if set(combo).issubset(actual_top3):
                    real_payout = self._lookup_payout(race_id, "ワイド", combo)
                    if real_payout:
                        total_ret += bet_amount / 100 * real_payout
                    else:
                        h1_odds = group[group["horse_number"] == axis_horse]["odds"].values[0] if len(group[group["horse_number"] == axis_horse]) > 0 else 1
                        h2_odds = group[group["horse_number"] == partner]["odds"].values[0] if len(group[group["horse_number"] == partner]) > 0 else 1
                        total_ret += bet_amount / 100 * max(110, (h1_odds + h2_odds) * 30)
                    hit = True

            bets.append({
                "race_id": race_id, "bet": total_bet, "return": total_ret, "hit": hit,
                "date": ranked.iloc[0].get("race_date"), "venue": ranked.iloc[0].get("venue", ""),
                "distance": ranked.iloc[0].get("distance", 0),
            })
        return bets

    # ====== 結果集計 ======

    def _compile_results(self, strategy, period, bets, data) -> BacktestResult:
        if not bets:
            return BacktestResult(strategy_name=strategy, period=period, total_races=0, total_bets=0, total_bet_amount=0, total_return=0, profit=0, roi_percent=0, hit_count=0, hit_rate=0, max_drawdown=0, max_drawdown_pct=0, best_month="", worst_month="")

        total_bet = sum(b["bet"] for b in bets)
        total_ret = sum(b["return"] for b in bets)
        hits = sum(1 for b in bets if b["hit"])

        # エクイティカーブ & ドローダウン
        equity = []
        cumulative = 0
        peak = 0
        max_dd = 0
        for b in bets:
            cumulative += b["return"] - b["bet"]
            equity.append(cumulative)
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        # 月別
        monthly = {}
        for b in bets:
            dt = b.get("date")
            if dt is not None and hasattr(dt, "strftime"):
                m = dt.strftime("%Y-%m")
            else:
                m = "unknown"
            if m not in monthly:
                monthly[m] = {"bet": 0, "return": 0, "hits": 0, "count": 0}
            monthly[m]["bet"] += b["bet"]
            monthly[m]["return"] += b["return"]
            monthly[m]["hits"] += 1 if b["hit"] else 0
            monthly[m]["count"] += 1

        monthly_list = []
        for m in sorted(monthly.keys()):
            v = monthly[m]
            roi = v["return"] / v["bet"] * 100 if v["bet"] > 0 else 0
            monthly_list.append({"month": m, "bet": v["bet"], "return": round(v["return"]), "profit": round(v["return"] - v["bet"]), "roi": round(roi, 1), "hit_rate": round(v["hits"] / v["count"] * 100, 1) if v["count"] > 0 else 0})

        best_m = max(monthly_list, key=lambda x: x["profit"])["month"] if monthly_list else ""
        worst_m = min(monthly_list, key=lambda x: x["profit"])["month"] if monthly_list else ""

        # 場別
        venue_results = {}
        for b in bets:
            v = b.get("venue", "不明")
            if v not in venue_results:
                venue_results[v] = {"bet": 0, "return": 0, "count": 0, "hits": 0}
            venue_results[v]["bet"] += b["bet"]
            venue_results[v]["return"] += b["return"]
            venue_results[v]["count"] += 1
            venue_results[v]["hits"] += 1 if b["hit"] else 0
        for v in venue_results:
            d = venue_results[v]
            d["roi"] = round(d["return"] / d["bet"] * 100, 1) if d["bet"] > 0 else 0

        return BacktestResult(
            strategy_name=strategy,
            period=period,
            total_races=data["race_id"].nunique(),
            total_bets=len(bets),
            total_bet_amount=total_bet,
            total_return=round(total_ret),
            profit=round(total_ret - total_bet),
            roi_percent=round(total_ret / total_bet * 100, 1) if total_bet > 0 else 0,
            hit_count=hits,
            hit_rate=round(hits / len(bets) * 100, 1) if bets else 0,
            max_drawdown=round(max_dd),
            max_drawdown_pct=round(max_dd / total_bet * 100, 1) if total_bet > 0 else 0,
            best_month=best_m,
            worst_month=worst_m,
            monthly_results=monthly_list,
            venue_results=venue_results,
            equity_curve=equity,
        )

    def print_comparison(self, results: List[BacktestResult]):
        """全戦略の比較テーブル"""
        print(f"\n{'='*80}")
        print(f"📊 バックテスト比較")
        print(f"{'='*80}")
        print(f"{'戦略':<20} {'投資額':>10} {'回収額':>10} {'損益':>10} {'回収率':>7} {'的中率':>7} {'最大DD':>8}")
        print(f"{'-'*80}")
        for r in sorted(results, key=lambda x: x.roi_percent, reverse=True):
            print(f"{r.strategy_name:<20} ¥{r.total_bet_amount:>9,} ¥{r.total_return:>9,} ¥{r.profit:>+9,} {r.roi_percent:>6.1f}% {r.hit_rate:>6.1f}% ¥{r.max_drawdown:>7,}")

    def export_json(self, results: List[BacktestResult], path: str = "data/backtest_results.json"):
        """結果をJSON出力"""
        from dataclasses import asdict
        data = [asdict(r) for r in results]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"💾 バックテスト結果: {path}")


import os

if __name__ == "__main__":
    from data_processor import SampleDataGenerator, RaceDataProcessor, FeatureEngineer
    from model_trainer import RacePredictionModel

    print("🏇 バックテスト テスト")

    raw = SampleDataGenerator().generate(n_races=300)
    proc = RaceDataProcessor()
    proc.raw_data = raw
    df = proc.preprocess()
    fe = FeatureEngineer()
    df = fe.create_features(df)
    cols = fe.get_feature_columns(df)

    # 学習
    train_df = df.head(int(len(df) * 0.7))
    test_df = df.tail(int(len(df) * 0.3))

    model = RacePredictionModel()
    model.train(train_df, cols, n_splits=2)
    preds = model.predict(test_df)

    # バックテスト
    bt = Backtester(test_df, preds)
    results = bt.run_all_strategies()
    bt.print_comparison(results)
    bt.export_json(results)
