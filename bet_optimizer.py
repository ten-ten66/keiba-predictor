"""
競馬予想AI - 馬券種別最適化
================================
馬券種別ごとに最適な買い方を計算する。

  単勝/複勝: ケリー基準
  馬連/ワイド: 組み合わせ確率の直接計算
  三連複/三連単: モンテカルロシミュレーションで期待値算出
"""

import numpy as np
from itertools import combinations, permutations
from typing import Dict, List, Tuple

# Kelly Criterion: f* = (bp - q) / b
# Fractional Kelly (1/4 Kelly) で保守的に運用
# 理由: フルKellyは破産リスクが高い。1/4 Kellyが実用的な上限。
KELLY_FRACTION = 0.25  # 設定可能な定数に

# 最大ベット額制限（bankrollの5%まで）
# 理由: 単一レースへの集中リスクを回避
MAX_BET_FRACTION = 0.05


def estimate_fukusho_odds(tanshow_odds: float, n_runners: int = None) -> float:
    """複勝オッズの推定（経験的な回帰モデル）

    単勝オッズとの関係: 複勝 ≈ (単勝^0.6) * 0.5 + 1.0
    低オッズ馬: 単勝2.0→複勝1.3, 単勝5.0→複勝1.9, 単勝10.0→複勝2.8

    Args:
        tanshow_odds: 単勝オッズ
        n_runners: 出走頭数（少頭数で補正あり）

    Returns:
        推定複勝オッズ
    """
    if n_runners and n_runners <= 7:
        # 少頭数では複勝オッズが低くなる（3着以内の割合が高い）
        base = (tanshow_odds ** 0.55) * 0.45 + 1.0
    else:
        base = (tanshow_odds ** 0.6) * 0.5 + 1.0
    return max(1.1, round(base, 1))


class BetOptimizer:
    """馬券種別ごとの最適買い目を算出"""

    def __init__(self, n_simulations: int = 50000):
        self.n_sim = n_simulations

    def optimize_all(
        self,
        horses: List[Dict],
        bankroll: int = 10000,
        bet_unit: int = 100,
        real_odds: Dict = None,
    ) -> Dict[str, List[Dict]]:
        """全馬券種別の推奨買い目を一括計算

        Args:
            horses: 各馬データのリスト
            bankroll: 資金額
            bet_unit: 賭け単位
            real_odds: fetch_all_odds() の戻り値（実オッズ）。
                       指定時は推定オッズの代わりに実オッズを使用。
        """
        results = {}
        results["単勝"] = self.optimize_tansho(horses, bankroll)
        results["複勝"] = self.optimize_fukusho(horses, bankroll)
        results["馬連"] = self.optimize_umaren(horses, bankroll, bet_unit, real_odds)
        results["ワイド"] = self.optimize_wide(horses, bankroll, bet_unit, real_odds)
        results["三連複"] = self.optimize_sanrenpuku(horses, bankroll, bet_unit, real_odds)
        results["三連単"] = self.optimize_sanrentan(horses, bankroll, bet_unit)
        return results

    # ====== 単勝 ======

    def optimize_tansho(self, horses: List[Dict], bankroll: int) -> List[Dict]:
        """単勝: 期待値プラスの馬にケリー基準で配分"""
        bets = []
        for h in horses:
            prob = h.get("prob_win", 0)
            odds = h.get("odds", 0)
            if odds <= 1 or prob <= 0:
                continue
            b = odds - 1
            q = 1 - prob
            ev = prob * odds
            # Kelly Criterion: f* = (bp - q) / b
            # Fractional Kelly (1/4 Kelly) で保守的に運用
            # 理由: フルKellyは破産リスクが高い。1/4 Kellyが実用的な上限。
            kelly = max(0, (b * prob - q) / b) * KELLY_FRACTION
            if ev <= 1.0:
                continue
            # 最大ベット額制限（bankrollの5%まで）でリスク管理
            amount = min(
                max(100, int(bankroll * kelly / 100) * 100),
                int(bankroll * MAX_BET_FRACTION / 100) * 100
            )
            bets.append({
                "horse_number": h["horse_number"],
                "horse_name": h.get("horse_name", ""),
                "amount": amount,
                "ev": round(ev, 3),
                "prob": round(prob, 4),
                "odds": odds,
            })
        return sorted(bets, key=lambda x: x["ev"], reverse=True)

    # ====== 複勝 ======

    def optimize_fukusho(self, horses: List[Dict], bankroll: int) -> List[Dict]:
        """複勝: 複勝率×複勝オッズの期待値で判断"""
        bets = []
        n_runners = len(horses)  # 出走頭数を複勝オッズ推定に利用
        for h in horses:
            prob = h.get("prob_top3", 0)
            odds = h.get("odds", 0)
            if odds <= 1 or prob <= 0:
                continue
            # 複勝オッズ推定: 経験的回帰モデル（単勝^0.6 * 0.5 + 1.0）
            # 旧方式 odds * 0.3 は低オッズ馬で過小、高オッズ馬で過大だった
            fukusho_odds = estimate_fukusho_odds(odds, n_runners)
            ev = prob * fukusho_odds
            if ev <= 1.05:
                continue
            amount = max(100, int(bankroll * 0.02 / 100) * 100)
            bets.append({
                "horse_number": h["horse_number"],
                "horse_name": h.get("horse_name", ""),
                "amount": amount,
                "ev": round(ev, 3),
                "prob_top3": round(prob, 4),
                "est_odds": round(fukusho_odds, 1),
            })
        return sorted(bets, key=lambda x: x["ev"], reverse=True)[:3]

    # ====== 馬連 ======

    def optimize_umaren(self, horses: List[Dict], bankroll: int, unit: int, real_odds: Dict = None) -> List[Dict]:
        """馬連: 1-2着の組み合わせ確率を直接計算。real_oddsがあれば実オッズを使用。"""
        if len(horses) < 2:
            return []
        probs = self._calc_position_probs(horses)
        bets = []

        # 実オッズの馬連データ
        real_umaren = {}
        if real_odds and "umaren" in real_odds:
            real_umaren = real_odds["umaren"]

        for h1, h2 in combinations(range(len(horses)), 2):
            # h1が1着かつh2が2着、またはその逆の確率
            p12 = probs[h1][0] * probs[h2][1] + probs[h2][0] * probs[h1][1]

            nums = sorted([horses[h1]["horse_number"], horses[h2]["horse_number"]])
            combo_key = tuple(nums)

            # 実オッズがあればそちらを使用
            if combo_key in real_umaren:
                odds_val = real_umaren[combo_key]
                odds_source = "real"
            else:
                o1 = horses[h1].get("odds", 0)
                o2 = horses[h2].get("odds", 0)
                if o1 <= 0 or o2 <= 0:
                    continue
                odds_val = max(2.0, o1 * o2 * 0.4)
                odds_source = "estimated"

            ev = p12 * odds_val
            if ev <= 1.0:
                continue

            bet_entry = {
                "combination": nums,
                "names": sorted([horses[h1].get("horse_name", ""), horses[h2].get("horse_name", "")]),
                "prob": round(p12, 4),
                "odds": round(odds_val, 1),
                "ev": round(ev, 3),
                "amount": unit,
                "odds_source": odds_source,
            }
            # 後方互換: est_odds も残す
            bet_entry["est_odds"] = bet_entry["odds"]
            bets.append(bet_entry)

        return sorted(bets, key=lambda x: x["ev"], reverse=True)[:5]

    # ====== ワイド ======

    def optimize_wide(self, horses: List[Dict], bankroll: int, unit: int, real_odds: Dict = None) -> List[Dict]:
        """ワイド: 3着以内の2頭の組み合わせ確率。real_oddsがあれば実オッズを使用。"""
        if len(horses) < 2:
            return []
        bets = []

        # 実オッズのワイドデータ
        real_wide = {}
        if real_odds and "wide" in real_odds:
            real_wide = real_odds["wide"]

        for h1, h2 in combinations(range(len(horses)), 2):
            p1 = horses[h1].get("prob_top3", 0)
            p2 = horses[h2].get("prob_top3", 0)
            # 両方が3着以内の確率（独立近似、実際は相関あり）
            p_both = p1 * p2 * 0.85  # 相関補正

            nums = sorted([horses[h1]["horse_number"], horses[h2]["horse_number"]])
            combo_key = tuple(nums)

            # 実オッズがあればそちらを使用
            if combo_key in real_wide:
                min_odds, max_odds = real_wide[combo_key]
                odds_val = min_odds  # 控えめに最小オッズで計算
                odds_source = "real"
                odds_range = [round(min_odds, 1), round(max_odds, 1)]
            else:
                o1 = horses[h1].get("odds", 0)
                o2 = horses[h2].get("odds", 0)
                if o1 <= 0 or o2 <= 0:
                    continue
                odds_val = max(1.1, (o1 + o2) * 0.3)
                odds_source = "estimated"
                odds_range = [round(odds_val, 1), round(odds_val, 1)]

            ev = p_both * odds_val
            if ev <= 1.0:
                continue

            bet_entry = {
                "combination": nums,
                "names": sorted([horses[h1].get("horse_name", ""), horses[h2].get("horse_name", "")]),
                "prob": round(p_both, 4),
                "odds": round(odds_val, 1),
                "odds_range": odds_range,
                "ev": round(ev, 3),
                "amount": unit,
                "odds_source": odds_source,
            }
            bet_entry["est_odds"] = bet_entry["odds"]
            bets.append(bet_entry)

        return sorted(bets, key=lambda x: x["ev"], reverse=True)[:5]

    # ====== 三連複 ======

    def optimize_sanrenpuku(self, horses: List[Dict], bankroll: int, unit: int, real_odds: Dict = None) -> List[Dict]:
        """三連複: モンテカルロシミュレーションで確率推定。real_oddsがあれば実オッズを使用。"""
        if len(horses) < 3:
            return []
        finish_probs = self._monte_carlo_finish(horses)
        bets = []

        # 実オッズの三連複データ
        real_sanrenpuku = {}
        if real_odds and "sanrenpuku" in real_odds:
            real_sanrenpuku = real_odds["sanrenpuku"]

        for h1, h2, h3 in combinations(range(len(horses)), 3):
            nums = tuple(sorted([horses[h1]["horse_number"], horses[h2]["horse_number"], horses[h3]["horse_number"]]))
            prob = finish_probs.get(("sanrenpuku", nums), 0)
            if prob <= 0:
                continue

            # 実オッズがあればそちらを使用
            if nums in real_sanrenpuku:
                odds_val = real_sanrenpuku[nums]
                odds_source = "real"
            else:
                o1, o2, o3 = horses[h1].get("odds", 0), horses[h2].get("odds", 0), horses[h3].get("odds", 0)
                if min(o1, o2, o3) <= 0:
                    continue
                odds_val = max(5.0, (o1 * o2 * o3) ** 0.5 * 2)
                odds_source = "estimated"

            ev = prob * odds_val
            if ev <= 0.8:
                continue

            bets.append({
                "combination": list(nums),
                "prob": round(prob, 5),
                "odds": round(odds_val, 1),
                "est_odds": round(odds_val, 0),
                "ev": round(ev, 3),
                "amount": unit,
                "odds_source": odds_source,
            })

        return sorted(bets, key=lambda x: x["ev"], reverse=True)[:10]

    # ====== 三連単 ======

    def optimize_sanrentan(self, horses: List[Dict], bankroll: int, unit: int) -> List[Dict]:
        """三連単: モンテカルロシミュレーション"""
        if len(horses) < 3:
            return []
        finish_probs = self._monte_carlo_finish(horses)
        bets = []

        top5 = sorted(horses, key=lambda x: x.get("prob_win", 0), reverse=True)[:5]

        for h1, h2, h3 in permutations(range(len(top5)), 3):
            nums = (top5[h1]["horse_number"], top5[h2]["horse_number"], top5[h3]["horse_number"])
            prob = finish_probs.get(("sanrentan", nums), 0)
            if prob <= 0:
                continue

            o1, o2, o3 = top5[h1].get("odds", 0), top5[h2].get("odds", 0), top5[h3].get("odds", 0)
            if min(o1, o2, o3) <= 0:
                continue
            est_odds = max(10.0, o1 * o2 * o3 * 0.5)

            ev = prob * est_odds
            if ev <= 0.5:
                continue

            bets.append({
                "combination": list(nums),
                "prob": round(prob, 5),
                "est_odds": round(est_odds, 0),
                "ev": round(ev, 3),
                "amount": unit,
            })

        return sorted(bets, key=lambda x: x["ev"], reverse=True)[:10]

    # ====== 内部計算 ======

    def _calc_position_probs(self, horses: List[Dict]) -> np.ndarray:
        """各馬の1着〜N着確率を推定（ソフトマックスベース）"""
        n = len(horses)
        if n == 0:
            return np.zeros((0, 0))
        probs = np.zeros((n, n))

        # 1着確率はprob_winをそのまま使用
        win_probs = np.array([h.get("prob_win", 1/n) for h in horses])
        total = win_probs.sum()
        win_probs = win_probs / total if total > 0 else np.ones(n) / n
        probs[:, 0] = win_probs

        # 2着以降は条件付き確率で近似
        for pos in range(1, min(n, 5)):
            remaining_prob = 1.0 - probs[:, :pos].sum(axis=1)
            remaining_prob = np.maximum(remaining_prob, 0)
            total = remaining_prob.sum()
            if total > 0:
                probs[:, pos] = remaining_prob / total
            else:
                probs[:, pos] = 1.0 / n

        return probs

    def _monte_carlo_finish(self, horses: List[Dict]) -> Dict:
        """モンテカルロシミュレーションで着順分布を推定"""
        n = len(horses)
        if n < 3:
            return {}
        abilities = np.array([h.get("prob_win", 0.05) for h in horses])
        total = abilities.sum()
        abilities = abilities / total * n if total > 0 else np.ones(n)

        counter = {}

        for _ in range(self.n_sim):
            # 能力値 + ランダムノイズ → 着順
            scores = abilities + np.random.gumbel(0, 1, n)
            order = np.argsort(-scores)

            top3_nums = tuple(sorted([horses[order[i]]["horse_number"] for i in range(min(3, n))]))
            top3_order = tuple([horses[order[i]]["horse_number"] for i in range(min(3, n))])

            # 三連複
            key_puku = ("sanrenpuku", top3_nums)
            counter[key_puku] = counter.get(key_puku, 0) + 1

            # 三連単
            key_tan = ("sanrentan", top3_order)
            counter[key_tan] = counter.get(key_tan, 0) + 1

        # 確率に変換
        return {k: v / self.n_sim for k, v in counter.items()}


if __name__ == "__main__":
    print("=" * 50)
    print("🎯 馬券最適化テスト")
    print("=" * 50)

    horses = [
        {"horse_number": 1, "horse_name": "サクラフローラ", "prob_win": 0.31, "prob_top3": 0.72, "odds": 2.8},
        {"horse_number": 2, "horse_name": "メイショウハルカ", "prob_win": 0.22, "prob_top3": 0.63, "odds": 4.5},
        {"horse_number": 3, "horse_name": "アドマイヤルナ", "prob_win": 0.16, "prob_top3": 0.56, "odds": 6.2},
        {"horse_number": 4, "horse_name": "ゴールドティアラ", "prob_win": 0.11, "prob_top3": 0.45, "odds": 8.1},
        {"horse_number": 5, "horse_name": "スターライトノヴァ", "prob_win": 0.08, "prob_top3": 0.33, "odds": 12.5},
        {"horse_number": 6, "horse_name": "ウインフェアリー", "prob_win": 0.05, "prob_top3": 0.29, "odds": 15.0},
        {"horse_number": 7, "horse_name": "ロードクイーン", "prob_win": 0.04, "prob_top3": 0.20, "odds": 22.3},
        {"horse_number": 8, "horse_name": "テイエムプリンス", "prob_win": 0.03, "prob_top3": 0.15, "odds": 35.0},
    ]

    opt = BetOptimizer(n_simulations=30000)
    results = opt.optimize_all(horses, bankroll=10000)

    for bet_type, bets in results.items():
        print(f"\n【{bet_type}】")
        if not bets:
            print("  推奨なし")
            continue
        for b in bets[:3]:
            if "combination" in b:
                sel = "-".join(str(s) for s in b["combination"])
            else:
                sel = str(b.get("horse_number", ""))
            print(f"  {sel} / EV={b['ev']} / ¥{b['amount']:,}")
