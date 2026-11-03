"""
競馬予想AI - 大規模サンプルデータ生成
======================================
実データが手に入るまでの開発用。
netkeiba.comの実データ分布を模倣した高精度サンプルを生成する。

生成規模:
  - 5年分（2020-2024）: 約17,000レース / 約220,000レコード
  - 1年あたり: 約3,400レース（JRA年間約3,400レース）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import os


# ===== JRA実データに基づく定数 =====

# 開催スケジュール（簡易版: 場コード→主要開催月）
VENUE_SCHEDULE = {
    "札幌": {"months": [7, 8, 9], "code": "01"},
    "函館": {"months": [6, 7, 8], "code": "02"},
    "福島": {"months": [4, 7, 11], "code": "03"},
    "新潟": {"months": [5, 8, 10], "code": "04"},
    "東京": {"months": [2, 5, 6, 10, 11], "code": "05"},
    "中山": {"months": [1, 3, 4, 9, 12], "code": "06"},
    "中京": {"months": [1, 3, 7, 12], "code": "07"},
    "京都": {"months": [1, 2, 4, 5, 10, 11], "code": "08"},
    "阪神": {"months": [3, 4, 6, 9, 12], "code": "09"},
    "小倉": {"months": [2, 7, 8], "code": "10"},
}

# 距離分布（JRA実績ベース）
DISTANCE_DISTRIBUTION = {
    "芝": {
        1000: 0.02, 1200: 0.12, 1400: 0.10, 1600: 0.20, 1800: 0.16,
        2000: 0.15, 2200: 0.08, 2400: 0.07, 2500: 0.04, 3000: 0.03,
        3200: 0.02, 3600: 0.01,
    },
    "ダート": {
        1000: 0.05, 1150: 0.03, 1200: 0.15, 1400: 0.12, 1600: 0.08,
        1700: 0.15, 1800: 0.18, 1900: 0.05, 2000: 0.08, 2100: 0.06,
        2400: 0.03, 2500: 0.02,
    },
}

# 代表的な騎手（上位50名）
TOP_JOCKEYS = [
    ("J001", "ルメール", 0.22), ("J002", "川田将雅", 0.20), ("J003", "横山武史", 0.16),
    ("J004", "戸崎圭太", 0.14), ("J005", "松山弘平", 0.13), ("J006", "Ｍ．デムーロ", 0.12),
    ("J007", "武豊", 0.11), ("J008", "池添謙一", 0.10), ("J009", "岩田望来", 0.10),
    ("J010", "田辺裕信", 0.09), ("J011", "吉田隼人", 0.09), ("J012", "横山和生", 0.09),
    ("J013", "坂井瑠星", 0.09), ("J014", "藤岡佑介", 0.08), ("J015", "福永祐一", 0.08),
    ("J016", "三浦皇成", 0.08), ("J017", "岩田康誠", 0.07), ("J018", "内田博幸", 0.07),
    ("J019", "浜中俊", 0.07), ("J020", "和田竜二", 0.07), ("J021", "鮫島克駿", 0.07),
    ("J022", "石橋脩", 0.06), ("J023", "北村宏司", 0.06), ("J024", "丸山元気", 0.06),
    ("J025", "柴田善臣", 0.05), ("J026", "津村明秀", 0.05), ("J027", "菅原明良", 0.06),
    ("J028", "団野大成", 0.06), ("J029", "西村淳也", 0.07), ("J030", "横山典弘", 0.06),
    ("J031", "藤岡康太", 0.05), ("J032", "幸英明", 0.05), ("J033", "秋山真一郎", 0.04),
    ("J034", "石川裕紀人", 0.06), ("J035", "菊沢一樹", 0.04), ("J036", "木幡巧也", 0.04),
    ("J037", "大野拓弥", 0.05), ("J038", "丹内祐次", 0.04), ("J039", "勝浦正樹", 0.04),
    ("J040", "柴田大知", 0.04), ("J041", "荻野極", 0.05), ("J042", "角田大和", 0.05),
    ("J043", "永野猛蔵", 0.05), ("J044", "小林脩斗", 0.04), ("J045", "佐々木大輔", 0.04),
    ("J046", "酒井学", 0.04), ("J047", "小沢大仁", 0.04), ("J048", "松岡正海", 0.04),
    ("J049", "富田暁", 0.04), ("J050", "泉谷楓真", 0.05),
]

# 基準タイム（距離別、芝、良馬場）
BASE_TIMES = {
    1000: 56.0, 1200: 68.5, 1400: 81.0, 1600: 94.0, 1800: 106.5,
    2000: 119.5, 2200: 132.0, 2400: 144.5, 2500: 151.0, 3000: 183.0,
    3200: 196.0, 3600: 222.0,
    1000: 58.5, 1150: 67.5, 1700: 103.0, 1900: 116.0, 2100: 128.5,
}


class LargeScaleSampleGenerator:
    """JRA実データ分布に準拠した大規模サンプルデータ生成"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

        # 馬プール: 5000頭（年間約2000頭が出走）
        self.n_horses = 5000
        self.n_trainers = 200

        # 馬の基礎能力（正規分布、上位と下位の差が大きい）
        self.horse_base_ability = self.rng.normal(50, 15, self.n_horses)
        # 馬の距離適性（短距離型/中距離型/長距離型）
        self.horse_dist_pref = self.rng.choice(
            ["sprint", "mile", "middle", "long"],
            self.n_horses,
            p=[0.25, 0.30, 0.30, 0.15]
        )
        # 馬のコース適性
        self.horse_surface_pref = self.rng.choice(
            ["芝", "ダート", "both"],
            self.n_horses,
            p=[0.40, 0.35, 0.25]
        )
        # 馬の年齢（生成時に2歳）
        self.horse_birth_year = self.rng.integers(2015, 2023, self.n_horses)
        # 馬の性別
        self.horse_sex = self.rng.choice(["牡", "牝", "セ"], self.n_horses, p=[0.50, 0.35, 0.15])

        # 調教師の能力
        self.trainer_skill = self.rng.normal(0, 3, self.n_trainers)
        # 馬→調教師の割当
        self.horse_trainer = self.rng.integers(0, self.n_trainers, self.n_horses)

        # 騎手リスト
        self.jockeys = TOP_JOCKEYS

    def generate(
        self,
        start_year: int = 2020,
        end_year: int = 2024,
        output_dir: str = "data",
    ) -> pd.DataFrame:
        """指定年範囲のレースデータを生成"""
        os.makedirs(output_dir, exist_ok=True)

        all_records = []
        total_races = 0

        for year in range(start_year, end_year + 1):
            year_records = self._generate_year(year)
            all_records.extend(year_records)

            # 年別CSV保存
            df_year = pd.DataFrame(year_records)
            year_csv = Path(output_dir) / f"races_{year}.csv"
            df_year.to_csv(year_csv, index=False, encoding="utf-8")

            n_races = df_year["race_id"].nunique()
            total_races += n_races
            print(f"  📅 {year}年: {n_races} レース / {len(year_records)} レコード → {year_csv}")

        # 全期間結合CSV
        df_all = pd.DataFrame(all_records)
        combined_csv = Path(output_dir) / f"races_{start_year}_{end_year}.csv"
        df_all.to_csv(combined_csv, index=False, encoding="utf-8")

        print(f"\n✅ 全期間: {total_races} レース / {len(df_all)} レコード")
        print(f"   💾 {combined_csv}")

        # 統計サマリー
        self._print_stats(df_all)

        return df_all

    def _generate_year(self, year: int) -> list:
        """1年分のレースを生成"""
        records = []
        race_counter = 0

        for venue_name, info in VENUE_SCHEDULE.items():
            venue_code = info["code"]

            for month in info["months"]:
                # 開催: 1開催あたり8日（4週×土日）
                n_days = self.rng.integers(6, 10)
                start_day = max(1, self.rng.integers(1, 15))

                for day_offset in range(n_days):
                    race_date = datetime(year, month, min(start_day + day_offset * 2, 28))

                    # 1日12レース
                    for race_num in range(1, 13):
                        race_counter += 1
                        race_id = f"{year}{venue_code}{month:02d}{day_offset+1:02d}{race_num:02d}"

                        race_records = self._generate_single_race(
                            race_id=race_id,
                            race_date=race_date,
                            venue=venue_name,
                            race_num=race_num,
                            year=year,
                        )
                        records.extend(race_records)

        return records

    def _generate_single_race(
        self, race_id: str, race_date: datetime, venue: str, race_num: int, year: int
    ) -> list:
        """1レース分のデータを生成"""
        # コースタイプ（後半レースは芝が多い傾向）
        turf_prob = 0.55 if race_num >= 5 else 0.40
        course_type = self.rng.choice(["芝", "ダート"], p=[turf_prob, 1 - turf_prob])

        # 距離
        dist_probs = DISTANCE_DISTRIBUTION[course_type]
        distances = list(dist_probs.keys())
        probs = list(dist_probs.values())
        probs = np.array(probs) / sum(probs)
        distance = int(self.rng.choice(distances, p=probs))

        # 天候・馬場
        weather = self.rng.choice(["晴", "曇", "小雨", "雨"], p=[0.45, 0.28, 0.15, 0.12])
        if weather in ["小雨", "雨"]:
            track = self.rng.choice(["良", "稍重", "重", "不良"], p=[0.10, 0.30, 0.35, 0.25])
        else:
            track = self.rng.choice(["良", "稍重", "重", "不良"], p=[0.70, 0.20, 0.07, 0.03])

        # 出走頭数（クラスによる）
        if race_num >= 11:  # メインレース
            n_horses = self.rng.integers(12, 19)
            race_name = self._generate_race_name(race_num, distance, course_type, True)
        elif race_num >= 9:
            n_horses = self.rng.integers(10, 17)
            race_name = self._generate_race_name(race_num, distance, course_type, False)
        else:
            n_horses = self.rng.integers(8, 17)
            race_name = f"{race_num}R {venue}"

        # 出走馬を選定（距離・コース適性を考慮）
        horse_indices = self._select_horses(n_horses, distance, course_type, year)

        # 騎手を割当
        jockey_indices = self.rng.choice(len(self.jockeys), n_horses, replace=False)

        # 能力値計算→着順決定
        abilities = []
        for i, h_idx in enumerate(horse_indices):
            ability = self._calc_ability(
                h_idx, jockey_indices[i], distance, course_type, track, year
            )
            abilities.append(ability)

        # 着順
        sorted_indices = np.argsort(-np.array(abilities))
        finish_positions = np.zeros(n_horses, dtype=int)
        for rank, idx in enumerate(sorted_indices):
            finish_positions[idx] = rank + 1

        # 基準タイム
        base_time = self._get_base_time(distance, course_type, track)

        records = []
        for i, h_idx in enumerate(horse_indices):
            pos = finish_positions[i]
            j_idx = jockey_indices[i]

            # タイム計算
            time_sec = base_time - abilities[i] * 0.015 + self.rng.normal(0, 0.2)
            minutes = int(time_sec // 60)
            secs = time_sec % 60

            # 上り3F（能力と脚質に依存）
            base_last3f = 33.5 + (pos - 1) * 0.15 + self.rng.normal(0, 0.8)

            # オッズ（能力逆順→対数分布）
            base_odds = np.exp((pos - 1) * 0.25 + self.rng.normal(0, 0.5))
            odds = max(1.1, round(base_odds, 1))

            # 通過順（脚質に依存）
            corners = self._generate_passing_order(pos, n_horses)

            # 馬体重
            base_weight = 460 + self.rng.normal(0, 25)
            if self.horse_sex[h_idx] == "牝":
                base_weight -= 20

            # 賞金（着順に応じた分配）
            prize = self._calc_prize(pos, race_num)

            # 馬齢
            age = year - self.horse_birth_year[h_idx]
            if age < 2 or age > 9:
                age = self.rng.integers(2, 8)

            records.append({
                "race_id": race_id,
                "race_date": race_date.strftime("%Y-%m-%d"),
                "race_name": race_name,
                "venue": venue,
                "course_type": course_type,
                "distance": distance,
                "weather": weather,
                "track_condition": track,
                "horse_id": f"H{h_idx+1:05d}",
                "horse_name": f"テスト馬{h_idx+1}",
                "age": age,
                "sex": self.horse_sex[h_idx],
                "weight": round(54 + self.rng.normal(0, 2), 1),
                "horse_weight": int(base_weight),
                "weight_diff": self.rng.integers(-16, 17),
                "jockey_id": self.jockeys[j_idx][0],
                "jockey_name": self.jockeys[j_idx][1],
                "trainer_id": f"T{self.horse_trainer[h_idx]+1:04d}",
                "trainer_name": f"調教師{self.horse_trainer[h_idx]+1}",
                "frame_number": (i % 8) + 1,
                "horse_number": i + 1,
                "odds": odds,
                "popularity": int(np.argsort(np.argsort([r["odds"] if isinstance(r, dict) else odds for r in records] + [odds]))[len(records)] + 1) if records else 1,
                "finish_position": pos,
                "finish_time": f"{minutes}:{secs:04.1f}",
                "prize_money": prize,
                "passing_order": "-".join(str(c) for c in corners),
                "last_3f": round(base_last3f, 1),
            })

        # 人気順を正しく計算
        odds_list = [(r["odds"], idx) for idx, r in enumerate(records)]
        odds_list.sort(key=lambda x: x[0])
        for rank, (_, idx) in enumerate(odds_list):
            records[idx]["popularity"] = rank + 1

        return records

    def _select_horses(self, n: int, distance: int, course_type: str, year: int) -> list:
        """距離・コース適性を考慮して出走馬を選定"""
        # 距離カテゴリ
        if distance <= 1400:
            dist_cat = "sprint"
        elif distance <= 1800:
            dist_cat = "mile"
        elif distance <= 2200:
            dist_cat = "middle"
        else:
            dist_cat = "long"

        # 適性に合う馬の確率を高く
        weights = np.ones(self.n_horses)
        for i in range(self.n_horses):
            # 距離適性
            if self.horse_dist_pref[i] == dist_cat:
                weights[i] *= 3.0
            # コース適性
            if self.horse_surface_pref[i] == course_type or self.horse_surface_pref[i] == "both":
                weights[i] *= 2.0
            # 年齢フィルタ
            age = year - self.horse_birth_year[i]
            if age < 2 or age > 9:
                weights[i] = 0.01

        weights /= weights.sum()
        return list(self.rng.choice(self.n_horses, n, replace=False, p=weights))

    def _calc_ability(
        self, horse_idx: int, jockey_idx: int,
        distance: int, course_type: str, track: str, year: int
    ) -> float:
        """総合能力値を計算"""
        base = self.horse_base_ability[horse_idx]

        # 騎手ボーナス
        jockey_win_rate = self.jockeys[jockey_idx][2]
        jockey_bonus = (jockey_win_rate - 0.08) * 50  # 勝率8%を基準

        # 調教師ボーナス
        trainer_bonus = self.trainer_skill[self.horse_trainer[horse_idx]]

        # 距離適性ボーナス
        dist_cat = "sprint" if distance <= 1400 else "mile" if distance <= 1800 else "middle" if distance <= 2200 else "long"
        dist_bonus = 5.0 if self.horse_dist_pref[horse_idx] == dist_cat else -3.0

        # コース適性
        surface_bonus = 3.0 if (self.horse_surface_pref[horse_idx] == course_type or self.horse_surface_pref[horse_idx] == "both") else -4.0

        # 馬場状態（重馬場得意/苦手）
        track_penalty = {"良": 0, "稍重": -1, "重": -3, "不良": -5}.get(track, 0)

        # ランダムファクター（当日の調子）
        daily_form = self.rng.normal(0, 8)

        return base + jockey_bonus + trainer_bonus + dist_bonus + surface_bonus + track_penalty + daily_form

    def _get_base_time(self, distance: int, course_type: str, track: str) -> float:
        """距離・馬場状態に応じた基準タイム"""
        # 近い距離の基準タイムを補間
        time = distance / 16.0  # 約16m/秒が基準

        # ダートは芝より遅い
        if course_type == "ダート":
            time *= 1.03

        # 馬場状態による補正
        track_adj = {"良": 0, "稍重": 0.5, "重": 1.5, "不良": 3.0}
        time += track_adj.get(track, 0)

        return time

    def _generate_passing_order(self, finish_pos: int, n_horses: int) -> list:
        """脚質に応じた通過順を生成"""
        # 脚質タイプをランダムに決定
        style = self.rng.choice(["逃げ", "先行", "差し", "追込"], p=[0.08, 0.32, 0.35, 0.25])

        corners = []
        for c in range(4):
            if style == "逃げ":
                pos = max(1, self.rng.integers(1, 4))
            elif style == "先行":
                pos = max(1, self.rng.integers(2, int(n_horses * 0.4) + 1))
            elif style == "差し":
                pos = max(1, self.rng.integers(int(n_horses * 0.3), int(n_horses * 0.7) + 1))
            else:  # 追込
                pos = max(1, self.rng.integers(int(n_horses * 0.6), n_horses + 1))

            # 後半にかけて着順に近づく
            if c >= 2:
                pos = int(pos * 0.7 + finish_pos * 0.3)

            corners.append(max(1, min(n_horses, pos)))

        return corners

    def _calc_prize(self, position: int, race_num: int) -> float:
        """着順とレース番号に応じた賞金"""
        # メインレースは賞金が高い
        base_prize = {11: 5000, 12: 8000, 10: 3000}.get(race_num, 1000)

        prize_rates = {1: 1.0, 2: 0.4, 3: 0.25, 4: 0.15, 5: 0.1}
        rate = prize_rates.get(position, 0)

        return round(base_prize * rate, 1)

    def _generate_race_name(self, race_num: int, distance: int, course_type: str, is_main: bool) -> str:
        """レース名を生成"""
        if is_main:
            names = [
                "特別", "ステークス", "カップ", "トロフィー", "賞",
                "記念", "オープン", "ハンデキャップ",
            ]
            prefixes = [
                "スプリング", "オータム", "サマー", "ウィンター",
                "ゴールデン", "シルバー", "ダイヤモンド", "エメラルド",
                "フラワー", "スター", "ムーン", "サン",
            ]
            return f"{self.rng.choice(prefixes)}{self.rng.choice(names)}"
        else:
            classes = ["未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン"]
            cls = classes[min(race_num // 3, 4)]
            return f"{course_type}{distance}m {cls}"

    def _print_stats(self, df: pd.DataFrame):
        """生成データの統計を表示"""
        print(f"\n{'='*60}")
        print("📊 生成データ統計")
        print(f"{'='*60}")
        print(f"  レース数:   {df['race_id'].nunique():,}")
        print(f"  レコード数: {len(df):,}")
        print(f"  馬数:       {df['horse_id'].nunique():,}")
        print(f"  騎手数:     {df['jockey_id'].nunique():,}")
        print(f"  期間:       {df['race_date'].min()} 〜 {df['race_date'].max()}")
        print(f"\n  コース別:")
        for ct in df["course_type"].unique():
            n = len(df[df["course_type"] == ct])
            print(f"    {ct}: {n:,} レコード ({n/len(df)*100:.1f}%)")
        print(f"\n  馬場状態別:")
        for tc in ["良", "稍重", "重", "不良"]:
            n = len(df[df["track_condition"] == tc])
            print(f"    {tc}: {n:,} レコード ({n/len(df)*100:.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="大規模サンプルデータ生成")
    parser.add_argument("--start", type=int, default=2020, help="開始年")
    parser.add_argument("--end", type=int, default=2024, help="終了年")
    parser.add_argument("--output", type=str, default="data", help="出力先ディレクトリ")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")

    args = parser.parse_args()

    print("🏇" * 20)
    print("  大規模サンプルデータ生成")
    print(f"  {args.start}年 〜 {args.end}年")
    print("🏇" * 20)

    gen = LargeScaleSampleGenerator(seed=args.seed)
    df = gen.generate(start_year=args.start, end_year=args.end, output_dir=args.output)
