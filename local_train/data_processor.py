"""
競馬予想AI - データ処理・特徴量エンジニアリング
==============================================
過去レースデータを読み込み、機械学習用の特徴量を生成する。

データ形式: CSVファイル（netkeiba.com等からのエクスポートを想定）
必須カラム:
  race_id, race_date, race_name, course_type (芝/ダート), distance,
  weather, track_condition (良/稍重/重/不良), horse_id, horse_name,
  age, sex, weight, weight_diff, jockey_id, jockey_name,
  trainer_id, trainer_name, frame_number, horse_number,
  odds, popularity, finish_position, finish_time,
  prize_money, passing_order, last_3f
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")


class RaceDataProcessor:
    """過去レースデータの読み込みと前処理"""

    # カラム名マッピング（日本語→英語）
    COLUMN_MAP = {
        "レースID": "race_id",
        "日付": "race_date",
        "レース名": "race_name",
        "コース": "course_type",
        "距離": "distance",
        "天候": "weather",
        "馬場": "track_condition",
        "馬ID": "horse_id",
        "馬名": "horse_name",
        "馬齢": "age",
        "性別": "sex",
        "斤量": "weight",
        "体重": "horse_weight",
        "体重変化": "weight_diff",
        "騎手ID": "jockey_id",
        "騎手": "jockey_name",
        "調教師ID": "trainer_id",
        "調教師": "trainer_name",
        "枠番": "frame_number",
        "馬番": "horse_number",
        "オッズ": "odds",
        "人気": "popularity",
        "着順": "finish_position",
        "タイム": "finish_time",
        "賞金": "prize_money",
        "通過順": "passing_order",
        "上り3F": "last_3f",
    }

    # 馬場状態エンコード
    TRACK_CONDITION_MAP = {"良": 0, "稍重": 1, "重": 2, "不良": 3}

    # 天候エンコード
    WEATHER_MAP = {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "小雪": 4, "雪": 5}

    # コースタイプエンコード
    COURSE_TYPE_MAP = {"芝": 0, "ダート": 1, "障害": 2}

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None

    def load_csv(self, filename: str, encoding: str = "utf-8") -> pd.DataFrame:
        """CSVファイルを読み込み、カラム名を正規化"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")

        df = pd.read_csv(filepath, encoding=encoding)

        # 日本語カラム名があれば英語に変換
        rename_map = {k: v for k, v in self.COLUMN_MAP.items() if k in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)

        self.raw_data = df
        print(f"✅ {len(df)} 件のレコードを読み込み ({filepath.name})")
        print(f"   カラム: {list(df.columns)}")
        return df

    def load_multiple_csv(self, pattern: str = "*.csv", encoding: str = "utf-8") -> pd.DataFrame:
        """複数CSVを一括読み込み"""
        files = sorted(self.data_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"CSVファイルが見つかりません: {self.data_dir}/{pattern}")

        dfs = []
        for f in files:
            df = pd.read_csv(f, encoding=encoding)
            rename_map = {k: v for k, v in self.COLUMN_MAP.items() if k in df.columns}
            if rename_map:
                df = df.rename(columns=rename_map)
            dfs.append(df)
            print(f"  📄 {f.name}: {len(df)} 件")

        self.raw_data = pd.concat(dfs, ignore_index=True)
        print(f"✅ 合計 {len(self.raw_data)} 件のレコードを読み込み")
        return self.raw_data

    def preprocess(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """前処理: 型変換、欠損値処理、カテゴリエンコード"""
        if df is None:
            df = self.raw_data.copy()
        else:
            df = df.copy()

        # 日付をdatetime型に変換（複数形式対応: YYYY-MM-DD, YYYY/MM/DD, YYYYMMDD）
        if "race_date" in df.columns:
            df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce", format="mixed")
            df["year"] = df["race_date"].dt.year
            df["month"] = df["race_date"].dt.month
            df["day_of_week"] = df["race_date"].dt.dayofweek

        # 数値型に変換
        numeric_cols = [
            "distance", "weight", "horse_weight", "weight_diff",
            "frame_number", "horse_number", "odds", "popularity",
            "finish_position", "prize_money", "last_3f", "age"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # タイム（mm:ss.s）を秒に変換
        if "finish_time" in df.columns:
            df["finish_time_sec"] = df["finish_time"].apply(self._parse_time)

        # 通過順を解析（例: "3-3-2-1" → 各コーナーの順位）
        if "passing_order" in df.columns:
            df = self._parse_passing_order(df)

        # カテゴリ変数のエンコード
        if "track_condition" in df.columns:
            df["track_condition_enc"] = df["track_condition"].map(self.TRACK_CONDITION_MAP)

        if "weather" in df.columns:
            df["weather_enc"] = df["weather"].map(self.WEATHER_MAP)

        if "course_type" in df.columns:
            df["course_type_enc"] = df["course_type"].map(self.COURSE_TYPE_MAP)

        # 性別エンコード
        if "sex" in df.columns:
            df["sex_enc"] = df["sex"].map({"牡": 0, "牝": 1, "セ": 2}).fillna(-1).astype(int)

        # 着順DNF処理（NaNは未実施レース＝予測対象として保持）
        if "finish_position" in df.columns:
            has_pos = df["finish_position"].notna()
            # 結果がある行のみフィルタ（0以下を除外）、NaN行は保持
            valid = has_pos & (df["finish_position"] > 0)
            df = df[valid | ~has_pos]

        self.processed_data = df
        print(f"✅ 前処理完了: {len(df)} 件 / {len(df.columns)} カラム")
        return df

    @staticmethod
    def _parse_time(time_str) -> Optional[float]:
        """タイム文字列をfloat秒に変換。M:SS.S or SS.S 形式のみ"""
        if pd.isna(time_str):
            return None
        try:
            time_str = str(time_str).strip()
            if not time_str or time_str == '---':
                return None
            if ":" in time_str:
                parts = time_str.split(":")
                if len(parts) != 2:
                    return None  # 1:2:3 等の不正形式を拒否
                minutes = float(parts[0])
                seconds = float(parts[1])
                if minutes < 0 or seconds < 0 or seconds >= 60:
                    return None
                return minutes * 60 + seconds
            val = float(time_str)
            return val if val > 0 else None
        except (ValueError, IndexError, TypeError):
            return None

    @staticmethod
    def _parse_passing_order(df: pd.DataFrame) -> pd.DataFrame:
        """通過順を各コーナーに分解"""
        def parse(val):
            if pd.isna(val):
                return [None] * 4
            parts = str(val).split("-")
            result = [None] * 4
            for i, p in enumerate(parts[:4]):
                try:
                    result[i] = int(p)
                except ValueError:
                    pass
            return result

        parsed = df["passing_order"].apply(parse)
        for i in range(4):
            df[f"corner_{i+1}"] = parsed.apply(lambda x: x[i])

        return df


class FeatureEngineer:
    """特徴量エンジニアリング: 過去成績・統計量から予測用特徴量を生成"""

    def __init__(self):
        self.feature_names = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """全特徴量を一括生成"""
        print("🔧 特徴量エンジニアリング開始...")

        df = df.sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)

        # 1. 馬の過去成績
        df = self._horse_history_features(df)

        # 2. 騎手の過去成績
        df = self._jockey_features(df)

        # 3. 調教師の過去成績
        df = self._trainer_features(df)

        # 4. 距離適性
        df = self._distance_features(df)

        # 5. コース適性（芝/ダート）
        df = self._course_type_features(df)

        # 6. 馬場状態適性
        df = self._track_condition_features(df)

        # 7. 枠番統計
        df = self._frame_features(df)

        # 8. レース内相対指標
        df = self._race_relative_features(df)

        # 9. スピード指数
        df = self._speed_index(df)

        # 10. ペース特徴量
        df = self._pace_features(df)

        # 11. 高NaN特徴量の補完（距離・コース適性は初出走時NaNになりやすい）
        df = self._fill_high_nan_features(df)

        print(f"✅ 特徴量エンジニアリング完了: {len(df.columns)} カラム")
        return df

    def _fill_high_nan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """NaN率が高い特徴量をグローバル中央値で補完"""
        fill_targets = [
            "horse_dist_avg_pos", "horse_dist_top3_rate",
            "horse_course_avg_pos", "horse_course_top3_rate",
            "days_since_last_race",
        ]
        for col in fill_targets:
            if col in df.columns:
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)
                else:
                    # 全NaNの場合はデフォルト値
                    defaults = {"avg_pos": 8.0, "top3_rate": 0.25, "days_since_last_race": 30.0}
                    for key, default in defaults.items():
                        if key in col:
                            df[col] = df[col].fillna(default)
                            break
                    else:
                        df[col] = df[col].fillna(0)
        return df

    def _horse_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬の過去N走の成績統計"""
        if "horse_id" not in df.columns:
            return df

        windows = [3, 5, 10]  # 直近3走、5走、10走

        for w in windows:
            group = df.groupby("horse_id")

            # 平均着順
            df[f"horse_avg_pos_{w}"] = group["finish_position"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )

            # 勝率（1着率）
            df[f"horse_win_rate_{w}"] = group["finish_position"].transform(
                lambda x: (x.shift(1) == 1).rolling(w, min_periods=1).mean()
            )

            # 複勝率（3着以内率）
            df[f"horse_top3_rate_{w}"] = group["finish_position"].transform(
                lambda x: (x.shift(1) <= 3).rolling(w, min_periods=1).mean()
            )

            # 平均上り3F
            if "last_3f" in df.columns:
                df[f"horse_avg_last3f_{w}"] = group["last_3f"].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                )

            # 平均賞金
            if "prize_money" in df.columns:
                df[f"horse_avg_prize_{w}"] = group["prize_money"].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                )

        # 通算成績
        df["horse_total_races"] = df.groupby("horse_id").cumcount()
        df["horse_total_wins"] = df.groupby("horse_id")["finish_position"].transform(
            lambda x: (x.shift(1) == 1).cumsum()
        )

        # 前走からの間隔（日数）
        if "race_date" in df.columns:
            df["days_since_last_race"] = df.groupby("horse_id")["race_date"].diff().dt.days

        return df

    def _jockey_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """騎手の過去成績"""
        if "jockey_id" not in df.columns:
            return df

        group = df.groupby("jockey_id")

        df["jockey_avg_pos_30"] = group["finish_position"].transform(
            lambda x: x.shift(1).rolling(30, min_periods=5).mean()
        )
        df["jockey_win_rate_30"] = group["finish_position"].transform(
            lambda x: (x.shift(1) == 1).rolling(30, min_periods=5).mean()
        )
        df["jockey_top3_rate_30"] = group["finish_position"].transform(
            lambda x: (x.shift(1) <= 3).rolling(30, min_periods=5).mean()
        )

        return df

    def _trainer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """調教師の過去成績"""
        if "trainer_id" not in df.columns:
            return df

        group = df.groupby("trainer_id")

        df["trainer_avg_pos_30"] = group["finish_position"].transform(
            lambda x: x.shift(1).rolling(30, min_periods=5).mean()
        )
        df["trainer_win_rate_30"] = group["finish_position"].transform(
            lambda x: (x.shift(1) == 1).rolling(30, min_periods=5).mean()
        )

        return df

    def _distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """距離適性（同距離帯での過去成績）"""
        if "distance" not in df.columns or "horse_id" not in df.columns:
            return df

        # 距離カテゴリ（短距離/マイル/中距離/長距離）
        df["distance_category"] = pd.cut(
            df["distance"],
            bins=[0, 1400, 1800, 2200, 9999],
            labels=["sprint", "mile", "middle", "long"]
        )

        # 同距離帯での平均着順
        df["horse_dist_avg_pos"] = df.groupby(["horse_id", "distance_category"])[
            "finish_position"
        ].transform(lambda x: x.shift(1).expanding().mean())

        # 同距離帯での複勝率
        df["horse_dist_top3_rate"] = df.groupby(["horse_id", "distance_category"])[
            "finish_position"
        ].transform(lambda x: (x.shift(1) <= 3).expanding().mean())

        return df

    def _course_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """芝/ダート適性"""
        if "course_type_enc" not in df.columns or "horse_id" not in df.columns:
            return df

        df["horse_course_avg_pos"] = df.groupby(["horse_id", "course_type_enc"])[
            "finish_position"
        ].transform(lambda x: x.shift(1).expanding().mean())

        df["horse_course_top3_rate"] = df.groupby(["horse_id", "course_type_enc"])[
            "finish_position"
        ].transform(lambda x: (x.shift(1) <= 3).expanding().mean())

        return df

    def _track_condition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬場状態適性"""
        if "track_condition_enc" not in df.columns or "horse_id" not in df.columns:
            return df

        # 重馬場（重・不良）での成績
        heavy_mask = df["track_condition_enc"] >= 2
        df["is_heavy_track"] = heavy_mask.astype(int)

        return df

    def _frame_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """枠番の統計"""
        if "frame_number" not in df.columns:
            return df

        # 枠番カテゴリ（内/中/外）
        df["frame_position"] = pd.cut(
            df["frame_number"],
            bins=[0, 3, 6, 8],
            labels=["inner", "middle", "outer"]
        )

        return df

    def _race_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """レース内での相対指標"""
        # オッズの相対位置
        if "odds" in df.columns:
            df["odds_rank"] = df.groupby("race_id")["odds"].rank()
            df["odds_ratio"] = df.groupby("race_id")["odds"].transform(
                lambda x: x / x.min()
            )

        # 体重の平均からの乖離
        if "horse_weight" in df.columns:
            df["weight_vs_avg"] = df.groupby("race_id")["horse_weight"].transform(
                lambda x: x - x.mean()
            )

        return df

    def _speed_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """スピード指数の算出"""
        if "finish_time_sec" not in df.columns or "distance" not in df.columns:
            return df

        # 基準タイム（同距離・同コースの中央値タイム）
        base_time = df.groupby(["distance", "course_type_enc"])["finish_time_sec"].transform("median")

        # スピード指数 = (基準タイム - 走破タイム) / 基準タイム * 100
        df["speed_index"] = ((base_time - df["finish_time_sec"]) / base_time * 100).round(2)

        # 過去3走の平均スピード指数
        if "horse_id" in df.columns:
            df["avg_speed_index_3"] = df.groupby("horse_id")["speed_index"].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean()
            )
            df["max_speed_index_5"] = df.groupby("horse_id")["speed_index"].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).max()
            )

        return df

    def _pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ペース関連特徴量"""
        # 通過順位の変化（追込み/逃げの傾向）
        if "corner_1" in df.columns and "corner_4" in df.columns:
            df["position_change"] = df["corner_1"] - df.get("corner_4", df["corner_1"])

        # 前走脚質判定
        if "corner_1" in df.columns and "horse_id" in df.columns:
            # 前走1コーナー順位の平均で脚質推定
            df["avg_early_position"] = df.groupby("horse_id")["corner_1"].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """モデルに投入する特徴量カラムのリストを取得"""
        exclude = {
            "race_id", "race_date", "race_name", "horse_id", "horse_name",
            "jockey_id", "jockey_name", "trainer_id", "trainer_name",
            "finish_position", "finish_time", "finish_time_sec",
            "prize_money", "passing_order", "course_type", "weather",
            "track_condition", "sex", "distance_category", "frame_position",
            "speed_index",       # 当該レースのタイムが必要 → リーク
            "corner_1", "corner_2", "corner_3", "corner_4",  # 当該レースの通過順 → リーク
            "last_3f",           # 当該レースの上がりタイム → リーク
            "position_change",   # corner_1-corner_4 → リーク
            "norm_position",     # finish_position ÷ n_runners → リーク
            "sire", "dam", "sire_sire", "dam_sire",  # 文字列カラム
        }
        feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64", "int32", "float32"]]
        self.feature_names = feature_cols
        return feature_cols


class SampleDataGenerator:
    """開発・テスト用のサンプルデータ生成"""

    @staticmethod
    def generate(n_races: int = 500, avg_horses: int = 14, seed: int = 42) -> pd.DataFrame:
        """リアルな分布に基づくサンプルデータを生成"""
        np.random.seed(seed)
        records = []

        # 仮想の馬プール
        n_horses = 300
        n_jockeys = 50
        n_trainers = 40

        horse_abilities = np.random.normal(50, 15, n_horses)  # 馬の基礎能力
        jockey_skills = np.random.normal(0, 5, n_jockeys)
        trainer_skills = np.random.normal(0, 3, n_trainers)

        start_date = pd.Timestamp("2020-01-01")

        for race_idx in range(n_races):
            n_horses_race = np.random.randint(8, 19)
            race_id = f"R{race_idx+1:05d}"
            race_date = start_date + pd.Timedelta(days=int(race_idx * 0.7))

            course_type = np.random.choice(["芝", "ダート"], p=[0.55, 0.45])
            distance = np.random.choice([1200, 1400, 1600, 1800, 2000, 2200, 2400, 2500, 3000, 3200],
                                         p=[0.12, 0.10, 0.18, 0.15, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03])
            weather = np.random.choice(["晴", "曇", "小雨", "雨"], p=[0.4, 0.3, 0.15, 0.15])
            track_cond = np.random.choice(["良", "稍重", "重", "不良"], p=[0.55, 0.25, 0.12, 0.08])

            # 出走馬を選択
            selected_horses = np.random.choice(n_horses, n_horses_race, replace=False)
            selected_jockeys = np.random.choice(n_jockeys, n_horses_race, replace=False)
            selected_trainers = np.random.choice(n_trainers, n_horses_race, replace=True)

            # 能力値からタイムを算出
            base_time = distance / 16.5  # 基準速度 (m/s)
            abilities = []
            for i, h in enumerate(selected_horses):
                ability = (
                    horse_abilities[h]
                    + jockey_skills[selected_jockeys[i]]
                    + trainer_skills[selected_trainers[i]]
                    + np.random.normal(0, 8)  # レースごとのブレ
                )
                abilities.append(ability)

            # 着順決定
            finish_order = np.argsort(-np.array(abilities)) + 1
            sorted_times = np.sort(np.array([base_time - a * 0.01 for a in abilities]))

            for i in range(n_horses_race):
                pos = int(finish_order[i])
                time_sec = base_time - abilities[i] * 0.01 + np.random.normal(0, 0.3)
                minutes = int(time_sec // 60)
                secs = time_sec % 60

                # 通過順生成
                corners = []
                for c in range(4):
                    cp = max(1, min(n_horses_race, pos + np.random.randint(-3, 4)))
                    corners.append(cp)

                records.append({
                    "race_id": race_id,
                    "race_date": race_date.strftime("%Y-%m-%d"),
                    "race_name": f"テストレース{race_idx+1}",
                    "course_type": course_type,
                    "distance": distance,
                    "weather": weather,
                    "track_condition": track_cond,
                    "horse_id": f"H{selected_horses[i]+1:04d}",
                    "horse_name": f"テスト馬{selected_horses[i]+1}",
                    "age": np.random.randint(2, 8),
                    "sex": np.random.choice(["牡", "牝", "セ"], p=[0.5, 0.35, 0.15]),
                    "weight": round(np.random.uniform(50, 58), 1),
                    "horse_weight": np.random.randint(420, 560),
                    "weight_diff": np.random.randint(-20, 21),
                    "jockey_id": f"J{selected_jockeys[i]+1:03d}",
                    "jockey_name": f"騎手{selected_jockeys[i]+1}",
                    "trainer_id": f"T{selected_trainers[i]+1:03d}",
                    "trainer_name": f"調教師{selected_trainers[i]+1}",
                    "frame_number": (i % 8) + 1,
                    "horse_number": i + 1,
                    "odds": round(max(1.1, np.random.exponential(15)), 1),
                    "popularity": pos,  # 簡易的にpos=人気とする
                    "finish_position": pos,
                    "finish_time": f"{minutes}:{secs:04.1f}",
                    "prize_money": max(0, (n_horses_race - pos + 1) * 100) if pos <= 5 else 0,
                    "passing_order": "-".join(str(c) for c in corners),
                    "last_3f": round(33 + np.random.normal(0, 1.5), 1),
                })

        return pd.DataFrame(records)


if __name__ == "__main__":
    # サンプルデータ生成 & 処理テスト
    print("=" * 60)
    print("🏇 競馬予想AI - データ処理テスト")
    print("=" * 60)

    # サンプルデータ生成
    gen = SampleDataGenerator()
    sample_df = gen.generate(n_races=500)
    sample_df.to_csv("data/sample_races.csv", index=False)
    print(f"\n📊 サンプルデータ生成: {len(sample_df)} レコード")

    # データ処理
    processor = RaceDataProcessor(data_dir="data")
    df = processor.load_csv("sample_races.csv")
    df = processor.preprocess(df)

    # 特徴量エンジニアリング
    fe = FeatureEngineer()
    df = fe.create_features(df)
    feature_cols = fe.get_feature_columns(df)

    print(f"\n📋 特徴量一覧 ({len(feature_cols)} 個):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    # 保存
    df.to_csv("data/processed_races.csv", index=False)
    print(f"\n💾 処理済みデータを保存: data/processed_races.csv")
