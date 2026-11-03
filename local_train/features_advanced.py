"""
競馬予想AI - 拡張特徴量エンジン
=================================
基本56特徴量に加え、精度向上に直結する高度な特徴量を生成。

追加特徴量:
  1. 血統（父・母父の統計）
  2. 斤量差分析
  3. ローテーション（間隔×前走成績の交互作用）
  4. 季節×馬齢 交互作用
  5. 馬場バイアス（内外有利）
  6. 正規化着順（出走頭数で正規化）
  7. レースレベル指標
  8. オッズ変動特徴量
"""

import pandas as pd
import numpy as np
from typing import Optional


class AdvancedFeatureEngineer:
    """基本FeatureEngineerの上に追加する拡張特徴量"""

    def create_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """全拡張特徴量を生成"""
        df = df.copy()
        df = self._normalized_position(df)
        df = self._weight_analysis(df)
        df = self._rotation_features(df)
        df = self._season_age_interaction(df)
        df = self._track_bias(df)
        df = self._race_level(df)
        df = self._bloodline_features(df)
        df = self._odds_movement(df)
        df = self._momentum_features(df)
        df = self._class_features(df)
        df = self._training_features(df)
        print(f"✅ 拡張特徴量完了: {len(df.columns)} カラム")
        return df

    def _normalized_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """正規化着順: 着順÷出走頭数（8頭と18頭を公平比較）"""
        if "finish_position" not in df.columns:
            return df

        # レースごとの出走頭数
        df["n_runners"] = df.groupby("race_id")["horse_number"].transform("count")

        # 正規化着順 (0=1着, 1=最下位)
        df["norm_position"] = (df["finish_position"] - 1) / (df["n_runners"] - 1).clip(lower=1)

        # 過去の正規化着順平均
        if "horse_id" in df.columns:
            for w in [3, 5, 10]:
                df[f"horse_norm_pos_avg_{w}"] = df.groupby("horse_id")["norm_position"].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                )

        return df

    def _weight_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """斤量の詳細分析"""
        if "weight" not in df.columns or "horse_id" not in df.columns:
            return df

        # 前走比の斤量変化
        df["weight_change"] = df.groupby("horse_id")["weight"].diff()

        # 同馬の平均斤量からの乖離
        df["weight_vs_self_avg"] = df.groupby("horse_id")["weight"].transform(
            lambda x: x - x.shift(1).expanding().mean()
        )

        # 馬体重の変化率
        if "horse_weight" in df.columns:
            df["horse_weight_change_rate"] = df.groupby("horse_id")["horse_weight"].transform(
                lambda x: x.pct_change()
            )
            # 馬体重が大幅に変わった（±10kg以上）フラグ
            if "weight_diff" in df.columns:
                df["big_weight_change"] = (df["weight_diff"].abs() >= 10).astype(int)

        return df

    def _rotation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ローテーション: 間隔×前走成績の交互作用"""
        if "horse_id" not in df.columns or "race_date" not in df.columns:
            return df

        # 前走からの間隔（日数）
        if "days_since_last_race" not in df.columns:
            df["days_since_last_race"] = df.groupby("horse_id")["race_date"].diff().dt.days

        # 間隔カテゴリ
        df["rest_category"] = pd.cut(
            df["days_since_last_race"].fillna(90),
            bins=[0, 14, 28, 56, 90, 180, 9999],
            labels=["連闘", "中1-3週", "中4-7週", "中8-12週", "半年以内", "半年以上"]
        )

        # 前走着順
        df["prev_position"] = df.groupby("horse_id")["finish_position"].shift(1)

        # 交互作用: 休み明け×前走好走 → 状態上昇の可能性
        df["rest_x_prev_pos"] = df["days_since_last_race"].fillna(90) * df["prev_position"].fillna(8)

        # 前走1着からの出走 → 連勝狙い度
        df["prev_win_flag"] = (df["prev_position"] == 1).astype(int)

        # 前走3着以内からの出走
        df["prev_top3_flag"] = (df["prev_position"] <= 3).astype(int)

        # 連続好走数（直近何走連続で3着以内か）
        if "horse_id" in df.columns:
            def calc_streak(series):
                streak = 0
                result = []
                for val in series:
                    if val <= 3:
                        streak += 1
                    else:
                        streak = 0
                    result.append(streak)
                return pd.Series(result, index=series.index)

            df["top3_streak"] = df.groupby("horse_id")["finish_position"].transform(
                lambda x: calc_streak(x.shift(1).fillna(99))
            )

        return df

    def _season_age_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """季節×馬齢の交互作用"""
        if "month" not in df.columns or "age" not in df.columns:
            return df

        # 季節カテゴリ
        df["season"] = pd.cut(
            df["month"],
            bins=[0, 3, 6, 9, 12],
            labels=["冬", "春", "夏", "秋"]
        )

        # 3歳馬の春（成長期）フラグ
        df["young_spring"] = ((df["age"] == 3) & (df["month"].between(3, 6))).astype(int)

        # 古馬の夏（暑さで落ちる）フラグ
        df["old_summer"] = ((df["age"] >= 5) & (df["month"].between(7, 9))).astype(int)

        # 馬齢×季節の交互作用（数値）
        season_enc = df["month"].map(lambda m: (m - 1) // 3)  # 0-3
        df["age_x_season"] = df["age"] * season_enc

        # 2歳戦フラグ（データが少なく荒れやすい）
        df["is_2yo_race"] = (df["age"] == 2).astype(int)

        return df

    def _track_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬場バイアス: その日の内外有利傾向"""
        if "race_date" not in df.columns or "frame_number" not in df.columns:
            return df

        # 同日・同場の内枠(1-3)vs外枠(6-8)の平均着順差
        df["is_inner"] = (df["frame_number"] <= 3).astype(int)
        df["is_outer"] = (df["frame_number"] >= 6).astype(int)

        # 当日バイアス（そのレースより前のレースから計算 → リーク防止は難しいが近似）
        if "venue" in df.columns:
            inner_avg = df.groupby(["race_date", "venue", "is_inner"])["finish_position"].transform("mean")
            df["daily_inner_bias"] = df.groupby(["race_date", "venue"])["finish_position"].transform("mean") - inner_avg

        return df

    def _race_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """レースレベル指標: 出走馬の平均実力"""
        if "race_id" not in df.columns:
            return df

        # レース内の平均オッズ（低い＝強い馬が多い＝ハイレベル）
        if "odds" in df.columns:
            df["race_avg_odds"] = df.groupby("race_id")["odds"].transform("mean")
            df["race_min_odds"] = df.groupby("race_id")["odds"].transform("min")
            # 自分のオッズ ÷ レース平均オッズ
            df["odds_vs_race_avg"] = df["odds"] / df["race_avg_odds"].clip(lower=1)

        # 出走頭数（多いほど勝ちにくい）
        if "n_runners" not in df.columns:
            df["n_runners"] = df.groupby("race_id")["horse_number"].transform("count")

        return df

    def _bloodline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        血統特徴量: 父・母父の統計。
        sire_id or sire(名前) カラムで集計。
        なくても動く（スキップする）。
        """
        # sire_id があればそちら、なければ sire(名前) / dam_sire(名前) を使う
        group_pairs = []
        if "sire_id" in df.columns:
            group_pairs.append(("sire_id", "sire"))
            if "broodmare_sire_id" in df.columns:
                group_pairs.append(("broodmare_sire_id", "bms"))
        elif "sire" in df.columns:
            group_pairs.append(("sire", "sire"))
            if "dam_sire" in df.columns:
                group_pairs.append(("dam_sire", "bms"))

        if not group_pairs:
            return df

        for group_col, prefix in group_pairs:
            if group_col not in df.columns:
                continue

                # 父産駒の通算勝率
                sire_stats = df.groupby(group_col).agg(
                    sire_races=("finish_position", "count"),
                    sire_wins=("finish_position", lambda x: (x == 1).sum()),
                    sire_avg_pos=("finish_position", "mean"),
                ).reset_index()
                sire_stats[f"{prefix}_win_rate"] = sire_stats["sire_wins"] / sire_stats["sire_races"]
                sire_stats[f"{prefix}_avg_pos"] = sire_stats["sire_avg_pos"]

                df = df.merge(
                    sire_stats[[group_col, f"{prefix}_win_rate", f"{prefix}_avg_pos"]],
                    on=group_col, how="left"
                )

                # 距離適性（父産駒の同距離帯での成績）
                if "distance" in df.columns:
                    dist_cat = pd.cut(df["distance"], bins=[0, 1400, 1800, 2200, 9999], labels=[0, 1, 2, 3])
                    temp_col = f"_dist_cat_{prefix}"
                    df[temp_col] = dist_cat
                    sire_dist = df.groupby([group_col, temp_col])["finish_position"].agg(["mean", "count"]).reset_index()
                    sire_dist.columns = [group_col, temp_col, f"{prefix}_dist_avg_pos", f"{prefix}_dist_count"]
                    df = df.merge(sire_dist, on=[group_col, temp_col], how="left")
                    df.drop(columns=[temp_col], inplace=True)

                # コース適性（父産駒の芝/ダート別成績）
                if "course_type_enc" in df.columns:
                    sire_course = df.groupby([group_col, "course_type_enc"])["finish_position"].mean().reset_index()
                    sire_course.columns = [group_col, "course_type_enc", f"{prefix}_course_avg_pos"]
                    df = df.merge(sire_course, on=[group_col, "course_type_enc"], how="left")

        return df

    def _odds_movement(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        オッズ変動特徴量。
        morning_odds / final_odds カラムがある場合に使用。
        """
        if "morning_odds" in df.columns and "odds" in df.columns:
            # オッズの変動率
            df["odds_change_rate"] = (df["odds"] - df["morning_odds"]) / df["morning_odds"].clip(lower=0.1)

            # 急激に売れた馬（オッズが30%以上下がった）
            df["odds_drop_flag"] = (df["odds_change_rate"] < -0.3).astype(int)

            # 急激に人気が落ちた馬
            df["odds_rise_flag"] = (df["odds_change_rate"] > 0.5).astype(int)

        # 複勝オッズ÷単勝オッズ比（プロ資金の流入指標）
        if "fukusho_odds" in df.columns and "odds" in df.columns:
            df["fukusho_tansho_ratio"] = df["fukusho_odds"] / df["odds"].clip(lower=0.1)

        return df

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """モメンタム: 馬の調子の上昇/下降トレンド"""
        if "horse_id" not in df.columns:
            return df

        # 直近3走の着順トレンド（傾き）
        def calc_trend(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            try:
                slope = np.polyfit(x, series.values, 1)[0]
                return -slope  # 着順が下がる（良くなる）→ 正の値
            except (np.linalg.LinAlgError, ValueError):
                return 0

        df["position_trend_3"] = df.groupby("horse_id")["finish_position"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=2).apply(calc_trend, raw=False)
        )

        # 上り3Fのトレンド
        if "last_3f" in df.columns:
            df["last3f_trend_3"] = df.groupby("horse_id")["last_3f"].transform(
                lambda x: x.shift(1).rolling(3, min_periods=2).apply(calc_trend, raw=False)
            )

        return df

    def _class_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """クラス（格）関連特徴量"""
        if "prize_money" not in df.columns or "horse_id" not in df.columns:
            return df

        # 通算獲得賞金
        df["cumulative_prize"] = df.groupby("horse_id")["prize_money"].transform(
            lambda x: x.shift(1).cumsum()
        )

        # レース内の賞金順位（クラス内での相対的な格）
        df["prize_rank_in_race"] = df.groupby("race_id")["cumulative_prize"].rank(ascending=False)

        return df

    def _training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """調教データ特徴量"""
        if "training_eval_score" not in df.columns:
            return df

        # 調教評価スコア（A=5, B=4, C=3, D=2, E=1）はそのまま使える
        df["training_eval_score"] = pd.to_numeric(df["training_eval_score"], errors="coerce").fillna(0)

        # レース内での調教評価の相対順位
        df["training_rank_in_race"] = df.groupby("race_id")["training_eval_score"].rank(
            ascending=False, method="min"
        )

        # 調教評価がレース内平均を上回るか
        race_mean = df.groupby("race_id")["training_eval_score"].transform("mean")
        df["training_above_avg"] = (df["training_eval_score"] > race_mean).astype(int)

        # 馬の過去の調教評価平均（調教成績の安定度）
        if "horse_id" in df.columns:
            df["training_horse_avg"] = df.groupby("horse_id")["training_eval_score"].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            # 今回の調教が過去平均を上回るか（仕上がり度合い）
            df["training_improvement"] = df["training_eval_score"] - df["training_horse_avg"].fillna(0)

        print(f"  ✅ 調教特徴量追加")
        return df

    def get_extended_feature_columns(self, df: pd.DataFrame, base_features: list) -> list:
        """拡張特徴量を含む全特徴量カラムのリスト"""
        exclude = {
            "race_id", "race_date", "race_name", "horse_id", "horse_name",
            "jockey_id", "jockey_name", "trainer_id", "trainer_name",
            "finish_position", "finish_time", "finish_time_sec",
            "prize_money", "passing_order", "course_type", "weather",
            "track_condition", "sex", "distance_category", "frame_position",
            "speed_index", "norm_position", "rest_category", "season",
            "sire_id", "broodmare_sire_id", "morning_odds", "fukusho_odds",
            "sire", "dam", "sire_sire", "dam_sire", "training_evaluation", "training_comment",
            # リーク特徴量: 当該レースの結果データ
            "corner_1", "corner_2", "corner_3", "corner_4",
            "last_3f", "position_change",
        }
        all_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64", "int32", "float32"]]
        return all_cols


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_processor import SampleDataGenerator, RaceDataProcessor, FeatureEngineer

    raw = SampleDataGenerator().generate(n_races=200)
    proc = RaceDataProcessor()
    proc.raw_data = raw
    df = proc.preprocess()

    fe = FeatureEngineer()
    df = fe.create_features(df)

    afe = AdvancedFeatureEngineer()
    df = afe.create_all(df)

    cols = afe.get_extended_feature_columns(df, [])
    print(f"\n特徴量数: {len(cols)}")
    new_cols = [c for c in cols if c not in fe.get_feature_columns(df)]
    print(f"新規追加: {len(new_cols)}")
    for c in sorted(new_cols):
        print(f"  + {c}")
