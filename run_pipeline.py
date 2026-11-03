"""
競馬予想AI - 実データ統合パイプライン（最適化版）
=================================================
groupby+rolling を高速化し、実データで全工程を実行。
"""

import sys
import os
import json
import time
import gc
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processor import RaceDataProcessor
from model_trainer import RacePredictionModel, export_results_json
from backtester import Backtester


def fast_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """高速版特徴量エンジニアリング: lambdaを避けてベクトル演算を活用"""

    df = df.sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)

    print("  馬の過去成績...")
    # --- 馬の過去成績 (shift+rolling, lambda不使用) ---
    df_sorted = df.sort_values(["horse_id", "race_date"]).copy()

    # shift済みカラムを先に作成
    df_sorted["fp_shifted"] = df_sorted.groupby("horse_id")["finish_position"].shift(1)
    df_sorted["win_shifted"] = (df_sorted["fp_shifted"] == 1).astype(float)
    df_sorted["top3_shifted"] = (df_sorted["fp_shifted"] <= 3).astype(float)

    if "last_3f" in df_sorted.columns:
        df_sorted["last3f_shifted"] = df_sorted.groupby("horse_id")["last_3f"].shift(1)
    if "prize_money" in df_sorted.columns:
        df_sorted["prize_shifted"] = df_sorted.groupby("horse_id")["prize_money"].shift(1)

    for w in [3, 5, 10]:
        df_sorted[f"horse_avg_pos_{w}"] = df_sorted.groupby("horse_id")["fp_shifted"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df_sorted[f"horse_win_rate_{w}"] = df_sorted.groupby("horse_id")["win_shifted"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df_sorted[f"horse_top3_rate_{w}"] = df_sorted.groupby("horse_id")["top3_shifted"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        if "last3f_shifted" in df_sorted.columns:
            df_sorted[f"horse_avg_last3f_{w}"] = df_sorted.groupby("horse_id")["last3f_shifted"].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
        if "prize_shifted" in df_sorted.columns:
            df_sorted[f"horse_avg_prize_{w}"] = df_sorted.groupby("horse_id")["prize_shifted"].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )

    # 通算成績
    df_sorted["horse_total_races"] = df_sorted.groupby("horse_id").cumcount()
    df_sorted["horse_total_wins"] = df_sorted.groupby("horse_id")["win_shifted"].cumsum()

    # 前走間隔
    df_sorted["days_since_last_race"] = df_sorted.groupby("horse_id")["race_date"].diff().dt.days

    # 元のインデックス順に戻す
    feat_cols = [c for c in df_sorted.columns if c not in df.columns]
    df = df.merge(
        df_sorted[["race_id", "horse_number"] + feat_cols],
        on=["race_id", "horse_number"],
        how="left",
    )
    del df_sorted
    gc.collect()

    print("  騎手・調教師成績...")
    # --- 騎手成績 ---
    for entity, name in [("jockey_id", "jockey"), ("trainer_id", "trainer")]:
        if entity not in df.columns:
            continue
        df_s = df.sort_values([entity, "race_date"]).copy()
        df_s["_fp_shift"] = df_s.groupby(entity)["finish_position"].shift(1)
        df_s["_win_shift"] = (df_s["_fp_shift"] == 1).astype(float)
        df_s["_top3_shift"] = (df_s["_fp_shift"] <= 3).astype(float)

        df_s[f"{name}_avg_pos_30"] = df_s.groupby(entity)["_fp_shift"].transform(
            lambda x: x.rolling(30, min_periods=5).mean()
        )
        df_s[f"{name}_win_rate_30"] = df_s.groupby(entity)["_win_shift"].transform(
            lambda x: x.rolling(30, min_periods=5).mean()
        )
        df_s[f"{name}_top3_rate_30"] = df_s.groupby(entity)["_top3_shift"].transform(
            lambda x: x.rolling(30, min_periods=5).mean()
        )

        new_cols = [f"{name}_avg_pos_30", f"{name}_win_rate_30", f"{name}_top3_rate_30"]
        df = df.merge(
            df_s[["race_id", "horse_number"] + new_cols],
            on=["race_id", "horse_number"],
            how="left",
        )
        del df_s
        gc.collect()

    print("  レース内相対指標...")
    # --- 距離カテゴリ ---
    if "distance" in df.columns:
        df["distance_category_enc"] = pd.cut(
            df["distance"], bins=[0, 1400, 1800, 2200, 9999], labels=[0, 1, 2, 3]
        ).astype(float)

    # --- レース内相対指標 ---
    if "odds" in df.columns:
        df["odds_rank"] = df.groupby("race_id")["odds"].rank()
        df["odds_ratio"] = df["odds"] / df.groupby("race_id")["odds"].transform("min")

    if "horse_weight" in df.columns:
        df["weight_vs_avg"] = df["horse_weight"] - df.groupby("race_id")["horse_weight"].transform("mean")

    # --- 重馬場フラグ ---
    if "track_condition_enc" in df.columns:
        df["is_heavy_track"] = (df["track_condition_enc"] >= 2).astype(int)

    # --- スピード指数 ---
    if "finish_time_sec" in df.columns and "distance" in df.columns:
        base_time = df.groupby(["distance", "course_type_enc"])["finish_time_sec"].transform("median")
        df["speed_index"] = ((base_time - df["finish_time_sec"]) / base_time * 100).round(2)

        df_s = df.sort_values(["horse_id", "race_date"]).copy()
        df_s["si_shifted"] = df_s.groupby("horse_id")["speed_index"].shift(1)
        df_s["avg_speed_index_3"] = df_s.groupby("horse_id")["si_shifted"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df_s["max_speed_index_5"] = df_s.groupby("horse_id")["si_shifted"].transform(
            lambda x: x.rolling(5, min_periods=1).max()
        )
        df = df.merge(
            df_s[["race_id", "horse_number", "avg_speed_index_3", "max_speed_index_5"]],
            on=["race_id", "horse_number"], how="left",
        )
        del df_s
        gc.collect()

    print("  通過順・ペース...")
    # --- ペース特徴量 ---
    if "corner_1" in df.columns and "corner_4" in df.columns:
        df["position_change"] = df["corner_1"] - df["corner_4"]

    if "corner_1" in df.columns:
        df_s = df.sort_values(["horse_id", "race_date"]).copy()
        df_s["c1_shifted"] = df_s.groupby("horse_id")["corner_1"].shift(1)
        df_s["avg_early_position"] = df_s.groupby("horse_id")["c1_shifted"].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df = df.merge(
            df_s[["race_id", "horse_number", "avg_early_position"]],
            on=["race_id", "horse_number"], how="left",
        )
        del df_s
        gc.collect()

    print("  拡張特徴量...")
    # --- 拡張特徴量 ---
    # 出走頭数
    df["n_runners"] = df.groupby("race_id")["horse_number"].transform("count")

    # レースレベル
    if "odds" in df.columns:
        df["race_avg_odds"] = df.groupby("race_id")["odds"].transform("mean")
        df["odds_vs_race_avg"] = df["odds"] / df["race_avg_odds"].clip(lower=1)

    # 斤量変化
    if "weight" in df.columns:
        df_s = df.sort_values(["horse_id", "race_date"]).copy()
        df_s["weight_change"] = df_s.groupby("horse_id")["weight"].diff()
        df = df.merge(
            df_s[["race_id", "horse_number", "weight_change"]],
            on=["race_id", "horse_number"], how="left",
        )
        del df_s

    # 馬体重大幅変化
    if "weight_diff" in df.columns:
        df["big_weight_change"] = (df["weight_diff"].abs() >= 10).astype(int)

    # ローテーション
    df["prev_position"] = df.sort_values(["horse_id", "race_date"]).groupby("horse_id")["finish_position"].shift(1)
    df["prev_win_flag"] = (df["prev_position"] == 1).astype(int)
    df["prev_top3_flag"] = (df["prev_position"] <= 3).astype(int)
    df["rest_x_prev_pos"] = df["days_since_last_race"].fillna(90) * df["prev_position"].fillna(8)

    # 季節×馬齢
    if "month" in df.columns and "age" in df.columns:
        df["young_spring"] = ((df["age"] == 3) & (df["month"].between(3, 6))).astype(int)
        df["old_summer"] = ((df["age"] >= 5) & (df["month"].between(7, 9))).astype(int)
        df["age_x_season"] = df["age"] * ((df["month"] - 1) // 3)
        df["is_2yo_race"] = (df["age"] == 2).astype(int)

    # 内外フラグ
    if "frame_number" in df.columns:
        df["is_inner"] = (df["frame_number"] <= 3).astype(int)
        df["is_outer"] = (df["frame_number"] >= 6).astype(int)

    # 通算賞金
    if "prize_money" in df.columns:
        df_s = df.sort_values(["horse_id", "race_date"]).copy()
        df_s["cumulative_prize"] = df_s.groupby("horse_id")["prize_money"].transform(
            lambda x: x.shift(1).cumsum()
        )
        df_s["prize_rank_in_race"] = df_s.groupby("race_id")["cumulative_prize"].rank(ascending=False)
        df = df.merge(
            df_s[["race_id", "horse_number", "cumulative_prize", "prize_rank_in_race"]],
            on=["race_id", "horse_number"], how="left",
        )
        del df_s
        gc.collect()

    # NaN補完
    fill_defaults = {
        "days_since_last_race": 30.0,
        "prev_position": 8.0,
    }
    for col, val in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # --- ペースカテゴリ（脚質分類） ---
    print("  ペースカテゴリ・脚質...")
    if "corner_1" in df.columns and "corner_4" in df.columns:
        # corner_1: 最初の通過順, corner_4: 最終コーナー通過順
        # first > last（後方→前方）= 差し/追込、first < last（前方→後方）= 逃げ/先行
        # ここではcorner_1が小さい=前にいる、corner_4が小さい=最終も前にいる
        df_s = df.sort_values(["horse_id", "race_date"]).copy()
        df_s["_c1_shift"] = df_s.groupby("horse_id")["corner_1"].shift(1)
        df_s["_c4_shift"] = df_s.groupby("horse_id")["corner_4"].shift(1)
        # 前走の脚質: corner_1 > corner_4 なら差し/追込(ポジション上げた), corner_1 < corner_4 なら逃げ/先行(前に行って下がった or キープ)
        # 0: 不明, 1: 逃げ/先行(front), 2: 中団, 3: 差し/追込(closer)
        pace = pd.Series(0, index=df_s.index, dtype=int)
        diff = df_s["_c1_shift"] - df_s["_c4_shift"]
        pace = np.where(diff > 1, 3, np.where(diff < -1, 1, 2))  # diff>1: 後方→前方=closer, diff<-1: 前→後=先行, それ以外=中団
        df_s["pace_category"] = pace
        # corner_1が3以内なら強制的に逃げ/先行
        df_s.loc[df_s["_c1_shift"] <= 3, "pace_category"] = 1
        df = df.merge(
            df_s[["race_id", "horse_number", "pace_category"]],
            on=["race_id", "horse_number"], how="left",
        )
        df["pace_category"] = df["pace_category"].fillna(0).astype(int)
        del df_s
        gc.collect()

    # --- 枠順バイアス ---
    print("  枠順バイアス...")
    if "frame_number" in df.columns and "venue" in df.columns and "distance" in df.columns:
        # 枠グループ: 1-4=内枠(0), 5-8=外枠(1)
        df["_frame_group"] = (df["frame_number"] >= 5).astype(int)
        ct_col = "course_type" if "course_type" in df.columns else ("course_type_enc" if "course_type_enc" in df.columns else None)
        if ct_col:
            df["_fb_key"] = df["venue"].astype(str) + "_" + df["distance"].astype(str) + "_" + df[ct_col].astype(str)
        else:
            df["_fb_key"] = df["venue"].astype(str) + "_" + df["distance"].astype(str)
        df["_is_win"] = (df["finish_position"] == 1).astype(float)
        df_s = df.sort_values("race_date").copy()
        # 枠グループ×条件ごとの過去勝率（shift済みでリーク防止）
        df_s["_fb_combined"] = df_s["_fb_key"] + "_" + df_s["_frame_group"].astype(str)
        df_s["frame_bias"] = df_s.groupby("_fb_combined")["_is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df = df.merge(
            df_s[["race_id", "horse_number", "frame_bias"]],
            on=["race_id", "horse_number"], how="left",
        )
        df["frame_bias"] = df["frame_bias"].fillna(0)
        df.drop(columns=["_frame_group", "_fb_key", "_is_win"], inplace=True, errors="ignore")
        del df_s
        gc.collect()

    # --- 人気乖離（予測順位 vs オッズ順位） ---
    print("  人気乖離...")
    if "odds" in df.columns:
        # オッズベースの人気順位（低オッズ = 高人気）
        df["_odds_rank"] = df.groupby("race_id")["odds"].rank(method="min")
        # 予測順位がまだないのでレース内の着順傾向指標として過去のオッズ順位と実際着順のギャップを使う
        # popularity_gap: オッズ人気順位 - 馬番順位（仮）→ 正の値 = 人気以上にオッズが高い（妙味あり）
        if "horse_avg_pos_5" in df.columns:
            # 過去5走平均着順でのレース内順位 vs オッズ順位
            df["_perf_rank"] = df.groupby("race_id")["horse_avg_pos_5"].rank(method="min")
            df["popularity_gap"] = df["_odds_rank"] - df["_perf_rank"]
            df.drop(columns=["_perf_rank"], inplace=True, errors="ignore")
        else:
            df["popularity_gap"] = 0
        df.drop(columns=["_odds_rank"], inplace=True, errors="ignore")

    # --- 馬体重変動の影響 ---
    print("  馬体重変動...")
    if "horse_weight" in df.columns:
        df_s = df.sort_values(["horse_id", "race_date"]).copy()
        df_s["_prev_hw"] = df_s.groupby("horse_id")["horse_weight"].shift(1)
        df_s["_hw_diff"] = df_s["horse_weight"] - df_s["_prev_hw"]
        # カテゴリ化: -3=大幅減(-20以下), -2=減(-10~-20), -1=微減(-1~-10), 0=変化なし, 1=微増, 2=増, 3=大幅増
        conditions = [
            df_s["_hw_diff"] <= -20,
            df_s["_hw_diff"].between(-19, -10),
            df_s["_hw_diff"].between(-9, -1),
            df_s["_hw_diff"].between(0, 0),
            df_s["_hw_diff"].between(1, 9),
            df_s["_hw_diff"].between(10, 19),
            df_s["_hw_diff"] >= 20,
        ]
        choices = [-3, -2, -1, 0, 1, 2, 3]
        df_s["weight_change_effect"] = np.select(conditions, choices, default=0)
        df = df.merge(
            df_s[["race_id", "horse_number", "weight_change_effect"]],
            on=["race_id", "horse_number"], how="left",
        )
        df["weight_change_effect"] = df["weight_change_effect"].fillna(0).astype(int)
        del df_s
        gc.collect()

    # --- 脚質×距離適性 ---
    print("  脚質×距離適性...")
    if "last_3f" in df.columns and "distance" in df.columns:
        df_s = df.sort_values(["horse_id", "race_date"]).copy()
        df_s["_last3f_shift"] = df_s.groupby("horse_id")["last_3f"].shift(1)
        df_s["_avg_last3f"] = df_s.groupby("horse_id")["_last3f_shift"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df = df.merge(
            df_s[["race_id", "horse_number", "_avg_last3f"]],
            on=["race_id", "horse_number"], how="left",
        )
        # 上がり3Fが速い（低い値）+ 長距離 = 差し追込適性が高い
        # last_3f median付近を基準に速い/遅いを判定
        last3f_median = df["_avg_last3f"].median()
        dist_median = df["distance"].median()
        # スコア: 上がりが速い(負方向) × 距離が長い(正方向) → 大きい値 = 差し馬×長距離でマッチ
        df["running_style_match"] = (
            (last3f_median - df["_avg_last3f"].fillna(last3f_median)) *
            (df["distance"] - dist_median) / dist_median
        ).fillna(0).round(3)
        df.drop(columns=["_avg_last3f"], inplace=True, errors="ignore")
        del df_s
        gc.collect()

    # --- 追加データ統合（血統・払戻） ---
    df = integrate_extra_data(df)

    print(f"  完了: {len(df.columns)} カラム")
    return df


def integrate_extra_data(df: pd.DataFrame) -> pd.DataFrame:
    """血統・払戻データがあれば特徴量として統合"""

    # 血統データ
    bloodline_path = "data/bloodline.csv"
    if os.path.exists(bloodline_path):
        print("  血統データ統合...")
        bl = pd.read_csv(bloodline_path)
        bl["horse_id"] = bl["horse_id"].astype(str)

        # 種牡馬エンコーディング（頻度ベース）
        for col in ["sire", "dam_sire"]:
            if col in bl.columns:
                freq = bl[col].value_counts()
                bl[f"{col}_freq"] = bl[col].map(freq).fillna(0)

                # 種牡馬の平均勝率（血統データとレースデータの結合）
                merged_tmp = df[["horse_id", "finish_position"]].merge(
                    bl[["horse_id", col]], on="horse_id", how="inner"
                )
                if not merged_tmp.empty:
                    sire_stats = merged_tmp.groupby(col)["finish_position"].agg(
                        [("win_rate", lambda x: (x == 1).mean()),
                         ("top3_rate", lambda x: (x <= 3).mean()),
                         ("avg_pos", "mean"),
                         ("n_races", "count")]
                    ).reset_index()
                    sire_stats = sire_stats[sire_stats["n_races"] >= 10]
                    sire_map_win = dict(zip(sire_stats[col], sire_stats["win_rate"]))
                    sire_map_top3 = dict(zip(sire_stats[col], sire_stats["top3_rate"]))
                    sire_map_pos = dict(zip(sire_stats[col], sire_stats["avg_pos"]))
                    bl[f"{col}_win_rate"] = bl[col].map(sire_map_win).fillna(0)
                    bl[f"{col}_top3_rate"] = bl[col].map(sire_map_top3).fillna(0)
                    bl[f"{col}_avg_pos"] = bl[col].map(sire_map_pos).fillna(8)

        # 結合
        bl_features = [c for c in bl.columns if c != "horse_id" and bl[c].dtype in ["float64", "int64"]]
        if bl_features:
            df["horse_id"] = df["horse_id"].astype(str)
            df = df.merge(bl[["horse_id"] + bl_features], on="horse_id", how="left")
            for c in bl_features:
                df[c] = df[c].fillna(0)
            print(f"    血統特徴量追加: {len(bl_features)} 個")

    # 調教評価データ
    training_path = "data/training.csv"
    if os.path.exists(training_path):
        print("  調教データ統合...")
        tr = pd.read_csv(training_path)
        tr["race_id"] = tr["race_id"].astype(str)
        df["race_id"] = df["race_id"].astype(str)

        if "training_eval_score" in tr.columns:
            tr_feat = tr[["race_id", "horse_number", "training_eval_score"]].drop_duplicates(
                subset=["race_id", "horse_number"], keep="first"
            )
            tr_feat["training_eval_rank"] = tr_feat.groupby("race_id")["training_eval_score"].rank(ascending=False)
            df = df.merge(tr_feat, on=["race_id", "horse_number"], how="left")
            df["training_eval_score"] = df["training_eval_score"].fillna(0)
            df["training_eval_rank"] = df["training_eval_rank"].fillna(0)
            print(f"    調教特徴量追加: training_eval_score, training_eval_rank")
        elif "final_time" in tr.columns:
            tr["final_time_num"] = pd.to_numeric(tr["final_time"], errors="coerce")
            tr_agg = tr.groupby(["race_id", "horse_number"]).agg(
                training_time_best=("final_time_num", "min"),
                training_time_mean=("final_time_num", "mean"),
            ).reset_index()
            tr_agg["training_rank"] = tr_agg.groupby("race_id")["training_time_best"].rank()
            df = df.merge(tr_agg, on=["race_id", "horse_number"], how="left")
            for c in ["training_time_best", "training_time_mean", "training_rank"]:
                if c in df.columns:
                    df[c] = df[c].fillna(0)
            print(f"    調教特徴量追加: training_time_best, training_time_mean, training_rank")

    # 払戻データ → 競馬場×距離の荒れ傾向を特徴量化（リーク防止: 過去データのみ使用）
    payouts_path = "data/payouts.csv"
    if os.path.exists(payouts_path) and "venue_code" in df.columns and "distance" in df.columns:
        print("  払戻データ統合...")
        pay = pd.read_csv(payouts_path)
        pay["race_id"] = pay["race_id"].astype(str)
        df["race_id"] = df["race_id"].astype(str)

        # 単勝配当をレースに紐付け（荒れ度の指標）
        tansho = pay[pay["bet_type"] == "単勝"][["race_id", "payout"]].rename(
            columns={"payout": "tansho_payout"}
        )
        df = df.merge(tansho, on="race_id", how="left")

        # 競馬場×距離帯の過去平均単勝配当（荒れやすさ）
        if "tansho_payout" in df.columns:
            dist_bin = (df["distance"] // 200) * 200  # 200m刻み
            df["_venue_dist"] = df["venue_code"].astype(str) + "_" + dist_bin.astype(str)

            # 過去の平均をgroupby+expanding（shift済み）で算出
            df_sorted = df.sort_values("race_date")
            venue_dist_avg = df_sorted.groupby("_venue_dist")["tansho_payout"].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            df["venue_dist_avg_tansho"] = venue_dist_avg.fillna(0)
            df["venue_dist_upset_tendency"] = (df["venue_dist_avg_tansho"] > 1000).astype(int)
            df.drop(columns=["_venue_dist", "tansho_payout"], inplace=True)
            print("    払戻特徴量追加: venue_dist_avg_tansho, venue_dist_upset_tendency")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """特徴量カラム抽出"""
    exclude = {
        "race_id", "race_date", "race_name", "horse_id", "horse_name",
        "jockey_id", "jockey_name", "trainer_id", "trainer_name",
        "finish_position", "finish_time", "finish_time_sec",
        "prize_money", "passing_order", "course_type", "weather",
        "track_condition", "sex", "distance_category", "frame_position",
        "speed_index", "norm_position", "rest_category", "season",
        "venue", "fp_shifted", "win_shifted", "top3_shifted",
        "last3f_shifted", "prize_shifted", "si_shifted", "c1_shifted",
        "_fp_shift", "_win_shift", "_top3_shift",
    }
    return [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64", "int32", "float32"]]


def main():
    start_time = time.time()

    print("=" * 70)
    print("  競馬予想AI - 実データ統合パイプライン")
    print("=" * 70)

    # ============================================================
    # STEP 1: データ読み込み
    # ============================================================
    print("\n[1/6] データ読み込み...")
    dfs = []
    for year in range(2015, 2026):
        path = f"data/races_{year}.csv"
        if os.path.exists(path):
            df_year = pd.read_csv(path)
            dfs.append(df_year)
            print(f"  {year}: {len(df_year):,} 件")
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    print(f"  合計: {len(df):,} 件")

    # ============================================================
    # STEP 2: 前処理
    # ============================================================
    print("\n[2/6] 前処理...")
    processor = RaceDataProcessor(data_dir="data")
    processor.raw_data = df
    df = processor.preprocess()
    print(f"  {len(df):,} 件 / {df['race_id'].nunique():,} レース")

    # ============================================================
    # STEP 3: 特徴量エンジニアリング
    # ============================================================
    print("\n[3/6] 特徴量エンジニアリング...")
    df = fast_feature_engineering(df)
    feature_cols = get_feature_columns(df)
    print(f"  特徴量: {len(feature_cols)} 個")

    # ============================================================
    # STEP 4: モデル学習
    # ============================================================
    print("\n[4/6] モデル学習...")
    train_df = df[df["year"] <= 2024].copy()
    test_df = df[df["year"] == 2025].copy()
    print(f"  学習: {len(train_df):,} 件 [2015-2024]")
    print(f"  テスト: {len(test_df):,} 件 [2025]")

    model = RacePredictionModel()
    model.train(train_df, feature_cols, n_splits=5)
    os.makedirs("models", exist_ok=True)
    model.save("keiba_model")
    del train_df
    gc.collect()

    # ============================================================
    # STEP 5: 精度評価
    # ============================================================
    print("\n[5/6] 2025年精度評価...")
    predictions = model.predict(test_df)

    merged = predictions.merge(
        test_df[["race_id", "horse_number", "finish_position", "odds"]],
        on=["race_id", "horse_number"], how="inner",
    )

    top1 = merged[merged["predicted_rank"] == 1]
    top1_acc = (top1["finish_position"] == 1).mean()
    top3_contains = merged.groupby("race_id").apply(
        lambda g: (g[g["predicted_rank"] <= 3]["finish_position"] == 1).any()
    ).mean()
    top3 = merged[merged["predicted_rank"] <= 3]
    top3_place = (top3["finish_position"] <= 3).mean()
    top5 = merged[merged["predicted_rank"] <= 5]
    top5_place = (top5["finish_position"] <= 3).mean()

    print(f"\n  予測1位→1着的中率:     {top1_acc*100:.1f}%")
    print(f"  上位3頭に1着馬含有率:  {top3_contains*100:.1f}%")
    print(f"  上位3頭の複勝的中率:   {top3_place*100:.1f}%")
    print(f"  上位5頭の複勝的中率:   {top5_place*100:.1f}%")

    if "popularity" in test_df.columns:
        top1_pop = top1.merge(
            test_df[["race_id", "horse_number", "popularity"]],
            on=["race_id", "horse_number"], how="left",
        )
        print("\n  【人気別】")
        for lo, hi, label in [(1, 3, "1-3人気"), (4, 6, "4-6人気"), (7, 99, "7人気~")]:
            mask = top1_pop["popularity"].between(lo, hi)
            if mask.sum() > 0:
                print(f"    {label}: {(top1_pop.loc[mask, 'finish_position'] == 1).mean()*100:.1f}% ({mask.sum()}件)")

    # ============================================================
    # STEP 6: バックテスト
    # ============================================================
    print("\n[6/6] バックテスト...")
    bt = Backtester(test_df, predictions)
    bt_results = bt.run_all_strategies(bet_amount=100, bankroll=100000)
    bt.print_comparison(bt_results)

    best = max(bt_results, key=lambda x: x.roi_percent)
    if best.monthly_results:
        print(f"\n  【{best.strategy_name} 月別】")
        for m in best.monthly_results:
            print(f"    {m['month']}: ROI {m['roi']:.1f}% / 的中 {m['hit_rate']:.1f}%")

    if best.venue_results:
        print(f"\n  【{best.strategy_name} 場別 (10件以上)】")
        for v, d in sorted(best.venue_results.items(), key=lambda x: x[1].get("roi", 0), reverse=True):
            if d["count"] >= 10:
                print(f"    {v}: ROI {d['roi']:.1f}% ({d['count']}件)")

    # ============================================================
    # 保存
    # ============================================================
    export_results_json(predictions, "data/predictions_2025.json")
    bt.export_json(bt_results, "data/backtest_results.json")

    summary = {
        "pipeline_run": datetime.now().isoformat(),
        "data": {"records": len(df), "test_records": len(test_df), "years": "2015-2025", "n_features": len(feature_cols)},
        "accuracy": {
            "top1_win_rate": round(top1_acc * 100, 1),
            "top3_contains_winner": round(top3_contains * 100, 1),
            "top3_place_rate": round(top3_place * 100, 1),
            "top5_place_rate": round(top5_place * 100, 1),
        },
        "model_evaluation": model.evaluation_results,
        "backtest": [
            {"strategy": r.strategy_name, "roi": r.roi_percent, "hit_rate": r.hit_rate, "profit": r.profit}
            for r in sorted(bt_results, key=lambda x: x.roi_percent, reverse=True)
        ],
        "top_features": {
            n: model.get_top_features(n, 15).to_dict("records")
            for n in ["position", "top3", "win"] if n in model.feature_importance
        },
    }
    with open("data/pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  完了！ ({elapsed/60:.1f} 分)")
    print(f"{'='*70}")
    print(f"  models/keiba_model.pkl       - モデル")
    print(f"  data/predictions_2025.json   - 予測結果")
    print(f"  data/backtest_results.json   - バックテスト")
    print(f"  data/pipeline_summary.json   - サマリー")


if __name__ == "__main__":
    main()
