"""
競馬予想AI - 高度学習スクリプト（ローカルPC用）
================================================
48GB RAM + NVIDIA GPU を活用した本格的な学習パイプライン。

サーバー版との違い:
  - 全データ使用（130万件、制限なし）
  - 拡張特徴量すべて有効（88+特徴量）
  - Optunaによるハイパーパラメータ自動最適化
  - LightGBM + XGBoost の2モデルスタッキング
  - SHAP による特徴量重要度分析
  - 詳細な評価レポート + グラフ出力

使い方:
  python train_advanced.py
"""

import os
import sys
import time
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
CONFIG = {
    "data_file": "data/_all_races.csv",
    "bloodline_file": "data/bloodline.csv",
    "output_dir": "output",
    "model_dir": "models",

    # 学習設定
    "use_all_data": True,          # True=全データ, False=直近10年
    "use_advanced_features": True,  # 拡張特徴量を使用
    "use_gpu": True,               # GPU使用（NVIDIA）
    "n_cv_folds": 5,               # CV分割数
    "optuna_trials": 100,          # ハイパーパラメータ探索回数

    # アンサンブル
    "use_xgboost": True,           # XGBoostも使用
    "use_stacking": True,          # スタッキングアンサンブル

    # 出力
    "save_shap": True,             # SHAP分析
    "save_charts": True,           # グラフ保存
}


# ──────────────────────────────────────────────
# データ処理（サーバー版と同じロジック）
# ──────────────────────────────────────────────
class DataProcessor:
    """データ読み込み・前処理"""

    TRACK_CONDITION_MAP = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
    WEATHER_MAP = {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "小雪": 4, "雪": 5}
    COURSE_TYPE_MAP = {"芝": 0, "ダート": 1, "障害": 2}

    def load(self, filepath: str) -> pd.DataFrame:
        print(f"📂 データ読み込み: {filepath}")
        df = pd.read_csv(filepath)
        print(f"   {len(df):,} 件 / {len(df.columns)} カラム")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🔧 前処理中...")
        df = df.copy()

        # 日付
        if "race_date" in df.columns:
            df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce", format="mixed")
            df["year"] = df["race_date"].dt.year
            df["month"] = df["race_date"].dt.month
            df["day_of_week"] = df["race_date"].dt.dayofweek

        # 数値変換
        numeric_cols = [
            "distance", "weight", "horse_weight", "weight_diff",
            "frame_number", "horse_number", "odds", "popularity",
            "finish_position", "prize_money", "last_3f", "age"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # タイム変換
        if "finish_time" in df.columns:
            df["finish_time_sec"] = df["finish_time"].apply(self._parse_time)

        # 通過順
        if "passing_order" in df.columns:
            df = self._parse_passing_order(df)

        # カテゴリエンコード
        if "track_condition" in df.columns:
            df["track_condition_enc"] = df["track_condition"].map(self.TRACK_CONDITION_MAP)
        if "weather" in df.columns:
            df["weather_enc"] = df["weather"].map(self.WEATHER_MAP)
        if "course_type" in df.columns:
            df["course_type_enc"] = df["course_type"].map(self.COURSE_TYPE_MAP)
        if "sex" in df.columns:
            df["sex_enc"] = df["sex"].map({"牡": 0, "牝": 1, "セ": 2}).fillna(-1).astype(int)

        # DNF除外（NaN=予測対象は保持）
        if "finish_position" in df.columns:
            has_pos = df["finish_position"].notna()
            valid = has_pos & (df["finish_position"] > 0)
            df = df[valid | ~has_pos]

        print(f"   前処理完了: {len(df):,} 件")
        return df

    @staticmethod
    def _parse_time(time_str):
        if pd.isna(time_str):
            return None
        try:
            time_str = str(time_str).strip()
            if not time_str or time_str == '---':
                return None
            if ":" in time_str:
                parts = time_str.split(":")
                if len(parts) != 2:
                    return None
                return float(parts[0]) * 60 + float(parts[1])
            val = float(time_str)
            return val if val > 0 else None
        except (ValueError, IndexError, TypeError):
            return None

    @staticmethod
    def _parse_passing_order(df):
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
            df[f"corner_{i+1}"] = parsed.apply(lambda x, idx=i: x[idx])
        return df


# ──────────────────────────────────────────────
# 特徴量エンジニアリング
# ──────────────────────────────────────────────
class FeatureEngine:
    """基本 + 拡張特徴量を一括生成"""

    # リーク特徴量（予測時に使えない情報）
    LEAK_FEATURES = {
        "finish_position", "finish_time", "finish_time_sec",
        "prize_money", "passing_order",
        "speed_index", "norm_position",
        "corner_1", "corner_2", "corner_3", "corner_4",
        "last_3f", "position_change",
    }

    # 除外カラム（ID・文字列等）
    EXCLUDE_COLS = {
        "race_id", "race_date", "race_name", "horse_id", "horse_name",
        "jockey_id", "jockey_name", "trainer_id", "trainer_name",
        "course_type", "weather", "track_condition", "sex",
        "distance_category", "frame_position", "rest_category", "season",
        "sire", "dam", "sire_sire", "dam_sire",
        "sire_id", "broodmare_sire_id",
        "morning_odds", "fukusho_odds",
        "training_evaluation", "training_comment",
    }

    def create_all(self, df: pd.DataFrame, use_advanced: bool = True) -> pd.DataFrame:
        print("🔧 特徴量エンジニアリング開始...")
        t0 = time.time()

        df = df.sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)

        # === 基本特徴量 ===
        print("  [1/10] 馬の過去成績...")
        df = self._horse_history(df)
        print("  [2/10] 騎手成績...")
        df = self._jockey_features(df)
        print("  [3/10] 調教師成績...")
        df = self._trainer_features(df)
        print("  [4/10] 距離適性...")
        df = self._distance_features(df)
        print("  [5/10] コース適性...")
        df = self._course_type_features(df)
        print("  [6/10] レース内相対指標...")
        df = self._race_relative(df)
        print("  [7/10] スピード指数...")
        df = self._speed_index(df)

        if use_advanced:
            # === 拡張特徴量 ===
            print("  [8/10] 拡張: 斤量・ローテーション...")
            df = self._weight_analysis(df)
            df = self._rotation_features(df)
            print("  [9/10] 拡張: モメンタム・クラス...")
            df = self._momentum_features(df)
            df = self._class_features(df)
            df = self._season_age_interaction(df)
            print("  [10/10] 拡張: レースレベル・馬場バイアス...")
            df = self._race_level(df)
            df = self._track_bias(df)
            df = self._normalized_position(df)

        # NaN補完
        df = self._fill_nan(df)

        elapsed = time.time() - t0
        print(f"✅ 特徴量エンジニアリング完了: {len(df.columns)} カラム ({elapsed:.0f}秒)")
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        exclude = self.LEAK_FEATURES | self.EXCLUDE_COLS
        return [c for c in df.columns
                if c not in exclude
                and df[c].dtype in ["float64", "int64", "int32", "float32"]]

    # ── 基本特徴量 ──

    def _horse_history(self, df):
        if "horse_id" not in df.columns:
            return df
        for w in [3, 5, 10]:
            g = df.groupby("horse_id")
            df[f"horse_avg_pos_{w}"] = g["finish_position"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            df[f"horse_win_rate_{w}"] = g["finish_position"].transform(
                lambda x: (x.shift(1) == 1).rolling(w, min_periods=1).mean())
            df[f"horse_top3_rate_{w}"] = g["finish_position"].transform(
                lambda x: (x.shift(1) <= 3).rolling(w, min_periods=1).mean())
            if "last_3f" in df.columns:
                df[f"horse_avg_last3f_{w}"] = g["last_3f"].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            if "prize_money" in df.columns:
                df[f"horse_avg_prize_{w}"] = g["prize_money"].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean())

        df["horse_total_races"] = df.groupby("horse_id").cumcount()
        df["horse_total_wins"] = df.groupby("horse_id")["finish_position"].transform(
            lambda x: (x.shift(1) == 1).cumsum())
        if "race_date" in df.columns:
            df["days_since_last_race"] = df.groupby("horse_id")["race_date"].diff().dt.days
        return df

    def _jockey_features(self, df):
        if "jockey_id" not in df.columns:
            return df
        g = df.groupby("jockey_id")
        df["jockey_avg_pos_30"] = g["finish_position"].transform(
            lambda x: x.shift(1).rolling(30, min_periods=5).mean())
        df["jockey_win_rate_30"] = g["finish_position"].transform(
            lambda x: (x.shift(1) == 1).rolling(30, min_periods=5).mean())
        df["jockey_top3_rate_30"] = g["finish_position"].transform(
            lambda x: (x.shift(1) <= 3).rolling(30, min_periods=5).mean())
        return df

    def _trainer_features(self, df):
        if "trainer_id" not in df.columns:
            return df
        g = df.groupby("trainer_id")
        df["trainer_avg_pos_30"] = g["finish_position"].transform(
            lambda x: x.shift(1).rolling(30, min_periods=5).mean())
        df["trainer_win_rate_30"] = g["finish_position"].transform(
            lambda x: (x.shift(1) == 1).rolling(30, min_periods=5).mean())
        return df

    def _distance_features(self, df):
        if "distance" not in df.columns or "horse_id" not in df.columns:
            return df
        df["distance_category"] = pd.cut(
            df["distance"], bins=[0, 1400, 1800, 2200, 9999],
            labels=["sprint", "mile", "middle", "long"])
        df["horse_dist_avg_pos"] = df.groupby(["horse_id", "distance_category"])[
            "finish_position"].transform(lambda x: x.shift(1).expanding().mean())
        df["horse_dist_top3_rate"] = df.groupby(["horse_id", "distance_category"])[
            "finish_position"].transform(lambda x: (x.shift(1) <= 3).expanding().mean())
        return df

    def _course_type_features(self, df):
        if "course_type_enc" not in df.columns or "horse_id" not in df.columns:
            return df
        df["horse_course_avg_pos"] = df.groupby(["horse_id", "course_type_enc"])[
            "finish_position"].transform(lambda x: x.shift(1).expanding().mean())
        df["horse_course_top3_rate"] = df.groupby(["horse_id", "course_type_enc"])[
            "finish_position"].transform(lambda x: (x.shift(1) <= 3).expanding().mean())
        return df

    def _race_relative(self, df):
        if "odds" in df.columns:
            df["odds_rank"] = df.groupby("race_id")["odds"].rank()
            df["odds_ratio"] = df.groupby("race_id")["odds"].transform(lambda x: x / x.min())
        if "horse_weight" in df.columns:
            df["weight_vs_avg"] = df.groupby("race_id")["horse_weight"].transform(lambda x: x - x.mean())
        df["is_heavy_track"] = (df.get("track_condition_enc", pd.Series(dtype=float)) >= 2).astype(int) if "track_condition_enc" in df.columns else 0
        return df

    def _speed_index(self, df):
        if "finish_time_sec" not in df.columns or "distance" not in df.columns:
            return df
        base_time = df.groupby(["distance", "course_type_enc"])["finish_time_sec"].transform("median")
        df["speed_index"] = ((base_time - df["finish_time_sec"]) / base_time * 100).round(2)
        if "horse_id" in df.columns:
            df["avg_speed_index_3"] = df.groupby("horse_id")["speed_index"].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean())
            df["max_speed_index_5"] = df.groupby("horse_id")["speed_index"].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).max())
        return df

    # ── 拡張特徴量 ──

    def _normalized_position(self, df):
        if "finish_position" not in df.columns:
            return df
        df["n_runners"] = df.groupby("race_id")["horse_number"].transform("count")
        df["norm_position"] = (df["finish_position"] - 1) / (df["n_runners"] - 1).clip(lower=1)
        if "horse_id" in df.columns:
            for w in [3, 5]:
                df[f"horse_norm_pos_avg_{w}"] = df.groupby("horse_id")["norm_position"].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        return df

    def _weight_analysis(self, df):
        if "weight" not in df.columns or "horse_id" not in df.columns:
            return df
        df["weight_change"] = df.groupby("horse_id")["weight"].diff()
        df["weight_vs_self_avg"] = df.groupby("horse_id")["weight"].transform(
            lambda x: x - x.shift(1).expanding().mean())
        if "horse_weight" in df.columns:
            df["horse_weight_change_rate"] = df.groupby("horse_id")["horse_weight"].transform(
                lambda x: x.pct_change())
            if "weight_diff" in df.columns:
                df["big_weight_change"] = (df["weight_diff"].abs() >= 10).astype(int)
        return df

    def _rotation_features(self, df):
        if "horse_id" not in df.columns or "race_date" not in df.columns:
            return df
        if "days_since_last_race" not in df.columns:
            df["days_since_last_race"] = df.groupby("horse_id")["race_date"].diff().dt.days
        df["prev_position"] = df.groupby("horse_id")["finish_position"].shift(1)
        df["rest_x_prev_pos"] = df["days_since_last_race"].fillna(90) * df["prev_position"].fillna(8)
        df["prev_win_flag"] = (df["prev_position"] == 1).astype(int)
        df["prev_top3_flag"] = (df["prev_position"] <= 3).astype(int)

        def calc_streak(series):
            streak = 0
            result = []
            for val in series:
                streak = streak + 1 if val <= 3 else 0
                result.append(streak)
            return pd.Series(result, index=series.index)

        df["top3_streak"] = df.groupby("horse_id")["finish_position"].transform(
            lambda x: calc_streak(x.shift(1).fillna(99)))
        return df

    def _momentum_features(self, df):
        if "horse_id" not in df.columns:
            return df

        def calc_trend(series):
            if len(series) < 2:
                return 0
            try:
                return -np.polyfit(np.arange(len(series)), series.values, 1)[0]
            except:
                return 0

        df["position_trend_3"] = df.groupby("horse_id")["finish_position"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=2).apply(calc_trend, raw=False))
        if "last_3f" in df.columns:
            df["last3f_trend_3"] = df.groupby("horse_id")["last_3f"].transform(
                lambda x: x.shift(1).rolling(3, min_periods=2).apply(calc_trend, raw=False))
        return df

    def _class_features(self, df):
        if "prize_money" not in df.columns or "horse_id" not in df.columns:
            return df
        df["cumulative_prize"] = df.groupby("horse_id")["prize_money"].transform(
            lambda x: x.shift(1).cumsum())
        df["prize_rank_in_race"] = df.groupby("race_id")["cumulative_prize"].rank(ascending=False)
        return df

    def _season_age_interaction(self, df):
        if "month" not in df.columns or "age" not in df.columns:
            return df
        df["young_spring"] = ((df["age"] == 3) & (df["month"].between(3, 6))).astype(int)
        df["old_summer"] = ((df["age"] >= 5) & (df["month"].between(7, 9))).astype(int)
        season_enc = (df["month"] - 1) // 3
        df["age_x_season"] = df["age"] * season_enc
        df["is_2yo_race"] = (df["age"] == 2).astype(int)
        return df

    def _race_level(self, df):
        if "odds" in df.columns:
            df["race_avg_odds"] = df.groupby("race_id")["odds"].transform("mean")
            df["race_min_odds"] = df.groupby("race_id")["odds"].transform("min")
            df["odds_vs_race_avg"] = df["odds"] / df["race_avg_odds"].clip(lower=1)
        if "n_runners" not in df.columns:
            df["n_runners"] = df.groupby("race_id")["horse_number"].transform("count")
        return df

    def _track_bias(self, df):
        if "frame_number" not in df.columns:
            return df
        df["is_inner"] = (df["frame_number"] <= 3).astype(int)
        df["is_outer"] = (df["frame_number"] >= 6).astype(int)
        return df

    def _fill_nan(self, df):
        fill_targets = [
            "horse_dist_avg_pos", "horse_dist_top3_rate",
            "horse_course_avg_pos", "horse_course_top3_rate",
            "days_since_last_race",
        ]
        for col in fill_targets:
            if col in df.columns:
                med = df[col].median()
                df[col] = df[col].fillna(med if pd.notna(med) else 0)
        return df


# ──────────────────────────────────────────────
# Optunaハイパーパラメータ最適化
# ──────────────────────────────────────────────
def optimize_lgbm(X_train, y_train, X_val, y_val, task="binary", n_trials=100, use_gpu=False):
    """Optunaでベストパラメータを探索"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "regression" if task == "regression" else "binary",
            "metric": "mae" if task == "regression" else "binary_logloss",
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,

            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }

        if use_gpu:
            params["device"] = "gpu"
            params["gpu_use_dp"] = True

        import lightgbm as lgb
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params, dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        if task == "regression":
            from sklearn.metrics import mean_absolute_error
            pred = model.predict(X_val)
            return mean_absolute_error(y_val, pred)
        else:
            from sklearn.metrics import roc_auc_score
            pred = model.predict(X_val)
            try:
                return roc_auc_score(y_val, pred)
            except ValueError:
                return 0.5

    direction = "minimize" if task == "regression" else "maximize"
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  ✅ ベストスコア: {study.best_value:.4f}")
    print(f"  ✅ ベストパラメータ: {study.best_params}")
    return study.best_params


# ──────────────────────────────────────────────
# メイン学習パイプライン
# ──────────────────────────────────────────────
def main():
    print("=" * 70)
    print("🏇 競馬予想AI - 高度学習パイプライン")
    print(f"   日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error,
        roc_auc_score, accuracy_score, log_loss
    )

    # GPU確認
    use_gpu = CONFIG["use_gpu"]
    if use_gpu:
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                print("\n🎮 NVIDIA GPU 検出:")
                for line in result.stdout.split("\n"):
                    if "NVIDIA" in line or "MiB" in line:
                        print(f"   {line.strip()}")
            else:
                print("⚠️ GPU未検出 → CPU版で実行")
                use_gpu = False
        except FileNotFoundError:
            print("⚠️ nvidia-smi未検出 → CPU版で実行")
            use_gpu = False

    # ──── 1. データ読み込み ────
    print("\n" + "=" * 70)
    print("📂 [1/6] データ読み込み")
    print("=" * 70)

    proc = DataProcessor()
    df = proc.load(CONFIG["data_file"])

    # 血統データ結合
    bloodline_path = Path(CONFIG["bloodline_file"])
    if bloodline_path.exists():
        bl = pd.read_csv(bloodline_path)
        print(f"   血統データ: {len(bl):,} 件")
        if "horse_id" in bl.columns and "horse_id" in df.columns:
            df = df.merge(bl, on="horse_id", how="left")
            print(f"   血統結合後: {len(df):,} 件")

    df = proc.preprocess(df)

    # データ範囲フィルタ
    if not CONFIG["use_all_data"] and "year" in df.columns:
        cutoff_year = datetime.now().year - 10
        df = df[df["year"] >= cutoff_year]
        print(f"   直近10年にフィルタ: {len(df):,} 件")

    # 結果のある行のみ学習対象
    df_train = df[df["finish_position"].notna()].copy()
    n_races = df_train["race_id"].nunique()
    print(f"\n   学習データ: {len(df_train):,} 件 / {n_races:,} レース")
    mem_mb = df_train.memory_usage(deep=True).sum() / 1024**2
    print(f"   メモリ使用量: {mem_mb:.0f} MB")

    # ──── 2. 特徴量エンジニアリング ────
    print("\n" + "=" * 70)
    print("🔧 [2/6] 特徴量エンジニアリング")
    print("=" * 70)

    fe = FeatureEngine()
    df_train = fe.create_all(df_train, use_advanced=CONFIG["use_advanced_features"])
    feature_cols = fe.get_feature_columns(df_train)
    print(f"\n   特徴量数: {len(feature_cols)} 個")

    # 特徴量リスト保存
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    with open(f"{CONFIG['output_dir']}/feature_columns.json", "w") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    # ──── 3. データ準備 ────
    print("\n" + "=" * 70)
    print("📊 [3/6] データ準備")
    print("=" * 70)

    X = df_train[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        med = X[col].median()
        X[col] = X[col].fillna(med if not pd.isna(med) else 0)

    y_position = df_train["finish_position"].values
    y_win = (df_train["finish_position"] == 1).astype(int).values
    y_top3 = (df_train["finish_position"] <= 3).astype(int).values

    # 時系列分割
    tscv = TimeSeriesSplit(n_splits=CONFIG["n_cv_folds"])
    splits = list(tscv.split(X))

    # 最終fold を最適化用に使用
    train_idx, val_idx = splits[-1]
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_pos, y_val_pos = y_position[train_idx], y_position[val_idx]
    y_train_win, y_val_win = y_win[train_idx], y_win[val_idx]
    y_train_top3, y_val_top3 = y_top3[train_idx], y_top3[val_idx]

    print(f"   訓練: {len(X_train):,} 件 / 検証: {len(X_val):,} 件")

    # ──── 4. Optunaハイパーパラメータ最適化 ────
    print("\n" + "=" * 70)
    print("🎯 [4/6] ハイパーパラメータ最適化 (Optuna)")
    print("=" * 70)

    n_trials = CONFIG["optuna_trials"]

    print(f"\n  --- 着順回帰モデル ({n_trials}試行) ---")
    best_params_pos = optimize_lgbm(X_train, y_train_pos, X_val, y_val_pos,
                                     task="regression", n_trials=n_trials, use_gpu=use_gpu)

    print(f"\n  --- 複勝分類モデル ({n_trials}試行) ---")
    best_params_top3 = optimize_lgbm(X_train, y_train_top3, X_val, y_val_top3,
                                      task="binary", n_trials=n_trials, use_gpu=use_gpu)

    print(f"\n  --- 勝利分類モデル ({n_trials}試行) ---")
    best_params_win = optimize_lgbm(X_train, y_train_win, X_val, y_val_win,
                                     task="binary", n_trials=n_trials, use_gpu=use_gpu)

    # パラメータ保存
    all_params = {
        "position": best_params_pos,
        "top3": best_params_top3,
        "win": best_params_win,
    }
    with open(f"{CONFIG['output_dir']}/best_params.json", "w") as f:
        json.dump(all_params, f, indent=2)

    # ──── 5. 最終モデル学習 ────
    print("\n" + "=" * 70)
    print("🤖 [5/6] 最終モデル学習")
    print("=" * 70)

    models = {}
    eval_results = {}
    feat_importances = {}

    model_configs = [
        ("position", "regression", y_position, best_params_pos),
        ("top3", "binary", y_top3, best_params_top3),
        ("win", "binary", y_win, best_params_win),
    ]

    for name, task, y, best_params in model_configs:
        print(f"\n  === {name} モデル ===")

        # LightGBM パラメータ組み立て
        lgb_params = {
            "objective": "regression" if task == "regression" else "binary",
            "metric": "mae" if task == "regression" else "binary_logloss",
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
            **best_params,
        }
        if use_gpu:
            lgb_params["device"] = "gpu"
            lgb_params["gpu_use_dp"] = True

        # CV学習
        fold_scores = []
        fold_models_lgb = []
        fold_models_xgb = []
        fold_importances = []

        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            if task == "binary" and (len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2):
                continue

            # LightGBM
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
            lgb_model = lgb.train(
                lgb_params, dtrain,
                num_boost_round=3000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )
            fold_models_lgb.append(lgb_model)

            lgb_pred = lgb_model.predict(X_va)

            # XGBoost
            xgb_pred = None
            if CONFIG["use_xgboost"]:
                xgb_params = {
                    "objective": "reg:squarederror" if task == "regression" else "binary:logistic",
                    "eval_metric": "mae" if task == "regression" else "logloss",
                    "max_depth": best_params.get("max_depth", 6),
                    "learning_rate": best_params.get("learning_rate", 0.05),
                    "subsample": best_params.get("bagging_fraction", 0.8),
                    "colsample_bytree": best_params.get("feature_fraction", 0.8),
                    "reg_alpha": best_params.get("reg_alpha", 0.1),
                    "reg_lambda": best_params.get("reg_lambda", 0.1),
                    "n_estimators": 2000,
                    "early_stopping_rounds": 100,
                    "verbosity": 0,
                    "random_state": 42,
                }
                if use_gpu:
                    xgb_params["tree_method"] = "hist"
                    xgb_params["device"] = "cuda"

                if task == "regression":
                    xgb_model = xgb.XGBRegressor(**xgb_params)
                else:
                    xgb_model = xgb.XGBClassifier(**xgb_params)

                xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                fold_models_xgb.append(xgb_model)

                if task == "regression":
                    xgb_pred = xgb_model.predict(X_va)
                else:
                    xgb_pred = xgb_model.predict_proba(X_va)[:, 1]

            # アンサンブル（LightGBM 60% + XGBoost 40%）
            if xgb_pred is not None:
                pred = lgb_pred * 0.6 + xgb_pred * 0.4
            else:
                pred = lgb_pred

            # 評価
            if task == "regression":
                mae = mean_absolute_error(y_va, pred)
                rmse = np.sqrt(mean_squared_error(y_va, pred))
                fold_scores.append({"mae": mae, "rmse": rmse})
                print(f"    Fold {fold_idx+1}: MAE={mae:.3f}, RMSE={rmse:.3f}")
            else:
                try:
                    auc = roc_auc_score(y_va, pred)
                except ValueError:
                    auc = 0.5
                acc = accuracy_score(y_va, (pred > 0.5).astype(int))
                fold_scores.append({"auc": auc, "accuracy": acc})
                print(f"    Fold {fold_idx+1}: AUC={auc:.4f}, Acc={acc:.4f}")

            # 特徴量重要度
            fold_importances.append(lgb_model.feature_importance(importance_type="gain"))

        # 平均スコア
        avg_scores = {}
        for key in fold_scores[0]:
            vals = [s[key] for s in fold_scores]
            avg_scores[f"mean_{key}"] = float(np.mean(vals))
            avg_scores[f"std_{key}"] = float(np.std(vals))
        eval_results[name] = avg_scores
        print(f"    📊 平均: {avg_scores}")

        # 最良foldのモデルを保存
        if task == "regression":
            best_idx = np.argmin([s["mae"] for s in fold_scores])
        else:
            best_idx = np.argmax([s["auc"] for s in fold_scores])
        models[name] = fold_models_lgb[best_idx]

        if fold_models_xgb:
            models[f"{name}_xgb"] = fold_models_xgb[min(best_idx, len(fold_models_xgb)-1)]

        # 特徴量重要度
        avg_imp = np.mean(fold_importances, axis=0)
        feat_importances[name] = pd.DataFrame({
            "feature": feature_cols,
            "importance": avg_imp,
        }).sort_values("importance", ascending=False)

    # ──── 6. 保存 ────
    print("\n" + "=" * 70)
    print("💾 [6/6] モデル保存")
    print("=" * 70)

    os.makedirs(CONFIG["model_dir"], exist_ok=True)

    # サーバー互換形式で保存（api_serverが読めるpklフォーマット）
    save_data = {
        "version": "2.0",
        "models": {k: v for k, v in models.items() if "_xgb" not in k},
        "feature_columns": feature_cols,
        "evaluation_results": eval_results,
        "feature_importance": {k: v.to_dict() for k, v in feat_importances.items()},
        "saved_at": datetime.now().isoformat(),
        "n_features": len(feature_cols),
        "trained_on": "local_advanced",
    }

    model_path = f"{CONFIG['model_dir']}/keiba_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"  ✅ サーバー互換モデル: {model_path}")

    # XGBoostモデルも別途保存
    xgb_models = {k: v for k, v in models.items() if "_xgb" in k}
    if xgb_models:
        xgb_path = f"{CONFIG['model_dir']}/keiba_model_xgb.pkl"
        with open(xgb_path, "wb") as f:
            pickle.dump(xgb_models, f)
        print(f"  ✅ XGBoostモデル: {xgb_path}")

    # パラメータ・評価結果保存
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_rows": len(df_train),
        "n_races": n_races,
        "n_features": len(feature_cols),
        "config": {k: v for k, v in CONFIG.items()},
        "evaluation": eval_results,
        "best_params": all_params,
    }
    with open(f"{CONFIG['output_dir']}/training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  ✅ レポート: {CONFIG['output_dir']}/training_report.json")

    # ──── SHAP分析 ────
    if CONFIG["save_shap"]:
        print("\n📊 SHAP特徴量重要度分析...")
        try:
            import shap
            # 勝利モデルのSHAP
            explainer = shap.TreeExplainer(models["win"])
            sample = X_val.sample(min(1000, len(X_val)), random_state=42)
            shap_values = explainer.shap_values(sample)

            if CONFIG["save_charts"]:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                # SHAP summary plot
                plt.figure(figsize=(12, 10))
                shap.summary_plot(shap_values, sample, show=False, max_display=30)
                plt.tight_layout()
                plt.savefig(f"{CONFIG['output_dir']}/shap_summary.png", dpi=150)
                plt.close()
                print(f"  ✅ SHAP図: {CONFIG['output_dir']}/shap_summary.png")

        except Exception as e:
            print(f"  ⚠️ SHAP分析スキップ: {e}")

    # ──── 特徴量重要度グラフ ────
    if CONFIG["save_charts"]:
        print("\n📊 特徴量重要度グラフ作成...")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            try:
                import japanize_matplotlib
            except ImportError:
                pass

            for name, imp_df in feat_importances.items():
                top30 = imp_df.head(30)
                plt.figure(figsize=(12, 10))
                plt.barh(range(len(top30)), top30["importance"].values)
                plt.yticks(range(len(top30)), top30["feature"].values)
                plt.xlabel("Importance (gain)")
                plt.title(f"Feature Importance - {name} model")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(f"{CONFIG['output_dir']}/importance_{name}.png", dpi=150)
                plt.close()
            print(f"  ✅ 重要度グラフ: {CONFIG['output_dir']}/importance_*.png")
        except Exception as e:
            print(f"  ⚠️ グラフ作成スキップ: {e}")

    # ──── 最終サマリー ────
    print("\n" + "=" * 70)
    print("🎉 学習完了！")
    print("=" * 70)
    for name, scores in eval_results.items():
        print(f"\n  【{name}】")
        for k, v in scores.items():
            print(f"    {k}: {v:.4f}")

    print(f"\n  📁 出力ファイル:")
    print(f"    モデル:     {CONFIG['model_dir']}/keiba_model.pkl")
    print(f"    レポート:   {CONFIG['output_dir']}/training_report.json")
    print(f"    パラメータ: {CONFIG['output_dir']}/best_params.json")
    print(f"    特徴量:     {CONFIG['output_dir']}/feature_columns.json")

    print(f"\n  🚀 サーバーにアップロード:")
    print(f"    scp {CONFIG['model_dir']}/keiba_model.pkl user@server:/opt/keiba-predictor/models/")
    print(f"    → APIサーバー再起動でモデル反映")

    return models, eval_results


if __name__ == "__main__":
    main()
