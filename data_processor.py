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
import re
from datetime import datetime
from collections import defaultdict

try:
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import IsolationForest
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

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

        # 障害レースを除外（平地レースとは特性が大きく異なり、混在するとモデル精度が低下）
        if "course_type" in df.columns:
            before = len(df)
            df = df[df["course_type"] != "障害"].copy()
            if len(df) < before:
                print(f"  障害レース除外: {before - len(df)} 件")

        # horse_weight=0は計測不能を意味するためNaNに変換し、後続のfillna処理に委ねる
        if "horse_weight" in df.columns:
            df.loc[df["horse_weight"] == 0, "horse_weight"] = np.nan

        # 着順DNF処理（NaNは未実施レース＝予測対象として保持）
        if "finish_position" in df.columns:
            has_pos = df["finish_position"].notna()
            # 結果がある行のみフィルタ（0以下を除外）、NaN行は保持
            valid = has_pos & (df["finish_position"] > 0)
            df = df[valid | ~has_pos]

        # ── IterativeImputer による多変量欠損値補完 ──
        df = self._iterative_impute(df)

        # ── Isolation Forest による外れ値フラグ ──
        df = self._outlier_flags(df)

        self.processed_data = df
        print(f"✅ 前処理完了: {len(df)} 件 / {len(df.columns)} カラム")
        return df

    def _iterative_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """IterativeImputer を用いた多変量欠損値補完。
        sklearn 未インストール時は中央値補完にフォールバック。
        学習データと予測データを分けず、fit_transform を全体に適用することで
        同一スコープ内でデータリークが発生しない（カラム間の相関を使うだけ）。
        """
        impute_cols = [
            c for c in ["distance", "weight", "horse_weight", "weight_diff",
                         "frame_number", "horse_number", "age"]
            if c in df.columns
        ]
        if not impute_cols:
            return df

        has_nan = df[impute_cols].isna().any(axis=1)
        if not has_nan.any():
            return df  # 欠損なしならスキップ

        if not _SKLEARN_AVAILABLE:
            # フォールバック: 中央値補完
            for col in impute_cols:
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)
            return df

        try:
            imputer = IterativeImputer(
                max_iter=5,
                random_state=42,
                initial_strategy="median",
                min_value=0,         # 負値を防ぐ
            )
            imputed = imputer.fit_transform(df[impute_cols])
            df[impute_cols] = np.clip(imputed, 0, None)
            print(f"  IterativeImputer 適用: {impute_cols}")
        except Exception as e:
            print(f"  ⚠️ IterativeImputer スキップ: {e}")
            # フォールバック
            for col in impute_cols:
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)
        return df

    def _outlier_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Isolation Forest で外れ値を検出し、フラグ特徴量として追加。
        除外ではなく外れ値フラグ（0/1）を特徴量に追加することで
        モデルが異常値を考慮できるようにする。

        注意: 入力はレース前に確定している列のみ使用する。
        finish_time_sec (走破タイム) は当該レースの結果であり、
        学習時にフラグへ混入するとリークになるため除外した (2026-07-02)。
        """
        outlier_cols = [
            c for c in ["weight", "horse_weight", "weight_diff"]
            if c in df.columns
        ]
        if not outlier_cols:
            return df

        if not _SKLEARN_AVAILABLE:
            df["is_outlier"] = 0
            return df

        try:
            subset = df[outlier_cols].copy()
            # 欠損を中央値で一時補完（IsolationForest はNaN非対応）
            for col in outlier_cols:
                subset[col] = subset[col].fillna(subset[col].median())

            iso = IsolationForest(
                n_estimators=100,
                contamination=0.02,   # 外れ値を全体の約2%と想定
                random_state=42,
                n_jobs=-1,
            )
            preds = iso.fit_predict(subset)
            # IsolationForest: -1=外れ値, 1=正常 → 0/1 フラグに変換
            df["is_outlier"] = (preds == -1).astype(int)
            n_out = df["is_outlier"].sum()
            print(f"  Isolation Forest: {n_out} 件を外れ値フラグ")
        except Exception as e:
            print(f"  ⚠️ Isolation Forest スキップ: {e}")
            df["is_outlier"] = 0
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
        """通過順を各コーナーに分解

        修正: 短距離レース（1200m以下）ではコーナーが2つしかなく
        corner_3, corner_4がNaNになる問題を線形補間で対処。
        3コーナーの場合（中距離の一部）も同様に補間する。
        """
        def parse(val):
            if pd.isna(val):
                return [None] * 4
            parts = str(val).split("-")
            result = [None] * 4

            # 2コーナー（短距離1200m以下）: corner_1とcorner_4に配置し、2,3を補間
            if len(parts) == 2:
                try:
                    result[0] = int(parts[0])  # corner_1
                    result[3] = int(parts[1])  # corner_4
                    # corner_2, corner_3を線形補間
                    result[1] = int(round((result[0] * 2 + result[3]) / 3))
                    result[2] = int(round((result[0] + result[3] * 2) / 3))
                except ValueError:
                    pass
            # 3コーナー: corner_1, corner_2, corner_4に配置し、corner_3を補間
            elif len(parts) == 3:
                try:
                    result[0] = int(parts[0])
                    result[1] = int(parts[1])
                    result[3] = int(parts[2])
                    result[2] = int(round((result[1] + result[3]) / 2))
                except ValueError:
                    pass
            # 4コーナー（通常）: そのまま配置
            else:
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

        # 14.5 v8: オッズ相対特徴量 (レース内相対 → リークなし)
        if "odds" in df.columns:
            import numpy as _np
            df["log_odds"] = _np.log1p(df["odds"].clip(1, 500))
            df["odds_rank_in_race"] = df.groupby("race_id")["odds"].rank(method="min")
            _race_mean = df.groupby("race_id")["odds"].transform("mean").clip(1)
            df["odds_vs_mean"] = df["odds"] / _race_mean
            df["is_favorite"] = (df["odds_rank_in_race"] == 1).astype(int)
            df["implied_prob"] = 1.0 / df["odds"].clip(1)
            _race_sum = df.groupby("race_id")["implied_prob"].transform("sum").clip(0.01)
            df["normalized_implied_prob"] = df["implied_prob"] / _race_sum
            # AI勝率 vs 市場暗黙勝率の乖離 (バリュー検出)
            if "horse_win_rate_5" in df.columns:
                df["value_gap"] = df["horse_win_rate_5"] - df["normalized_implied_prob"]
            del _race_mean, _race_sum

        # 15. 馬場状態×馬体重の相互作用
        if "track_condition_enc" in df.columns and "horse_weight" in df.columns:
            df["heavy_track_x_hw"] = (df["track_condition_enc"] >= 2).astype(int) * df["horse_weight"]
            df["good_track_x_hw"] = (df["track_condition_enc"] == 0).astype(int) * df["horse_weight"]

        # 16. 距離変更ショック
        if "horse_id" in df.columns and "distance" in df.columns:
            prev_dist = df.groupby("horse_id")["distance"].shift(1)
            df["distance_change"] = df["distance"] - prev_dist
            df["distance_change_abs"] = df["distance_change"].abs()
            df["is_distance_shortening"] = (df["distance_change"] < -200).astype(int)
            df["is_distance_extending"] = (df["distance_change"] > 200).astype(int)
            df["distance_change"] = df["distance_change"].fillna(0)
            df["distance_change_abs"] = df["distance_change_abs"].fillna(0)
            df["is_distance_shortening"] = df["is_distance_shortening"].fillna(0)
            df["is_distance_extending"] = df["is_distance_extending"].fillna(0)

        # 19. 使い込み度（直近10走でのローリングカウント）
        if "horse_id" in df.columns:
            df["races_last_10"] = df.groupby("horse_id")["finish_position"].transform(
                lambda x: x.shift(1).rolling(10, min_periods=1).count()
            )
            df["races_last_10"] = df["races_last_10"].fillna(0)

        # 22. 昇級/降級の差別化
        if "class_change_formal" in df.columns:
            df["is_grade_up"] = (df["class_change_formal"] > 0).astype(int)
            df["is_grade_down"] = (df["class_change_formal"] < 0).astype(int)
            df["grade_change_magnitude"] = df["class_change_formal"].abs()
            df["is_grade_up"] = df["is_grade_up"].fillna(0)
            df["is_grade_down"] = df["is_grade_down"].fillna(0)
            df["grade_change_magnitude"] = df["grade_change_magnitude"].fillna(0)

        print(f"✅ 特徴量エンジニアリング完了: {len(df.columns)} カラム")

        # 12. GOD特徴量（自動適用、エラー時はスキップ）
        try:
            df = self.create_god_features(df)
        except Exception as e:
            print(f"⚠️ GOD特徴量スキップ: {e}")

        # 13. GOD特徴量生成後に再度NaN補完
        df = self._fill_high_nan_features(df)

        return df

    def _fill_high_nan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """NaN率が高い特徴量をグローバル中央値で補完"""
        fill_targets = [
            "horse_dist_avg_pos", "horse_dist_top3_rate",
            "horse_course_avg_pos", "horse_course_top3_rate",
            "days_since_last_race",
            # コンボ特徴量（初回組み合わせ時にNaN）
            "jh_combo_avg_pos", "jh_combo_win_rate", "jh_combo_races",
            "jt_combo_avg_pos", "jt_combo_win_rate",
            "horse_venue_avg_pos", "horse_venue_win_rate",
            "horse_dist_combo_avg", "horse_dist_combo_win",
            "jockey_venue_avg_pos",
            "jockey_dist_avg_pos", "jockey_dist_win_rate",
            "jockey_course_avg_pos", "jockey_course_win_rate",
            # 新規特徴量
            "venue_avg_pos", "venue_win_rate", "venue_track_avg_pos",
            "trainer_career_avg_pos", "trainer_career_win_rate", "trainer_career_top3_rate",
            "horse_ewm_pos", "horse_ewm_top3", "horse_ewm_last3f",
            "running_style", "running_style_std",
            "avg_race_interval",
            # トレンド・ペース
            "position_trend_3", "last3f_trend_3",
            "horse_avg_early_pace", "horse_pace_consistency",
            "horse_avg_pos_change",
            # クラス
            "race_class_formal", "class_change_formal",
            # ベイズTarget Encoding（初出走カテゴリはNaNになる）
            "bte_jockey_id_win", "bte_jockey_id_top3",
            "bte_jockey_id_win_conf", "bte_jockey_id_top3_conf",
            "bte_trainer_id_win", "bte_trainer_id_top3",
            "bte_trainer_id_win_conf", "bte_trainer_id_top3_conf",
            "bte_horse_id_win", "bte_horse_id_top3",
            "bte_horse_id_win_conf", "bte_horse_id_top3_conf",
            # 馬体重時系列パターン
            "weight_trend_3", "weight_std_3", "weight_vs_best",
            "best_weight", "weight_stability", "weight_trend",
            # 円形特徴量（race_date がない場合はNaN）
            "month_sin", "month_cos", "dow_sin", "dow_cos",
            "doy_sin", "doy_cos",
            # 馬場×馬体重、距離変更、使い込み度、昇級降級
            "heavy_track_x_hw", "good_track_x_hw",
            "distance_change", "distance_change_abs",
            "is_distance_shortening", "is_distance_extending",
            "races_last_10",
            "is_grade_up", "is_grade_down", "grade_change_magnitude",
        ]
        for col in fill_targets:
            if col in df.columns:
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)
                else:
                    # 全NaNの場合はデフォルト値（カラム名のパターンで決定）
                    # 長いパターンから先にマッチさせる（部分文字列の誤マッチ防止）
                    defaults = [
                        ("running_style_std", 0.2),
                        ("running_style", 1.5),
                        ("race_class_formal", 3.0),
                        ("class_change_formal", 0.0),
                        ("days_since_last_race", 30.0),
                        ("avg_race_interval", 30.0),
                        ("ewm_last3f", 35.0),
                        ("ewm_pos", 8.0),
                        ("ewm_top3", 0.25),
                        ("pos_change", 0.0),    # horse_avg_pos_change: 通過順変化
                        ("consistency", 0.2),
                        ("avg_pos", 8.0),
                        ("top3_rate", 0.25),
                        # ベイズTE: _conf はサンプル数ゼロ=0, win_rate/top3は全体平均で埋める
                        ("_conf", 0.0),
                        ("bte_", 0.08),         # bte_ prefix: global mean 相当
                        ("win_rate", 0.08),
                        ("trend", 0.0),
                        ("weight_std_3", 0.0),
                        ("weight_trend_3", 0.0),
                        ("weight_vs_best", 0.0),
                        ("best_weight", 0.0),
                        ("weight_stability", 0.0),
                        # 円形特徴量: 月中旬・週中相当のデフォルト
                        ("month_sin", 0.0),
                        ("month_cos", 0.0),
                        ("dow_sin", 0.0),
                        ("dow_cos", 1.0),
                        ("doy_sin", 0.0),
                        ("doy_cos", 1.0),
                        ("early_pace", 0.5),    # horse_avg_early_pace
                        ("pace", 0.5),
                        ("races", 0.0),
                    ]
                    filled = False
                    for key, default in defaults:
                        if key in col:
                            df[col] = df[col].fillna(default)
                            filled = True
                            break
                    if not filled:
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
        """スピード指数の算出 (temporal-safe: expanding median for base_time)"""
        if "finish_time_sec" not in df.columns or "distance" not in df.columns:
            return df

        # Data must be sorted by race_date for expanding() to be temporal-safe
        # (create_features already sorts by race_date at the top)

        # 基準タイム: shift(1) + expanding median（当該レースの走破タイムを除外）
        # expanding().median() は当該レース自身を含んでしまうため shift(1) で1つずらす
        base_time = df.groupby(["distance", "course_type_enc"])["finish_time_sec"].transform(
            lambda x: x.shift(1).expanding().median()
        )
        # 初回（基準タイムがNaN）は距離から概算（将来データを使わない）
        # 芝: 約16.7m/s, ダート: 約16.0m/s を基準
        fallback_time = df["distance"] / 16.3  # 概算基準タイム
        base_time = base_time.fillna(fallback_time)

        # スピード指数 = (基準タイム - 走破タイム) / 基準タイム * 100
        df["speed_index"] = ((base_time - df["finish_time_sec"]) / base_time * 100).round(2)

        # 過去3走の平均スピード指数 (shift(1) ensures no leakage of current race)
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

    # ==================================================================
    # GOD-level feature engineering (62 additional features)
    # Ported from local_training/train_god_v4.py + train_advanced.py
    # ==================================================================

    def create_god_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all 62 missing features for v4 GOD model."""
        print("🔧 GOD特徴量エンジニアリング開始...")

        df = self._normalized_position(df)
        df = self._weight_analysis(df)
        df = self._rotation_features(df)
        df = self._momentum_features(df)
        df = self._class_features(df)
        df = self._season_age_interaction(df)
        df = self._race_level_features(df)
        df = self._elo_features(df)
        df = self._target_encoding_features(df)
        df = self._god_interactions(df)
        df = self._god_race_relative(df)
        df = self._god_bloodline(df)
        df = self._god_pace(df)
        df = self._god_class_transition(df)
        df = self._god_combinations(df)
        df = self._god_domain_features(df)

        df = self._venue_features(df)
        df = self._jockey_horse_combo(df)
        df = self._jockey_aptitude(df)
        df = self._trainer_career_stats(df)
        df = self._exponential_decay_features(df)
        df = self._running_style(df)
        df = self._fitness_indicator(df)

        # v6.0 追加特徴量
        df = self._trainer_jockey_synergy(df)
        df = self._course_shape_features(df)
        df = self._race_level_estimation(df)
        df = self._opponent_strength(df)
        df = self._bloodline_3way_interaction(df)
        df = self._weight_pattern(df)
        df = self._season_course_week(df)
        df = self._circular_time_features(df)
        df = self._auto_interactions(df)

        print(f"✅ GOD特徴量エンジニアリング完了: {len(df.columns)} カラム")
        return df

    # ── Normalized position ──
    def _normalized_position(self, df):
        if "finish_position" not in df.columns:
            return df
        if "n_runners" not in df.columns:
            df["n_runners"] = df.groupby("race_id")["horse_number"].transform("count")
        df["norm_position"] = (df["finish_position"] - 1) / (df["n_runners"] - 1).clip(lower=1)
        if "horse_id" in df.columns:
            for w in [3, 5]:
                df[f"horse_norm_pos_avg_{w}"] = df.groupby("horse_id")["norm_position"].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        return df

    # ── Weight analysis ──
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

    # ── Rotation / previous race features ──
    def _rotation_features(self, df):
        if "horse_id" not in df.columns:
            return df
        if "days_since_last_race" not in df.columns and "race_date" in df.columns:
            df["days_since_last_race"] = df.groupby("horse_id")["race_date"].diff().dt.days
        df["prev_position"] = df.groupby("horse_id")["finish_position"].shift(1)
        df["rest_x_prev_pos"] = df["days_since_last_race"].fillna(90) * df["prev_position"].fillna(8)
        df["prev_top3_flag"] = (df["prev_position"] <= 3).astype(int)
        return df

    # ── Momentum / trend features ──
    def _momentum_features(self, df):
        if "horse_id" not in df.columns:
            return df

        def calc_trend(series):
            if len(series) < 2:
                return 0
            try:
                return -np.polyfit(np.arange(len(series)), series.values, 1)[0]
            except Exception:
                return 0

        df["position_trend_3"] = df.groupby("horse_id")["finish_position"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=2).apply(calc_trend, raw=False))
        if "last_3f" in df.columns:
            df["last3f_trend_3"] = df.groupby("horse_id")["last_3f"].transform(
                lambda x: x.shift(1).rolling(3, min_periods=2).apply(calc_trend, raw=False))
        return df

    # ── Class features (cumulative prize) ──
    def _class_features(self, df):
        if "prize_money" not in df.columns or "horse_id" not in df.columns:
            return df
        df["cumulative_prize"] = df.groupby("horse_id")["prize_money"].transform(
            lambda x: x.shift(1).cumsum())
        df["prize_rank_in_race"] = df.groupby("race_id")["cumulative_prize"].rank(ascending=False)
        return df

    # ── Season x age interaction ──
    def _season_age_interaction(self, df):
        if "month" not in df.columns or "age" not in df.columns:
            return df
        season_enc = (df["month"] - 1) // 3
        df["age_x_season"] = df["age"] * season_enc
        return df

    # ── Race-level odds features ──
    def _race_level_features(self, df):
        if "odds" in df.columns:
            df["race_avg_odds"] = df.groupby("race_id")["odds"].transform("mean")
            df["race_min_odds"] = df.groupby("race_id")["odds"].transform("min")
        if "n_runners" not in df.columns:
            df["n_runners"] = df.groupby("race_id")["horse_number"].transform("count")
        return df

    # ── Elo / Glicko ratings ──
    def _elo_features(self, df):
        if "horse_id" not in df.columns:
            return df

        print("  [Elo] Computing ratings...")
        k_horse, k_jockey, k_trainer = 20, 12, 8

        horse_elo = _EloSystem(k_horse)
        horse_glicko = _GlickoSystem(initial_rd=350, c=15)
        jockey_elo = _EloSystem(k_jockey)
        trainer_elo = _EloSystem(k_trainer)

        horse_elo_col = np.full(len(df), np.nan)
        horse_glicko_col = np.full(len(df), np.nan)
        horse_glicko_rd_col = np.full(len(df), np.nan)
        jockey_elo_col = np.full(len(df), np.nan)
        trainer_elo_col = np.full(len(df), np.nan)

        # Pre-extract columns as numpy arrays for speed
        horse_ids = df["horse_id"].values
        jockey_ids = df["jockey_id"].values if "jockey_id" in df.columns else np.full(len(df), np.nan)
        trainer_ids = df["trainer_id"].values if "trainer_id" in df.columns else np.full(len(df), np.nan)
        finish_pos = df["finish_position"].values
        race_ids = df["race_id"].values

        # Build race groups as index ranges
        race_boundaries = []
        current_race = race_ids[0]
        start = 0
        for i in range(1, len(race_ids)):
            if race_ids[i] != current_race:
                race_boundaries.append((current_race, start, i))
                current_race = race_ids[i]
                start = i
        race_boundaries.append((current_race, start, len(race_ids)))

        n_races = len(race_boundaries)
        period = 0
        for ri, (race_id, s, e) in enumerate(race_boundaries):
            if ri % 10000 == 0:
                print(f"  [Elo] {ri}/{n_races} races...", end="\r")

            # Snapshot pre-race ratings
            for i in range(s, e):
                hid = horse_ids[i]
                if pd.notna(hid):
                    horse_elo_col[i] = horse_elo.get(hid)
                    horse_glicko_col[i] = horse_glicko.get_rating(hid)
                    horse_glicko_rd_col[i] = horse_glicko.get_rd(hid, period)
                jid = jockey_ids[i]
                if pd.notna(jid):
                    jockey_elo_col[i] = jockey_elo.get(jid)
                tid = trainer_ids[i]
                if pd.notna(tid):
                    trainer_elo_col[i] = trainer_elo.get(tid)

            # Update ratings with results
            horse_parts = [(horse_ids[i], finish_pos[i])
                           for i in range(s, e)
                           if pd.notna(horse_ids[i]) and pd.notna(finish_pos[i])]
            jockey_parts = [(jockey_ids[i], finish_pos[i])
                            for i in range(s, e)
                            if pd.notna(jockey_ids[i]) and pd.notna(finish_pos[i])]
            trainer_parts = [(trainer_ids[i], finish_pos[i])
                             for i in range(s, e)
                             if pd.notna(trainer_ids[i]) and pd.notna(finish_pos[i])]
            if len(horse_parts) >= 2:
                horse_elo.update_race(horse_parts)
                horse_glicko.update_race(horse_parts, period)
            if len(jockey_parts) >= 2:
                jockey_elo.update_race(jockey_parts)
            if len(trainer_parts) >= 2:
                trainer_elo.update_race(trainer_parts)
            period += 1
        print()

        df["horse_elo"] = horse_elo_col
        df["horse_glicko"] = horse_glicko_col
        df["horse_glicko_rd"] = horse_glicko_rd_col
        df["jockey_elo"] = jockey_elo_col
        df["trainer_elo"] = trainer_elo_col

        df["horse_elo_rank"] = df.groupby("race_id")["horse_elo"].rank(ascending=False)
        df["jockey_elo_rank"] = df.groupby("race_id")["jockey_elo"].rank(ascending=False)
        # combined_elo: 欠損コンポーネントは初期レート1500として合成する。
        # (id未収集 = 「レート未知のエンティティ」であり、Eloの初期値と同じ扱い。
        #  NaNのまま合成すると、trainer_id欠損行 (2026年スクレイプ分は約48%) で
        #  combined_elo/combined_elo_rank がNaN連鎖し、最重要特徴量が消失する)
        df["combined_elo"] = (df["horse_elo"].fillna(1500.0) * 0.6 +
                              df["jockey_elo"].fillna(1500.0) * 0.25 +
                              df["trainer_elo"].fillna(1500.0) * 0.15)
        df["combined_elo_rank"] = df.groupby("race_id")["combined_elo"].rank(ascending=False)

        print(f"  [Elo] Done: {len(horse_elo.ratings)} horses / {len(jockey_elo.ratings)} jockeys")
        return df

    # ── Bayesian Target Encoding (階層的ベイズTarget Encoding) ──
    def _target_encoding_features(self, df):
        """階層的ベイズTarget Encoding。
        小サンプルカテゴリにスムージングを適用し、Prior（全体平均）との混合比を
        サンプル数に基づいて自動調整する。信頼度スコアも特徴量として追加。
        データリーク防止: TimeSeriesSplit + 時系列順 shift ベースの2方式を使い分ける。
        """
        print("  [BayesTE] Computing Bayesian target encodings...")

        has_result = df["finish_position"].notna()
        df_wr = df[has_result].copy()
        df_wr["_is_win"] = (df_wr["finish_position"] == 1).astype(float)
        df_wr["_is_top3"] = (df_wr["finish_position"] <= 3).astype(float)

        encode_cols = [c for c in ["jockey_id", "trainer_id", "horse_id"] if c in df.columns]

        # スムージング強度: カテゴリ出現数がこの値に達すると Prior 比率が50%になる
        SMOOTH_K = 20

        for col in encode_cols:
            for target_name, target_col in [("win", "_is_win"), ("top3", "_is_top3")]:
                new_col = f"bte_{col}_{target_name}"
                conf_col = f"bte_{col}_{target_name}_conf"

                global_mean = df_wr[target_col].mean()

                # 時系列順に累積統計（shift(1) でリーク防止）
                g = df_wr.groupby(col)[target_col]
                cum_sum = g.transform(lambda x: x.shift(1).cumsum().fillna(0))
                cum_cnt = g.transform(lambda x: x.shift(1).expanding().count().fillna(0))

                # ベイズスムージング: (観測合計 + k * prior_mean) / (観測数 + k)
                # → サンプル数が多いほど観測値に信頼を置く
                smooth_val = (cum_sum + SMOOTH_K * global_mean) / (cum_cnt + SMOOTH_K)

                # 信頼度スコア: サンプル数 / (サンプル数 + k) ∈ [0, 1)
                conf_val = cum_cnt / (cum_cnt + SMOOTH_K)

                df.loc[has_result, new_col] = smooth_val.values
                df.loc[has_result, conf_col] = conf_val.values

                # 結果未確定行（予測対象）は全体のcategoryカウントで埋める
                # （fit済みの全データ統計を使用 → リークなし：未来の結果を使わない）
                cat_stats = df_wr.groupby(col)[target_col].agg(["sum", "count"])
                cat_smooth = (cat_stats["sum"] + SMOOTH_K * global_mean) / (cat_stats["count"] + SMOOTH_K)
                cat_conf = cat_stats["count"] / (cat_stats["count"] + SMOOTH_K)

                nan_mask = df[new_col].isna()
                df.loc[nan_mask, new_col] = df.loc[nan_mask, col].map(cat_smooth).fillna(global_mean)
                df.loc[nan_mask, conf_col] = df.loc[nan_mask, col].map(cat_conf).fillna(0)

                # Inf/NaN の最終クリーンアップ
                df[new_col] = df[new_col].replace([np.inf, -np.inf], global_mean).fillna(global_mean)
                df[conf_col] = df[conf_col].replace([np.inf, -np.inf], 0).fillna(0).clip(0, 1)

        # 後方互換: te_jockey_id_win / te_trainer_id_win も保持
        for col in ["jockey_id", "trainer_id"]:
            bte_col = f"bte_{col}_win"
            te_col = f"te_{col}_win"
            if bte_col in df.columns and te_col not in df.columns:
                df[te_col] = df[bte_col]

        print(f"  [BayesTE] Done: {len(encode_cols)} cols x 2 targets = {len(encode_cols)*2} encodings (+confidence)")
        return df

    # ── GOD interactions ──
    def _god_interactions(self, df):
        if "horse_weight" in df.columns and "distance" in df.columns:
            df["hw_per_dist"] = df["horse_weight"] / df["distance"].clip(lower=1)
        if "age" in df.columns and "distance" in df.columns:
            df["age_x_dist"] = df["age"] * df["distance"]
        if "weight" in df.columns and "horse_weight" in df.columns:
            df["weight_x_hw"] = df["weight"] * df["horse_weight"]
            hw_safe = df["horse_weight"].replace(0, np.nan)
            df["weight_ratio_hw"] = df["weight"] / hw_safe
        if "odds" in df.columns:
            df["odds"] = df["odds"].replace(0, np.nan)
            df["odds_missing"] = df["odds"].isna().astype(int)
        return df

    # ── GOD race-relative features ──
    def _god_race_relative(self, df):
        if "horse_weight" in df.columns:
            df["hw_rank_pct"] = df.groupby("race_id")["horse_weight"].rank(pct=True)
            race_std_hw = df.groupby("race_id")["horse_weight"].transform("std").clip(lower=0.01)
            race_mean_hw = df.groupby("race_id")["horse_weight"].transform("mean")
            df["hw_zscore"] = (df["horse_weight"] - race_mean_hw) / race_std_hw
        if "age" in df.columns:
            df["age_rank_pct"] = df.groupby("race_id")["age"].rank(pct=True)
            race_mean_age = df.groupby("race_id")["age"].transform("mean")
            df["age_vs_race_mean"] = df["age"] - race_mean_age
        if "weight" in df.columns:
            df["weight_rank_pct"] = df.groupby("race_id")["weight"].rank(pct=True)
        return df

    # ── GOD bloodline stats ──
    def _god_bloodline(self, df):
        """Compute sire/dam/sire_sire/dam_sire historical statistics.

        Uses shift(1) + expanding() to prevent future-data leakage. Columns
        are created only when the upstream lineage column exists and has
        at least one non-null value.
        """
        fp = df.get("finish_position")
        if fp is None:
            return df

        # sire / dam: win_rate, top3_rate, avg_pos, std_pos, n_runners
        for parent in ["sire", "dam"]:
            if parent not in df.columns:
                continue
            if df[parent].notna().sum() == 0:
                continue
            g = df.groupby(parent)["finish_position"]
            df[f"{parent}_win_rate"] = g.transform(
                lambda x: (x.shift(1) == 1).expanding().mean())
            df[f"{parent}_top3_rate"] = g.transform(
                lambda x: (x.shift(1) <= 3).expanding().mean())
            df[f"{parent}_avg_pos"] = g.transform(
                lambda x: x.shift(1).expanding().mean())
            df[f"{parent}_std_pos"] = g.transform(
                lambda x: x.shift(1).expanding().std())
            df[f"{parent}_n_runners"] = g.transform(
                lambda x: x.shift(1).expanding().count())

        # sire_sire: win_rate, avg_pos
        if "sire_sire" in df.columns and df["sire_sire"].notna().sum() > 0:
            g = df.groupby("sire_sire")["finish_position"]
            df["sire_sire_win_rate"] = g.transform(
                lambda x: (x.shift(1) == 1).expanding().mean())
            df["sire_sire_avg_pos"] = g.transform(
                lambda x: x.shift(1).expanding().mean())

        # dam_sire (broodmare sire): win_rate, top3_rate, avg_pos
        if "dam_sire" in df.columns and df["dam_sire"].notna().sum() > 0:
            g = df.groupby("dam_sire")["finish_position"]
            df["dam_sire_win_rate"] = g.transform(
                lambda x: (x.shift(1) == 1).expanding().mean())
            df["dam_sire_top3_rate"] = g.transform(
                lambda x: (x.shift(1) <= 3).expanding().mean())
            df["dam_sire_avg_pos"] = g.transform(
                lambda x: x.shift(1).expanding().mean())

        return df

    # ── GOD pace features ──
    def _god_pace(self, df):
        if "corner_1" not in df.columns or "horse_id" not in df.columns:
            return df

        c1 = pd.to_numeric(df["corner_1"], errors="coerce")
        if "n_runners" not in df.columns:
            df["n_runners"] = df.groupby("race_id")["horse_number"].transform("count")
        early_pace_pct = c1 / df["n_runners"].clip(lower=1)

        df["horse_avg_early_pace"] = df.groupby("horse_id")[early_pace_pct.rename("_ep")].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()) if False else np.nan
        # Compute horse_avg_early_pace properly
        df["_early_pace_pct"] = early_pace_pct
        df["horse_avg_early_pace"] = df.groupby("horse_id")["_early_pace_pct"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())

        # Pace consistency (std)
        df["horse_pace_consistency"] = df.groupby("horse_id")["_early_pace_pct"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).std())

        # Position change (corner1 vs corner4)
        if "corner_4" in df.columns:
            c4 = pd.to_numeric(df["corner_4"], errors="coerce")
            pos_change = c1 - c4
            df["horse_avg_pos_change"] = df.groupby("horse_id")[pos_change.values].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()) if False else np.nan
            df["_pos_change_c1c4"] = pos_change
            df["horse_avg_pos_change"] = df.groupby("horse_id")["_pos_change_c1c4"].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            df.drop(columns=["_pos_change_c1c4"], inplace=True, errors='ignore')

        # Field pace prediction
        if "horse_avg_early_pace" in df.columns:
            is_front = (df["horse_avg_early_pace"] < 0.25).astype(float)
            df["race_n_front_runners"] = df.groupby("race_id")[is_front].transform("sum") if False else np.nan
            df["_is_front"] = is_front
            df["race_n_front_runners"] = df.groupby("race_id")["_is_front"].transform("sum")
            df["predicted_pace"] = df.groupby("race_id")["horse_avg_early_pace"].transform("mean")
            df["pace_advantage"] = np.where(
                df["horse_avg_early_pace"] < 0.3,
                1.0 / df["race_n_front_runners"].clip(lower=1),
                0.0
            )
            df.drop(columns=["_is_front"], inplace=True, errors='ignore')

        df.drop(columns=["_early_pace_pct"], inplace=True, errors='ignore')
        return df

    # ── GOD class transition ──
    def _god_class_transition(self, df):
        if "race_name" not in df.columns:
            return df
        class_map = {"新馬": 1, "未勝利": 2, "1勝": 3, "2勝": 4, "3勝": 5,
                     "オープン": 6, "OP": 6, "リステッド": 7, "L": 7,
                     "G3": 8, "G2": 9, "G1": 10, "GI": 10, "GII": 9, "GIII": 8}

        def _parse_class(name):
            if pd.isna(name):
                return np.nan
            name = str(name)
            for pattern, val in class_map.items():
                if pattern in name:
                    return val
            return np.nan

        df["race_class_formal"] = df["race_name"].apply(_parse_class)
        if "horse_id" in df.columns:
            prev_class = df.groupby("horse_id")["race_class_formal"].shift(1)
            df["class_change_formal"] = df["race_class_formal"] - prev_class
        return df

    # ── GOD combination stats ──
    def _god_combinations(self, df):
        # Jockey-trainer combo
        if "jockey_id" in df.columns and "trainer_id" in df.columns:
            combo = df["jockey_id"].astype(str) + "_" + df["trainer_id"].astype(str)
            df["_jt_combo"] = combo
            g = df.groupby("_jt_combo")["finish_position"]
            df["jt_combo_avg_pos"] = g.transform(lambda x: x.shift(1).expanding().mean())
            df["jt_combo_win_rate"] = g.transform(lambda x: (x.shift(1) == 1).expanding().mean())
            df.drop(columns=["_jt_combo"], inplace=True)

        # Horse-distance combo
        if "horse_id" in df.columns and "distance" in df.columns:
            dist_bucket = (df["distance"] // 200 * 200).astype(int)
            combo = df["horse_id"].astype(str) + "_" + dist_bucket.astype(str)
            df["_hd_combo"] = combo
            g = df.groupby("_hd_combo")["finish_position"]
            df["horse_dist_combo_avg"] = g.transform(lambda x: x.shift(1).expanding().mean())
            df["horse_dist_combo_win"] = g.transform(lambda x: (x.shift(1) == 1).expanding().mean())
            df.drop(columns=["_hd_combo"], inplace=True)

        # Horse-venue combo
        if "horse_id" in df.columns and "venue" in df.columns:
            combo = df["horse_id"].astype(str) + "_" + df["venue"].astype(str)
            df["_hv_combo"] = combo
            g = df.groupby("_hv_combo")["finish_position"]
            df["horse_venue_avg_pos"] = g.transform(lambda x: x.shift(1).expanding().mean())
            df["horse_venue_win_rate"] = g.transform(lambda x: (x.shift(1) == 1).expanding().mean())
            df.drop(columns=["_hv_combo"], inplace=True)

        # Jockey-venue combo
        if "jockey_id" in df.columns and "venue" in df.columns:
            combo = df["jockey_id"].astype(str) + "_" + df["venue"].astype(str)
            df["_jv_combo"] = combo
            g = df.groupby("_jv_combo")["finish_position"]
            df["jockey_venue_avg_pos"] = g.transform(lambda x: x.shift(1).expanding().mean())
            df.drop(columns=["_jv_combo"], inplace=True)

        return df

    # ── GOD domain features ──
    def _god_domain_features(self, df):
        # Jockey change detection
        if "horse_id" in df.columns and "jockey_id" in df.columns:
            prev_jockey = df.groupby("horse_id")["jockey_id"].shift(1)
            df["jockey_changed"] = (df["jockey_id"] != prev_jockey).astype(int)
            df.loc[prev_jockey.isna(), "jockey_changed"] = 0
            if "jockey_elo" in df.columns:
                prev_jockey_elo = df.groupby("horse_id")["jockey_elo"].shift(1)
                df["jockey_elo_delta"] = df["jockey_elo"] - prev_jockey_elo

        # Distance change
        if "horse_id" in df.columns and "distance" in df.columns:
            prev_dist = df.groupby("horse_id")["distance"].shift(1)
            df["distance_change"] = df["distance"] - prev_dist
            df["is_distance_shortening"] = (df["distance_change"] < 0).astype(int)
            df["is_distance_extending"] = (df["distance_change"] > 0).astype(int)

        return df

    # ── Venue-specific features (競馬場別バイアス) ──
    def _venue_features(self, df):
        if "venue" not in df.columns:
            return df

        # 競馬場ごとの平均着順（レース単位で集約してからshift → 同一レースリーク防止）
        if "race_id" in df.columns:
            race_venue = df.groupby(["race_id", "venue"]).agg(
                finish_position_mean=("finish_position", "mean"),
                win_rate=("finish_position", lambda x: (x == 1).mean())
            ).reset_index()
            race_venue = race_venue.sort_values("race_id").reset_index(drop=True)
            race_venue["venue_avg_pos"] = race_venue.groupby("venue")["finish_position_mean"].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            race_venue["venue_win_rate"] = race_venue.groupby("venue")["win_rate"].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            merge_cols = ["race_id", "venue", "venue_avg_pos", "venue_win_rate"]
            df = df.merge(race_venue[merge_cols].drop_duplicates(),
                          on=["race_id", "venue"], how="left", suffixes=("_old", ""))
            for col in ["venue_avg_pos_old", "venue_win_rate_old"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
        else:
            df["venue_avg_pos"] = df.groupby("venue")["finish_position"].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            df["venue_win_rate"] = df.groupby("venue")["finish_position"].transform(
                lambda x: (x.shift(1) == 1).expanding().mean()
            )

        # 馬場状態×競馬場 交互作用
        if "track_condition_enc" in df.columns:
            combo = df["venue"].astype(str) + "_" + df["track_condition_enc"].astype(str)
            df["_vtc_combo"] = combo
            df["venue_track_avg_pos"] = df.groupby("_vtc_combo")["finish_position"].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            df.drop(columns=["_vtc_combo"], inplace=True)

        return df

    # ── Jockey × Horse combination features ──
    def _jockey_horse_combo(self, df):
        if "jockey_id" not in df.columns or "horse_id" not in df.columns:
            return df
        combo = df["horse_id"].astype(str) + "_" + df["jockey_id"].astype(str)
        df["_jh_combo"] = combo
        g = df.groupby("_jh_combo")["finish_position"]
        df["jh_combo_avg_pos"] = g.transform(lambda x: x.shift(1).expanding().mean())
        df["jh_combo_races"] = g.transform(lambda x: x.shift(1).expanding().count())
        df["jh_combo_win_rate"] = g.transform(lambda x: (x.shift(1) == 1).expanding().mean())
        df.drop(columns=["_jh_combo"], inplace=True)
        return df

    # ── Jockey × distance/course aptitude ──
    def _jockey_aptitude(self, df):
        if "jockey_id" not in df.columns:
            return df

        # 騎手×距離カテゴリ
        if "distance_category" in df.columns:
            combo = df["jockey_id"].astype(str) + "_" + df["distance_category"].astype(str)
            df["_jd_combo"] = combo
            g = df.groupby("_jd_combo")["finish_position"]
            df["jockey_dist_avg_pos"] = g.transform(lambda x: x.shift(1).expanding().mean())
            df["jockey_dist_win_rate"] = g.transform(lambda x: (x.shift(1) == 1).expanding().mean())
            df.drop(columns=["_jd_combo"], inplace=True)

        # 騎手×コースタイプ（芝/ダート）
        if "course_type_enc" in df.columns:
            combo = df["jockey_id"].astype(str) + "_" + df["course_type_enc"].astype(str)
            df["_jc_combo"] = combo
            g = df.groupby("_jc_combo")["finish_position"]
            df["jockey_course_avg_pos"] = g.transform(lambda x: x.shift(1).expanding().mean())
            df["jockey_course_win_rate"] = g.transform(lambda x: (x.shift(1) == 1).expanding().mean())
            df.drop(columns=["_jc_combo"], inplace=True)

        return df

    # ── Trainer career stats (not just 30-day rolling) ──
    def _trainer_career_stats(self, df):
        if "trainer_id" not in df.columns:
            return df
        g = df.groupby("trainer_id")["finish_position"]
        df["trainer_career_avg_pos"] = g.transform(lambda x: x.shift(1).expanding().mean())
        df["trainer_career_win_rate"] = g.transform(lambda x: (x.shift(1) == 1).expanding().mean())
        df["trainer_career_top3_rate"] = g.transform(lambda x: (x.shift(1) <= 3).expanding().mean())
        df["trainer_total_races"] = g.transform(lambda x: x.shift(1).expanding().count())
        return df

    # ── Exponential decay weighting for recency ──
    def _exponential_decay_features(self, df):
        """直近の成績を重視する指数減衰加重平均"""
        if "horse_id" not in df.columns:
            return df

        def ewm_mean(series):
            """指数加重移動平均（halflife=3走）"""
            return series.shift(1).ewm(halflife=3, min_periods=1).mean()

        df["horse_ewm_pos"] = df.groupby("horse_id")["finish_position"].transform(ewm_mean)
        df["horse_ewm_top3"] = df.groupby("horse_id")["finish_position"].transform(
            lambda x: (x <= 3).astype(float).shift(1).ewm(halflife=3, min_periods=1).mean()
        )

        if "last_3f" in df.columns:
            df["horse_ewm_last3f"] = df.groupby("horse_id")["last_3f"].transform(ewm_mean)

        return df

    # ── Running style classification (脚質分類) ──
    def _running_style(self, df):
        """逃げ/先行/差し/追込 の脚質分類"""
        if "horse_id" not in df.columns or "corner_1" not in df.columns:
            return df

        c1 = pd.to_numeric(df["corner_1"], errors="coerce")
        if "n_runners" not in df.columns:
            df["n_runners"] = df.groupby("race_id")["horse_number"].transform("count")

        # 相対ポジション (0=先頭, 1=最後尾)
        rel_pos = (c1 - 1) / (df["n_runners"] - 1).clip(lower=1)

        # 過去5走の平均相対ポジション
        df["_rel_early_pos"] = rel_pos
        avg_rel = df.groupby("horse_id")["_rel_early_pos"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean()
        )

        # 脚質分類 (0=逃げ, 1=先行, 2=差し, 3=追込)
        df["running_style"] = pd.cut(
            avg_rel,
            bins=[-0.01, 0.2, 0.45, 0.7, 1.01],
            labels=[0, 1, 2, 3]
        ).astype(float)

        # 脚質の安定度（std が低い = 安定した脚質）
        df["running_style_std"] = df.groupby("horse_id")["_rel_early_pos"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).std()
        )

        df.drop(columns=["_rel_early_pos"], inplace=True)
        return df

    # ── Horse fitness indicator ──
    def _fitness_indicator(self, df):
        """馬のフィットネス指標: 直近N日間の出走回数"""
        if "horse_id" not in df.columns or "race_date" not in df.columns:
            return df

        # 直近90日の出走回数
        df["races_last_90d"] = df.groupby("horse_id")["race_date"].transform(
            lambda x: x.rolling("90D", on=x).count().shift(1)
        ) if False else np.nan  # rolling on datetime requires index

        # 簡易実装: days_since_last_race から逆算
        if "days_since_last_race" in df.columns:
            # 過去3走の平均間隔（当該レースの間隔は含めない → shift(1)）
            df["avg_race_interval"] = df.groupby("horse_id")["days_since_last_race"].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean()
            )
            # 高頻度出走フラグ（2週間以内に連戦）
            df["is_consecutive"] = (df["days_since_last_race"] <= 14).astype(int)
            # 長期休養明けフラグ（半年以上）
            df["is_long_rest"] = (df["days_since_last_race"] >= 180).astype(int)

        # races_last_90d のNaN列を削除
        if "races_last_90d" in df.columns and df["races_last_90d"].isna().all():
            df.drop(columns=["races_last_90d"], inplace=True)

        return df

    # ── #7 調教師×騎手シナジー ──
    def _trainer_jockey_synergy(self, df):
        if "trainer_id" not in df.columns or "jockey_id" not in df.columns:
            return df
        combo = df["trainer_id"].astype(str) + "_" + df["jockey_id"].astype(str)
        df["_tj_combo"] = combo
        g = df.groupby("_tj_combo")["finish_position"]
        df["tj_combo_avg_pos"] = g.transform(lambda x: x.shift(1).expanding().mean())
        df["tj_combo_win_rate"] = g.transform(lambda x: (x.shift(1) == 1).expanding().mean())
        df["tj_combo_races"] = g.transform(lambda x: x.shift(1).expanding().count())
        df.drop(columns=["_tj_combo"], inplace=True)
        # コンボ初回はNaN → 全体平均で補完
        fill_avg = df["finish_position"].mean() if "finish_position" in df.columns else 8.0
        df["tj_combo_avg_pos"] = df["tj_combo_avg_pos"].fillna(fill_avg)
        df["tj_combo_win_rate"] = df["tj_combo_win_rate"].fillna(0.08)
        df["tj_combo_races"] = df["tj_combo_races"].fillna(0)
        return df

    # ── #8 コース形状特徴量 ──
    COURSE_PROFILES = {
        "東京": {"straight": 525.9, "has_slope": True, "corners": 2, "inner": True},
        "中山": {"straight": 310.0, "has_slope": True, "corners": 4, "inner": True},
        "阪神": {"straight": 473.6, "has_slope": True, "corners": 4, "inner": True},
        "京都": {"straight": 403.7, "has_slope": True, "corners": 4, "inner": True},
        "中京": {"straight": 412.5, "has_slope": True, "corners": 4, "inner": False},
        "新潟": {"straight": 658.7, "has_slope": False, "corners": 2, "inner": False},
        "福島": {"straight": 292.0, "has_slope": False, "corners": 4, "inner": True},
        "小倉": {"straight": 293.0, "has_slope": False, "corners": 4, "inner": True},
        "札幌": {"straight": 266.1, "has_slope": False, "corners": 4, "inner": True},
        "函館": {"straight": 262.1, "has_slope": False, "corners": 4, "inner": True},
    }

    def _course_shape_features(self, df):
        if "venue" not in df.columns:
            return df
        straight = df["venue"].map(lambda v: self.COURSE_PROFILES.get(v, {}).get("straight", 400))
        has_slope = df["venue"].map(lambda v: self.COURSE_PROFILES.get(v, {}).get("has_slope", False)).astype(int)
        corners = df["venue"].map(lambda v: self.COURSE_PROFILES.get(v, {}).get("corners", 4))

        df["course_straight_len"] = straight.astype(float)
        df["course_has_slope"] = has_slope
        df["course_corners"] = corners.astype(int)
        # 直線率: 直線距離 / 全距離
        if "distance" in df.columns:
            df["straight_ratio"] = df["course_straight_len"] / df["distance"].clip(lower=800)
        return df

    # ── #11 レースレベル推定（出走馬のElo平均） ──
    def _race_level_estimation(self, df):
        for col in ["horse_elo", "combined_elo"]:
            if col in df.columns:
                df[f"race_avg_{col}"] = df.groupby("race_id")[col].transform("mean")
                df[f"race_max_{col}"] = df.groupby("race_id")[col].transform("max")
                # 自分のEloとレース平均の差 → 相対的な実力
                df[f"{col}_vs_race"] = df[col] - df[f"race_avg_{col}"]
        return df

    # ── #12 同レース相手関係 ──
    def _opponent_strength(self, df):
        if "horse_elo" not in df.columns:
            return df
        # レース内の自分を除いた他馬のElo統計
        race_sum = df.groupby("race_id")["horse_elo"].transform("sum")
        race_count = df.groupby("race_id")["horse_elo"].transform("count")
        # 自分を除いた平均（単独馬はNaN）
        denom = race_count - 1
        df["opponents_avg_elo"] = np.where(denom > 0,
            (race_sum - df["horse_elo"]) / denom, np.nan)
        df["elo_advantage"] = df["horse_elo"] - df["opponents_avg_elo"]

        # 「格上相手の数」: 自分よりEloが高い馬の頭数
        df["n_stronger_opponents"] = df.groupby("race_id")["horse_elo"].transform(
            lambda x: x.rank(ascending=True, method="min") - 1
        )
        return df

    # ── #13 血統×距離×馬場 3次元交互作用 ──
    def _bloodline_3way_interaction(self, df):
        if "sire" not in df.columns or "distance_category" not in df.columns:
            return df
        if "course_type_enc" not in df.columns:
            return df
        # sire × distance × course_type
        combo = (df["sire"].astype(str) + "_" +
                 df["distance_category"].astype(str) + "_" +
                 df["course_type_enc"].astype(str))
        df["_sdc_combo"] = combo
        g = df.groupby("_sdc_combo")["finish_position"]
        df["sire_dist_course_avg"] = g.transform(lambda x: x.shift(1).expanding().mean())
        df["sire_dist_course_top3"] = g.transform(lambda x: (x.shift(1) <= 3).expanding().mean())
        df.drop(columns=["_sdc_combo"], inplace=True)

        # dam_sire × distance × track_condition
        if "dam" in df.columns and "track_condition_enc" in df.columns:
            combo2 = (df["dam"].astype(str) + "_" +
                      df["distance_category"].astype(str) + "_" +
                      df["track_condition_enc"].astype(str))
            df["_ddt_combo"] = combo2
            g2 = df.groupby("_ddt_combo")["finish_position"]
            df["dam_dist_track_avg"] = g2.transform(lambda x: x.shift(1).expanding().mean())
            df.drop(columns=["_ddt_combo"], inplace=True)
        return df

    # ── #14 馬体重変動パターン（拡張版） ──
    def _weight_pattern(self, df):
        if "horse_weight" not in df.columns or "horse_id" not in df.columns:
            return df
        g = df.groupby("horse_id")["horse_weight"]

        # 過去5走の体重変動の標準偏差（安定度）
        df["weight_stability"] = g.transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).std()
        )

        # ── 過去3走の体重変化トレンド（増加傾向/減少傾向）──
        # polyfit の傾き: 正=増加傾向, 負=減少傾向
        def weight_slope(s):
            if len(s) < 2:
                return 0.0
            try:
                slope = np.polyfit(np.arange(len(s)), s.values, 1)[0]
                return float(slope)
            except Exception:
                return 0.0

        # 体重トレンド（過去5走の傾き）
        df["weight_trend"] = g.transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).apply(weight_slope, raw=False)
        )

        # 過去3走の体重変化トレンド（短期集中版）
        df["weight_trend_3"] = g.transform(
            lambda x: x.shift(1).rolling(3, min_periods=2).apply(weight_slope, raw=False)
        )
        # トレンド方向フラグ（増加=1, 減少=-1, 安定=0）
        df["weight_trend_dir"] = np.sign(df["weight_trend_3"].fillna(0)).astype(int)

        # ── 体重変化の標準偏差（過去3走・安定性）──
        df["weight_std_3"] = g.transform(
            lambda x: x.shift(1).rolling(3, min_periods=2).std()
        )
        # 安定フラグ（過去3走の標準偏差が3kg未満）
        df["weight_stable_flag"] = (df["weight_std_3"].fillna(99) < 3).astype(int)

        # ── 急減フラグ（weight_diff が -10kg 以上の減少）──
        # weight_diff = 前走比の体重変化（前処理済みの生データカラム）
        if "weight_diff" in df.columns:
            df["weight_sharp_drop"] = (
                pd.to_numeric(df["weight_diff"], errors="coerce").fillna(0) <= -10
            ).astype(int)
            # 前走からの体重変化（weight_diff が存在しない場合の代替）
        else:
            # horse_weight の差分から計算（shift(1) でリーク防止）
            hw_diff = g.transform(lambda x: x.diff())
            df["weight_sharp_drop"] = (hw_diff.fillna(0) <= -10).astype(int)

        # ── ベスト体重からの乖離（過去1着時の平均体重との差）──
        if "finish_position" in df.columns:
            win_mask = df["finish_position"] == 1
            df["_win_hw"] = np.where(win_mask, df["horse_weight"], np.nan)
            df["best_weight"] = df.groupby("horse_id")["_win_hw"].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            df["weight_vs_best"] = df["horse_weight"] - df["best_weight"]
            df.drop(columns=["_win_hw"], inplace=True)

        # NaN/Inf を安全にクリーンアップ
        for col in ["weight_trend_3", "weight_trend_dir", "weight_std_3",
                    "weight_stable_flag", "weight_sharp_drop"]:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

        return df

    # ── #15 季節×コース開催週（馬場の荒れ具合） ──
    def _season_course_week(self, df):
        if "race_date" not in df.columns or "venue" not in df.columns:
            return df
        dates = pd.to_datetime(df["race_date"], errors="coerce")
        # 開催ごとのレース通し番号（同一venue+同月内での日数経過）
        year_month = dates.dt.to_period("M").astype(str)
        meeting_key = df["venue"].astype(str) + "_" + year_month
        df["_meeting_key"] = meeting_key
        # 開催内の何日目か（開幕週=0, 最終週=大きい値）
        df["meeting_day_num"] = df.groupby("_meeting_key")["race_date"].transform(
            lambda x: (pd.to_datetime(x) - pd.to_datetime(x).min()).dt.days
        )
        # 開催後半は馬場が荒れる → 内枠不利、差し有利の傾向
        df["is_late_meeting"] = (df["meeting_day_num"] >= 14).astype(int)
        # 開催日数 × フレーム番号 交互作用
        if "frame_number" in df.columns:
            df["meeting_day_x_frame"] = df["meeting_day_num"] * df["frame_number"]
        df.drop(columns=["_meeting_key"], inplace=True)
        return df

    # ── #16 円形特徴量（月の周期性・曜日効果） ──
    def _circular_time_features(self, df):
        """月・曜日を円形（sin/cos）エンコーディングで連続化。
        例: 12月と1月が近い距離になるよう周期性を保持する。
        """
        # ── 月の周期性 ──
        if "month" in df.columns:
            month = pd.to_numeric(df["month"], errors="coerce").fillna(1)
            # sin(month * π/6): 月を [0, 2π) 上の角度にマッピング
            df["month_sin"] = np.sin(month * np.pi / 6).round(6)
            df["month_cos"] = np.cos(month * np.pi / 6).round(6)

        # ── 曜日効果のエンコーディング ──
        # day_of_week は preprocess() で生成済み (0=月曜 ... 6=日曜)
        if "day_of_week" in df.columns:
            dow = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0)
            # 1週間 = 7日の周期
            df["dow_sin"] = np.sin(dow * 2 * np.pi / 7).round(6)
            df["dow_cos"] = np.cos(dow * 2 * np.pi / 7).round(6)
            # 週末フラグ（土=5, 日=6）← 競馬は主に土日開催
            df["is_weekend"] = (dow >= 5).astype(int)
        elif "race_date" in df.columns:
            # day_of_week カラムがない場合は race_date から生成
            dates = pd.to_datetime(df["race_date"], errors="coerce")
            dow = dates.dt.dayofweek.fillna(0)
            df["dow_sin"] = np.sin(dow * 2 * np.pi / 7).round(6)
            df["dow_cos"] = np.cos(dow * 2 * np.pi / 7).round(6)
            df["is_weekend"] = (dow >= 5).astype(int)

        # ── 年間通算日（1年 = 365日の周期） ──
        if "race_date" in df.columns:
            dates = pd.to_datetime(df["race_date"], errors="coerce")
            day_of_year = dates.dt.dayofyear.fillna(1).astype(float)
            df["doy_sin"] = np.sin(day_of_year * 2 * np.pi / 365).round(6)
            df["doy_cos"] = np.cos(day_of_year * 2 * np.pi / 365).round(6)

        return df

    # ── #6 重要特徴量の自動交互作用 ──
    TOP_INTERACTION_FEATURES = [
        "horse_elo", "jockey_elo", "trainer_elo", "combined_elo",
        "horse_ewm_pos", "weight", "distance", "age",
        "days_since_last_race", "n_runners",
    ]

    def _auto_interactions(self, df):
        """重要特徴量Top10の2次交互作用を自動生成（生値の積、正規化はモデルに任せる）"""
        available = [c for c in self.TOP_INTERACTION_FEATURES if c in df.columns]
        count = 0
        for i, c1 in enumerate(available):
            for c2 in available[i+1:]:
                name = f"ix_{c1}_x_{c2}"
                if name not in df.columns:
                    # GBDTはNaN対応なのでNaN伝播のまま（fillna(0)は不適切）
                    df[name] = df[c1] * df[c2]
                    count += 1
                if count >= 20:  # 最大20個まで（特徴量爆発防止）
                    break
            if count >= 20:
                break
        return df

    # オッズ関連カラム（デフォルトで学習から除外、予測時にブレンド用として保持）
    # 生オッズ
    ODDS_FEATURES_RAW = {
        "odds", "popularity", "odds_missing",
        "odds_change_ratio", "odds_change_rate", "smart_money_signal",
    }
    # オッズ派生特徴量 (レース内相対も含む)
    #
    # ポリシー (2026-07-02 統一): デフォルトでは全オッズ派生を学習から除外する。
    # 理由:
    #  1. 学習データのオッズは「確定オッズ」だが、予測時に使えるのは朝〜直前
    #     オッズであり時点が一致しない (train/serve 不整合)。
    #  2. 市場情報を特徴量にすると予測確率が市場確率へ回帰し、EVベット戦略の
    #     エッジ源泉 (モデル確率と市場確率の乖離) が消える。
    #  3. 本番モデル v7.2 (220特徴量) は実際にオッズ系ゼロで学習されており、
    #     この既定が現行挙動とも一致する。
    # 市場込みモデルを意図的に作る場合のみ include_odds=True を指定する。
    ODDS_FEATURES_SAFE = {
        "odds_rank", "odds_ratio",
        "race_avg_odds", "race_min_odds",
        "odds_vs_race_avg",
        "log_odds", "odds_rank_in_race", "odds_vs_mean",
        "is_favorite", "normalized_implied_prob",
        "implied_prob", "value_gap",
    }
    ODDS_FEATURES = ODDS_FEATURES_RAW | ODDS_FEATURES_SAFE  # 全オッズ派生

    # 血統統計特徴量（prepare_retrain.py の compute_sire_dam_stats() が生成）
    # sire_win_rate, sire_avg_pos, dam_top3_rate 等は数値で学習対象に含める
    BLOODLINE_STAT_FEATURES = {
        # sire overall
        "sire_win_rate", "sire_top3_rate", "sire_avg_pos",
        "sire_n_runners", "sire_std_pos",
        # sire by distance
        "sire_dist_win_rate", "sire_dist_avg_pos",
        # sire by surface
        "sire_surface_win_rate", "sire_surface_avg_pos",
        # sire by condition
        "sire_cond_win_rate",
        # dam overall
        "dam_win_rate", "dam_top3_rate", "dam_avg_pos",
        "dam_n_runners", "dam_std_pos",
        # dam by distance
        "dam_dist_win_rate", "dam_dist_avg_pos",
        # dam by surface
        "dam_surface_win_rate", "dam_surface_avg_pos",
        # dam by condition
        "dam_cond_win_rate",
        # dam_sire (broodmare sire)
        "dam_sire_win_rate", "dam_sire_top3_rate", "dam_sire_avg_pos",
        # sire_sire (paternal grandsire)
        "sire_sire_win_rate", "sire_sire_avg_pos",
        # interaction
        "sire_dist_n",
        # data_processor 内の3次元交互作用
        "sire_dist_course_avg", "sire_dist_course_top3",
        "dam_dist_track_avg",
    }

    # 調教評価特徴量（integrate_extra_data.py が生成）
    # training_eval_score: A=5, B=4, C=3, D=2, E=1, 不明=0
    TRAINING_FEATURES = {
        "training_eval_score",
    }

    # 予想印特徴量（integrate_extra_data.py が生成）
    # mark_score: ◎=5, ○=4, ▲=3, △=2, ×=1 の重み付き合計
    MARKS_FEATURES = {
        "mark_honmei", "mark_taikou", "mark_tanana", "mark_renka",
        "mark_hoshi", "mark_score", "mark_score_norm",
    }

    # 気象特徴量（weather_model.py が生成）
    WEATHER_FEATURES = {
        "weather_temperature", "weather_humidity",
        "weather_rainfall_1h", "weather_rainfall_3h",
        "weather_wind_speed", "weather_precip_prob",
        "weather_available",
    }

    def get_feature_columns(self, df: pd.DataFrame, include_odds: bool = False) -> list:
        """モデルに投入する特徴量カラムのリストを取得

        Args:
            include_odds: Trueならオッズ系特徴量も含める（デフォルト: False=除外）

        含まれる特徴量グループ:
          - 基本特徴量（race/horse 属性）
          - Eloレーティング、EWM着順統計
          - 血統統計: BLOODLINE_STAT_FEATURES（sire_win_rate, dam_top3_rate 等）
          - 調教評価: TRAINING_FEATURES（training_eval_score）
          - 予想印:  MARKS_FEATURES（mark_score 等）
          - 気象:    WEATHER_FEATURES（weather_temperature 等）
          - include_odds=True 時: ODDS_FEATURES（odds_change_ratio 等も含む）
        """
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
            "sire", "dam", "sire_sire", "dam_sire",  # 文字列カラム（統計量はBLOODLINE_STAT_FEATURESで利用）
            "venue",             # 文字列カラム（特徴量はvenue_avg_pos等で間接利用）
            "training_evaluation",  # 文字列カラム（スコアはtraining_eval_scoreで利用）
        }
        if not include_odds:
            # 生オッズ + 全オッズ派生を除外 (時点不整合とエッジ消失の防止。
            # ODDS_FEATURES_SAFE のコメント参照)
            exclude |= self.ODDS_FEATURES
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        self.feature_names = feature_cols
        return feature_cols


class _EloSystem:
    """Elo rating system for horses/jockeys/trainers."""
    def __init__(self, k_factor=16, initial=1500):
        self.k = k_factor
        self.initial = initial
        self.ratings = {}

    def get(self, eid):
        return self.ratings.get(eid, self.initial)

    def update_race(self, participants):
        n = len(participants)
        if n < 2:
            return
        ratings = {pid: self.get(pid) for pid, _ in participants}
        deltas = defaultdict(list)
        for i, (id_i, pos_i) in enumerate(participants):
            delta = 0.0
            for j, (id_j, pos_j) in enumerate(participants):
                if i == j:
                    continue
                expected = 1.0 / (1.0 + 10 ** ((ratings[id_j] - ratings[id_i]) / 400.0))
                actual = 1.0 if pos_i < pos_j else (0.5 if pos_i == pos_j else 0.0)
                delta += self.k * (actual - expected) / (n - 1)
            deltas[id_i].append(delta)
        for eid, dlist in deltas.items():
            self.ratings[eid] = ratings[eid] + np.mean(dlist)


class _GlickoSystem:
    """Glicko rating system."""
    def __init__(self, initial_rating=1500, initial_rd=350, c=15):
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.c = c
        self.ratings = {}
        self.rds = {}
        self.last_race = {}

    def get_rating(self, eid):
        return self.ratings.get(eid, self.initial_rating)

    def get_rd(self, eid, period=0):
        rd = self.rds.get(eid, self.initial_rd)
        last = self.last_race.get(eid, period)
        return min(350, np.sqrt(rd ** 2 + self.c ** 2 * (period - last)))

    def update_race(self, participants, period=0):
        n = len(participants)
        if n < 2:
            return
        pre_r = {pid: self.get_rating(pid) for pid, _ in participants}
        pre_rd = {pid: self.get_rd(pid, period) for pid, _ in participants}

        for pid, pos in participants:
            r, rd = pre_r[pid], pre_rd[pid]
            d2_inv = 0
            delta_sum = 0
            for oid, opos in participants:
                if pid == oid:
                    continue
                rd_o = pre_rd[oid]
                g_val = 1.0 / np.sqrt(1 + 3 * (rd_o / np.pi) ** 2 / 400 ** 2)
                e_val = 1.0 / (1 + 10 ** (-g_val * (r - pre_r[oid]) / 400))
                actual = 1.0 if pos < opos else (0.5 if pos == opos else 0.0)
                d2_inv += g_val ** 2 * e_val * (1 - e_val)
                delta_sum += g_val * (actual - e_val)
            if d2_inv == 0:
                continue
            d2 = 1.0 / (d2_inv / (n - 1))
            new_rd = 1.0 / np.sqrt(1 / rd ** 2 + 1 / d2)
            self.ratings[pid] = r + (new_rd ** 2) * delta_sum / (n - 1)
            self.rds[pid] = new_rd
            self.last_race[pid] = period


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
