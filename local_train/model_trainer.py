"""
競馬予想AI - モデル学習・予測エンジン
=====================================
LightGBM + アンサンブル学習で着順予測・回収率最適化を行う。

3種類のモデルを並列学習:
  1. 着順予測モデル（回帰）: 着順を直接予測
  2. 複勝予測モデル（分類）: 3着以内に入るか
  3. 1着予測モデル（分類）: 勝つかどうか

最終スコア = 重み付きアンサンブル
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pickle
import json
from datetime import datetime

# scikit-learn
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    log_loss,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# LightGBMが利用可能か確認
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("⚠️ LightGBMが未インストール: pip install lightgbm")

# XGBoostが利用可能か確認
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


class RacePredictionModel:
    """競馬予測モデル: 学習・評価・予測を一括管理"""

    # LightGBMハイパーパラメータ（チューニング済みベースライン）
    LGBM_PARAMS_REGRESSION = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    LGBM_PARAMS_BINARY = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.models: Dict[str, object] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        self.evaluation_results: Dict[str, dict] = {}
        self.feature_columns: List[str] = []
        self.scaler = StandardScaler()

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_col: str = "finish_position",
        n_splits: int = 5,
    ) -> Dict[str, dict]:
        """モデル学習（時系列バリデーション付き）"""
        self.feature_columns = feature_columns

        print("=" * 60)
        print("🤖 モデル学習開始")
        print(f"   データ: {len(df)} 件 / 特徴量: {len(feature_columns)} 個")
        print("=" * 60)

        # 欠損値をカラムごとの中央値で補完
        X = df[feature_columns].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0)

        # ターゲット変数
        y_position = df[target_col].values
        y_win = (df[target_col] == 1).astype(int).values
        y_top3 = (df[target_col] <= 3).astype(int).values

        # 時系列分割（未来のデータでリークしない）
        if "race_date" in df.columns:
            # 日付順でソート済みなのでTimeSeriesSplitを使用
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = list(tscv.split(X))
        else:
            # race_idでグループ分割
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = list(tscv.split(X))

        # --- 1. 着順回帰モデル ---
        print("\n📊 1/3: 着順回帰モデル")
        self.models["position"], self.evaluation_results["position"], self.feature_importance["position"] = (
            self._train_model(X, y_position, splits, task="regression")
        )

        # --- 2. 複勝（3着以内）分類モデル ---
        print("\n📊 2/3: 複勝（3着以内）分類モデル")
        self.models["top3"], self.evaluation_results["top3"], self.feature_importance["top3"] = (
            self._train_model(X, y_top3, splits, task="binary")
        )

        # --- 3. 勝利分類モデル ---
        print("\n📊 3/3: 勝利（1着）分類モデル")
        self.models["win"], self.evaluation_results["win"], self.feature_importance["win"] = (
            self._train_model(X, y_win, splits, task="binary")
        )

        self._print_summary()
        return self.evaluation_results

    def _train_model(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        splits: list,
        task: str = "regression",
    ) -> Tuple[object, dict, pd.DataFrame]:
        """個別モデルの学習（CV付き）"""
        scores = []
        importances = []
        best_model = None
        best_score = float("inf") if task == "regression" else -float("inf")

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # バイナリ分類で訓練or検証に1クラスしかない場合はスキップ
            if task == "binary":
                if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                    print(f"  Fold {fold_idx+1}: スキップ（クラス不足）")
                    continue

            if HAS_LGBM:
                model = self._train_lgbm(X_train, y_train, X_val, y_val, task)
                importance = pd.DataFrame({
                    "feature": self.feature_columns,
                    "importance": model.feature_importance(importance_type="gain"),
                })
            else:
                model = self._train_sklearn(X_train, y_train, task)
                importance = pd.DataFrame({
                    "feature": self.feature_columns,
                    "importance": model.feature_importances_,
                })

            importances.append(importance)

            # 評価
            if task == "regression":
                pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, pred)
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                scores.append({"mae": mae, "rmse": rmse})
                print(f"  Fold {fold_idx+1}: MAE={mae:.3f}, RMSE={rmse:.3f}")

                if mae < best_score:
                    best_score = mae
                    best_model = model
            else:
                pred_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else model.predict(X_val)
                if hasattr(pred_proba, "shape") and len(pred_proba.shape) > 1:
                    pred_proba = pred_proba[:, 1]

                try:
                    auc = roc_auc_score(y_val, pred_proba)
                except ValueError:
                    auc = 0.5

                pred_label = (pred_proba > 0.5).astype(int)
                acc = accuracy_score(y_val, pred_label)
                scores.append({"auc": auc, "accuracy": acc})
                print(f"  Fold {fold_idx+1}: AUC={auc:.4f}, Accuracy={acc:.4f}")

                if auc > best_score:
                    best_score = auc
                    best_model = model

        # 全foldスキップされた場合のフォールバック
        if not scores or not importances or best_model is None:
            print(f"  ⚠️ 有効なfoldがありません。sklearnフォールバックで全データ学習")
            if task == "regression":
                best_model = self._train_sklearn(X, y, task)
                pred = best_model.predict(X)
                scores = [{"mae": mean_absolute_error(y, pred), "rmse": np.sqrt(mean_squared_error(y, pred))}]
            else:
                # 全データでも1クラスなら回帰にフォールバック
                if len(np.unique(y)) < 2:
                    best_model = self._train_sklearn(X, y, "regression")
                    scores = [{"auc": 0.5, "accuracy": 0.0}]
                else:
                    best_model = self._train_sklearn(X, y, task)
                    scores = [{"auc": 0.5, "accuracy": 0.0}]
            importances = [pd.DataFrame({"feature": self.feature_columns, "importance": best_model.feature_importances_})]

        # 特徴量重要度の平均
        avg_importance = importances[0].copy()
        avg_importance["importance"] = np.mean(
            [imp["importance"].values for imp in importances], axis=0
        )
        avg_importance = avg_importance.sort_values("importance", ascending=False)

        # 平均スコア
        avg_scores = {}
        for key in scores[0].keys():
            vals = [s[key] for s in scores]
            avg_scores[f"mean_{key}"] = np.mean(vals)
            avg_scores[f"std_{key}"] = np.std(vals)

        return best_model, avg_scores, avg_importance

    def _train_lgbm(self, X_train, y_train, X_val, y_val, task):
        """LightGBMで学習"""
        params = self.LGBM_PARAMS_REGRESSION.copy() if task == "regression" else self.LGBM_PARAMS_BINARY.copy()

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        return model

    def _train_sklearn(self, X_train, y_train, task):
        """sklearn GBDTで学習（LightGBMが無い場合のフォールバック）"""
        if task == "regression":
            model = GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
        model.fit(X_train, y_train)
        return model

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """予測実行: 各モデルのスコアを統合"""
        if not self.models:
            raise RuntimeError("モデルが未学習です。先にtrain()を実行してください。")

        if df.empty:
            return pd.DataFrame(columns=["race_id", "horse_number", "predicted_rank", "composite_score"])

        # 必要カラムの存在チェック - 不足分は0埋め（拡張特徴量が未計算のケース対応）
        missing_cols = [c for c in self.feature_columns if c not in df.columns]
        if missing_cols:
            import logging
            logging.getLogger(__name__).warning(
                f"特徴量 {len(missing_cols)}個が不足 → 0埋めで補完: {missing_cols[:5]}..."
            )
            for c in missing_cols:
                df[c] = 0

        X = df[self.feature_columns].copy()
        # Inf → NaN → 中央値で補完
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0)

        results = df[["race_id", "horse_number", "horse_name"]].copy() if "horse_name" in df.columns else df[["race_id", "horse_number"]].copy()

        # 各モデルで予測
        if "position" in self.models:
            model = self.models["position"]
            results["pred_position"] = model.predict(X)

        if "top3" in self.models:
            model = self.models["top3"]
            pred = model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)
            results["prob_top3"] = pred[:, 1] if hasattr(pred, "shape") and len(pred.shape) > 1 else pred

        if "win" in self.models:
            model = self.models["win"]
            pred = model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)
            results["prob_win"] = pred[:, 1] if hasattr(pred, "shape") and len(pred.shape) > 1 else pred

        # 統合スコア（着順が低い＝良い、確率が高い＝良い）
        if "pred_position" in results.columns and "prob_top3" in results.columns and "prob_win" in results.columns:
            # 着順を正規化（反転: 低い着順＝高スコア）
            pos_score = 1 - (results["pred_position"] - results["pred_position"].min()) / (
                results["pred_position"].max() - results["pred_position"].min() + 1e-8
            )
            results["composite_score"] = (
                pos_score * 0.3
                + results["prob_top3"] * 0.3
                + results["prob_win"] * 0.4
            )
            # NaN/Inf を 0 に置換
            results["composite_score"] = results["composite_score"].replace([np.inf, -np.inf], 0).fillna(0)

        # レース内ランキング
        results["predicted_rank"] = results.groupby("race_id")["composite_score"].rank(
            ascending=False, method="min"
        ).astype(int)

        return results.sort_values(["race_id", "predicted_rank"])

    def simulate_betting(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        strategy: str = "top1_win",
        bet_amount: int = 100,
    ) -> Dict:
        """回収率シミュレーション"""
        merged = predictions.merge(
            actuals[["race_id", "horse_number", "finish_position", "odds"]],
            on=["race_id", "horse_number"],
            how="inner",
        )

        total_bet = 0
        total_return = 0
        wins = 0
        bets = 0

        for race_id, race in merged.groupby("race_id"):
            if strategy == "top1_win":
                # 予測1位に単勝
                top = race.sort_values("predicted_rank").iloc[0]
                total_bet += bet_amount
                bets += 1
                if top["finish_position"] == 1:
                    total_return += bet_amount * top["odds"]
                    wins += 1

            elif strategy == "top3_wide":
                # 予測上位3頭に複勝
                tops = race.sort_values("predicted_rank").head(3)
                for _, row in tops.iterrows():
                    total_bet += bet_amount
                    bets += 1
                    if row["finish_position"] <= 3:
                        total_return += bet_amount * (row["odds"] * 0.3)  # 複勝概算
                        wins += 1

            elif strategy == "value_bet":
                # 期待値プラスの馬のみ購入
                for _, row in race.iterrows():
                    expected_value = row["prob_win"] * row["odds"]
                    if expected_value > 1.2:  # 期待値1.2倍以上
                        total_bet += bet_amount
                        bets += 1
                        if row["finish_position"] == 1:
                            total_return += bet_amount * row["odds"]
                            wins += 1

        roi = (total_return / total_bet * 100) if total_bet > 0 else 0
        profit = total_return - total_bet

        result = {
            "strategy": strategy,
            "total_bets": bets,
            "total_bet_amount": total_bet,
            "total_return": round(total_return, 0),
            "profit": round(profit, 0),
            "roi_percent": round(roi, 2),
            "win_rate_percent": round(wins / bets * 100, 2) if bets > 0 else 0,
            "wins": wins,
        }

        return result

    MODEL_VERSION = "2.0"

    def save(self, name: str = "keiba_model"):
        """モデル保存"""
        save_path = self.model_dir / f"{name}.pkl"
        data = {
            "version": self.MODEL_VERSION,
            "models": self.models,
            "feature_columns": self.feature_columns,
            "evaluation_results": self.evaluation_results,
            "feature_importance": {k: v.to_dict() for k, v in self.feature_importance.items()},
            "saved_at": datetime.now().isoformat(),
            "n_features": len(self.feature_columns),
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 モデル保存: {save_path} (v{self.MODEL_VERSION}, {len(self.feature_columns)}特徴量)")

    def load(self, name: str = "keiba_model"):
        """モデル読み込み"""
        load_path = self.model_dir / f"{name}.pkl"
        if not load_path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {load_path}")

        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as e:
            raise RuntimeError(f"モデルファイル破損: {load_path} ({e})")

        # バージョンチェック
        saved_ver = data.get("version", "unknown")
        if saved_ver != self.MODEL_VERSION:
            print(f"⚠️ モデルバージョン不一致: 保存={saved_ver}, 現在={self.MODEL_VERSION}")
            print(f"   再学習を推奨します")

        # 必須キーチェック
        required = ["models", "feature_columns"]
        for key in required:
            if key not in data:
                raise RuntimeError(f"モデルファイルに必須キー '{key}' がありません")

        self.models = data["models"]
        self.feature_columns = data["feature_columns"]
        self.evaluation_results = data.get("evaluation_results", {})
        self.feature_importance = {
            k: pd.DataFrame(v) for k, v in data.get("feature_importance", {}).items()
        }
        print(f"✅ モデル読み込み: {load_path} (v{saved_ver}, {len(self.feature_columns)}特徴量)")

    def get_top_features(self, model_name: str = "position", top_n: int = 20) -> pd.DataFrame:
        """重要特徴量Top-N"""
        if model_name not in self.feature_importance:
            return pd.DataFrame()
        return self.feature_importance[model_name].head(top_n)

    def _print_summary(self):
        """学習結果サマリー"""
        print("\n" + "=" * 60)
        print("📈 学習結果サマリー")
        print("=" * 60)
        for name, scores in self.evaluation_results.items():
            print(f"\n【{name}】")
            for k, v in scores.items():
                print(f"  {k}: {v:.4f}")

        print("\n🏆 重要特徴量 Top10:")
        for name in ["position", "top3", "win"]:
            if name in self.feature_importance:
                top10 = self.feature_importance[name].head(10)
                print(f"\n  [{name}]")
                for _, row in top10.iterrows():
                    print(f"    {row['feature']:35s} : {row['importance']:.1f}")


def export_results_json(predictions: pd.DataFrame, output_path: str = "predictions.json"):
    """予測結果をJSON形式でエクスポート（UI連携用）"""
    races = {}
    for race_id, group in predictions.groupby("race_id"):
        horses = []
        for _, row in group.iterrows():
            horse = {
                "horse_number": int(row["horse_number"]),
                "horse_name": row.get("horse_name", f"馬{int(row['horse_number'])}"),
                "predicted_rank": int(row["predicted_rank"]),
                "composite_score": round(float(row.get("composite_score", 0)), 4),
                "prob_win": round(float(row.get("prob_win", 0)), 4),
                "prob_top3": round(float(row.get("prob_top3", 0)), 4),
                "pred_position": round(float(row.get("pred_position", 0)), 2),
            }
            horses.append(horse)
        races[race_id] = sorted(horses, key=lambda x: x["predicted_rank"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(races, f, ensure_ascii=False, indent=2)
    print(f"💾 予測結果をJSONエクスポート: {output_path}")


if __name__ == "__main__":
    # --- テスト実行 ---
    from data_processor import RaceDataProcessor, FeatureEngineer, SampleDataGenerator

    print("=" * 60)
    print("🏇 競馬予想AI - モデル学習テスト")
    print("=" * 60)

    # 1. サンプルデータ生成
    gen = SampleDataGenerator()
    raw_df = gen.generate(n_races=800)

    # 2. 前処理
    processor = RaceDataProcessor()
    processor.raw_data = raw_df
    df = processor.preprocess()

    # 3. 特徴量生成
    fe = FeatureEngineer()
    df = fe.create_features(df)
    feature_cols = fe.get_feature_columns(df)

    # 4. 学習
    model = RacePredictionModel()
    results = model.train(df, feature_cols)

    # 5. 予測
    test_df = df.tail(500)
    predictions = model.predict(test_df)
    print("\n🏇 予測結果 (先頭10件):")
    print(predictions.head(10).to_string())

    # 6. 回収率シミュレーション
    print("\n💰 回収率シミュレーション:")
    for strategy in ["top1_win", "top3_wide", "value_bet"]:
        result = model.simulate_betting(predictions, test_df, strategy=strategy)
        print(f"\n  [{result['strategy']}]")
        print(f"    購入回数: {result['total_bets']}")
        print(f"    投資額:   ¥{result['total_bet_amount']:,.0f}")
        print(f"    回収額:   ¥{result['total_return']:,.0f}")
        print(f"    収支:     ¥{result['profit']:+,.0f}")
        print(f"    回収率:   {result['roi_percent']}%")
        print(f"    的中率:   {result['win_rate_percent']}%")

    # 7. 保存
    model.save("test_model")
    export_results_json(predictions, "data/predictions.json")
