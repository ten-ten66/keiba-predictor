"""
競馬予想AI - モデルチューニング
================================
Optuna自動ハイパーパラメータ最適化 + カスタム損失関数。

改善点:
  1. Optunaで LightGBM/XGBoost のパラメータ自動探索
  2. 正規化着順ターゲット（出走頭数で正規化）
  3. 回収率最適化カスタム目的関数
  4. アンサンブル重みの自動最適化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import warnings
import json
import os

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠️ pip install optuna で自動チューニングが有効になります")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


class ModelTuner:
    """Optunaによるハイパーパラメータ自動チューニング"""

    def __init__(self, n_trials: int = 50, n_splits: int = 3):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.best_params: Dict[str, Dict] = {}
        self.study_results: Dict[str, Dict] = {}

    def tune_all(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_col: str = "finish_position",
    ) -> Dict[str, Dict]:
        """全モデルのハイパーパラメータをチューニング"""
        if not HAS_OPTUNA:
            print("⚠️ Optunaなし。デフォルトパラメータを使用")
            return self._default_params()

        X = df[feature_columns].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            med = X[col].median()
            X[col] = X[col].fillna(med if not pd.isna(med) else 0)

        y_pos = df[target_col].values

        # 正規化着順（出走頭数で割る）
        if "n_runners" in df.columns:
            y_norm = ((df[target_col] - 1) / (df["n_runners"] - 1).clip(lower=1)).values
        else:
            y_norm = y_pos / y_pos.max()

        y_top3 = (df[target_col] <= 3).astype(int).values
        y_win = (df[target_col] == 1).astype(int).values

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        print("🔧 ハイパーパラメータチューニング開始")
        print(f"   試行数: {self.n_trials} / CV: {self.n_splits}fold")

        # 1. 正規化着順回帰
        print("\n  [1/3] 正規化着順回帰...")
        self.best_params["position"] = self._tune_regression(X, y_norm, tscv)

        # 2. 複勝分類
        print("  [2/3] 複勝分類...")
        self.best_params["top3"] = self._tune_classification(X, y_top3, tscv)

        # 3. 勝利分類
        print("  [3/3] 勝利分類...")
        self.best_params["win"] = self._tune_classification(X, y_win, tscv)

        # 保存
        self._save_params()

        return self.best_params

    def _tune_regression(self, X, y, tscv) -> Dict:
        """回帰モデルのチューニング"""
        def objective(trial):
            params = self._suggest_params(trial, task="regression")
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                if HAS_LGBM:
                    train_data = lgb.Dataset(X_tr, label=y_tr)
                    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                    model = lgb.train(
                        params, train_data, num_boost_round=500,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
                    )
                    pred = model.predict(X_val)
                else:
                    model = GradientBoostingRegressor(
                        n_estimators=params.get("n_estimators", 200),
                        max_depth=params.get("max_depth", 5),
                        learning_rate=params.get("learning_rate", 0.05),
                        subsample=params.get("subsample", 0.8),
                        random_state=42,
                    )
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_val)

                scores.append(mean_absolute_error(y_val, pred))
            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        print(f"    最良MAE: {study.best_value:.4f}")
        self.study_results["position"] = {"best_value": study.best_value, "n_trials": self.n_trials}
        return study.best_params

    def _tune_classification(self, X, y, tscv) -> Dict:
        """分類モデルのチューニング"""
        def objective(trial):
            params = self._suggest_params(trial, task="binary")
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                if HAS_LGBM:
                    train_data = lgb.Dataset(X_tr, label=y_tr)
                    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                    model = lgb.train(
                        params, train_data, num_boost_round=500,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
                    )
                    pred = model.predict(X_val)
                else:
                    model = GradientBoostingClassifier(
                        n_estimators=params.get("n_estimators", 200),
                        max_depth=params.get("max_depth", 5),
                        learning_rate=params.get("learning_rate", 0.05),
                        subsample=params.get("subsample", 0.8),
                        random_state=42,
                    )
                    model.fit(X_tr, y_tr)
                    pred = model.predict_proba(X_val)[:, 1]

                try:
                    scores.append(roc_auc_score(y_val, pred))
                except ValueError:
                    scores.append(0.5)

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        print(f"    最良AUC: {study.best_value:.4f}")
        self.study_results["classification"] = {"best_value": study.best_value}
        return study.best_params

    def _suggest_params(self, trial, task: str = "regression") -> Dict:
        """Optunaのtrial から パラメータを提案"""
        if HAS_LGBM:
            params = {
                "objective": task if task == "regression" else "binary",
                "metric": "mae" if task == "regression" else "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
                "verbose": -1,
                "n_jobs": -1,
                "seed": 42,
            }
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }
        return params

    def _default_params(self) -> Dict[str, Dict]:
        """Optunaなし時のデフォルト"""
        return {
            "position": {"num_leaves": 63, "learning_rate": 0.05},
            "top3": {"num_leaves": 63, "learning_rate": 0.05},
            "win": {"num_leaves": 63, "learning_rate": 0.05},
        }

    def _save_params(self, path: str = "data/tuned_params.json"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"params": self.best_params, "results": self.study_results}, f, indent=2)
        print(f"  💾 パラメータ保存: {path}")


class EnsembleOptimizer:
    """アンサンブル重みの最適化"""

    @staticmethod
    def optimize_weights(
        pos_scores: np.ndarray,
        top3_probs: np.ndarray,
        win_probs: np.ndarray,
        actual_positions: np.ndarray,
        actual_odds: np.ndarray,
        n_trials: int = 200,
    ) -> Tuple[float, float, float]:
        """回収率を最大化する重みを探索"""
        if not HAS_OPTUNA:
            return 0.3, 0.3, 0.4

        def objective(trial):
            w1 = trial.suggest_float("w_pos", 0.0, 0.5)
            w2 = trial.suggest_float("w_top3", 0.0, 0.5)
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                return -9999

            composite = pos_scores * w1 + top3_probs * w2 + win_probs * w3

            # 各レースで予測1位の馬が実際に1着だった場合の回収率をシミュレート
            total_bet = 0
            total_return = 0

            # race_idでグループ化するために十分な情報がないので簡易計算
            # ここではcomposite上位の的中率×平均オッズで回収率を近似
            top_mask = composite > np.percentile(composite, 90)
            if top_mask.sum() == 0:
                return 0

            hits = actual_positions[top_mask] == 1
            if hits.sum() == 0:
                return 0

            roi = hits.mean() * actual_odds[top_mask][hits].mean()
            return roi

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        w1 = study.best_params["w_pos"]
        w2 = study.best_params["w_top3"]
        w3 = 1.0 - w1 - w2

        print(f"  最適重み: pos={w1:.2f} / top3={w2:.2f} / win={w3:.2f}")
        print(f"  推定ROI: {study.best_value:.3f}")

        return w1, w2, max(0, w3)


class MultiObjectiveTuner:
    """MAE + Spearman 相関の同時最適化 (Pareto Front 探索)。

    Optuna の NSGAIISampler を使い、2 目的を同時最小化する。
      - 目的 1: CV MAE (着順予測誤差、小さい方が良い)
      - 目的 2: -mean Spearman (順位相関の負値、小さい方が良い = 相関が高い)

    Pareto Front の試行を返す。
    """

    def __init__(self, n_trials: int = 50, n_splits: int = 3):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.study: Optional[object] = None
        self.pareto_params: List[Dict] = []

    def tune(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_col: str = "finish_position",
    ) -> List[Dict]:
        """MAE + Spearman 相関の同時最適化を実行。

        Returns:
            Pareto Front の試行パラメータリスト (best_params dict のリスト)
        """
        if not HAS_OPTUNA:
            print("Optuna 未インストール。Multi-Objective チューニングをスキップ")
            return []

        try:
            from scipy.stats import spearmanr as _spearmanr
        except ImportError:
            print("scipy 未インストール。Multi-Objective チューニングをスキップ")
            return []

        X = df[feature_columns].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            med = X[col].median()
            X[col] = X[col].fillna(med if not pd.isna(med) else 0)

        y = df[target_col].values
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        def objective(trial):
            params: Dict = {}
            if HAS_LGBM:
                params = {
                    "objective": "regression",
                    "metric": "mae",
                    "boosting_type": "gbdt",
                    "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
                    "verbose": -1,
                    "n_jobs": -1,
                    "seed": 42,
                }
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                }

            mae_scores = []
            spearman_scores = []

            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                try:
                    if HAS_LGBM:
                        dtrain = lgb.Dataset(X_tr, label=y_tr)
                        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
                        m = lgb.train(
                            params, dtrain, num_boost_round=500,
                            valid_sets=[dval],
                            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
                        )
                        pred = m.predict(X_val)
                    else:
                        from sklearn.ensemble import GradientBoostingRegressor as GBR
                        m = GBR(
                            n_estimators=params.get("n_estimators", 200),
                            max_depth=params.get("max_depth", 5),
                            learning_rate=params.get("learning_rate", 0.05),
                            subsample=params.get("subsample", 0.8),
                            random_state=42,
                        )
                        m.fit(X_tr, y_tr)
                        pred = m.predict(X_val)

                    mae_scores.append(mean_absolute_error(y_val, pred))
                    corr, _ = _spearmanr(y_val, pred)
                    spearman_scores.append(corr if not np.isnan(corr) else 0.0)
                except Exception:
                    mae_scores.append(9999.0)
                    spearman_scores.append(0.0)

            avg_mae = float(np.mean(mae_scores))
            avg_spearman = float(np.mean(spearman_scores))
            # 目的 1: MAE (minimize), 目的 2: -Spearman (minimize = 相関最大化)
            return avg_mae, -avg_spearman

        print(f"\nMulti-Objective チューニング開始 (trials={self.n_trials}, CV={self.n_splits})")
        print("  目的 1: MAE 最小化 / 目的 2: Spearman 相関 最大化")

        sampler = optuna.samplers.NSGAIISampler(seed=42)
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=sampler,
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self.study = study

        # Pareto Front の試行を取得
        pareto_trials = study.best_trials
        self.pareto_params = [t.params for t in pareto_trials]

        print(f"\n  Pareto Front: {len(pareto_trials)} 試行")
        print(f"  {'Trial':>6}  {'MAE':>8}  {'Spearman':>10}  {'num_leaves':>10}  {'lr':>8}")
        for t in sorted(pareto_trials, key=lambda x: x.values[0])[:10]:
            mae_val = t.values[0]
            spearman_val = -t.values[1]
            nl = t.params.get("num_leaves", "-")
            lr = t.params.get("learning_rate", float("nan"))
            print(f"  {t.number:>6}  {mae_val:>8.4f}  {spearman_val:>10.4f}  {str(nl):>10}  {lr:>8.4f}")

        return self.pareto_params

    def best_balanced_params(self) -> Optional[Dict]:
        """Pareto Front から MAE と Spearman のバランスが最も良い試行を返す。

        正規化後の 2 目的の距離が原点（理想点）に最も近い試行を選ぶ。
        """
        if self.study is None:
            return None
        pareto_trials = self.study.best_trials
        if not pareto_trials:
            return None

        vals = np.array([t.values for t in pareto_trials])
        # 正規化
        mins = vals.min(axis=0)
        maxs = vals.max(axis=0)
        denom = np.where(maxs - mins > 1e-8, maxs - mins, 1.0)
        normalized = (vals - mins) / denom
        # 原点からのユークリッド距離が最小の試行
        distances = np.linalg.norm(normalized, axis=1)
        best_idx = int(np.argmin(distances))
        best_trial = pareto_trials[best_idx]
        mae_val = best_trial.values[0]
        sp_val = -best_trial.values[1]
        print(f"\n  バランス最良試行 #{best_trial.number}: MAE={mae_val:.4f}, Spearman={sp_val:.4f}")
        return best_trial.params

    def save_pareto(self, path: str = "data/pareto_params.json"):
        """Pareto Front のパラメータを JSON に保存する。"""
        if not self.pareto_params:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "pareto_trials": len(self.pareto_params),
            "params": self.pareto_params,
            "balanced_best": self.best_balanced_params(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Pareto Front 保存: {path}")


if __name__ == "__main__":
    print("=" * 50)
    print("🔧 モデルチューニング テスト")
    print("=" * 50)

    from data_processor import SampleDataGenerator, RaceDataProcessor, FeatureEngineer
    from features_advanced import AdvancedFeatureEngineer

    raw = SampleDataGenerator().generate(n_races=100)
    proc = RaceDataProcessor()
    proc.raw_data = raw
    df = proc.preprocess()

    fe = FeatureEngineer()
    df = fe.create_features(df)

    afe = AdvancedFeatureEngineer()
    df = afe.create_all(df)

    cols = afe.get_extended_feature_columns(df, [])
    print(f"特徴量数: {len(cols)}")

    tuner = ModelTuner(n_trials=5, n_splits=2)  # テスト用に少なく
    params = tuner.tune_all(df, cols)
    print(f"\n最適パラメータ: {json.dumps(params, indent=2)}")
