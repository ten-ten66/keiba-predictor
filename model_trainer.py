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
    ndcg_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from scipy.stats import spearmanr

import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

from collections import defaultdict

# imblearn (SMOTE)
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


class EloRatingSystem:
    def __init__(self, k_factor=16, initial_rating=1500):
        self.k = k_factor
        self.initial = initial_rating
        self.ratings = {}

    def get_rating(self, entity_id):
        return self.ratings.get(entity_id, self.initial)

    def expected_score(self, ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update_race(self, participants):
        n = len(participants)
        if n < 2:
            return
        ratings = {pid: self.get_rating(pid) for pid, _ in participants}
        deltas = defaultdict(list)
        for i, (id_i, pos_i) in enumerate(participants):
            delta = 0.0
            for j, (id_j, pos_j) in enumerate(participants):
                if i == j:
                    continue
                expected = self.expected_score(ratings[id_i], ratings[id_j])
                actual = 1.0 if pos_i < pos_j else (0.5 if pos_i == pos_j else 0.0)
                delta += self.k * (actual - expected) / (n - 1)
            deltas[id_i].append(delta)
        for eid, delta_list in deltas.items():
            self.ratings[eid] = ratings[eid] + np.mean(delta_list)


class GlickoRatingSystem:
    def __init__(self, initial_rating=1500, initial_rd=350, c=15):
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.c = c
        self.ratings = {}
        self.rds = {}
        self.last_race = {}

    def get_rating(self, entity_id):
        return self.ratings.get(entity_id, self.initial_rating)

    def get_rd(self, entity_id, current_period=0):
        rd = self.rds.get(entity_id, self.initial_rd)
        last = self.last_race.get(entity_id, current_period)
        periods_inactive = current_period - last
        return min(350, np.sqrt(rd ** 2 + self.c ** 2 * periods_inactive))

    def g(self, rd):
        q = np.log(10) / 400.0
        return 1.0 / np.sqrt(1 + 3 * (q * rd / np.pi) ** 2)

    def expected(self, r, r_opp, rd_opp):
        return 1.0 / (1 + 10 ** (-self.g(rd_opp) * (r - r_opp) / 400))

    def update_race(self, participants, period=0):
        n = len(participants)
        if n < 2:
            return
        pre_ratings = {pid: self.get_rating(pid) for pid, _ in participants}
        pre_rds = {pid: self.get_rd(pid, period) for pid, _ in participants}
        for pid, pos in participants:
            r = pre_ratings[pid]
            rd = pre_rds[pid]
            d2_inv = 0
            delta_sum = 0
            for oid, opos in participants:
                if pid == oid:
                    continue
                r_opp = pre_ratings[oid]
                rd_opp = pre_rds[oid]
                g_val = self.g(rd_opp)
                e_val = self.expected(r, r_opp, rd_opp)
                actual = 1.0 if pos < opos else (0.5 if pos == opos else 0.0)
                d2_inv += g_val ** 2 * e_val * (1 - e_val)
                delta_sum += g_val * (actual - e_val)
            if d2_inv == 0:
                continue
            d2 = 1.0 / (d2_inv / (n - 1))
            new_rd = 1.0 / np.sqrt(1 / rd ** 2 + 1 / d2)
            new_r = r + (new_rd ** 2) * delta_sum / (n - 1)
            self.ratings[pid] = new_r
            self.rds[pid] = new_rd
            self.last_race[pid] = period


class PlattScaler:
    """Isotonic Regression キャリブレーション（Platt Scalingより頑健）

    Platt Scaling（2パラメータロジスティック）は単調増加のS字を仮定するため、
    競馬のように「高確率帯で過大推定」が起きるとうまく補正できない。
    Isotonic Regressionは形状の仮定なしで単調な変換を学習する。

    フォールバック: IsotonicRegressionが使えない場合はPlatt Scalingを使用。
    """
    def __init__(self):
        self._isotonic = None
        self.a = 1.0
        self.b = 0.0

    def fit(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        try:
            from sklearn.isotonic import IsotonicRegression
            self._isotonic = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self._isotonic.fit(y_pred, y_true)
        except Exception:
            # フォールバック: 2パラメータPlatt
            self._isotonic = None
            y_pred_c = np.clip(y_pred, 1e-7, 1 - 1e-7)
            f = np.log(y_pred_c / (1 - y_pred_c))
            from scipy.optimize import minimize
            def nll(params):
                a, b = params
                p = np.clip(1.0 / (1.0 + np.exp(-(a * f + b))), 1e-7, 1 - 1e-7)
                return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
            result = minimize(nll, [1.0, 0.0], method='Nelder-Mead')
            self.a, self.b = result.x
        return self

    def transform(self, y_pred):
        y_pred = np.asarray(y_pred, dtype=float)
        if self._isotonic is not None:
            return np.clip(self._isotonic.predict(y_pred), 0.001, 0.999)
        y_pred_c = np.clip(y_pred, 1e-7, 1 - 1e-7)
        f = np.log(y_pred_c / (1 - y_pred_c))
        return 1.0 / (1.0 + np.exp(-(self.a * f + self.b)))


def _focal_loss_lgbm(gamma=2.0, alpha=0.25):
    """Focal Loss for LightGBM: 難しい例を重視するカスタム損失関数

    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    where pt = p if y==positive else (1-p)

    注意: 正例判定は `y_true > 0.5` で行う。Label Smoothing 済みラベル
    (例: 正例=0.975, 負例=0.025) では `y_true == 1` が一度も成立せず、
    全サンプルが負例扱いになって focal 重みが逆転する (正例ほど学習されない)
    バグが v7.2 学習時に存在した。残差 (p - y_true) は平滑化ラベルに
    向けたままにし、smoothing の過信防止効果は維持する。
    """
    def focal_loss_obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        is_pos = y_true > 0.5  # 平滑化ラベル対応の正例判定
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        pt = np.where(is_pos, p, 1 - p)
        at = np.where(is_pos, alpha, 1 - alpha)
        # gradient: d(FL)/d(y_pred) の簡略化: focal_weight * (p - y_true)
        focal_w = at * np.power(1 - pt, gamma)
        grad = focal_w * (p - y_true)
        # hessian（2次近似、正定値保証）
        hess = focal_w * p * (1 - p)
        hess = np.maximum(hess, 1e-8)
        return grad, hess

    def focal_loss_eval(y_pred, dtrain):
        y_true = dtrain.get_label()
        is_pos = y_true > 0.5
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        pt = np.where(is_pos, p, 1 - p)
        at = np.where(is_pos, alpha, 1 - alpha)
        loss = -at * np.power(1 - pt, gamma) * np.log(pt + 1e-8)
        return "focal_loss", np.mean(loss), False

    return focal_loss_obj, focal_loss_eval


def _ordinal_regression_lgbm(n_classes: int = 18):
    """Ordinal Regression custom objective for LightGBM.

    着順の順序関係（1着 < 2着 < 3着 ...）を明示的に学習する。
    累積ロジットモデル（proportional odds）を近似的に実装。

    ラベルは「レース内で正規化済みの [0,1] 値」((着順-1)/(頭数-1)) を渡すこと
    (2026-07-02 変更)。旧実装は生着順をデータ全体の最大着順で割っており、
    8頭立ての8着と18頭立ての8着が同じターゲットになる歪みがあった。
    n_classes 引数は後方互換のため残置 (未使用)。

    参考: Frank & Hall (2001) "A Simple Approach to Ordinal Classification"
    """
    def ordinal_obj(y_pred, dtrain):
        y_norm = dtrain.get_label()  # 事前にレース内正規化済み [0,1]
        # sigmoid で予測を [0,1] に圧縮
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        residual = p - y_norm
        # 順序ペナルティ: 両端 (勝ち馬・大敗) ほど重みを乗算
        order_weight = 1.0 + np.abs(y_norm - 0.5) * 2.0
        grad = order_weight * residual * p * (1 - p)
        hess = order_weight * p * (1 - p)
        hess = np.maximum(hess, 1e-8)
        return grad, hess

    def ordinal_eval(y_pred, dtrain):
        y_norm = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        loss = np.mean((p - y_norm) ** 2)
        return "ordinal_mse", loss, False

    return ordinal_obj, ordinal_eval


def _lr_decay_callback(init_lr=0.03, min_lr=0.005, decay_rate=0.995):
    """Cosine Annealing風の学習率減衰コールバック（LightGBM用）"""
    def callback(env):
        lr = max(min_lr, init_lr * (decay_rate ** env.iteration))
        env.params["learning_rate"] = lr
    callback.order = 0
    return callback


class _LGBScorerWrapper:
    def __init__(self, m):
        self.m = m
    def fit(self, X, y):
        return self
    def predict(self, X):
        return self.m.predict(X)
    def score(self, X, y):
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(y, self.m.predict(X))
        except ValueError:
            return 0.5


class PurgedTimeSeriesSplit:
    def __init__(self, n_splits=5, purge_days=7, embargo_days=7):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, X, dates=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        dates_ns = None
        if dates is not None and len(dates) > 0:
            try:
                dates_ns = pd.to_datetime(pd.Series(dates), errors="coerce").values.astype("int64")
                purge_ns = int(pd.Timedelta(days=self.purge_days).value)
                embargo_ns = int(pd.Timedelta(days=self.embargo_days).value)
            except Exception:
                dates_ns = None
        for i in range(self.n_splits):
            val_start = fold_size * (i + 1)
            val_end = fold_size * (i + 2) if i < self.n_splits - 1 else n
            train_idx = np.arange(0, val_start)
            val_idx = np.arange(val_start, val_end)
            if dates_ns is not None:
                try:
                    val_start_ns = dates_ns[val_start]
                    if val_start_ns > 0:
                        purge_cutoff_ns = val_start_ns - purge_ns
                        train_mask = dates_ns[train_idx] < purge_cutoff_ns
                        train_idx = train_idx[train_mask]
                        if self.embargo_days > 0:
                            embargo_end_ns = val_start_ns + embargo_ns
                            val_mask = dates_ns[val_idx] >= embargo_end_ns
                            val_idx = val_idx[val_mask]
                except Exception:
                    pass
            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx


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

# CatBoostが利用可能か確認
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# TabNetが利用可能か確認
try:
    from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Import GOD ensemble classes for pickle deserialization.
try:
    from god_ensemble import _GodEnsembleModel, _EnsembleLGBModel, TemperatureScaler
    HAS_GOD_ENSEMBLE = True
except ImportError:
    HAS_GOD_ENSEMBLE = False


class _BlendedModel:
    """複数モデルタイプ×複数foldのブレンド予測ラッパー。
    predict()/predict_proba()/feature_importance() を提供し、
    単一モデルと同じインターフェースで透過的に使える。
    """

    def __init__(self, fold_models_by_type, task_type, feature_names=None):
        """
        Args:
            fold_models_by_type: {"lgbm": [m1,m2,...], "xgb": [...], ...}
            task_type: "regression" | "binary" | "binary_win"
            feature_names: 特徴量名リスト（importance用）
        """
        self.fold_models_by_type = fold_models_by_type
        self.task_type = task_type
        self.feature_names = feature_names or []

    def _predict_single(self, model, model_type, X):
        """1つのモデルで予測"""
        try:
            if model_type == "lgbm":
                pred = model.predict(X)
                # Focal Loss等のcustom objectiveの場合、raw outputなのでsigmoid変換
                if self.task_type in ("binary", "binary_win"):
                    # raw outputの判定: 範囲が[0,1]外ならsigmoid変換
                    if pred.min() < -0.01 or pred.max() > 1.01:
                        pred = 1.0 / (1.0 + np.exp(-pred))
                return pred
            elif model_type == "tabnet":
                X_np = np.asarray(X, dtype=np.float32)
                if self.task_type == "regression":
                    return model.predict(X_np).flatten()
                else:
                    return model.predict_proba(X_np)[:, 1]
            else:  # xgb, catboost
                if self.task_type == "regression":
                    return model.predict(X)
                elif hasattr(model, "predict_proba"):
                    return model.predict_proba(X)[:, 1]
                else:
                    return model.predict(X)
        except Exception:
            return None

    def predict(self, X):
        """全モデルタイプ×全foldの平均予測（Trimmed Mean + 並列化）

        各モデルタイプの予測をThreadPoolExecutorで並列実行し、
        推論時間を短縮する。GILの制約はあるが、C拡張ベースの
        LightGBM/XGBoost/CatBoostではGILが解放されるため効果あり。
        """
        from concurrent.futures import ThreadPoolExecutor

        def predict_type(mtype, models):
            """1つのモデルタイプ内の全foldを予測してTrimmed Meanを返す"""
            fold_preds = []
            for m in models:
                p = self._predict_single(m, mtype, X)
                if p is not None:
                    fold_preds.append(p)
            if fold_preds:
                # Trimmed Mean: 3モデル以上で最低・最高を除外
                if len(fold_preds) >= 3:
                    stacked = np.sort(np.array(fold_preds), axis=0)
                    return stacked[1:-1].mean(axis=0)
                return np.mean(fold_preds, axis=0)
            return None

        type_preds = []

        # 並列実行: モデルタイプごとに予測（C拡張はGIL解放するため効果あり）
        try:
            with ThreadPoolExecutor(max_workers=min(3, len(self.fold_models_by_type))) as ex:
                futures = {
                    ex.submit(predict_type, mtype, models): mtype
                    for mtype, models in self.fold_models_by_type.items()
                }
                for f in futures:
                    result = f.result()
                    if result is not None:
                        type_preds.append(result)
        except Exception:
            # フォールバック: 並列化失敗時は逐次実行
            for mtype, models in self.fold_models_by_type.items():
                result = predict_type(mtype, models)
                if result is not None:
                    type_preds.append(result)

        if not type_preds:
            return np.zeros(len(X))
        return np.mean(type_preds, axis=0)

    def predict_proba(self, X):
        """2列 [1-p, p] 形式（分類モデル互換）"""
        p = self.predict(X)
        return np.column_stack([1 - p, p])

    def distill(self, X_train, y_train=None):
        """重いアンサンブル -> 軽いシングルLGBモデルに蒸留 (Knowledge Distillation)

        Teacher（このアンサンブル）のsoft labelsを使って、
        軽量なLightGBMシングルモデルを学習する。
        推論速度100x高速化、モデルサイズ1/10を実現。

        Args:
            X_train: 学習用特徴量（DataFrame or ndarray）
            y_train: Hard labels（正解ラベル）。指定時はsoft/hardの混合で学習。
                     Noneの場合はteacher predictionsのみで学習。

        Returns:
            student: 蒸留済みLightGBM Booster
        """
        import lightgbm as lgb

        # Teacher predictions (soft labels)
        teacher_pred = self.predict(X_train)

        if y_train is not None:
            # Hard labelとSoft labelの混合 (alpha=0.5)
            # 理由: soft labelだけだとteacherのバイアスを継承する。
            # hard labelを混ぜることで汎化性能を維持。
            alpha = 0.5
            target = alpha * np.asarray(y_train) + (1 - alpha) * teacher_pred
        else:
            target = teacher_pred

        # Student model (軽量LGB)
        params = {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
        }

        ds = lgb.Dataset(X_train, label=target)
        student = lgb.train(params, ds, num_boost_round=500)

        return student  # 推論100x高速化、サイズ1/10

    def feature_importance(self, importance_type="gain"):
        """全モデルの特徴量重要度を正規化して平均"""
        all_imps = []
        for mtype, models in self.fold_models_by_type.items():
            for m in models:
                try:
                    if mtype == "lgbm":
                        imp = m.feature_importance(importance_type=importance_type).astype(float)
                    elif hasattr(m, "feature_importances_"):
                        imp = np.array(m.feature_importances_, dtype=float)
                    else:
                        continue
                    # 正規化（各モデルのスケールを揃える）
                    total = imp.sum()
                    if total > 0:
                        imp = imp / total
                    all_imps.append(imp)
                except Exception:
                    continue
        if not all_imps:
            return np.zeros(len(self.feature_names))
        return np.mean(all_imps, axis=0)


class RacePredictionModel:
    """競馬予測モデル: 学習・評価・予測を一括管理"""

    # LightGBMハイパーパラメータ（チューニング済みベースライン）
    # v8最適化: 過学習抑制 + 汎化性能向上
    LGBM_PARAMS_REGRESSION = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 95,
        "learning_rate": 0.025,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.65,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "max_depth": 7,
        "reg_alpha": 0.3,
        "reg_lambda": 0.3,
        "min_gain_to_split": 0.01,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    LGBM_PARAMS_BINARY = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 95,
        "learning_rate": 0.025,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.65,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "max_depth": 7,
        "reg_alpha": 0.3,
        "reg_lambda": 0.3,
        "min_gain_to_split": 0.01,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    # Winモデル専用パラメータ（クラス不均衡対策: 勝率~8%に対応）
    LGBM_PARAMS_WIN = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.02,
        "feature_fraction": 0.55,
        "bagging_fraction": 0.6,
        "bagging_freq": 5,
        "min_child_samples": 30,
        "max_depth": 6,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "min_gain_to_split": 0.02,
        "is_unbalance": True,
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

        # インスタンスコピー（apply_tuned_paramsでクラス変数を汚染しない）
        self.LGBM_PARAMS_REGRESSION = self.__class__.LGBM_PARAMS_REGRESSION.copy()
        self.LGBM_PARAMS_BINARY = self.__class__.LGBM_PARAMS_BINARY.copy()
        self.LGBM_PARAMS_WIN = self.__class__.LGBM_PARAMS_WIN.copy()

    def apply_tuned_params(self, tuned_params: Dict):
        """Optunaチューニング結果をLGBMパラメータに適用"""
        if "position" in tuned_params:
            p = tuned_params["position"]
            self.LGBM_PARAMS_REGRESSION.update({
                k: v for k, v in p.items()
                if k in ("num_leaves", "learning_rate", "feature_fraction",
                         "bagging_fraction", "bagging_freq", "min_child_samples", "reg_alpha", "reg_lambda")
            })
            print(f"  ✅ Position パラメータ適用: {p}")
        if "top3" in tuned_params:
            p = tuned_params["top3"]
            self.LGBM_PARAMS_BINARY.update({
                k: v for k, v in p.items()
                if k in ("num_leaves", "learning_rate", "feature_fraction",
                         "bagging_fraction", "bagging_freq", "min_child_samples", "reg_alpha", "reg_lambda")
            })
        if "win" in tuned_params:
            p = tuned_params["win"]
            self.LGBM_PARAMS_WIN.update({
                k: v for k, v in p.items()
                if k in ("num_leaves", "learning_rate", "feature_fraction",
                         "bagging_fraction", "bagging_freq", "min_child_samples", "reg_alpha", "reg_lambda")
            })

    # 利用可能なモデルタイプ
    AVAILABLE_MODELS = {
        "lgbm": HAS_LGBM,
        "xgb": HAS_XGB,
        "catboost": HAS_CATBOOST,
        "tabnet": HAS_TABNET,
    }

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_col: str = "finish_position",
        n_splits: int = 5,
        ensemble_models: Optional[List[str]] = None,
        time_decay_halflife: Optional[float] = 365 * 2,
    ) -> Dict[str, dict]:
        """モデル学習（時系列バリデーション付き）

        Args:
            ensemble_models: 使用するモデルタイプのリスト。
                None → LightGBMのみ（後方互換）
                ["lgbm", "xgb", "catboost"] → 3モデルブレンド
                ["lgbm", "xgb", "catboost", "tabnet"] → 4モデルブレンド
            time_decay_halflife: 時間減衰の半減期（日数）。
                最新データの重み=1.0、halflife日前のデータの重み=0.5。
                None で減衰なし（従来通り均等重み）。
                デフォルト: 730日（2年で半減）
        """
        self.feature_columns = feature_columns

        # 使用モデル決定
        if ensemble_models is None:
            self._ensemble_types = ["lgbm"] if HAS_LGBM else []
        else:
            self._ensemble_types = [m for m in ensemble_models if self.AVAILABLE_MODELS.get(m, False)]
            skipped = [m for m in ensemble_models if m not in self._ensemble_types]
            if skipped:
                print(f"  ⚠️ 未インストールのモデルをスキップ: {skipped}")
        if not self._ensemble_types:
            self._ensemble_types = ["sklearn"]

        use_blend = len(self._ensemble_types) > 1

        print("=" * 60)
        print("🤖 モデル学習開始")
        print(f"   データ: {len(df)} 件 / 特徴量: {len(feature_columns)} 個")
        print(f"   モデル: {self._ensemble_types}" + (" (ブレンド)" if use_blend else ""))
        if time_decay_halflife:
            print(f"   時間減衰: 半減期 {time_decay_halflife:.0f}日")
        print("=" * 60)

        # 時間減衰サンプル重み計算
        if time_decay_halflife and "race_date" in df.columns:
            race_dates = pd.to_datetime(df["race_date"], errors="coerce")
            max_date = race_dates.max()
            days_ago = (max_date - race_dates).dt.days.fillna(0).values.astype(float)
            # 指数減衰: weight = 0.5 ^ (days_ago / halflife)
            self._sample_weights = np.power(0.5, days_ago / time_decay_halflife)
            # 最小重み0.05（古すぎるデータを完全に無視しない）
            self._sample_weights = np.clip(self._sample_weights, 0.05, 1.0)
            oldest_w = self._sample_weights.min()
            newest_w = self._sample_weights.max()
            print(f"   重み範囲: {oldest_w:.3f} (最古) ~ {newest_w:.3f} (最新)")
        else:
            self._sample_weights = None

        # 欠損値をカラムごとの中央値で補完
        # 補完値は保存し、predict() でも同じ値を使う (train/serve 整合。
        # 旧実装はサーブ時にレース内中央値で補完しており分布がズレていた)
        X = df[feature_columns].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        self._feature_medians = {}
        for col in X.columns:
            median_val = X[col].median()
            fill_val = median_val if not pd.isna(median_val) else 0
            self._feature_medians[col] = float(fill_val)
            X[col] = X[col].fillna(fill_val)

        # ターゲット変数
        y_position = df[target_col].values
        y_win = (df[target_col] == 1).astype(int).values
        y_top3 = (df[target_col] <= 3).astype(int).values

        # Label Smoothing: Win/Top3の学習ターゲットを平滑化（過信防止）
        alpha = 0.05
        y_top3_smooth = y_top3.astype(float) * (1 - alpha) + alpha * 0.5
        y_win_smooth = y_win.astype(float) * (1 - alpha) + alpha * 0.5

        self._y_targets = {"top3": y_top3, "win": y_win}

        # 時系列分割（未来のデータでリークしない）
        if "race_date" in df.columns:
            # PurgedTimeSeriesSplit: パージ＆エンバーゴ付き時系列分割
            dates = df["race_date"].values
            tscv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_days=7, embargo_days=7)
            splits = list(tscv.split(X, dates=dates))
            if not splits:
                # フォールバック
                tscv = TimeSeriesSplit(n_splits=n_splits)
                splits = list(tscv.split(X))
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = list(tscv.split(X))

        # --- 0. Adversarial Validation: train vs test 分布チェック ---
        self._run_adversarial_validation(X, df)

        # OOF予測を保存（L2スタッキング用）
        self._oof_preds = {
            "position": np.full(len(X), np.nan),
            "top3": np.full(len(X), np.nan),
            "win": np.full(len(X), np.nan),
        }

        # --- 1. 着順回帰モデル ---
        print("\n📊 1/3: 着順回帰モデル")
        self.models["position"], self.evaluation_results["position"], self.feature_importance["position"] = (
            self._train_model(X, y_position, splits, task="regression")
        )

        # --- 2. 複勝（3着以内）分類モデル（Label Smoothing適用） ---
        print("\n📊 2/3: 複勝（3着以内）分類モデル [Label Smoothing]")
        self.models["top3"], self.evaluation_results["top3"], self.feature_importance["top3"] = (
            self._train_model(X, y_top3_smooth, splits, task="binary", y_binary=y_top3)
        )

        # --- 3. 勝利分類モデル（SMOTE + Focal Loss + クラス不均衡対策 + Label Smoothing） ---
        print("\n📊 3/3: 勝利（1着）分類モデル [Focal Loss + SMOTE + Label Smoothing]")
        X_win, y_win_sm = self._apply_smote(X, y_win)
        # SMOTE 後は splits を再構築せず全データ学習用スプリットに差し替える
        if X_win is not X:
            # SMOTE でサンプルが増えた場合: OOF は元の X で行う（リーク回避）
            # SMOTE 済みデータは最終モデル（全データ）学習のみに使用
            self.models["win"], self.evaluation_results["win"], self.feature_importance["win"] = (
                self._train_model(X, y_win_smooth, splits, task="binary_win", y_binary=y_win)
            )
            # 最終モデルのみ SMOTE 済みデータで上書き学習
            try:
                self.models["win"] = self._train_model_full(X_win, y_win_sm, task="binary_win")
                print("  SMOTE適用済みデータで最終モデルを上書き学習")
            except Exception as e:
                logger.warning(f"SMOTE最終モデル学習失敗（元モデルを維持）: {e}")
        else:
            self.models["win"], self.evaluation_results["win"], self.feature_importance["win"] = (
                self._train_model(X, y_win_smooth, splits, task="binary_win", y_binary=y_win)
            )

        # 最適閾値を ROC 曲線から決定
        self._win_threshold = self._find_optimal_threshold(X, y_win, splits)

        # --- 3.5. Ordinal Regression モデル（着順の順序関係を明示的に学習） ---
        print("\n📊 3.5/3: Ordinal Regression モデル [着順順序学習]")
        self._train_ordinal_model(X, y_position, splits, df)

        # --- 4. Plattキャリブレーション（L2の前に実行: L2がキャリブレ済み入力で学習） ---
        self._train_calibrators(splits)

        # OOFにPlattを適用してからL2に渡す
        for key in ["top3", "win"]:
            cal = self._calibrators.get(key)
            if cal is not None:
                valid = ~np.isnan(self._oof_preds[key])
                if valid.any():
                    self._oof_preds[key][valid] = cal.transform(self._oof_preds[key][valid])
        # L2/L4 はキャリブレ済みOOFで学習される → predict() 側も
        # キャリブレ済み確率をメタ入力に使う必要がある (フラグで明示)
        self._l2_inputs_calibrated = True

        # --- 5. L2スタッキング: キャリブレ済みOOF予測でメタ学習 ---
        self._train_l2_stacking(X, y_position, y_top3, y_win, splits, df)

        # --- 6. LambdaRank（順位最適化） ---
        self._train_lambdarank(X, y_position, splits, df)

        # --- 7. NDCG/Spearman評価 ---
        self._evaluate_ranking(X, y_position, splits, df)

        # --- 8. SHAP特徴量選抜（底辺20%自動削除 + レポート）---
        self._shap_feature_report(X)

        # --- 9. L4スタッキング (L1->L2->L3->L4 拡張) ---
        self._train_l4_stacking(X, y_position)

        # --- 10. Conformal Prediction (95%信頼区間キャリブレーション) ---
        # L4 の後に実行し、predict() が実際に返す最終スケール (L2/L4適用後) の
        # 残差で較正する (旧: L1 OOF 残差 → 適用先とスケール不一致だった)
        self._train_conformal_prediction(X, y_position, splits)

        self._print_summary()
        return self.evaluation_results

    def _train_model(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        splits: list,
        task: str = "regression",
        y_binary: np.ndarray = None,
    ) -> Tuple[object, dict, pd.DataFrame]:
        """個別モデルの学習（CV付き、マルチモデルブレンド対応）

        Args:
            y_binary: Label Smoothing前の元のバイナリラベル。
                      XGBoost/CatBoostの学習とaccuracy計算に使用。
        """
        scores = []
        importances = []

        # シングルモデルモード用
        best_model = None
        best_score = float("inf") if task == "regression" else -float("inf")

        # マルチモデルモード用
        use_blend = len(self._ensemble_types) > 1
        fold_models_by_type = {mtype: [] for mtype in self._ensemble_types}

        # モデルタイプ → 学習関数マッピング
        train_funcs = {
            "lgbm": self._train_lgbm,
            "xgb": self._train_xgb,
            "catboost": self._train_catboost,
            "tabnet": self._train_tabnet,
            "sklearn": self._train_sklearn,
        }

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # バイナリ評価用ラベル（Label Smoothing前）
            if y_binary is not None:
                y_train_bin = y_binary[train_idx]
                y_val_bin = y_binary[val_idx]
            else:
                y_train_bin = y_train
                y_val_bin = y_val

            # 時間減衰重み（fold用に切り出し）
            w_train = self._sample_weights[train_idx] if self._sample_weights is not None else None

            if task in ("binary", "binary_win"):
                # クラス数チェックはバイナリラベルで行う
                check_y = y_val_bin if y_binary is not None else y_val
                if len(np.unique(np.round(check_y))) < 2:
                    print(f"  Fold {fold_idx+1}: スキップ（クラス不足）")
                    continue

            # 各モデルタイプで学習
            fold_preds = []
            for mtype in self._ensemble_types:
                try:
                    fn = train_funcs[mtype]
                    # XGBoost/CatBoostは{0,1}バイナリラベルが必要
                    if mtype in ("xgb", "catboost") and y_binary is not None:
                        y_tr = y_train_bin
                        y_vl = y_val_bin
                    else:
                        y_tr = y_train
                        y_vl = y_val
                    if mtype == "sklearn":
                        model = fn(X_train, y_tr, task, sample_weight=w_train)
                    else:
                        model = fn(X_train, y_tr, X_val, y_vl, task, sample_weight=w_train)
                    fold_models_by_type[mtype].append(model)

                    # fold予測を収集（ブレンド評価用）
                    if mtype == "lgbm":
                        pred = model.predict(X_val)
                        # Focal Loss等custom objectiveはraw output → sigmoid
                        if task in ("binary", "binary_win") and (pred.min() < -0.01 or pred.max() > 1.01):
                            pred = 1.0 / (1.0 + np.exp(-pred))
                    elif mtype == "tabnet":
                        xv = np.asarray(X_val, dtype=np.float32)
                        pred = model.predict(xv).flatten() if task == "regression" else model.predict_proba(xv)[:, 1]
                    elif hasattr(model, "predict_proba") and task != "regression":
                        pred = model.predict_proba(X_val)[:, 1]
                    else:
                        pred = model.predict(X_val)
                    fold_preds.append(pred)
                except Exception as e:
                    print(f"    ⚠️ {mtype} Fold {fold_idx+1} 失敗: {e}")

            if not fold_preds:
                continue

            # ブレンド予測で評価
            blended_pred = np.mean(fold_preds, axis=0)

            # OOF予測保存（L2スタッキング用）
            task_key = "position" if task == "regression" else ("win" if task == "binary_win" else "top3")
            if hasattr(self, "_oof_preds") and task_key in self._oof_preds:
                self._oof_preds[task_key][val_idx] = blended_pred

            # 特徴量重要度（利用可能なモデルから収集）
            fold_imp = np.zeros(len(self.feature_columns))
            imp_count = 0
            for mtype in self._ensemble_types:
                if not fold_models_by_type[mtype]:
                    continue
                m = fold_models_by_type[mtype][-1]
                try:
                    if mtype == "lgbm":
                        raw = m.feature_importance(importance_type="gain").astype(float)
                    elif hasattr(m, "feature_importances_"):
                        raw = np.array(m.feature_importances_, dtype=float)
                    else:
                        continue
                    total = raw.sum()
                    if total > 0:
                        raw = raw / total  # 正規化
                    fold_imp += raw
                    imp_count += 1
                except Exception:
                    pass
            if imp_count > 0:
                fold_imp /= imp_count
            importances.append(pd.DataFrame({
                "feature": self.feature_columns,
                "importance": fold_imp,
            }))

            # 評価
            if task == "regression":
                mae = mean_absolute_error(y_val, blended_pred)
                rmse = np.sqrt(mean_squared_error(y_val, blended_pred))
                scores.append({"mae": mae, "rmse": rmse})
                model_names = "+".join(self._ensemble_types)
                print(f"  Fold {fold_idx+1} [{model_names}]: MAE={mae:.3f}, RMSE={rmse:.3f}")
                if mae < best_score:
                    best_score = mae
                    best_model = fold_models_by_type[self._ensemble_types[0]][-1]
            else:
                # AUCとaccuracyはバイナリラベルで計算
                eval_y = y_val_bin if y_binary is not None else y_val
                try:
                    auc = roc_auc_score(eval_y, blended_pred)
                except ValueError:
                    auc = 0.5
                pred_label = (blended_pred > 0.5).astype(int)
                acc = accuracy_score(eval_y, pred_label)
                scores.append({"auc": auc, "accuracy": acc})
                model_names = "+".join(self._ensemble_types)
                print(f"  Fold {fold_idx+1} [{model_names}]: AUC={auc:.4f}, Acc={acc:.4f}")
                if auc > best_score:
                    best_score = auc
                    best_model = fold_models_by_type[self._ensemble_types[0]][-1]

        # 全foldスキップ時のフォールバック
        if not scores or best_model is None:
            print(f"  ⚠️ 有効なfoldがありません。sklearnフォールバックで全データ学習")
            # sklearnにはバイナリラベルを渡す（Label Smoothed値は非対応）
            y_fallback = y_binary if y_binary is not None else y
            if task == "regression":
                best_model = self._train_sklearn(X, y, task)
                pred = best_model.predict(X)
                scores = [{"mae": mean_absolute_error(y, pred), "rmse": np.sqrt(mean_squared_error(y, pred))}]
            else:
                if len(np.unique(np.round(y_fallback))) < 2:
                    best_model = self._train_sklearn(X, y_fallback, "regression")
                    scores = [{"auc": 0.5, "accuracy": 0.0}]
                else:
                    best_model = self._train_sklearn(X, y_fallback, task)
                    scores = [{"auc": 0.5, "accuracy": 0.0}]
            importances = [pd.DataFrame({"feature": self.feature_columns, "importance": best_model.feature_importances_})]
            use_blend = False

        # 最終モデル構築
        if use_blend:
            # 有効なモデルのみ残す
            active = {k: v for k, v in fold_models_by_type.items() if v}
            if len(active) > 1:
                final_model = _BlendedModel(active, task, self.feature_columns)
            elif active:
                # 1タイプしか成功しなかった場合でもBlendedModelでラップ（fold平均のため）
                final_model = _BlendedModel(active, task, self.feature_columns)
            else:
                final_model = best_model
        else:
            final_model = best_model

        # 特徴量重要度の平均
        if importances:
            avg_importance = importances[0].copy()
            avg_importance["importance"] = np.mean(
                [imp["importance"].values for imp in importances], axis=0
            )
        else:
            avg_importance = pd.DataFrame({
                "feature": self.feature_columns,
                "importance": np.zeros(len(self.feature_columns)),
            })
        avg_importance = avg_importance.sort_values("importance", ascending=False)

        # 平均スコア
        avg_scores = {}
        for key in scores[0].keys():
            vals = [s[key] for s in scores]
            avg_scores[f"mean_{key}"] = np.mean(vals)
            avg_scores[f"std_{key}"] = np.std(vals)

        return final_model, avg_scores, avg_importance

    def _train_lgbm(self, X_train, y_train, X_val, y_val, task, sample_weight=None):
        """LightGBMで学習（Focal Loss対応）"""
        if task == "regression":
            params = self.LGBM_PARAMS_REGRESSION.copy()
        elif task == "binary_win":
            params = self.LGBM_PARAMS_WIN.copy()
        else:
            params = self.LGBM_PARAMS_BINARY.copy()

        # Label Smoothing 等の連続ラベル対応:
        # objective="binary" は {0,1} 以外のラベルを全て正例として扱い、
        # 定数1.0を返す退化モデルになる (v7.2 の top3 LGBM で実際に発生)。
        # [0,1] 内の連続ラベルは cross_entropy 目的関数で学習する。
        if task in ("binary", "binary_win"):
            uniq = np.unique(np.asarray(y_train, dtype=float))
            if not np.all(np.isin(uniq, (0.0, 1.0))):
                params["objective"] = "cross_entropy"
                params["metric"] = "cross_entropy"
                params.pop("is_unbalance", None)  # cross_entropy では非対応

        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Early Stopping: タスクごとに異なるpatience設定
        if task == "binary_win":
            es_patience = 80
        else:
            es_patience = 50

        train_kwargs = dict(
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(es_patience),
                lgb.log_evaluation(0),
                _lr_decay_callback(init_lr=params.get("learning_rate", 0.03), min_lr=0.005, decay_rate=0.995),
            ],
        )

        # Focal Loss: binary_win タスクで使用（難しい例を重視）
        if task == "binary_win" and getattr(self, "_use_focal_loss", True):
            focal_obj, focal_eval = _focal_loss_lgbm(gamma=2.0, alpha=0.25)
            # LightGBM 4.x: objective にcallableを渡す
            params["objective"] = focal_obj
            params.pop("metric", None)
            train_kwargs["feval"] = focal_eval

        model = lgb.train(params, train_data, **train_kwargs)
        return model

    def _train_xgb(self, X_train, y_train, X_val, y_val, task, sample_weight=None):
        """XGBoostで学習"""
        common = dict(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1,
            early_stopping_rounds=50, verbosity=0,
        )
        if task == "regression":
            model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="mae", **common)
        elif task == "binary_win":
            pos_ratio = max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1))
            model = xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                scale_pos_weight=pos_ratio, **common,
            )
        else:
            model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  sample_weight=sample_weight, verbose=False)
        return model

    def _train_catboost(self, X_train, y_train, X_val, y_val, task, sample_weight=None):
        """CatBoostで学習"""
        common = dict(
            iterations=1000, depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=42,
            early_stopping_rounds=50, verbose=0,
        )
        if task == "regression":
            model = CatBoostRegressor(loss_function="RMSE", **common)
        elif task == "binary_win":
            # CatBoostはsample_weightとclass_weightsの併用不可、sample_weightを優先
            model = CatBoostClassifier(loss_function="Logloss", **common)
        else:
            model = CatBoostClassifier(loss_function="Logloss", **common)

        fit_params = dict(eval_set=(X_val, y_val), verbose=0)
        if sample_weight is not None:
            # CatBoost: Pool で重み指定
            from catboost import Pool
            train_pool = Pool(X_train, label=y_train, weight=sample_weight)
            val_pool = Pool(X_val, label=y_val)
            model.fit(train_pool, eval_set=val_pool, verbose=0)
        else:
            model.fit(X_train, y_train, **fit_params)
        return model

    def _train_tabnet(self, X_train, y_train, X_val, y_val, task, sample_weight=None):
        """TabNetで学習"""
        X_tr = np.asarray(X_train, dtype=np.float32)
        X_vl = np.asarray(X_val, dtype=np.float32)
        y_tr = np.asarray(y_train).reshape(-1)
        y_vl = np.asarray(y_val).reshape(-1)

        common = dict(
            n_d=16, n_a=16, n_steps=3,
            gamma=1.5, lambda_sparse=1e-3,
            optimizer_params=dict(lr=0.02),
            scheduler_params=dict(step_size=50, gamma=0.9),
            scheduler_fn=__import__("torch.optim.lr_scheduler", fromlist=["StepLR"]).StepLR,
            mask_type="entmax", seed=42, verbose=0,
        )
        if task == "regression":
            model = TabNetRegressor(**common)
            y_tr = y_tr.reshape(-1, 1).astype(np.float32)
            y_vl = y_vl.reshape(-1, 1).astype(np.float32)
        else:
            model = TabNetClassifier(**common)
        fit_kwargs = dict(
            eval_set=[(X_vl, y_vl)],
            eval_metric=["mae"] if task == "regression" else ["logloss"],
            max_epochs=100, patience=20, batch_size=1024,
        )
        if sample_weight is not None:
            fit_kwargs["weights"] = 1  # TabNet: 1=use sample weights
            # TabNet expects weights as a separate argument in some versions
            # Pass via the training data wrapper
        model.fit(X_tr, y_tr, **fit_kwargs)
        return model

    def _train_sklearn(self, X_train, y_train, task, sample_weight=None):
        """sklearn GBDTで学習（LightGBMが無い場合のフォールバック）"""
        if task == "regression":
            model = GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
        else:
            # binary_win も binary と同じ分類器（sklearnはサンプル重み不要）
            model = GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model

    # ================================================================
    # L2スタッキング: base3モデルのOOF予測をメタ特徴量にして再学習
    # ================================================================
    def _train_l2_stacking(self, X, y_position, y_top3, y_win, splits, df):
        """OOF予測を使ったL2メタ学習"""
        oof_pos = self._oof_preds.get("position", None)
        oof_top3 = self._oof_preds.get("top3", None)
        oof_win = self._oof_preds.get("win", None)
        if oof_pos is None or oof_top3 is None or oof_win is None:
            return

        # OOFが埋まっている行のみ使用
        valid_mask = ~(np.isnan(oof_pos) | np.isnan(oof_top3) | np.isnan(oof_win))
        if valid_mask.sum() < 100:
            print("  ⚠️ L2スタッキング: OOFデータ不足、スキップ")
            return

        print("\n📊 4/7: L2スタッキング (メタ学習)")
        meta_X = np.column_stack([oof_pos[valid_mask], oof_top3[valid_mask], oof_win[valid_mask]])

        # Position用のL2メタ: Ridge回帰
        y_pos_valid = y_position[valid_mask]
        self._l2_position = Ridge(alpha=1.0)
        self._l2_position.fit(meta_X, y_pos_valid)
        l2_pred = self._l2_position.predict(meta_X)
        l2_mae = mean_absolute_error(y_pos_valid, l2_pred)
        print(f"  L2 Position MAE: {l2_mae:.4f}")

        # Top3用のL2メタ: LogisticRegression
        y_top3_valid = y_top3[valid_mask]
        self._l2_top3 = LogisticRegression(C=1.0, max_iter=1000)
        self._l2_top3.fit(meta_X, y_top3_valid)
        l2_top3_pred = self._l2_top3.predict_proba(meta_X)[:, 1]
        try:
            l2_top3_auc = roc_auc_score(y_top3_valid, l2_top3_pred)
        except ValueError:
            l2_top3_auc = 0.5
        print(f"  L2 Top3 AUC: {l2_top3_auc:.4f}")

        # Win用のL2メタ: LogisticRegression
        y_win_valid = y_win[valid_mask]
        self._l2_win = LogisticRegression(C=1.0, max_iter=1000)
        self._l2_win.fit(meta_X, y_win_valid)
        l2_win_pred = self._l2_win.predict_proba(meta_X)[:, 1]
        try:
            l2_win_auc = roc_auc_score(y_win_valid, l2_win_pred)
        except ValueError:
            l2_win_auc = 0.5
        print(f"  L2 Win AUC: {l2_win_auc:.4f}")

        # BMA重み学習: ベクトル化Spearmanグリッドサーチ（高速版）
        print("\n📊 5/7: BMA重み学習 (Spearmanグリッドサーチ)")

        y_position_valid = y_position[valid_mask]

        # レースIDを取得
        if "race_id" in df.columns:
            race_ids_valid = df.iloc[np.where(valid_mask)[0]]["race_id"].values
        else:
            race_ids_valid = np.arange(valid_mask.sum()) // 12

        # ── 事前にレース内正規化済みのpos_scoreを計算（1回だけ） ──
        oof_p = oof_pos[valid_mask]
        oof_t = oof_top3[valid_mask]
        oof_w = oof_win[valid_mask]

        pos_norm = np.zeros_like(oof_p)
        top3_norm = np.zeros_like(oof_t)
        win_norm = np.zeros_like(oof_w)

        # pandas groupby でベクトル化（forループ回避）
        _df_tmp = pd.DataFrame({
            "rid": race_ids_valid, "pos": oof_p, "top3": oof_t, "win": oof_w
        })
        for col, arr in [("pos", pos_norm), ("top3", top3_norm), ("win", win_norm)]:
            gmin = _df_tmp.groupby("rid")[col].transform("min")
            gmax = _df_tmp.groupby("rid")[col].transform("max")
            rng = (gmax - gmin).values + 1e-8
            if col == "pos":
                arr[:] = 1 - (_df_tmp[col].values - gmin.values) / rng  # 反転
            else:
                arr[:] = (_df_tmp[col].values - gmin.values) / rng
        del _df_tmp

        # ── グリッドサーチ ──
        # 探索範囲は各重み [0.05, 0.90] (2026-07-02 拡張)。
        # 旧グリッドは w_pos/w_top3 の上限0.575・w_win 下限0.10 で、
        # 過去に最適だった pos 支配的な解 (例: v7.2 の pos=0.857) を
        # 探索範囲外に締め出していた。
        from scipy.stats import spearmanr
        best_corr = -1.0
        best_weights = np.array([0.30, 0.35, 0.35])

        for w_p in np.arange(0.05, 0.925, 0.025):
            for w_t in np.arange(0.05, 0.925, 0.025):
                w_w = 1.0 - w_p - w_t
                if w_w < 0.05 or w_w > 0.90:
                    continue
                scores = pos_norm * w_p + top3_norm * w_t + win_norm * w_w
                corr, _ = spearmanr(y_position_valid, -scores)
                if corr > best_corr:
                    best_corr = corr
                    best_weights = np.array([w_p, w_t, w_w])

        self._bma_weights = best_weights
        print(f"  BMA重み: pos={best_weights[0]:.3f}, top3={best_weights[1]:.3f}, win={best_weights[2]:.3f}")
        print(f"  最良Spearman: {best_corr:.4f}")

    # ================================================================
    # LambdaRank: 順位最適化
    # ================================================================
    def _train_lambdarank(self, X, y_position, splits, df):
        """LightGBM LambdaRank for ranking optimization.

        全foldで学習してOOFスコアを蓄積する (2026-07-02 変更)。
        旧実装は最終foldのみ学習し、L4スタッキングへは学習データ自身への
        in-sample予測を渡していたため、L4がLambdaRankを過大評価する
        メタ過学習のリスクがあった。OOF化により L4 の入力が正直になる。
        サービング用には最終fold (最も多くのデータで学習) のモデルを保持する。
        """
        self.lambdarank_model = None
        self._lambdarank_oof = None
        if not HAS_LGBM:
            return
        if "race_id" not in df.columns:
            return
        if not splits:
            return

        print("\n📊 6/7: LambdaRank (順位最適化, 全fold OOF)")
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5],
            "num_leaves": 95,
            "learning_rate": 0.03,
            "feature_fraction": 0.65,
            "bagging_fraction": 0.65,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "max_depth": 7,
            "reg_alpha": 0.2,
            "reg_lambda": 0.2,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }

        # relevance = レース内逆着順 (1着=max, 最下位=0) ※レースごとに正規化
        relevance = np.zeros_like(y_position, dtype=int)
        for rid in df["race_id"].unique():
            mask = df["race_id"].values == rid
            race_pos = y_position[mask]
            race_max = int(race_pos.max())
            relevance[mask] = np.clip(race_max - race_pos, 0, race_max).astype(int)

        race_ids = df["race_id"].values
        full_counts = pd.Series(race_ids).value_counts()

        def complete_race_subset(idx_arr):
            """purging で一部の馬が欠落したレースを除いた index を返す"""
            rids = race_ids[idx_arr]
            part_counts = pd.Series(rids).value_counts()
            complete = set(r for r in part_counts.index if part_counts[r] == full_counts.get(r, 0))
            return idx_arr[np.isin(rids, list(complete))]

        # 連続ブロック方式でグループサイズ計算
        def contiguous_groups(rids):
            groups = []
            prev = None
            for r in rids:
                if r != prev:
                    groups.append(0)
                    prev = r
                groups[-1] += 1
            return np.array(groups)

        oof = np.full(len(X), np.nan)
        last_model = None
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            tr = complete_race_subset(np.asarray(train_idx))
            vl = complete_race_subset(np.asarray(val_idx))
            if len(tr) == 0 or len(vl) == 0:
                continue

            w_train = self._sample_weights[tr] if self._sample_weights is not None else None
            train_data = lgb.Dataset(
                X.iloc[tr], label=relevance[tr],
                group=contiguous_groups(race_ids[tr]), weight=w_train
            )
            val_data = lgb.Dataset(
                X.iloc[vl], label=relevance[vl],
                group=contiguous_groups(race_ids[vl]), reference=train_data
            )
            try:
                m = lgb.train(
                    params, train_data,
                    num_boost_round=500,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0),
                               _lr_decay_callback(init_lr=0.03, min_lr=0.005, decay_rate=0.995)],
                )
            except Exception as e:
                print(f"  ⚠️ LambdaRank fold {fold_idx+1} スキップ: {e}")
                continue
            oof[vl] = m.predict(X.iloc[vl])
            last_model = m

        if last_model is not None:
            self.lambdarank_model = last_model
            self._lambdarank_oof = oof
            n_oof = int(np.isfinite(oof).sum())
            print(f"  LambdaRank: {len(splits)}fold 学習完了, OOFカバレッジ {n_oof:,}/{len(X):,} 行")
        else:
            print("  ⚠️ LambdaRank: 有効なfoldなし、スキップ")

    # ================================================================
    # Adversarial Validation
    # ================================================================
    # Adversarial Validation で除外するカレンダー直結特徴量
    # (時系列分割では自明に train/test を分離できてしまい、実質的な
    #  特徴量ドリフトの検出を妨げるため)
    ADVERSARIAL_EXCLUDE_COLS = {
        "year", "month", "day_of_week", "is_weekend",
        "month_sin", "month_cos", "dow_sin", "dow_cos",
        "doy_sin", "doy_cos",
    }

    def _run_adversarial_validation(self, X: pd.DataFrame, df: pd.DataFrame):
        """train vs test の分布差を判定するバイナリ分類器。

        時系列順の直近 20% に label=1、残りに label=0 を付け、
        シャッフルした 70/30 分割で「期間を判別できるか」の AUC を測る。
        AUC > 0.65 の場合はデータシフト警告を出す。

        旧実装は学習側=全てlabel 0 / 評価側=全てlabel 1 という単一クラス
        構成になっており、roc_auc_score が常に ValueError → 0.5 を返して
        分布シフトを一切検出できていなかった (v7.2 の adversarial_auc=0.5 は
        この壊れた実装の産物であり「分布差なし」の根拠にならない)。
        """
        try:
            n = len(X)
            if n < 200:
                self._adversarial_auc = None
                return
            split_idx = int(n * 0.8)
            adv_y = np.zeros(n, dtype=int)
            adv_y[split_idx:] = 1  # 直近 20% = test 期間

            # カレンダー直結列は除外して実質的なドリフトを測る
            drop_cols = [c for c in X.columns if c in self.ADVERSARIAL_EXCLUDE_COLS]
            adv_X = X.drop(columns=drop_cols, errors="ignore")

            # シャッフル分割 (両クラスが学習・評価の双方に入るように)
            rng = np.random.default_rng(42)
            perm = rng.permutation(n)
            n_tr = int(n * 0.7)
            tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
            if len(np.unique(adv_y[tr_idx])) < 2 or len(np.unique(adv_y[te_idx])) < 2:
                self._adversarial_auc = None
                return

            X_tr = adv_X.iloc[tr_idx]
            y_tr = adv_y[tr_idx]
            X_te = adv_X.iloc[te_idx]
            y_te = adv_y[te_idx]

            if HAS_LGBM:
                params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "n_jobs": -1,
                    "verbose": -1,
                    "seed": 42,
                }
                dtrain = lgb.Dataset(X_tr, label=y_tr)
                dval = lgb.Dataset(X_te, label=y_te, reference=dtrain)
                adv_model = lgb.train(
                    params, dtrain, num_boost_round=200,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
                )
                adv_pred = adv_model.predict(X_te)
            else:
                adv_clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
                adv_clf.fit(X_tr, y_tr)
                adv_pred = adv_clf.predict_proba(X_te)[:, 1]

            try:
                adv_auc = roc_auc_score(y_te, adv_pred)
            except ValueError:
                adv_auc = 0.5
            if np.isnan(adv_auc):
                adv_auc = 0.5
            msg = f"Adversarial Validation AUC = {adv_auc:.4f}"
            if adv_auc > 0.65:
                logger.warning(
                    "[データシフト警告] %s (>0.65): "
                    "train と test の特徴量分布が大きく異なります。"
                    "時系列リーク・特徴量ドリフトを確認してください。", msg
                )
                print(f"  *** データシフト警告 *** {msg} (閾値 0.65 超)")
            else:
                logger.info(msg)
                print(f"  Adversarial Validation AUC: {adv_auc:.4f} (分布差なし)")

            self._adversarial_auc = adv_auc
        except Exception as e:
            logger.warning("Adversarial Validation スキップ: %s", e)
            self._adversarial_auc = None

    # ================================================================
    # SMOTE オーバーサンプリング
    # ================================================================
    def _apply_smote(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        max_samples: int = 200_000,
        min_minority_samples: int = 50,
    ):
        """Win タスク向け SMOTE オーバーサンプリング。

        - imblearn が無い場合はフォールバック（元データをそのまま返す）
        - サンプル総数が max_samples を超える場合はスキップ（メモリ節約）
        - 少数クラスサンプル数が min_minority_samples 未満の場合はスキップ
        """
        minority_count = int(y.sum())

        if not HAS_IMBLEARN:
            logger.info("imblearn 未インストール。SMOTE をスキップします (pip install imbalanced-learn)")
            print("  SMOTE スキップ (imblearn 未インストール)")
            return X, y

        if len(X) > max_samples:
            logger.info("SMOTE スキップ: サンプル数 %d > 上限 %d", len(X), max_samples)
            print(f"  SMOTE スキップ (サンプル数 {len(X):,} が上限 {max_samples:,} 超)")
            return X, y

        if minority_count < min_minority_samples:
            logger.info("SMOTE スキップ: 少数クラスサンプル数 %d < %d", minority_count, min_minority_samples)
            print(f"  SMOTE スキップ (少数クラス {minority_count} サンプル不足)")
            return X, y

        try:
            k = min(5, minority_count - 1)
            sm = SMOTE(k_neighbors=k, random_state=42)
            X_res, y_res = sm.fit_resample(X, y)
            gain = len(X_res) - len(X)
            print(f"  SMOTE 完了: {len(X):,} -> {len(X_res):,} (+{gain:,} サンプル)")
            logger.info("SMOTE: %d -> %d samples", len(X), len(X_res))
            return pd.DataFrame(X_res, columns=X.columns), y_res
        except Exception as e:
            logger.warning("SMOTE 失敗（元データを維持）: %s", e)
            print(f"  SMOTE 失敗 -> 元データを使用: {e}")
            return X, y

    # ================================================================
    # ROC 曲線から最適閾値を決定
    # ================================================================
    def _find_optimal_threshold(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        splits: list,
    ) -> float:
        """OOF 予測の ROC 曲線から Youden's J 統計量で最適閾値を決定する。"""
        try:
            from sklearn.metrics import roc_curve
            oof = self._oof_preds.get("win")
            if oof is None:
                return 0.5
            valid = ~np.isnan(oof)
            if valid.sum() < 50:
                return 0.5

            fpr, tpr, thresholds = roc_curve(y[valid], oof[valid])
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            optimal_threshold = float(np.clip(thresholds[best_idx], 0.05, 0.95))
            print(f"  Win 最適閾値 (Youden's J): {optimal_threshold:.4f}")
            logger.info("Win optimal threshold: %.4f", optimal_threshold)
            return optimal_threshold
        except Exception as e:
            logger.warning("最適閾値計算スキップ: %s", e)
            return 0.5

    # ================================================================
    # 全データ学習（最終モデル: SMOTE 適用後データ用）
    # ================================================================
    def _train_model_full(self, X: pd.DataFrame, y: np.ndarray, task: str = "binary_win"):
        """CV なしで全データを使って最終モデルを学習する（SMOTE 後データ用）。
        20%をvalidation setに分割してearly_stoppingで過学習防止。
        """
        if HAS_LGBM:
            # 20%をvalidation setに分割（early_stoppingで過学習防止）
            # SMOTE 出力は「元データ→合成サンプル」の順で並ぶため、末尾を
            # 切り出すと検証側が合成サンプルだけになる。シャッフルしてから分割する。
            n = len(X)
            val_size = int(n * 0.2)
            if val_size > 10:
                rng = np.random.default_rng(42)
                perm = rng.permutation(n)
                tr_i, vl_i = perm[val_size:], perm[:val_size]
                y_arr = np.asarray(y)
                X_train_full = X.iloc[tr_i] if hasattr(X, 'iloc') else X[tr_i]
                X_val_full = X.iloc[vl_i] if hasattr(X, 'iloc') else X[vl_i]
                y_train_full = y_arr[tr_i]
                y_val_full = y_arr[vl_i]
                use_early_stop = True
            else:
                X_train_full = X
                y_train_full = y
                use_early_stop = False

            if task == "binary_win":
                params = self.LGBM_PARAMS_WIN.copy()
                focal_obj, focal_eval = _focal_loss_lgbm(gamma=2.0, alpha=0.25)
                params["objective"] = focal_obj
                params.pop("metric", None)
                dtrain = lgb.Dataset(X_train_full, label=y_train_full)
                callbacks = [
                    lgb.log_evaluation(0),
                    _lr_decay_callback(init_lr=params.get("learning_rate", 0.03), min_lr=0.005, decay_rate=0.995),
                ]
                train_kwargs = dict(
                    num_boost_round=800,
                    feval=focal_eval,
                )
                if use_early_stop:
                    dval = lgb.Dataset(X_val_full, label=y_val_full, reference=dtrain)
                    train_kwargs["valid_sets"] = [dval]
                    callbacks.append(lgb.early_stopping(80))
                train_kwargs["callbacks"] = callbacks
                model = lgb.train(params, dtrain, **train_kwargs)
            else:
                params = self.LGBM_PARAMS_BINARY.copy()
                # 連続ラベル (Label Smoothing 済み) は cross_entropy で学習
                uniq = np.unique(np.asarray(y_train_full, dtype=float))
                if not np.all(np.isin(uniq, (0.0, 1.0))):
                    params["objective"] = "cross_entropy"
                    params["metric"] = "cross_entropy"
                    params.pop("is_unbalance", None)
                dtrain = lgb.Dataset(X_train_full, label=y_train_full)
                callbacks = [
                    lgb.log_evaluation(0),
                    _lr_decay_callback(init_lr=params.get("learning_rate", 0.03), min_lr=0.005, decay_rate=0.995),
                ]
                train_kwargs = dict(num_boost_round=800)
                if use_early_stop:
                    dval = lgb.Dataset(X_val_full, label=y_val_full, reference=dtrain)
                    train_kwargs["valid_sets"] = [dval]
                    callbacks.append(lgb.early_stopping(50))
                train_kwargs["callbacks"] = callbacks
                model = lgb.train(params, dtrain, **train_kwargs)
            return model
        else:
            model = GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
            model.fit(X, y)
            return model

    # ================================================================
    # Ordinal Regression モデル
    # ================================================================
    def _train_ordinal_model(
        self,
        X: pd.DataFrame,
        y_position: np.ndarray,
        splits: list,
        df: pd.DataFrame,
    ):
        """着順の順序関係を明示的に学習する Ordinal Regression モデルを学習。

        LightGBM の custom objective として _ordinal_regression_lgbm を使用。
        既存の position モデルと並行して models["ordinal"] に格納する。
        """
        if not HAS_LGBM:
            logger.info("LightGBM 未インストール。Ordinal モデルをスキップ")
            return

        try:
            ordinal_obj, ordinal_eval = _ordinal_regression_lgbm()

            # レース内正規化ターゲット: (着順-1)/(頭数-1) ∈ [0,1]
            # (8頭立てと18頭立てで同じ着順が異なる意味を持つことを反映)
            if "race_id" in df.columns:
                n_runners = df.groupby("race_id")["race_id"].transform("count").values.astype(float)
            else:
                n_runners = np.full(len(X), float(max(int(np.nanmax(y_position)), 2)))
            denom = np.clip(n_runners - 1.0, 1.0, None)
            y_norm_all = (np.asarray(y_position, dtype=float) - 1.0) / denom

            params = self.LGBM_PARAMS_REGRESSION.copy()
            params["objective"] = ordinal_obj
            params.pop("metric", None)

            fold_models = []
            oof_preds = np.full(len(X), np.nan)
            mae_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_val = y_position[val_idx]
                w_tr = self._sample_weights[train_idx] if self._sample_weights is not None else None

                dtrain = lgb.Dataset(X_tr, label=y_norm_all[train_idx], weight=w_tr)
                dval = lgb.Dataset(X_val, label=y_norm_all[val_idx], reference=dtrain)

                m = lgb.train(
                    params, dtrain, num_boost_round=800,
                    valid_sets=[dval],
                    feval=ordinal_eval,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0),
                               _lr_decay_callback(init_lr=params.get("learning_rate", 0.03), min_lr=0.005, decay_rate=0.995)],
                )
                fold_models.append(m)

                # raw スコアを着順スケールに変換 (sigmoid → [0,1] → 実頭数スケール)
                raw = m.predict(X_val)
                p = 1.0 / (1.0 + np.exp(-raw))
                pred_pos = 1.0 + p * denom[val_idx]
                oof_preds[val_idx] = pred_pos

                mae = mean_absolute_error(y_val, pred_pos)
                mae_scores.append(mae)
                print(f"  Ordinal Fold {fold_idx+1}: MAE={mae:.3f}")

            if fold_models:
                self.models["ordinal"] = _BlendedModel(
                    {"lgbm": fold_models}, "regression", list(X.columns)
                )
                avg_mae = float(np.mean(mae_scores))
                self.evaluation_results["ordinal"] = {
                    "mean_mae": avg_mae,
                    "std_mae": float(np.std(mae_scores)),
                }
                self._oof_preds["ordinal"] = oof_preds
                print(f"  Ordinal Regression: 平均MAE={avg_mae:.4f}")
                logger.info("Ordinal Regression MAE: %.4f", avg_mae)
            else:
                logger.warning("Ordinal Regression: 有効な fold がありません")
        except Exception as e:
            logger.warning("Ordinal Regression スキップ: %s", e)
            print(f"  *** Ordinal Regression スキップ: {e}")

    # ================================================================
    # Incremental Learning (継続学習)
    # ================================================================
    def incremental_train(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_col: str = "finish_position",
        num_boost_round: int = 100,
    ) -> Dict[str, dict]:
        """新規データ到着時に init_model で継続学習する。

        Args:
            df: 新規データ（追加分のみ）
            feature_columns: 特徴量カラムリスト（既存モデルと同じ順序）
            target_col: ターゲットカラム名
            num_boost_round: 追加学習ラウンド数

        Returns:
            更新後の評価結果 dict
        """
        if not self.models:
            raise RuntimeError("事前に train() でモデルを学習してください。")

        if not HAS_LGBM:
            logger.warning("incremental_train: LightGBM 未インストールのためスキップ")
            return {}

        print(f"\n[Incremental Learning] 新規データ {len(df):,} 件で継続学習")

        # 特徴量を既存モデルと合わせる
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            logger.warning(
                "incremental_train: %d 特徴量不足 -> 0 補完: %s",
                len(missing), missing[:5],
            )
        X_new = df.reindex(columns=self.feature_columns, fill_value=0).copy()
        X_new = X_new.replace([np.inf, -np.inf], np.nan)
        for col in X_new.columns:
            med = X_new[col].median()
            X_new[col] = X_new[col].fillna(med if not pd.isna(med) else 0)

        y_pos = df[target_col].values
        y_win = (df[target_col] == 1).astype(int).values
        y_top3 = (df[target_col] <= 3).astype(int).values

        updated: Dict[str, dict] = {}

        task_map = {
            "position": (y_pos, self.LGBM_PARAMS_REGRESSION),
            "top3": (y_top3, self.LGBM_PARAMS_BINARY),
            "win": (y_win, self.LGBM_PARAMS_WIN),
        }

        for task_name, (y_task, base_params) in task_map.items():
            model_obj = self.models.get(task_name)
            if model_obj is None:
                continue

            # _BlendedModel から最初の lgbm モデルを取得
            if isinstance(model_obj, _BlendedModel):
                lgbm_list = model_obj.fold_models_by_type.get("lgbm", [])
                init_model = lgbm_list[0] if lgbm_list else None
            elif hasattr(model_obj, "predict"):
                init_model = model_obj
            else:
                init_model = None

            if init_model is None:
                logger.info("incremental_train: %s の init_model が取得できません", task_name)
                continue

            try:
                params = base_params.copy()
                # focal loss を使わず通常 binary で継続学習（安定性優先）
                if task_name == "win":
                    params["objective"] = "binary"
                    params["metric"] = "binary_logloss"

                dtrain = lgb.Dataset(X_new, label=y_task)
                new_model = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    init_model=init_model,
                    callbacks=[lgb.log_evaluation(0)],
                )

                # モデルを更新（_BlendedModel の場合は先頭を置き換え）
                if isinstance(model_obj, _BlendedModel):
                    model_obj.fold_models_by_type["lgbm"][0] = new_model
                else:
                    self.models[task_name] = new_model

                # 評価
                pred = new_model.predict(X_new)
                if task_name == "position":
                    mae = mean_absolute_error(y_task, pred)
                    updated[task_name] = {"incremental_mae": float(mae)}
                    print(f"  [{task_name}] 継続学習完了: MAE={mae:.4f}")
                else:
                    # raw logit かどうかチェックして sigmoid
                    if pred.min() < -0.01 or pred.max() > 1.01:
                        pred_prob = np.clip(1.0 / (1.0 + np.exp(-pred)), 0, 1)
                    else:
                        pred_prob = np.clip(pred, 0, 1)
                    try:
                        auc = roc_auc_score(y_task, pred_prob)
                    except ValueError:
                        auc = 0.5
                    updated[task_name] = {"incremental_auc": float(auc)}
                    print(f"  [{task_name}] 継続学習完了: AUC={auc:.4f}")

                logger.info("incremental_train [%s]: +%d rounds", task_name, num_boost_round)

            except Exception as e:
                logger.warning("incremental_train [%s] 失敗: %s", task_name, e)
                print(f"  *** [{task_name}] 継続学習失敗: {e}")

        self.evaluation_results.update(updated)
        print("[Incremental Learning] 完了")
        return updated

    # ================================================================
    # Platt Scaling キャリブレーション
    # ================================================================
    def _train_calibrators(self, splits):
        """OOF予測でPlatt Scalingキャリブレーターを学習

        修正: 全OOF予測でfit→同じOOFに適用するとデータリークになるため、
        最後のfoldのvalidation setのみをcalibration setとして使用する。
        """
        self._calibrators = {}

        if not splits:
            return

        # 最後のfoldのvalidation indexのみをキャリブレーション用に使用
        _, last_val_idx = splits[-1]

        for key in ["top3", "win"]:
            oof = self._oof_preds.get(key)
            y_true = self._y_targets.get(key)
            if oof is None or y_true is None:
                continue
            # 最後のfoldのval_idxのみで学習（リーク防止）
            cal_pred = oof[last_val_idx]
            cal_true = y_true[last_val_idx]
            valid = ~np.isnan(cal_pred)
            if valid.sum() < 50:
                continue
            try:
                scaler = PlattScaler()
                scaler.fit(cal_true[valid], cal_pred[valid])
                self._calibrators[key] = scaler
                # キャリブレーション前後の比較
                calibrated = scaler.transform(cal_pred[valid])
                print(f"  Platt {key}: before=[{cal_pred[valid].min():.3f},{cal_pred[valid].max():.3f}] → after=[{calibrated.min():.3f},{calibrated.max():.3f}]")
            except Exception as e:
                print(f"  ⚠️ Platt {key} スキップ: {e}")

    # ================================================================
    # NDCG / Spearman τ 評価
    # ================================================================
    def _evaluate_ranking(self, X, y_position, splits, df):
        """ランキング精度の直接的な評価"""
        if "race_id" not in df.columns:
            return

        print("\n📊 7/7: ランキング精度評価")
        ndcg_scores = []
        spearman_scores = []

        # OOF予測を使用（最終モデルで評価するとデータリークになる）
        oof_pos = self._oof_preds.get("position")
        if oof_pos is None:
            return

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            y_val = y_position[val_idx]
            race_ids_val = df.iloc[val_idx]["race_id"].values
            pred_pos = oof_pos[val_idx]
            if np.any(np.isnan(pred_pos)):
                continue

            # レースごとにNDCG/Spearman計算
            for rid in np.unique(race_ids_val):
                mask = race_ids_val == rid
                if mask.sum() < 2:
                    continue
                true_pos = y_val[mask]
                pred = pred_pos[mask]

                # NDCG: relevance = max_pos - pos (高い=良い)
                max_p = true_pos.max()
                true_rel = np.clip(max_p - true_pos, 0, max_p)
                pred_rel = np.clip(max_p - pred, 0, max_p)
                try:
                    ndcg = ndcg_score([true_rel], [pred_rel])
                    ndcg_scores.append(ndcg)
                except Exception:
                    pass

                # Spearman相関
                try:
                    corr, _ = spearmanr(true_pos, pred)
                    if not np.isnan(corr):
                        spearman_scores.append(corr)
                except Exception:
                    pass

        if ndcg_scores:
            avg_ndcg = np.mean(ndcg_scores)
            avg_spearman = np.mean(spearman_scores) if spearman_scores else 0
            self.evaluation_results["ranking"] = {
                "mean_ndcg": avg_ndcg,
                "mean_spearman": avg_spearman,
            }
            print(f"  NDCG@全頭: {avg_ndcg:.4f}")
            print(f"  Spearman τ: {avg_spearman:.4f}")

    # ================================================================
    # SHAP 特徴量重要度レポート + 自動削除 (改善5)
    # ================================================================
    def _shap_feature_report(self, X):
        """SHAP値ベースの特徴量重要度（情報表示 + 底辺20%自動削除）"""
        try:
            import shap
        except ImportError:
            return  # SHAP未インストール時はスキップ

        if "position" not in self.models:
            return

        model = self.models["position"]
        # サンプル数制限（SHAP計算は重い）
        sample_n = min(2000, len(X))
        X_sample = X.sample(n=sample_n, random_state=42)

        try:
            if isinstance(model, _BlendedModel):
                # BlendedModelの場合、最初のLGBMモデルで計算
                lgbm_models = model.fold_models_by_type.get("lgbm", [])
                if not lgbm_models:
                    return
                explainer = shap.TreeExplainer(lgbm_models[0])
            elif hasattr(model, "predict"):
                explainer = shap.TreeExplainer(model)
            else:
                return

            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # regression

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_importance = pd.DataFrame({
                "feature": self.feature_columns,
                "shap_importance": mean_abs_shap,
            }).sort_values("shap_importance", ascending=False)

            # 低重要度の特徴量を表示
            low_features = shap_importance[shap_importance["shap_importance"] < 1e-4]
            if len(low_features) > 0:
                print(f"\n  SHAP低重要度特徴量 ({len(low_features)}個): 次回の学習で除外検討")
                for _, row in low_features.head(10).iterrows():
                    print(f"    {row['feature']:35s}: {row['shap_importance']:.6f}")

            self._shap_importance = shap_importance

            # 改善5: 底辺20%を自動削除
            self._apply_shap_feature_pruning(shap_importance)

        except Exception as e:
            print(f"  SHAP計算スキップ: {e}")

    def _apply_shap_feature_pruning(self, shap_importance: pd.DataFrame):
        """SHAP重要度底辺35%の特徴量を削除し、削除リストを保存する。

        predict() 時に不要特徴量が自動的にスキップされる。
        最低40特徴量は残す安全弁を設ける。
        """
        try:
            n_total = len(shap_importance)
            n_remove = max(0, int(n_total * 0.35))
            # 安全弁: 残り特徴量が40未満にならないようにする
            n_remove = min(n_remove, max(0, n_total - 40))

            if n_remove == 0:
                self._shap_pruned_features = []
                return

            # 重要度昇順で底辺 n_remove 個を取得
            pruned = shap_importance.nsmallest(n_remove, "shap_importance")["feature"].tolist()
            self._shap_pruned_features = pruned

            # 注意: feature_columns は変更しない（既学習モデルとの整合性維持）
            # pruned_features は次回学習時の参考情報として保存のみ

            print(f"\n  SHAP特徴量選択: {len(pruned)}個を低重要度として記録（次回学習で除外候補）")
            print(f"  削除例: {pruned[:5]}")

            # 削除リストをJSONで保存（デバッグ・再現性用）
            try:
                pruned_path = self.model_dir / "shap_pruned_features.json"
                with open(pruned_path, "w", encoding="utf-8") as fp:
                    json.dump({"pruned": pruned, "remaining": len(self.feature_columns)}, fp, ensure_ascii=False, indent=2)
            except Exception:
                pass

        except Exception as e:
            print(f"  SHAP特徴量選択スキップ: {e}")
            self._shap_pruned_features = []

    # ================================================================
    # 改善3: Conformal Prediction (calibration set で補正係数を計算)
    # ================================================================
    def _refine_positions_for_rows(self, row_idx, X):
        """OOF予測を predict() と同じ L2→L4 経路で最終スケールに精緻化する。

        Conformal 較正専用ヘルパー。predict() 側の L2/L4 適用ロジック
        (メタ入力 = [pos, top3, win](キャリブレ済みOOF) + 上位SHAP特徴 + LambdaRank)
        をミラーする。predict() を変更したらここも追従させること。
        """
        oof_pos = self._oof_preds.get("position")
        oof_top3 = self._oof_preds.get("top3")
        oof_win = self._oof_preds.get("win")
        if oof_pos is None or oof_top3 is None or oof_win is None:
            return None

        pos = np.asarray(oof_pos[row_idx], dtype=float).copy()
        top3 = np.asarray(oof_top3[row_idx], dtype=float)
        win = np.asarray(oof_win[row_idx], dtype=float)

        l2 = getattr(self, "_l2_position", None)
        if l2 is not None:
            pos = l2.predict(np.column_stack([pos, top3, win]))

        l4 = getattr(self, "_l4_position", None)
        if l4 is not None:
            l4_in = np.column_stack([pos, top3, win])
            feats = [f for f in getattr(self, "_l4_top_feats", []) if f in X.columns]
            if feats:
                extra = np.nan_to_num(X[feats].values[row_idx], nan=0.0)
                l4_in = np.hstack([l4_in, extra])
            if getattr(self, "_l4_has_lambdarank", False):
                lr_model = getattr(self, "lambdarank_model", None)
                lr_oof = getattr(self, "_lambdarank_oof", None)
                if lr_oof is not None:
                    lr = np.asarray(lr_oof, dtype=float)[row_idx].copy()
                    nan_mask = ~np.isfinite(lr)
                    if nan_mask.any() and lr_model is not None:
                        lr[nan_mask] = lr_model.predict(X.iloc[row_idx][nan_mask])
                elif lr_model is not None:
                    lr = lr_model.predict(X.iloc[row_idx])
                else:
                    return None  # 学習時と同じ入力構造を再現できない
                l4_in = np.column_stack([l4_in, lr])
            pos = l4.predict(l4_in)
        return pos

    def _train_conformal_prediction(self, X, y_position, splits):
        """Conformal Prediction の補正係数 (nonconformity scores) を計算する。

        最後の fold の validation set を calibration set として利用し、
        「predict() が実際に返す最終スケール (L2/L4適用後)」の残差で
        非適合スコアを計算する (2026-07-02 変更。旧実装は L1 ブレンド直後の
        OOF 残差を使っており、適用先の L4 出力とスケール不一致だった)。
        注: L4 は全OOF行で学習しているため cal set 残差はやや楽観的になる。
        """
        self._conformal_scores = None
        self._conformal_q95 = None

        if not splits:
            return

        try:
            _, val_idx = splits[-1]
            oof_pos = self._oof_preds.get("position")
            if oof_pos is None:
                return

            cal_pred = np.asarray(oof_pos[val_idx], dtype=float).copy()
            cal_true = y_position[val_idx]
            valid = ~np.isnan(cal_pred)

            # L2/L4 の最終スケールへ精緻化 (入力OOFが欠ける行は除外)
            oof_top3 = self._oof_preds.get("top3")
            oof_win = self._oof_preds.get("win")
            if oof_top3 is not None and oof_win is not None:
                valid = (valid
                         & ~np.isnan(np.asarray(oof_top3[val_idx], dtype=float))
                         & ~np.isnan(np.asarray(oof_win[val_idx], dtype=float)))
                try:
                    refined = self._refine_positions_for_rows(val_idx, X)
                    if refined is not None:
                        cal_pred = np.asarray(refined, dtype=float)
                except Exception as e:
                    logger.warning("Conformal: L2/L4精緻化失敗、L1 OOFで較正: %s", e)

            valid = valid & np.isfinite(cal_pred)
            if valid.sum() < 20:
                return

            # 非適合スコア = |真の着順 - 予測着順|
            nonconformity = np.abs(cal_true[valid] - cal_pred[valid])
            self._conformal_scores = np.sort(nonconformity)
            # 95% quantile (alpha=0.05)
            q_idx = int(np.ceil(0.95 * (len(self._conformal_scores) + 1))) - 1
            q_idx = min(q_idx, len(self._conformal_scores) - 1)
            self._conformal_q95 = float(self._conformal_scores[q_idx])
            print(f"\n  Conformal Prediction: calibration サンプル={valid.sum()}, "
                  f"Q95={self._conformal_q95:.3f} (95%信頼区間幅 +-{self._conformal_q95:.3f}着順)")
        except Exception as e:
            print(f"  Conformal Prediction スキップ: {e}")

    # ================================================================
    # 改善4: 4層スタッキング (L4 Ridge)
    # ================================================================
    def _train_l4_stacking(self, X, y_position):
        """L3出力 + 上位SHAP特徴量(最大10個) -> L4 (Ridge) 最終Meta-Learner。

        _train_l2_stacking() の後に呼ばれることを前提とする。
        L4入力 = [l3_pos_pred, l3_top3_pred, l3_win_pred, 上位SHAP特徴量]。
        過学習防止のため Ridge(alpha=10) を使用。
        """
        self._l4_position = None
        self._l4_top_feats = []

        try:
            oof_pos = self._oof_preds.get("position")
            oof_top3 = self._oof_preds.get("top3")
            oof_win = self._oof_preds.get("win")
            if oof_pos is None or oof_top3 is None or oof_win is None:
                return

            valid_mask = ~(np.isnan(oof_pos) | np.isnan(oof_top3) | np.isnan(oof_win))
            if valid_mask.sum() < 100:
                print("  L4スタッキング: OOFデータ不足、スキップ")
                return

            meta_X = np.column_stack([oof_pos[valid_mask], oof_top3[valid_mask], oof_win[valid_mask]])

            # L3の出力を取得
            l3_preds = []
            if getattr(self, "_l2_position", None) is not None:
                l3_preds.append(self._l2_position.predict(meta_X))
            if getattr(self, "_l2_top3", None) is not None:
                l3_preds.append(self._l2_top3.predict_proba(meta_X)[:, 1])
            if getattr(self, "_l2_win", None) is not None:
                l3_preds.append(self._l2_win.predict_proba(meta_X)[:, 1])

            if not l3_preds:
                return

            l3_matrix = np.column_stack(l3_preds)

            # 上位SHAP特徴量（最大10個）をL4の入力に加える
            l4_extra = l3_matrix
            avail_feats = []
            shap_imp = getattr(self, "_shap_importance", None)
            if shap_imp is not None and hasattr(X, "iloc"):
                top_feats = shap_imp.head(10)["feature"].tolist()
                avail_feats = [f for f in top_feats if f in X.columns]
                if avail_feats:
                    extra = X[avail_feats].values[valid_mask]
                    extra = np.nan_to_num(extra, nan=0.0)
                    l4_extra = np.hstack([l4_extra, extra])

            # LambdaRankスコアをL4入力に追加
            # OOFスコアを優先 (in-sample予測だとL4がLambdaRankを過大評価する)。
            # OOFが無い行のみサービング用モデルの予測でフォールバック。
            self._l4_has_lambdarank = False
            if hasattr(self, 'lambdarank_model') and self.lambdarank_model is not None:
                try:
                    lr_oof = getattr(self, "_lambdarank_oof", None)
                    if lr_oof is not None:
                        lr_pred = np.asarray(lr_oof, dtype=float).copy()
                        nan_mask = ~np.isfinite(lr_pred)
                        if nan_mask.any():
                            lr_pred[nan_mask] = self.lambdarank_model.predict(
                                X[self.feature_columns].iloc[nan_mask]
                            )
                    else:
                        lr_pred = self.lambdarank_model.predict(X[self.feature_columns])
                    l4_extra = np.column_stack([l4_extra, lr_pred[valid_mask]])
                    self._l4_has_lambdarank = True
                except Exception:
                    pass  # LambdaRank予測失敗時はスキップ

            l4_X = l4_extra

            # L4: Ridge回帰（過学習防止）
            print("\n  L4スタッキング: Ridge meta-learner (L1->L2->L3->L4)")
            self._l4_position = Ridge(alpha=10.0)
            self._l4_position.fit(l4_X, y_position[valid_mask])
            l4_pred = self._l4_position.predict(l4_X)
            l4_mae = mean_absolute_error(y_position[valid_mask], l4_pred)
            print(f"    L4 Position MAE: {l4_mae:.4f}")
            self._l4_top_feats = avail_feats

        except Exception as e:
            print(f"  L4スタッキング スキップ: {e}")

    def predict(self, df: pd.DataFrame, odds_weight: float = None) -> pd.DataFrame:
        """予測実行: 各モデルのスコアを統合

        Args:
            odds_weight: オッズ情報のブレンド比率 (0.0=不使用, 0.1=10%参考, 最大0.3)
                         Noneの場合はモデル保存時の設定値を使用

        新規出力カラム (改善2,3):
            ev_mean, ev_std, ev_p25, ev_p75 : EV分布（モンテカルロサンプリング）
            pred_rank_lower95, pred_rank_upper95 : 予測ランクの95%信頼区間（Conformal）
        """
        if not self.models:
            raise RuntimeError("モデルが未学習です。先にtrain()を実行してください。")

        if df.empty:
            return pd.DataFrame(columns=["race_id", "horse_number", "predicted_rank", "composite_score"])

        if odds_weight is None:
            odds_weight = getattr(self, "odds_weight", 0.0)
        odds_weight = max(0.0, min(odds_weight, 0.3))  # 0〜0.3にクリップ

        # 全特徴量を使用（SHAP pruned_features は次回学習の参考情報のみ）
        active_features = self.feature_columns

        # 必要カラムの存在チェック - 不足分は0埋め（拡張特徴量が未計算のケース対応）
        missing_cols = [c for c in active_features if c not in df.columns]
        if missing_cols:
            import logging
            logging.getLogger(__name__).warning(
                f"特徴量 {len(missing_cols)}個が不足 -> 0埋めで補完: {missing_cols[:5]}..."
            )

        # コピーして特徴量マトリクスを構築（入力dfを変更しない）
        available = [c for c in active_features if c in df.columns]
        X = df[available].copy()
        for c in missing_cols:
            X[c] = 0
        X = X[active_features]  # カラム順を揃える
        # Inf → NaN → 補完
        #  - 新モデル (feature_medians 保存済み): 学習時と同じ中央値で補完
        #    (旧来のレース内中央値補完はサーブ時だけ分布が変わる train/serve 不整合)
        #  - 旧モデル: 従来のレース内中央値補完を維持 (挙動保存)
        X = X.replace([np.inf, -np.inf], np.nan)
        train_medians = getattr(self, "_feature_medians", None)
        _legacy_impute = not bool(train_medians)
        if not _legacy_impute and X.isna().any().any():
            X = X.fillna(value={c: train_medians.get(c, 0.0) for c in X.columns})
        try:
            if _legacy_impute and "race_id" in df.columns and X.isna().any().any():
                # カラム数が多い場合はグローバル中央値にフォールバック（パフォーマンス対策）
                if len(X.columns) <= 500:
                    for race_id in df["race_id"].unique():
                        mask = df["race_id"] == race_id
                        for col in X.columns:
                            col_mask = mask & X[col].isna()
                            if col_mask.sum() > 0:
                                median_val = X.loc[mask & ~X[col].isna(), col].median()
                                X.loc[col_mask, col] = median_val if not pd.isna(median_val) else 0
                else:
                    # グローバル中央値フォールバック
                    for col in X.columns:
                        median_val = X[col].median()
                        X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0)
            elif _legacy_impute:
                for col in X.columns:
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0)
        except Exception:
            # フォールバック: グローバル中央値 → 0
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

        # キャリブレーション前の生スコアを保持（旧モデルのL2/L4用）
        raw_prob_top3 = results["prob_top3"].values.copy() if "prob_top3" in results.columns else None
        raw_prob_win = results["prob_win"].values.copy() if "prob_win" in results.columns else None

        # Isotonic/Plattキャリブレーション適用 → prob_top3/prob_win を「実確率」に変換
        calibrators = getattr(self, "_calibrators", {})
        if "top3" in calibrators and "prob_top3" in results.columns:
            try:
                results["prob_top3"] = calibrators["top3"].transform(results["prob_top3"].values)
            except Exception:
                pass
        if "win" in calibrators and "prob_win" in results.columns:
            try:
                results["prob_win"] = calibrators["win"].transform(results["prob_win"].values)
            except Exception:
                pass

        # L2/L4 に渡す確率スケールの選択:
        #  - 新モデル (_l2_inputs_calibrated=True): train() がキャリブレ済みOOFで
        #    L2/L4 を学習しているため、キャリブレ済み確率を渡す (train/serve 整合)
        #  - 旧モデル (フラグなし): 学習時のコード状態が不明なため従来挙動 (生スコア) を維持
        _use_cal_meta = bool(getattr(self, "_l2_inputs_calibrated", False))
        if _use_cal_meta:
            meta_prob_top3 = results["prob_top3"].values if "prob_top3" in results.columns else raw_prob_top3
            meta_prob_win = results["prob_win"].values if "prob_win" in results.columns else raw_prob_win
        else:
            meta_prob_top3 = raw_prob_top3
            meta_prob_win = raw_prob_win

        # L2スタッキング: メタ入力で pred_position のみ更新。
        # prob_top3/prob_win はキャリブレ済み値を保持 (EV計算に使うため)。
        l2_pos = getattr(self, "_l2_position", None)
        if (l2_pos is not None and
            meta_prob_top3 is not None and meta_prob_win is not None and
            "pred_position" in results.columns):
            try:
                meta_X = np.column_stack([
                    results["pred_position"].values,
                    meta_prob_top3,
                    meta_prob_win,
                ])
                results["pred_position"] = l2_pos.predict(meta_X)
                # prob_top3/prob_winはキャリブレーション済みの実確率を維持
                # → L2で上書きしない（EV計算に使うため正確な確率が必要）
            except Exception:
                pass

        # L4スタッキング: pred_positionのみ精緻化（確率は触らない）
        l4_pos = getattr(self, "_l4_position", None)
        l4_top_feats = getattr(self, "_l4_top_feats", [])
        if (l4_pos is not None and
            meta_prob_top3 is not None and meta_prob_win is not None and
            "pred_position" in results.columns):
            try:
                l3_matrix = np.column_stack([
                    results["pred_position"].values,
                    meta_prob_top3,
                    meta_prob_win,
                ])
                l4_X = l3_matrix
                if l4_top_feats:
                    avail_l4 = [f for f in l4_top_feats if f in X.columns]
                    if avail_l4:
                        extra = np.nan_to_num(X[avail_l4].values, nan=0.0)
                        l4_X = np.hstack([l4_X, extra])
                # LambdaRankスコアをL4入力に追加（学習時と同じ構造にする）
                if getattr(self, "_l4_has_lambdarank", False):
                    lambdarank_model = getattr(self, "lambdarank_model", None)
                    if lambdarank_model is not None:
                        try:
                            lr_pred = lambdarank_model.predict(X)
                            l4_X = np.column_stack([l4_X, lr_pred])
                        except Exception:
                            pass
                results["pred_position"] = l4_pos.predict(l4_X)
            except Exception:
                pass  # L4失敗時はL3出力をそのまま使用

        # LambdaRank score (if available)
        lambdarank_model = getattr(self, "lambdarank_model", None)
        if lambdarank_model is not None:
            try:
                results["lambdarank_score"] = lambdarank_model.predict(X)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"LambdaRank prediction failed: {e}")

        # 統合スコア — 全コンポーネントをレース内で正規化してから合成
        if "pred_position" in results.columns and "prob_top3" in results.columns and "prob_win" in results.columns:
            # 各スコアをレース単位で0-1正規化（全て「高い=良い」に統一）
            def _race_normalize(series, ascending=True):
                """レース内で0-1正規化。ascending=True: 元の値が大きい→スコア高い"""
                return results.groupby("race_id")[series.name].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8) if ascending
                    else 1 - (x - x.min()) / (x.max() - x.min() + 1e-8)
                )

            # 着順: 低い=良い → 反転正規化
            pos_score = _race_normalize(results["pred_position"], ascending=False)
            # prob_top3, prob_win: 高い=良い → そのまま正規化
            top3_score = _race_normalize(results["prob_top3"], ascending=True)
            win_score = _race_normalize(results["prob_win"], ascending=True)

            # BMA重み取得
            bma_weights = getattr(self, "_bma_weights", None)
            if bma_weights is not None and len(bma_weights) == 3:
                w_pos, w_top3, w_win = float(bma_weights[0]), float(bma_weights[1]), float(bma_weights[2])
            else:
                w_pos, w_top3, w_win = self._compute_ensemble_weights()

            if "lambdarank_score" in results.columns:
                lr_score = _race_normalize(results["lambdarank_score"], ascending=True)
                # LambdaRankに15%割当、残りをBMA比率配分
                lr_share = 0.15
                remaining = 1.0 - lr_share
                results["composite_score"] = (
                    pos_score * w_pos * remaining
                    + top3_score * w_top3 * remaining
                    + win_score * w_win * remaining
                    + lr_score * lr_share
                )
            else:
                results["composite_score"] = (
                    pos_score * w_pos
                    + top3_score * w_top3
                    + win_score * w_win
                )
            # NaN/Inf を 0 に置換
            results["composite_score"] = results["composite_score"].replace([np.inf, -np.inf], 0).fillna(0)

        # 改善2: EV分布計算（モンテカルロサンプリング）
        results = self._compute_ev_distribution(results, X)

        # 改善3: Conformal Prediction による予測ランクの95%信頼区間
        results = self._apply_conformal_prediction(results)

        # オッズブレンド: 実力スコアにオッズ情報を軽く加味
        if odds_weight > 0 and "odds" in df.columns:
            # オッズを「人気度スコア」に変換 (オッズ低い=人気=高スコア)
            # レース内で正規化して 0〜1 にする
            odds_score = pd.Series(index=df.index, dtype=float)
            for rid, grp in df.groupby("race_id"):
                mask = grp.index
                o = grp["odds"].fillna(grp["odds"].median())
                o = o.replace([np.inf, -np.inf], o.median())
                if o.max() > o.min():
                    # 低オッズ(人気)→高スコア: 1 - 正規化
                    normalized = 1 - (o - o.min()) / (o.max() - o.min())
                else:
                    normalized = pd.Series(0.5, index=mask)
                odds_score.loc[mask] = normalized.values

            # 実力スコア × (1 - odds_weight) + オッズスコア × odds_weight
            ability_score = results["composite_score"].values
            odds_vals = odds_score.reindex(df.index).fillna(0.5).values
            results["composite_score"] = ability_score * (1 - odds_weight) + odds_vals * odds_weight
            results["odds_score"] = odds_vals

        # タイブレーカー: composite_scoreが同値の場合、prob_winで微小差をつける
        if "prob_win" in results.columns:
            tiebreak = results["prob_win"] * 1e-6
            results["composite_score"] = results["composite_score"] + tiebreak

        # レース内ランキング（method="first"で同値でも一意のランクを保証）
        results["predicted_rank"] = results.groupby("race_id")["composite_score"].rank(
            ascending=False, method="first"
        ).astype(int)

        return results.sort_values(["race_id", "predicted_rank"])

    # ================================================================
    # 改善1: 動的アンサンブル重み
    # ================================================================
    def _compute_dynamic_weights(self, race_row) -> tuple:
        """レースタイプ（距離/馬場/クラス）に応じた動的重み調整。

        race_row: DataFrameの1行（pd.Series）またはNone。
        Returns: (w_pos, w_top3, w_win) の正規化済みタプル。

        距離別ルール:
          長距離(>=2400m): 着順回帰モデル重視 (pos重み+25%)
          短距離(<=1400m): 勝利モデル重視     (win重み+25%)
          中距離:          複勝モデル重視      (変更なし)
        馬場別ルール:
          ダート(surface==2): 着順安定性高い -> pos +10%
        """
        # ベース重み（BMA or バリデーションスコアから）
        bma = getattr(self, "_bma_weights", None)
        if bma is not None and len(bma) == 3:
            base_pos, base_top3, base_win = float(bma[0]), float(bma[1]), float(bma[2])
        else:
            base_pos, base_top3, base_win = self._compute_ensemble_weights()

        if race_row is None:
            return base_pos, base_top3, base_win

        try:
            distance = float(race_row.get("distance", 0) or 0)
            surface = int(race_row.get("surface", 0) or 0)

            # 距離による補正係数
            if distance >= 2400:
                adj_pos, adj_top3, adj_win = 1.25, 1.0, 0.75
            elif 0 < distance <= 1400:
                adj_pos, adj_top3, adj_win = 0.75, 1.0, 1.25
            else:
                adj_pos, adj_top3, adj_win = 1.0, 1.0, 1.0

            # ダート補正
            if surface == 2:
                adj_pos *= 1.1
                adj_win *= 0.9

            w_pos = base_pos * adj_pos
            w_top3 = base_top3 * adj_top3
            w_win = base_win * adj_win

            total = w_pos + w_top3 + w_win
            if total <= 0:
                return base_pos, base_top3, base_win
            return w_pos / total, w_top3 / total, w_win / total

        except Exception:
            return base_pos, base_top3, base_win

    # ================================================================
    # 改善2: EV分布計算（モンテカルロサンプリング）
    # ================================================================
    def _compute_ev_distribution(self, results: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """予測確率の不確実性をモンテカルロサンプリングで推定する。

        手法1: BlendedModelの各foldモデルから予測を個別収集して分布を計算。
        手法2: foldモデルが取得できない場合はBernoulliノイズで近似。
        出力カラム: ev_mean, ev_std, ev_p25, ev_p75
        """
        try:
            n_mc_samples = 100  # フォールバック用モンテカルロサンプル数

            fold_preds_list = []
            win_model = self.models.get("win")
            if win_model is not None and isinstance(win_model, _BlendedModel):
                for mtype, models in win_model.fold_models_by_type.items():
                    for m in models:
                        try:
                            p = win_model._predict_single(m, mtype, X)
                            if p is not None:
                                fold_preds_list.append(np.asarray(p, dtype=float))
                        except Exception:
                            continue

            if len(fold_preds_list) >= 2:
                # foldの実予測分散を使用（最も信頼性が高い）
                fold_arr = np.array(fold_preds_list)  # (n_folds, n_horses)
                ev_mean = fold_arr.mean(axis=0)
                ev_std = fold_arr.std(axis=0)
                ev_p25 = np.percentile(fold_arr, 25, axis=0)
                ev_p75 = np.percentile(fold_arr, 75, axis=0)
            else:
                # フォールバック: Bernoulliノイズで近似
                if "prob_win" in results.columns:
                    base_pred = results["prob_win"].values.astype(float)
                else:
                    base_pred = np.full(len(results), 0.1)

                sigma_est = np.sqrt(np.maximum(base_pred * (1 - base_pred), 1e-6)) * 0.3
                rng = np.random.default_rng(seed=42)
                mc_samples = rng.normal(
                    loc=base_pred[np.newaxis, :],
                    scale=sigma_est[np.newaxis, :],
                    size=(n_mc_samples, len(base_pred))
                )
                mc_samples = np.clip(mc_samples, 0.0, 1.0)
                ev_mean = mc_samples.mean(axis=0)
                ev_std = mc_samples.std(axis=0)
                ev_p25 = np.percentile(mc_samples, 25, axis=0)
                ev_p75 = np.percentile(mc_samples, 75, axis=0)

            results["ev_mean"] = np.round(ev_mean, 6)
            results["ev_std"] = np.round(ev_std, 6)
            results["ev_p25"] = np.round(ev_p25, 6)
            results["ev_p75"] = np.round(ev_p75, 6)

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"EV分布計算スキップ: {e}")
            for col in ("ev_mean", "ev_std", "ev_p25", "ev_p75"):
                if col not in results.columns:
                    results[col] = np.nan

        return results

    # ================================================================
    # 改善3: Conformal Prediction の適用
    # ================================================================
    def _apply_conformal_prediction(self, results: pd.DataFrame) -> pd.DataFrame:
        """Conformal Prediction で予測ランクの80%/95%信頼区間を付与する。

        Outputs:
            pred_rank_lower80, pred_rank_upper80: tighter band (default for UI)
            pred_rank_lower95, pred_rank_upper95: conservative band
        Both are clipped to [1, n_horses_in_race] so the bounds never exceed
        the number of runners — essential for small fields.
        """
        try:
            if "pred_position" not in results.columns:
                return results

            scores = getattr(self, "_conformal_scores", None)
            q95 = getattr(self, "_conformal_q95", None)

            # Derive Q80 from raw scores if available; else fall back to 0.7*Q95
            if scores is not None and len(scores) > 0:
                q80 = float(np.percentile(scores, 80))
                if q95 is None:
                    q95 = float(np.percentile(scores, 95))
            else:
                if q95 is None:
                    return results
                q80 = 0.7 * q95  # heuristic fallback

            pred_pos = results["pred_position"].values

            # Per-race horse count for clipping
            if "race_id" in results.columns:
                n_by_race = results.groupby("race_id")["pred_position"].transform("count")
                n_arr = n_by_race.values.astype(float)
            else:
                n_arr = np.full(len(pred_pos), len(pred_pos), dtype=float)

            for q, tag in [(q80, "80"), (q95, "95")]:
                lo = np.maximum(1, np.floor(pred_pos - q)).astype(int)
                hi = np.minimum(n_arr, np.ceil(pred_pos + q)).astype(int)
                results[f"pred_rank_lower{tag}"] = lo
                results[f"pred_rank_upper{tag}"] = hi

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Conformal Prediction 適用スキップ: {e}")

        return results
    def _compute_ensemble_weights(self):
        """バリデーションスコアからアンサンブル重みを算出"""
        # デフォルト重み
        w_pos, w_top3, w_win = 0.25, 0.35, 0.40

        if not self.evaluation_results:
            return w_pos, w_top3, w_win

        # 各モデルの品質を0-1のスケールに統一
        # Position: MAE を品質スコアに変換（MAE=0→1.0, MAE≥5→0.0）
        pos_mae = self.evaluation_results.get("position", {}).get("mean_mae", 3.0)
        pos_quality = max(0.0, 1.0 - pos_mae / 5.0)

        # Top3: AUCを0-1スケールに（AUC=0.5→0.0, AUC=1.0→1.0）
        top3_auc = self.evaluation_results.get("top3", {}).get("mean_auc", 0.5)
        top3_quality = max(0.0, (top3_auc - 0.5) * 2.0)

        # Win: AUCを0-1スケールに
        win_auc = self.evaluation_results.get("win", {}).get("mean_auc", 0.5)
        win_quality = max(0.0, (win_auc - 0.5) * 2.0)

        scores = [
            max(pos_quality, 0.01),
            max(top3_quality, 0.01),
            max(win_quality, 0.01),
        ]

        total = sum(scores)
        w_pos = scores[0] / total
        w_top3 = scores[1] / total
        w_win = scores[2] / total

        return w_pos, w_top3, w_win

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

    MODEL_VERSION = "7.2-advanced"
    COMPATIBLE_VERSIONS = {"2.0", "2.1-tuned", "3.0-god", "4.0-god", "5.0-blend", "6.0-ultimate", "7.1-advanced", "7.2-advanced"}

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
            "odds_weight": getattr(self, "odds_weight", 0.0),
            # v6.0 追加
            "lambdarank_model": getattr(self, "lambdarank_model", None),
            "l2_position": getattr(self, "_l2_position", None),
            "l2_top3": getattr(self, "_l2_top3", None),
            "l2_win": getattr(self, "_l2_win", None),
            "bma_weights": getattr(self, "_bma_weights", None),
            "calibrators": getattr(self, "_calibrators", {}),
            # v7.0 追加: 動的アンサンブル・Conformal・L4・SHAP pruning
            "conformal_q95": getattr(self, "_conformal_q95", None),
            "conformal_scores": getattr(self, "_conformal_scores", None),
            "l4_position": getattr(self, "_l4_position", None),
            "l4_top_feats": getattr(self, "_l4_top_feats", []),
            "shap_pruned_features": getattr(self, "_shap_pruned_features", []),
            "shap_importance": (getattr(self, "_shap_importance", pd.DataFrame()).to_dict()
                                if getattr(self, "_shap_importance", None) is not None else {}),
            # v7.1 追加: Adversarial Validation / SMOTE / Ordinal / Incremental
            "adversarial_auc": getattr(self, "_adversarial_auc", None),
            "win_threshold": getattr(self, "_win_threshold", 0.5),
            # v7.2 追加: L4 LambdaRank統合フラグ
            "l4_has_lambdarank": getattr(self, "_l4_has_lambdarank", False),
            # v7.3 追加: L2/L4 がキャリブレ済みOOFで学習されたかのフラグ
            # (predict() のメタ入力スケール選択に使用。旧pklはFalse扱い=従来挙動)
            "l2_inputs_calibrated": getattr(self, "_l2_inputs_calibrated", False),
            # v7.3 追加: 学習時の欠損補完中央値 (predict() で同じ値を使う)
            "feature_medians": getattr(self, "_feature_medians", None),
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"モデル保存: {save_path} (v{self.MODEL_VERSION}, {len(self.feature_columns)}特徴量)")

    def load(self, name: str = "keiba_model"):
        """モデル読み込み"""
        load_path = self.model_dir / f"{name}.pkl"
        if not load_path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {load_path}")

        # Custom unpickler to resolve classes saved from __main__
        import model_trainer as _this_module

        class _ModelUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "__main__" and hasattr(_this_module, name):
                    return getattr(_this_module, name)
                return super().find_class(module, name)

        try:
            with open(load_path, "rb") as f:
                data = _ModelUnpickler(f).load()
        except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as e:
            raise RuntimeError(f"モデルファイル破損: {load_path} ({e})")

        # バージョンチェック
        saved_ver = data.get("version", "unknown")
        if saved_ver not in self.COMPATIBLE_VERSIONS:
            print(f"⚠️ モデルバージョン不一致: 保存={saved_ver}, 対応={self.COMPATIBLE_VERSIONS}")
            print(f"   再学習を推奨します")

        # 必須キーチェック
        required = ["models", "feature_columns"]
        for key in required:
            if key not in data:
                raise RuntimeError(f"モデルファイルに必須キー '{key}' がありません")

        self.models = data["models"]
        self.feature_columns = data["feature_columns"]
        self.evaluation_results = data.get("evaluation_results", {})
        self.lambdarank_model = data.get("lambdarank_model", None)
        self.model_type = data.get("model_type", "legacy")
        self.odds_weight = data.get("odds_weight", 0.0)
        # v6.0 追加コンポーネント
        self._l2_position = data.get("l2_position", None)
        self._l2_top3 = data.get("l2_top3", None)
        self._l2_win = data.get("l2_win", None)
        self._bma_weights = data.get("bma_weights", None)
        self._calibrators = data.get("calibrators", {})
        # v7.0 追加コンポーネント
        self._conformal_q95 = data.get("conformal_q95", None)
        self._conformal_scores = data.get("conformal_scores", None)
        self._l4_position = data.get("l4_position", None)
        self._l4_top_feats = data.get("l4_top_feats", [])
        self._shap_pruned_features = data.get("shap_pruned_features", [])
        _shap_imp_dict = data.get("shap_importance", {})
        self._shap_importance = pd.DataFrame(_shap_imp_dict) if _shap_imp_dict else None
        # v7.1 追加コンポーネント
        self._adversarial_auc = data.get("adversarial_auc", None)
        self._win_threshold = data.get("win_threshold", 0.5)
        # v7.2 追加コンポーネント
        self._l4_has_lambdarank = data.get("l4_has_lambdarank", False)
        # v7.3 追加: L2/L4 メタ入力スケールフラグ (旧pklは False = 生スコア)
        self._l2_inputs_calibrated = data.get("l2_inputs_calibrated", False)
        # v7.3 追加: 学習時補完中央値 (旧pklは None = レース内中央値補完)
        self._feature_medians = data.get("feature_medians", None)

        model_info = data.get("model_type", "legacy")
        if model_info == "god_ensemble":
            print(f"  GOD Ensemble loaded: full stacking pipeline (LGB+XGB+CB+TabNet -> L2 -> L3)")
            for task_name, m in self.models.items():
                if hasattr(m, "fold_models"):
                    counts = {k: len(v) for k, v in m.fold_models.items() if len(v) > 0}
                    has_meta = "yes" if m.meta_model is not None else "no"
                    has_l3 = "yes" if m.l3_model is not None else "no"
                    print(f"    {task_name}: models={counts}, meta={has_meta}, l3={has_l3}")
            if self.lambdarank_model:
                print(f"    LambdaRank: loaded")
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

        # BMA重み
        bma = getattr(self, "_bma_weights", None)
        if bma is not None:
            print(f"\n🎯 BMA最適重み: pos={bma[0]:.3f}, top3={bma[1]:.3f}, win={bma[2]:.3f}")

        # LambdaRank
        if getattr(self, "lambdarank_model", None):
            print("✅ LambdaRank: 有効")

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
