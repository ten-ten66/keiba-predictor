"""
GOD Ensemble Model - Full stacking pipeline for server inference.
================================================================
Shared between training (train_god.py) and serving (model_trainer.py).

This module defines the _GodEnsembleModel class at module level so that
pickle can find it during deserialization on the server side.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class TemperatureScaler:
    """Temperature scaling for probability calibration."""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, y_true, y_pred):
        best_t, best_nll = 1.0, float("inf")
        for t in np.linspace(0.1, 5.0, 50):
            scaled = self._scale(y_pred, t)
            nll = -np.mean(y_true * np.log(scaled + 1e-10) +
                           (1 - y_true) * np.log(1 - scaled + 1e-10))
            if nll < best_nll:
                best_nll = nll
                best_t = t
        self.temperature = best_t
        return self

    def predict(self, y_pred):
        return self._scale(y_pred, self.temperature)

    def _scale(self, probs, temperature):
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        log_odds = np.log(probs / (1 - probs))
        return 1.0 / (1.0 + np.exp(-log_odds / temperature))


class _GodEnsembleModel:
    """Full GOD stacking ensemble for server-side inference.

    Runs the complete stacking pipeline:
      1. Base models (LGB/XGB/CatBoost/TabNet) - average across folds per type
      2. BMA-weighted blending of base model outputs
      3. L2 meta-learner (LGB) on stacked predictions
      4. L3 model (Ridge/LogReg) on meta-features
      5. Isotonic calibration (for binary tasks)
      6. Temperature scaling (for binary tasks)

    Gracefully degrades when optional packages (xgboost, catboost, pytorch-tabnet)
    are not installed - those model types are simply skipped.

    Defined at module level for pickle compatibility.
    """

    def __init__(self, fold_models, meta_model, l3_model,
                 bma_weights=None, calibrator=None, temperature_scaler=None,
                 task_type="binary", models_config_names=None):
        """
        Args:
            fold_models: dict of {model_name: [fold_model_1, fold_model_2, ...]}
                         e.g. {"lightgbm": [...], "xgboost": [...], "catboost": [...], "tabnet": [...]}
            meta_model:  LGB Booster (L2 meta-learner), or None
            l3_model:    Ridge/LogisticRegression (L3), or None
            bma_weights: numpy array of per-model-type weights, or None (equal weight)
            calibrator:  IsotonicRegression calibrator, or None
            temperature_scaler: TemperatureScaler, or None
            task_type:   "binary" or "regression"
            models_config_names: ordered list of model names matching bma_weights order
                                 e.g. ["lightgbm", "xgboost", "catboost", "tabnet"]
        """
        self.fold_models = fold_models
        self.meta_model = meta_model
        self.l3_model = l3_model
        self.bma_weights = bma_weights
        self.calibrator = calibrator
        self.temperature_scaler = temperature_scaler
        self.task_type = task_type
        self.models_config_names = models_config_names or list(fold_models.keys())

    def _predict_base_models(self, X):
        """Get per-model-type averaged predictions from fold models.

        Returns:
            preds_per_type: dict {model_name: averaged_prediction_array}
            ordered_preds:  list of prediction arrays in models_config_names order
        """
        preds_per_type = {}

        for model_name in self.models_config_names:
            fold_list = self.fold_models.get(model_name, [])
            if not fold_list:
                continue

            fold_preds = []
            for m in fold_list:
                try:
                    if model_name == "tabnet":
                        pred = self._predict_tabnet(m, X)
                    elif model_name in ("xgboost",):
                        pred = self._predict_xgb(m, X)
                    elif model_name in ("catboost",):
                        pred = self._predict_catboost(m, X)
                    else:
                        # lightgbm Booster: .predict() returns probabilities/values
                        pred = m.predict(X)

                    if pred is not None and not np.all(np.isnan(pred)):
                        fold_preds.append(pred)
                except Exception as e:
                    logger.debug(f"Fold model {model_name} prediction failed: {e}")
                    continue

            if fold_preds:
                preds_per_type[model_name] = np.mean(fold_preds, axis=0)

        # Build ordered list matching models_config_names order
        ordered_preds = []
        for name in self.models_config_names:
            if name in preds_per_type:
                ordered_preds.append(preds_per_type[name])

        return preds_per_type, ordered_preds

    def _predict_tabnet(self, model, X):
        """Predict with TabNet, handling import errors gracefully."""
        try:
            X_f32 = np.asarray(X, dtype=np.float32)
            if self.task_type == "regression":
                return model.predict(X_f32).flatten()
            else:
                return model.predict_proba(X_f32)[:, 1]
        except Exception as e:
            logger.debug(f"TabNet prediction failed: {e}")
            return None

    def _predict_xgb(self, model, X):
        """Predict with XGBoost model."""
        try:
            if self.task_type == "regression":
                return model.predict(X)
            else:
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(X)[:, 1]
                return model.predict(X)
        except Exception as e:
            logger.debug(f"XGBoost prediction failed: {e}")
            return None

    def _predict_catboost(self, model, X):
        """Predict with CatBoost model."""
        try:
            if self.task_type == "regression":
                return model.predict(X)
            else:
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(X)[:, 1]
                return model.predict(X)
        except Exception as e:
            logger.debug(f"CatBoost prediction failed: {e}")
            return None

    def predict(self, X):
        """Run the full stacking pipeline and return final predictions.

        Pipeline:
          1. Get base model predictions (averaged across folds per type)
          2. Stack into meta-feature matrix
          3. If meta/l3 models available: run L2 -> L3 pipeline
          4. Otherwise: BMA-weighted average of base predictions
          5. Apply calibration + temperature scaling (binary only)
        """
        X_arr = np.asarray(X)
        preds_per_type, ordered_preds = self._predict_base_models(X_arr)

        if not ordered_preds:
            logger.warning("No base model produced predictions, returning zeros")
            return np.zeros(len(X_arr))

        # Stack base predictions into meta-feature matrix
        # Shape: (n_samples, n_model_types_with_predictions)
        X_meta = np.column_stack(ordered_preds)

        # We need the meta matrix to match the expected number of columns
        # If some model types failed, pad with zeros to match training shape
        n_expected = len(self.models_config_names)
        if X_meta.shape[1] < n_expected:
            # Build full matrix, filling missing model types with the MEAN of
            # available models (not zeros, which skews the meta-learner)
            available_mean = np.mean(ordered_preds, axis=0) if ordered_preds else np.zeros(len(X_arr))
            full_meta = np.tile(available_mean.reshape(-1, 1), (1, n_expected))
            for i, name in enumerate(self.models_config_names):
                if name in preds_per_type:
                    full_meta[:, i] = preds_per_type[name]
            X_meta = full_meta
            logger.info(f"Padded {n_expected - len(preds_per_type)} missing model types with mean of available models")

        # BMA-weighted average of base predictions (always compute as fallback)
        bma_result = self._compute_bma(preds_per_type, ordered_preds, X_meta)

        # L2 + L3 stacking
        if self.meta_model is not None:
            try:
                meta_pred = self.meta_model.predict(X_meta)

                if self.l3_model is not None:
                    X_l3 = np.column_stack([X_meta, meta_pred.reshape(-1, 1)])
                    try:
                        if hasattr(self.l3_model, "predict_proba") and self.task_type == "binary":
                            result = self.l3_model.predict_proba(X_l3)[:, 1]
                        else:
                            result = self.l3_model.predict(X_l3)
                    except Exception as e:
                        logger.warning(f"L3 prediction failed, using meta prediction: {e}")
                        result = meta_pred
                else:
                    result = meta_pred

                # Sanity check: fall back to BMA if stacking output is degenerate.
                # For win-probability (1 positive per ~16 horses), a low mean is expected,
                # so we check for (a) saturation at the top, or (b) loss of discrimination.
                if self.task_type == "binary":
                    saturated_high = np.mean(result > 0.95) > 0.5
                    low_discrimination = float(np.std(result)) < 1e-3
                    if saturated_high or low_discrimination:
                        logger.warning(
                            f"Stacking output degenerate (mean={np.mean(result):.3f}, "
                            f"std={np.std(result):.4f}), falling back to BMA"
                        )
                        result = bma_result
            except Exception as e:
                logger.warning(f"Meta model failed, using BMA: {e}")
                result = bma_result
        else:
            result = bma_result

        # Apply calibration (binary tasks only)
        # NOTE: Skip calibration - the IsotonicRegression calibrator trained on
        # stacking OOF data maps values > ~0.6 to 1.0, destroying discrimination.
        # This is because the training distribution had very few high-confidence
        # false positives, so the calibrator learned to map any "high" value to 1.0.
        # The BMA/stacking output is already reasonably calibrated.

        # Apply temperature scaling (binary tasks only)
        if self.temperature_scaler is not None:
            try:
                result = self.temperature_scaler.predict(result)
            except Exception as e:
                logger.debug(f"Temperature scaling failed: {e}")

        return result

    def _compute_bma(self, preds_per_type, ordered_preds, X_meta):
        """Compute BMA-weighted average of base predictions."""
        if (self.bma_weights is not None and len(self.bma_weights) > 0
                and len(ordered_preds) > 0):
            weights = []
            avail_preds = []
            for i, name in enumerate(self.models_config_names):
                if name in preds_per_type and i < len(self.bma_weights):
                    weights.append(self.bma_weights[i])
                    avail_preds.append(preds_per_type[name])
            if weights and len(weights) == len(avail_preds):
                w = np.array(weights)
                w = w / w.sum()
                return np.average(avail_preds, axis=0, weights=w)
            else:
                return np.mean(ordered_preds, axis=0)
        else:
            return np.mean(ordered_preds, axis=0) if ordered_preds else np.zeros(X_meta.shape[0])

    def predict_proba(self, X):
        """Return 2-column array [1-p, p] for server compatibility."""
        p = self.predict(X)
        return np.column_stack([1 - p, p])

    def feature_importance(self, importance_type="gain"):
        """Average feature importance from LightGBM fold models."""
        lgb_models = self.fold_models.get("lightgbm", [])
        if not lgb_models:
            return None
        imps = []
        for m in lgb_models:
            try:
                imps.append(m.feature_importance(importance_type=importance_type))
            except Exception:
                pass
        if imps:
            return np.mean(imps, axis=0)
        return None


class _EnsembleLGBModel:
    """Legacy wrapper: multiple LGB fold models with .predict() interface.

    Kept for backward compatibility with existing keiba_model.pkl files
    that were saved with this class from train_god.py.
    """
    def __init__(self, models, bma_weights=None, calibrator=None):
        self.models = models
        if bma_weights is not None and len(bma_weights) > 0:
            self.fold_weights = None
        else:
            self.fold_weights = None
        self.calibrator = calibrator

    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        if self.fold_weights is not None:
            result = np.average(preds, axis=0, weights=self.fold_weights)
        else:
            result = np.mean(preds, axis=0)
        if self.calibrator is not None:
            try:
                result = self.calibrator.predict(result)
            except Exception:
                pass
        return result

    def predict_proba(self, X):
        p = self.predict(X)
        return np.column_stack([1 - p, p])

    def feature_importance(self, importance_type="gain"):
        imps = [m.feature_importance(importance_type=importance_type) for m in self.models]
        return np.mean(imps, axis=0)


# ================================================================
# 改善6: Mixture of Experts (MoE)
# ================================================================

class GateNetwork:
    """Gate Network: 入力特徴量から各 Expert の重みをソフトマックスで出力する。

    軽量なロジスティック回帰ベースのゲーティング。
    race_type_features（距離・馬場・クラス等）から重みを計算する。
    """

    def __init__(self, n_experts: int, gate_feature_names: list = None):
        """
        Args:
            n_experts: Expert の数
            gate_feature_names: ゲーティングに使う特徴量名リスト
                                 Noneの場合は全特徴量を使用
        """
        self.n_experts = n_experts
        self.gate_feature_names = gate_feature_names or []
        self._coef = None  # shape: (n_experts, n_gate_features)
        self._bias = None  # shape: (n_experts,)
        self._is_fitted = False

    def fit(self, X_gate: np.ndarray, expert_performances: np.ndarray):
        """Gate を学習する。

        Args:
            X_gate: shape (n_samples, n_gate_features)
            expert_performances: shape (n_samples, n_experts)
                各サンプルで各 Expert が正解に近かった程度（高いほど良い）
        """
        try:
            from sklearn.linear_model import LogisticRegression as _LR
            n_samples, n_feats = X_gate.shape
            # 各サンプルで最良の Expert を教師ラベルとして使用
            best_expert = np.argmax(expert_performances, axis=1)
            clf = _LR(C=1.0, max_iter=500, random_state=42, multi_class="ovr")
            clf.fit(X_gate, best_expert)
            self._coef = clf.coef_  # (n_experts, n_feats)
            self._bias = clf.intercept_  # (n_experts,)
            self._is_fitted = True
        except Exception as e:
            logger.warning(f"GateNetwork.fit failed: {e}")

    def predict_weights(self, X_gate: np.ndarray) -> np.ndarray:
        """各サンプルの Expert 重みを返す。

        Returns:
            weights: shape (n_samples, n_experts), 各行の和=1
        """
        if not self._is_fitted or self._coef is None:
            # 未学習の場合は均等重み
            n = len(X_gate)
            return np.full((n, self.n_experts), 1.0 / self.n_experts)

        try:
            logits = X_gate @ self._coef.T + self._bias  # (n, n_experts)
            # Softmax
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            weights = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            return weights
        except Exception as e:
            logger.warning(f"GateNetwork.predict_weights failed: {e}")
            n = len(X_gate)
            return np.full((n, self.n_experts), 1.0 / self.n_experts)


class MixtureOfExperts:
    """Mixture of Experts (MoE) モデル。

    条件別の専門家モデル（Expert）と、特徴量ベースのゲートネットワークを組み合わせる。

    Expert の例:
        - short_distance: 短距離専門（<=1400m）
        - long_distance:  長距離専門（>=2400m）
        - dirt:           ダート専門
        - turf:           芝専門

    Gate Network が各サンプルの特徴量（距離・馬場・クラス等）を見て
    どの Expert を何%信用するかのソフトな重みを出力する。
    各 Expert の予測を重み付き平均して最終予測とする。

    定義はモジュールレベルで行い、pickle 互換性を確保する。
    """

    # ゲーティングに使う特徴量名（学習データに含まれる列名）
    GATE_FEATURE_NAMES = ["distance", "surface", "race_class", "n_horses"]

    def __init__(self, task_type: str = "binary"):
        """
        Args:
            task_type: "binary" または "regression"
        """
        self.task_type = task_type
        self.experts: dict = {}          # {expert_name: model}
        self.expert_names: list = []     # 順序を保持
        self.gate: GateNetwork = None
        self._is_fitted = False

    # ----------------------------------------------------------
    # Expert 定義（条件別ルールベース特徴量選択）
    # ----------------------------------------------------------

    _EXPERT_CONDITIONS = {
        "short_distance": lambda row: float(row.get("distance", 0) or 0) <= 1400,
        "long_distance":  lambda row: float(row.get("distance", 0) or 0) >= 2400,
        "dirt":           lambda row: int(row.get("surface", 0) or 0) == 2,
        "turf":           lambda row: int(row.get("surface", 0) or 0) == 1,
        "general":        lambda row: True,  # 全条件に適用される汎用 Expert
    }

    def fit(
        self,
        fold_models_by_expert: dict,
        X_gate_train: np.ndarray,
        expert_oof_preds: dict,
        y_true: np.ndarray,
    ):
        """MoE を学習する。

        Args:
            fold_models_by_expert: {expert_name: model} の辞書
                各 expert は .predict(X) インターフェースを持つ
            X_gate_train: shape (n_samples, n_gate_features)
                ゲーティングに使う特徴量（OOFデータ）
            expert_oof_preds: {expert_name: oof_pred_array}
                各 Expert の OOF 予測
            y_true: 正解ラベル
        """
        try:
            self.experts = fold_models_by_expert
            self.expert_names = list(fold_models_by_expert.keys())
            n_experts = len(self.expert_names)

            if n_experts == 0:
                logger.warning("MoE.fit: no experts provided")
                return

            # Expert パフォーマンス行列を構築 (高いほど良い)
            n_samples = len(y_true)
            perf_matrix = np.zeros((n_samples, n_experts))
            for i, name in enumerate(self.expert_names):
                oof = expert_oof_preds.get(name)
                if oof is None:
                    continue
                if self.task_type == "regression":
                    # 回帰: 予測誤差が小さいほど良い -> -|error| を使用
                    err = np.abs(y_true - oof)
                    perf_matrix[:, i] = -err
                else:
                    # 分類: ログ尤度（高いほど良い）
                    oof_clipped = np.clip(oof, 1e-7, 1 - 1e-7)
                    perf_matrix[:, i] = (
                        y_true * np.log(oof_clipped) +
                        (1 - y_true) * np.log(1 - oof_clipped)
                    )

            # Gate Network を学習
            self.gate = GateNetwork(n_experts, self.GATE_FEATURE_NAMES)
            self.gate.fit(X_gate_train, perf_matrix)
            self._is_fitted = True
            logger.info(f"MoE fitted: {n_experts} experts, gate_features={self.GATE_FEATURE_NAMES}")

        except Exception as e:
            logger.warning(f"MoE.fit failed: {e}")

    def predict(self, X: np.ndarray, X_gate: np.ndarray = None) -> np.ndarray:
        """MoE 予測を実行する。

        Args:
            X: 予測用特徴量行列 (n_samples, n_features)
            X_gate: ゲーティング用特徴量 (n_samples, n_gate_features)
                    Noneの場合は均等重み

        Returns:
            predictions: shape (n_samples,)
        """
        if not self.experts:
            logger.warning("MoE.predict: no experts, returning zeros")
            return np.zeros(len(X))

        X_arr = np.asarray(X)
        n_samples = len(X_arr)

        # 各 Expert の予測を収集
        expert_preds = []
        valid_names = []
        for name in self.expert_names:
            expert = self.experts.get(name)
            if expert is None:
                continue
            try:
                pred = expert.predict(X_arr)
                if pred is not None and not np.all(np.isnan(pred)):
                    expert_preds.append(pred)
                    valid_names.append(name)
            except Exception as e:
                logger.debug(f"MoE expert {name} prediction failed: {e}")
                continue

        if not expert_preds:
            return np.zeros(n_samples)

        if len(expert_preds) == 1:
            return expert_preds[0]

        expert_matrix = np.column_stack(expert_preds)  # (n_samples, n_valid_experts)

        # Gate Network で重みを計算
        if X_gate is not None and self.gate is not None and self._is_fitted:
            try:
                X_gate_arr = np.asarray(X_gate, dtype=float)
                X_gate_arr = np.nan_to_num(X_gate_arr, nan=0.0)
                # valid_names に対応する重みだけ使う
                all_weights = self.gate.predict_weights(X_gate_arr)  # (n, n_all_experts)
                # valid_names のインデックスを取得
                valid_idx = [self.expert_names.index(n) for n in valid_names
                             if n in self.expert_names]
                if len(valid_idx) == len(valid_names):
                    weights = all_weights[:, valid_idx]  # (n, n_valid)
                    # 再正規化
                    w_sum = weights.sum(axis=1, keepdims=True)
                    w_sum = np.where(w_sum <= 0, 1.0, w_sum)
                    weights = weights / w_sum
                else:
                    weights = np.full((n_samples, len(expert_preds)), 1.0 / len(expert_preds))
            except Exception as e:
                logger.debug(f"MoE gate weighting failed: {e}")
                weights = np.full((n_samples, len(expert_preds)), 1.0 / len(expert_preds))
        else:
            # 均等重み（gate 未学習 or X_gate 未提供）
            weights = np.full((n_samples, len(expert_preds)), 1.0 / len(expert_preds))

        # 重み付き平均
        result = (expert_matrix * weights).sum(axis=1)
        return result

    def predict_proba(self, X: np.ndarray, X_gate: np.ndarray = None) -> np.ndarray:
        """2列 [1-p, p] 形式で返す（分類互換）。"""
        p = self.predict(X, X_gate)
        return np.column_stack([1 - p, p])

    @classmethod
    def build_from_god_ensemble(
        cls,
        god_model: "_GodEnsembleModel",
        task_type: str = "binary",
    ) -> "MixtureOfExperts":
        """既存の _GodEnsembleModel から MoE を構築するファクトリメソッド。

        _GodEnsembleModel の fold_models を Expert として登録する。
        Gate Network は学習データがないため均等重みのまま。
        train_god.py 側で fit() を呼び出すことで Gate を学習できる。

        Args:
            god_model: 学習済みの _GodEnsembleModel
            task_type: "binary" または "regression"
        """
        moe = cls(task_type=task_type)

        # 各モデルタイプを Expert として登録
        for model_name, fold_list in god_model.fold_models.items():
            if not fold_list:
                continue
            # fold の平均予測を行う軽量ラッパー
            expert = _FoldAverageExpert(fold_list, model_name, task_type)
            moe.experts[model_name] = expert
            moe.expert_names.append(model_name)

        # Gate は未学習（均等重みで動作）
        moe.gate = GateNetwork(len(moe.expert_names), cls.GATE_FEATURE_NAMES)
        moe._is_fitted = False

        logger.info(f"MoE built from GodEnsemble: experts={moe.expert_names}")
        return moe


class _FoldAverageExpert:
    """fold モデルリストの平均予測を行う軽量 Expert ラッパー。

    MixtureOfExperts の Expert として使用する。
    pickle 互換のためモジュールレベルで定義する。
    """

    def __init__(self, fold_models: list, model_name: str, task_type: str = "binary"):
        self.fold_models = fold_models
        self.model_name = model_name
        self.task_type = task_type

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for m in self.fold_models:
            try:
                if self.model_name == "tabnet":
                    X_f32 = np.asarray(X, dtype=np.float32)
                    if self.task_type == "regression":
                        p = m.predict(X_f32).flatten()
                    else:
                        p = m.predict_proba(X_f32)[:, 1]
                elif self.model_name == "xgboost":
                    if self.task_type == "regression":
                        p = m.predict(X)
                    elif hasattr(m, "predict_proba"):
                        p = m.predict_proba(X)[:, 1]
                    else:
                        p = m.predict(X)
                elif self.model_name == "catboost":
                    if self.task_type == "regression":
                        p = m.predict(X)
                    elif hasattr(m, "predict_proba"):
                        p = m.predict_proba(X)[:, 1]
                    else:
                        p = m.predict(X)
                else:
                    # lightgbm
                    p = m.predict(X)
                    if self.task_type != "regression":
                        if p.min() < -0.01 or p.max() > 1.01:
                            p = 1.0 / (1.0 + np.exp(-p))
                if p is not None and not np.all(np.isnan(p)):
                    preds.append(p)
            except Exception as e:
                logger.debug(f"_FoldAverageExpert {self.model_name} fold prediction failed: {e}")
                continue

        if not preds:
            return None
        return np.mean(preds, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.predict(X)
        if p is None:
            p = np.zeros(len(X))
        return np.column_stack([1 - p, p])
