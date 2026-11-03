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

        # L2 + L3 stacking
        if self.meta_model is not None:
            meta_pred = self.meta_model.predict(X_meta)

            if self.l3_model is not None:
                # L3 input: base model predictions + meta-learner prediction
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
        else:
            # No meta model: BMA-weighted average of base predictions
            if (self.bma_weights is not None and len(self.bma_weights) > 0
                    and len(ordered_preds) > 0):
                # Map BMA weights to available model types
                weights = []
                for i, name in enumerate(self.models_config_names):
                    if name in preds_per_type and i < len(self.bma_weights):
                        weights.append(self.bma_weights[i])
                if weights and len(weights) == len(ordered_preds):
                    w = np.array(weights)
                    w = w / w.sum()  # normalize
                    result = np.average(ordered_preds, axis=0, weights=w)
                else:
                    result = np.mean(ordered_preds, axis=0)
            else:
                result = np.mean(ordered_preds, axis=0)

        # Apply calibration (binary tasks only)
        if self.calibrator is not None:
            try:
                result = self.calibrator.predict(result)
            except Exception as e:
                logger.debug(f"Calibration failed: {e}")

        # Apply temperature scaling (binary tasks only)
        if self.temperature_scaler is not None:
            try:
                result = self.temperature_scaler.predict(result)
            except Exception as e:
                logger.debug(f"Temperature scaling failed: {e}")

        return result

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
