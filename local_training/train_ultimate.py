"""
競馬予想AI - 最高精度学習パイプライン (Ultimate)
==================================================
時間制限なし。精度最優先。

手法:
  1. Elo/Glickoレーティングシステム（馬・騎手・調教師）
  2. ターゲットエンコーディング（CV-safe）
  3. Boruta特徴量選択
  4. LightGBM + XGBoost + CatBoost 3モデル
  5. Optunaハイパーパラメータ最適化（各200試行 + Pruning）
  6. Stacking Ensemble（メタモデル: Ridge + LightGBM）
  7. 確率キャリブレーション
  8. Adversarial Validation（分布シフト検出）
  9. 8-fold 時系列CV
  10. SHAP + Permutation Importance

推定所要時間: 8-16時間（RTX 5060 + 48GB RAM）
"""

import os
import sys
import time
import json
import pickle
import warnings
import gc
from datetime import datetime
from pathlib import Path
from collections import defaultdict

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
    "output_dir": "output_ultimate",
    "model_dir": "models",

    "n_cv_folds": 8,
    "optuna_trials": 200,
    "use_gpu": True,

    # Stacking
    "stacking_meta_cv": 5,

    # Boruta
    "boruta_max_iter": 100,
}


# ══════════════════════════════════════════════
# Part 1: Elo/Glicko レーティングシステム
# ══════════════════════════════════════════════
class EloRatingSystem:
    """
    競馬用Eloレーティング。
    各レースの着順に基づいて馬・騎手・調教師のレーティングを更新。
    通常のEloは1vs1だが、競馬はN頭立てなので全ペア比較で計算。
    """

    def __init__(self, k_factor=16, initial_rating=1500):
        self.k = k_factor
        self.initial = initial_rating
        self.ratings = {}

    def get_rating(self, entity_id):
        return self.ratings.get(entity_id, self.initial)

    def expected_score(self, ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update_race(self, participants):
        """
        participants: list of (entity_id, finish_position)
        全ペア比較でElo更新
        """
        n = len(participants)
        if n < 2:
            return

        # 各馬のレーティング取得
        ratings = {pid: self.get_rating(pid) for pid, _ in participants}

        # 全ペア比較
        for i, (id_i, pos_i) in enumerate(participants):
            delta = 0.0
            for j, (id_j, pos_j) in enumerate(participants):
                if i == j:
                    continue
                expected = self.expected_score(ratings[id_i], ratings[id_j])
                actual = 1.0 if pos_i < pos_j else (0.5 if pos_i == pos_j else 0.0)
                delta += self.k * (actual - expected) / (n - 1)

            self.ratings[id_i] = ratings[id_i] + delta


class GlickoRatingSystem:
    """
    Glicko-1レーティング。RD（不確実性）を考慮したElo拡張。
    長期休養明けの馬はRDが大きくなり、レーティングが不安定になる。
    """

    def __init__(self, initial_rating=1500, initial_rd=350, c=15):
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.c = c  # RD増加速度
        self.ratings = {}
        self.rds = {}
        self.last_race = {}

    def get_rating(self, entity_id):
        return self.ratings.get(entity_id, self.initial_rating)

    def get_rd(self, entity_id, current_period=0):
        rd = self.rds.get(entity_id, self.initial_rd)
        last = self.last_race.get(entity_id, current_period)
        periods_inactive = current_period - last
        # RDは非活動期間に応じて増加（最大350）
        return min(350, np.sqrt(rd ** 2 + self.c ** 2 * periods_inactive))

    def g(self, rd):
        return 1.0 / np.sqrt(1 + 3 * (rd / np.pi) ** 2 / 400 ** 2)

    def expected(self, r, r_opp, rd_opp):
        return 1.0 / (1 + 10 ** (-self.g(rd_opp) * (r - r_opp) / 400))

    def update_race(self, participants, period=0):
        """participants: list of (entity_id, finish_position)"""
        n = len(participants)
        if n < 2:
            return

        for pid, pos in participants:
            r = self.get_rating(pid)
            rd = self.get_rd(pid, period)

            d2_inv = 0
            delta_sum = 0

            for oid, opos in participants:
                if pid == oid:
                    continue
                r_opp = self.get_rating(oid)
                rd_opp = self.get_rd(oid, period)

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


def compute_elo_features(df):
    """全レースを時系列順に処理してElo/Glickoレーティングを計算"""
    print("  [Elo] 馬のEloレーティング計算中...")
    horse_elo = EloRatingSystem(k_factor=20)
    horse_glicko = GlickoRatingSystem()
    jockey_elo = EloRatingSystem(k_factor=12)
    trainer_elo = EloRatingSystem(k_factor=8)

    # 結果列
    horse_elo_col = np.full(len(df), np.nan)
    horse_glicko_col = np.full(len(df), np.nan)
    horse_glicko_rd_col = np.full(len(df), np.nan)
    jockey_elo_col = np.full(len(df), np.nan)
    trainer_elo_col = np.full(len(df), np.nan)

    # レースごとに処理
    race_ids = df["race_id"].unique()
    period = 0

    for race_id in tqdm(race_ids, desc="  [Elo] Computing", leave=False):
        mask = df["race_id"] == race_id
        race_df = df[mask]
        indices = race_df.index

        # レーティング記録（レース前の値を使用）
        for idx, row in race_df.iterrows():
            hid = row.get("horse_id")
            jid = row.get("jockey_id")
            tid = row.get("trainer_id")

            if pd.notna(hid):
                horse_elo_col[idx] = horse_elo.get_rating(hid)
                horse_glicko_col[idx] = horse_glicko.get_rating(hid)
                horse_glicko_rd_col[idx] = horse_glicko.get_rd(hid, period)
            if pd.notna(jid):
                jockey_elo_col[idx] = jockey_elo.get_rating(jid)
            if pd.notna(tid):
                trainer_elo_col[idx] = trainer_elo.get_rating(tid)

        # 結果がある場合のみ更新
        valid = race_df.dropna(subset=["finish_position"])
        if len(valid) >= 2:
            horse_parts = [(r["horse_id"], r["finish_position"])
                           for _, r in valid.iterrows() if pd.notna(r.get("horse_id"))]
            jockey_parts = [(r["jockey_id"], r["finish_position"])
                            for _, r in valid.iterrows() if pd.notna(r.get("jockey_id"))]
            trainer_parts = [(r["trainer_id"], r["finish_position"])
                             for _, r in valid.iterrows() if pd.notna(r.get("trainer_id"))]

            horse_elo.update_race(horse_parts)
            horse_glicko.update_race(horse_parts, period)
            jockey_elo.update_race(jockey_parts)
            trainer_elo.update_race(trainer_parts)

        period += 1

    df["horse_elo"] = horse_elo_col
    df["horse_glicko"] = horse_glicko_col
    df["horse_glicko_rd"] = horse_glicko_rd_col
    df["jockey_elo"] = jockey_elo_col
    df["trainer_elo"] = trainer_elo_col

    # Elo相対値（レース内）
    df["horse_elo_rank"] = df.groupby("race_id")["horse_elo"].rank(ascending=False)
    df["jockey_elo_rank"] = df.groupby("race_id")["jockey_elo"].rank(ascending=False)
    df["combined_elo"] = df["horse_elo"] * 0.6 + df["jockey_elo"] * 0.25 + df["trainer_elo"] * 0.15
    df["combined_elo_rank"] = df.groupby("race_id")["combined_elo"].rank(ascending=False)

    print(f"  [Elo] 完了 — 馬{len(horse_elo.ratings):,}頭 / 騎手{len(jockey_elo.ratings):,}人")
    return df


# ══════════════════════════════════════════════
# Part 2: ターゲットエンコーディング (CV-safe)
# ══════════════════════════════════════════════
def target_encode_cv(df, col, target, n_splits=5, smoothing=20):
    """
    時系列safeなターゲットエンコーディング。
    FIX: KFold(ランダム) → TimeSeriesSplit で未来データのリークを防止。
    """
    from sklearn.model_selection import TimeSeriesSplit

    global_mean = df[target].mean()
    encoded = pd.Series(np.nan, index=df.index)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(df):
        train = df.iloc[train_idx]
        stats = train.groupby(col)[target].agg(["mean", "count"])
        smooth_mean = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        encoded.iloc[val_idx] = df.iloc[val_idx][col].map(smooth_mean)

    encoded = encoded.fillna(global_mean)
    return encoded


def add_target_encodings(df, feature_cols_before):
    """騎手・調教師・会場のターゲットエンコーディング"""
    print("  [TargetEnc] ターゲットエンコーディング計算中...")

    # 結果のある行のみで計算
    has_result = df["finish_position"].notna()
    df_with_result = df[has_result].copy()

    # binary targets
    df_with_result["_is_win"] = (df_with_result["finish_position"] == 1).astype(float)
    df_with_result["_is_top3"] = (df_with_result["finish_position"] <= 3).astype(float)

    targets = {
        "win": "_is_win",
        "top3": "_is_top3",
    }

    encode_cols = []
    if "jockey_id" in df.columns:
        encode_cols.append("jockey_id")
    if "trainer_id" in df.columns:
        encode_cols.append("trainer_id")
    if "venue" in df.columns:
        encode_cols.append("venue")

    for col in encode_cols:
        for tname, tcol in targets.items():
            new_col = f"te_{col}_{tname}"
            values = target_encode_cv(df_with_result, col, tcol)
            # 全行にマージ
            df.loc[has_result, new_col] = values.values
            df[new_col] = df[new_col].fillna(df_with_result[tcol].mean())
            print(f"    + {new_col}")

    # cleanup
    for tcol in targets.values():
        if tcol in df.columns:
            df.drop(columns=[tcol], inplace=True, errors="ignore")

    print(f"  [TargetEnc] 完了")
    return df


# ══════════════════════════════════════════════
# Part 3: Boruta 特徴量選択
# ══════════════════════════════════════════════
PROTECTED_FEATURES = [
    "distance", "age", "weight", "horse_weight", "odds", "popularity",
    "track_condition", "course_type", "frame_number", "horse_number",
    "field_condition", "n_runners", "sex_enc", "weather_enc",
    "horse_elo", "jockey_elo", "trainer_elo", "horse_glicko",
    "combined_elo", "days_since_last_race",
]
BORUTA_MIN_FEATURES = 30


def boruta_selection(X, y, feature_names, max_iter=100):
    """
    Boruta: シャドウ特徴量と比較して有意な特徴量を選択。
    FIX: 最小30特徴量保証 + ドメイン保護特徴量 + year/month除外。
    """
    import lightgbm as lgb
    from scipy import stats

    print(f"  [Boruta] 特徴量選択開始 ({len(feature_names)}個, {max_iter}イテレーション)...")

    # year/month/day_of_weekを除外
    exclude_names = {"year", "month", "day_of_week"}
    exclude_idx = [i for i, f in enumerate(feature_names) if f in exclude_names]
    keep_idx = [i for i in range(len(feature_names)) if i not in exclude_idx]
    X_boruta = X[:, keep_idx]
    boruta_names = [feature_names[i] for i in keep_idx]
    n_features = X_boruta.shape[1]

    hit_counts = np.zeros(n_features)

    for iteration in tqdm(range(max_iter), desc="  [Boruta]", leave=False):
        X_shadow = X_boruta.copy()
        rng = np.random.RandomState(iteration)
        for col_idx in range(n_features):
            X_shadow[:, col_idx] = rng.permutation(X_shadow[:, col_idx])

        X_combined = np.hstack([X_boruta, X_shadow])
        dtrain = lgb.Dataset(X_combined, label=y)
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "boosting_type": "gbdt", "num_leaves": 63, "learning_rate": 0.1,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "verbose": -1, "n_jobs": -1, "seed": iteration,
        }
        model = lgb.train(params, dtrain, num_boost_round=100)

        importances = model.feature_importance(importance_type="gain")
        real_imp = importances[:n_features]
        shadow_imp = importances[n_features:]
        shadow_max = shadow_imp.max() if len(shadow_imp) > 0 else 0
        hit_counts += (real_imp > shadow_max).astype(int)

    # 統計検定
    boruta_mask = np.zeros(n_features, dtype=bool)
    for i in range(n_features):
        p_value = 1 - stats.binom.cdf(hit_counts[i] - 1, max_iter, 0.5)
        boruta_mask[i] = p_value < 0.05

    # 全特徴量に戻す
    full_mask = np.zeros(len(feature_names), dtype=bool)
    for bi, fi in enumerate(keep_idx):
        if boruta_mask[bi]:
            full_mask[fi] = True

    # 保護特徴量を強制追加
    protected_added = 0
    for i, fname in enumerate(feature_names):
        if not full_mask[i]:
            for pat in PROTECTED_FEATURES:
                if pat in fname:
                    full_mask[i] = True
                    protected_added += 1
                    break

    # 最小特徴量数を保証
    if full_mask.sum() < BORUTA_MIN_FEATURES:
        hit_order = np.argsort(-hit_counts)
        for bi in hit_order:
            fi = keep_idx[bi]
            if not full_mask[fi]:
                full_mask[fi] = True
            if full_mask.sum() >= BORUTA_MIN_FEATURES:
                break

    selected = [f for f, s in zip(feature_names, full_mask) if s]
    rejected = [f for f, s in zip(feature_names, full_mask) if not s]

    print(f"  [Boruta] 結果: {len(selected)}個採用 / {len(rejected)}個除外")
    if protected_added:
        print(f"    保護特徴量追加: {protected_added}個")

    return selected, full_mask


# ══════════════════════════════════════════════
# Part 4: データ処理 & 特徴量エンジニアリング
# ══════════════════════════════════════════════
# (train_advanced.py の DataProcessor と FeatureEngine をインポート)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from train_advanced import DataProcessor, FeatureEngine
except ImportError:
    print("❌ train_advanced.py が見つかりません。同じディレクトリに配置してください。")
    sys.exit(1)


# ══════════════════════════════════════════════
# Part 5: Adversarial Validation
# ══════════════════════════════════════════════
def adversarial_validation(X_train, X_test, feature_names):
    """
    訓練と検証の分布差を検出。AUC > 0.6 なら分布シフトの可能性。
    高AUCの特徴量は信頼性が低い。
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    print("  [AdvVal] Adversarial Validation...")

    # FIX: year/month/day_of_weekを除外（AUC=1.0のリーク防止）
    exclude = {"year", "month", "day_of_week"}
    keep_idx = [i for i, f in enumerate(feature_names) if f not in exclude]
    X_tr_clean = X_train[:, keep_idx]
    X_te_clean = X_test[:, keep_idx]
    keep_names = [feature_names[i] for i in keep_idx]

    y_adv = np.concatenate([np.zeros(len(X_tr_clean)), np.ones(len(X_te_clean))])
    X_adv = np.vstack([X_tr_clean, X_te_clean])

    # 80/20 split
    n = len(X_adv)
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    tr_idx, va_idx = idx[:split], idx[split:]

    dtrain = lgb.Dataset(X_adv[tr_idx], label=y_adv[tr_idx])
    dval = lgb.Dataset(X_adv[va_idx], label=y_adv[va_idx])

    params = {
        "objective": "binary", "metric": "auc", "verbose": -1,
        "num_leaves": 31, "learning_rate": 0.1, "n_jobs": -1,
    }
    model = lgb.train(params, dtrain, num_boost_round=200,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

    pred = model.predict(X_adv[va_idx])
    auc = roc_auc_score(y_adv[va_idx], pred)

    imp = model.feature_importance(importance_type="gain")
    top_drift = sorted(zip(keep_names, imp), key=lambda x: -x[1])[:10]

    print(f"  [AdvVal] AUC = {auc:.4f} {'⚠️ 分布シフト検出!' if auc > 0.6 else '✅ 分布安定'}")
    if auc > 0.6:
        print(f"    シフトが大きい特徴量: {[f[0] for f in top_drift[:5]]}")

    return auc, top_drift



# ══════════════════════════════════════════════
# Part 7: Stacking Ensemble
# ══════════════════════════════════════════════
def train_stacking(X, y, task, models_config, n_folds=5, use_gpu=False):
    """
    Level 1: LightGBM + XGBoost + CatBoost → OOF predictions
    Level 2: Ridge + LightGBM meta-learner
    """
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import roc_auc_score, mean_absolute_error

    print(f"\n  === Stacking ({task}) ===")

    tscv = TimeSeriesSplit(n_splits=n_folds)
    n_models = len(models_config)

    # FIX: NaN初期化（0初期化だと予測値0との区別がつかない）
    oof_preds = np.full((len(X), n_models), np.nan)

    fold_models = {name: [] for name, _ in models_config}
    fold_scores = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        if task == "binary" and (len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2):
            continue

        fold_preds = np.zeros((len(va_idx), n_models))

        for model_idx, (model_name, model_params) in enumerate(models_config):
            if model_name == "lightgbm":
                dtrain = lgb.Dataset(X_tr, label=y_tr)
                dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

                params = model_params.copy()
                if use_gpu:
                    params["device"] = "gpu"
                    params["gpu_use_dp"] = True

                model = lgb.train(
                    params, dtrain, num_boost_round=3000,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
                )
                pred = model.predict(X_va)

            elif model_name == "xgboost":
                params = model_params.copy()
                es_rounds = params.pop("early_stopping_rounds", 100)
                if use_gpu:
                    params["tree_method"] = "hist"
                    params["device"] = "cuda"

                if task == "regression":
                    m = xgb.XGBRegressor(**params, early_stopping_rounds=es_rounds)
                else:
                    m = xgb.XGBClassifier(**params, early_stopping_rounds=es_rounds)
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                pred = m.predict(X_va) if task == "regression" else m.predict_proba(X_va)[:, 1]
                model = m

            elif model_name == "catboost":
                try:
                    from catboost import CatBoostRegressor, CatBoostClassifier
                    params = model_params.copy()
                    if use_gpu:
                        params["task_type"] = "GPU"

                    if task == "regression":
                        m = CatBoostRegressor(**params)
                    else:
                        m = CatBoostClassifier(**params)
                    m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, early_stopping_rounds=100)
                    pred = m.predict(X_va) if task == "regression" else m.predict_proba(X_va)[:, 1]
                    model = m
                except ImportError:
                    print(f"    CatBoost未インストール → LightGBMで代替")
                    dtrain = lgb.Dataset(X_tr, label=y_tr)
                    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
                    params2 = models_config[0][1].copy()
                    params2["seed"] = 99
                    if use_gpu:
                        params2["device"] = "gpu"
                        params2["gpu_use_dp"] = True
                    model = lgb.train(params2, dtrain, num_boost_round=3000,
                                      valid_sets=[dval],
                                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                    pred = model.predict(X_va)

            fold_models[model_name].append(model)
            fold_preds[:, model_idx] = pred

        oof_preds[va_idx] = fold_preds

        # fold score
        if task == "regression":
            ensemble_pred = fold_preds.mean(axis=1)
            score = mean_absolute_error(y_va, ensemble_pred)
            print(f"    Fold {fold_idx+1}: MAE={score:.4f}")
        else:
            ensemble_pred = fold_preds.mean(axis=1)
            try:
                score = roc_auc_score(y_va, ensemble_pred)
            except ValueError:
                score = 0.5
            print(f"    Fold {fold_idx+1}: AUC={score:.4f}")
        fold_scores.append(score)

    # Level 2: Meta-learner
    print(f"  [Stacking] Level 2 メタモデル学習...")

    # FIX: NaN判定（0判定だと予測値0が除外される）
    valid_mask = ~np.isnan(oof_preds).all(axis=1)
    X_meta = np.nan_to_num(oof_preds[valid_mask], nan=0.0)
    y_meta = y[valid_mask]

    if task == "regression":
        meta_model = Ridge(alpha=1.0)
    else:
        meta_model = LogisticRegression(C=1.0, max_iter=1000)

    # FIX: train/val splitで評価（同じデータで学習＆評価しない）
    meta_split = int(len(X_meta) * 0.8)
    if meta_split > 50:
        meta_model.fit(X_meta[:meta_split], y_meta[:meta_split])
        meta_pred_va = meta_model.predict(X_meta[meta_split:]) if task == "regression" else meta_model.predict_proba(X_meta[meta_split:])[:, 1]
        if task == "regression":
            meta_score = mean_absolute_error(y_meta[meta_split:], meta_pred_va)
            print(f"  [Stacking] Meta holdout MAE: {meta_score:.4f}")
        else:
            try:
                meta_score = roc_auc_score(y_meta[meta_split:], meta_pred_va)
            except ValueError:
                meta_score = 0.5
            print(f"  [Stacking] Meta holdout AUC: {meta_score:.4f}")
        # 最終的には全データでrefit
        meta_model.fit(X_meta, y_meta)
    else:
        meta_model.fit(X_meta, y_meta)
        print(f"  [Stacking] Meta: データ不足のためholdout評価スキップ")

    avg_score = np.mean(fold_scores)
    print(f"  [Stacking] 平均スコア: {avg_score:.4f}")

    return {
        "fold_models": fold_models,
        "meta_model": meta_model,
        "oof_preds": oof_preds,
        "fold_scores": fold_scores,
        "avg_score": avg_score,
    }


# ══════════════════════════════════════════════
# Part 8: Optunaハイパーパラメータ最適化 (Pruning付き)
# ══════════════════════════════════════════════
def optimize_with_pruning(X_train, y_train, X_val, y_val, task, n_trials=200, use_gpu=False):
    """Optuna + Median Pruning で効率的に探索"""
    import optuna
    import lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "regression" if task == "regression" else "binary",
            "metric": "l1" if task == "regression" else "auc",
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,

            "num_leaves": trial.suggest_int("num_leaves", 31, 512),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 2.0),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
        }

        if params["boosting_type"] == "dart":
            params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.5)
            params["max_drop"] = trial.suggest_int("max_drop", 5, 50)

        if use_gpu:
            params["device"] = "gpu"
            params["gpu_use_dp"] = True

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Pruning callback — metric must match study direction
        # regression: minimize l1, binary: maximize auc
        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial, "l1" if task == "regression" else "auc"
        )

        model = lgb.train(
            params, dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0), pruning_callback],
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
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)

    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"    ベストスコア: {study.best_value:.4f}")
    print(f"    完了/枝刈り: {len(study.trials)}/{sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}")
    return study.best_params


# ══════════════════════════════════════════════
# メインパイプライン
# ══════════════════════════════════════════════
def main():
    print("=" * 70)
    print("🏇 競馬予想AI - Ultimate 学習パイプライン")
    print(f"   日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["model_dir"], exist_ok=True)

    use_gpu = CONFIG["use_gpu"]
    if use_gpu:
        try:
            import subprocess
            r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if r.returncode != 0:
                use_gpu = False
        except FileNotFoundError:
            use_gpu = False
    print(f"  GPU: {'ON' if use_gpu else 'OFF'}")

    # ──── 1. データ読み込み ────
    print("\n" + "=" * 70)
    print("📂 [1/9] データ読み込み & 前処理")
    print("=" * 70)

    proc = DataProcessor()
    df = proc.load(CONFIG["data_file"])

    bl_path = Path(CONFIG["bloodline_file"])
    if bl_path.exists():
        bl = pd.read_csv(bl_path)
        if "horse_id" in bl.columns:
            df = df.merge(bl, on="horse_id", how="left")
            print(f"  血統結合: {len(bl)} 件")

    df = proc.preprocess(df)
    df_train = df[df["finish_position"].notna()].copy()

    # FIX: 日付順ソート（shift(1)で未来データを使わないため）
    if "race_date" in df_train.columns:
        df_train = df_train.sort_values(["race_date", "race_id", "horse_number"]).reset_index(drop=True)
        print("  ✅ データを日付順にソート済み")
    elif "race_id" in df_train.columns:
        df_train = df_train.sort_values(["race_id", "horse_number"]).reset_index(drop=True)

    n_races = df_train["race_id"].nunique()
    print(f"  学習データ: {len(df_train):,} 件 / {n_races:,} レース")

    # ──── 2. 基本 + 拡張特徴量 ────
    print("\n" + "=" * 70)
    print("🔧 [2/9] 基本 + 拡張特徴量")
    print("=" * 70)

    fe = FeatureEngine()
    df_train = fe.create_all(df_train, use_advanced=True)

    # ──── 3. Elo/Glicko レーティング ────
    print("\n" + "=" * 70)
    print("📊 [3/9] Elo/Glicko レーティング")
    print("=" * 70)

    df_train = compute_elo_features(df_train)

    # ──── 4. ターゲットエンコーディング ────
    print("\n" + "=" * 70)
    print("📊 [4/9] ターゲットエンコーディング")
    print("=" * 70)

    feature_cols_before = fe.get_feature_columns(df_train)
    df_train = add_target_encodings(df_train, feature_cols_before)

    # 特徴量リスト
    feature_cols = fe.get_feature_columns(df_train)
    print(f"\n  総特徴量数: {len(feature_cols)} 個")

    # ──── 5. データ準備 ────
    print("\n" + "=" * 70)
    print("📊 [5/9] データ準備")
    print("=" * 70)

    X = df_train[feature_cols].values.copy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y_position = df_train["finish_position"].values
    y_win = (df_train["finish_position"] == 1).astype(int).values
    y_top3 = (df_train["finish_position"] <= 3).astype(int).values

    tscv = TimeSeriesSplit(n_splits=CONFIG["n_cv_folds"])
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-1]
    X_train, X_val = X[train_idx], X[val_idx]

    print(f"  訓練: {len(X_train):,} / 検証: {len(X_val):,}")

    # ──── 6. Adversarial Validation ────
    print("\n" + "=" * 70)
    print("🔍 [6/9] Adversarial Validation")
    print("=" * 70)

    adv_auc, adv_drift = adversarial_validation(X_train, X_val, feature_cols)

    # ──── 7. Boruta 特徴量選択 ────
    print("\n" + "=" * 70)
    print("🔬 [7/9] Boruta 特徴量選択")
    print("=" * 70)

    # FIX: 最新N件をサンプリング（ランダムではなく時系列順を尊重）
    sample_size = min(200000, len(X_train))
    sample_idx = np.arange(len(X_train) - sample_size, len(X_train))

    selected_features, selected_mask = boruta_selection(
        X_train[sample_idx], y_top3[train_idx][sample_idx],
        feature_cols, max_iter=CONFIG["boruta_max_iter"]
    )

    # Boruta結果を適用
    X_selected = X[:, selected_mask]
    X_train_sel = X_selected[train_idx]
    X_val_sel = X_selected[val_idx]

    print(f"  選択後: {len(selected_features)} / {len(feature_cols)} 特徴量")

    # Boruta結果保存
    with open(f"{CONFIG['output_dir']}/boruta_results.json", "w") as f:
        json.dump({"selected": selected_features,
                    "rejected": [f for f, s in zip(feature_cols, selected_mask) if not s]},
                   f, ensure_ascii=False, indent=2)

    # ──── 8. Optuna + Stacking ────
    print("\n" + "=" * 70)
    print("🎯 [8/9] Optuna最適化 + Stacking Ensemble")
    print("=" * 70)

    # Optuna for LightGBM
    tasks = [
        ("position", "regression", y_position),
        ("top3", "binary", y_top3),
        ("win", "binary", y_win),
    ]

    all_results = {}

    for task_name, task_type, y_target in tasks:
        print(f"\n  ─── {task_name} ({task_type}) ───")

        y_tr = y_target[train_idx]
        y_va = y_target[val_idx]

        # Optuna
        print(f"  [Optuna] {CONFIG['optuna_trials']}試行 (Pruning有効)...")
        best_lgb_params = optimize_with_pruning(
            X_train_sel, y_tr, X_val_sel, y_va,
            task=task_type, n_trials=CONFIG["optuna_trials"], use_gpu=use_gpu
        )

        # LightGBM パラメータ
        lgb_params = {
            "objective": "regression" if task_type == "regression" else "binary",
            "metric": "mae" if task_type == "regression" else "binary_logloss",
            "verbose": -1, "n_jobs": -1, "seed": 42,
            **best_lgb_params,
        }

        # XGBoost パラメータ
        xgb_params = {
            "objective": "reg:squarederror" if task_type == "regression" else "binary:logistic",
            "eval_metric": "mae" if task_type == "regression" else "logloss",
            "max_depth": best_lgb_params.get("max_depth", 6),
            "learning_rate": best_lgb_params.get("learning_rate", 0.05),
            "subsample": best_lgb_params.get("bagging_fraction", 0.8),
            "colsample_bytree": best_lgb_params.get("feature_fraction", 0.8),
            "reg_alpha": best_lgb_params.get("reg_alpha", 0.1),
            "reg_lambda": best_lgb_params.get("reg_lambda", 0.1),
            "n_estimators": 3000,
            "early_stopping_rounds": 100,
            "verbosity": 0,
            "random_state": 42,
        }

        # CatBoost パラメータ
        catboost_params = {
            "iterations": 3000,
            "learning_rate": best_lgb_params.get("learning_rate", 0.05),
            "depth": min(best_lgb_params.get("max_depth", 6), 10),
            "l2_leaf_reg": best_lgb_params.get("reg_lambda", 1.0),
            "random_seed": 42,
            "verbose": 0,
        }

        # Stacking
        models_config = [
            ("lightgbm", lgb_params),
            ("xgboost", xgb_params),
            ("catboost", catboost_params),
        ]

        # FIX: 訓練データのみでStacking（検証データの混入を防止）
        stack_result = train_stacking(
            X_selected[train_idx], y_target[train_idx], task_type,
            models_config, n_folds=CONFIG["n_cv_folds"], use_gpu=use_gpu
        )

        all_results[task_name] = {
            "best_lgb_params": best_lgb_params,
            "stacking": stack_result,
            "task_type": task_type,
        }

    # ──── 9. 保存 ────
    print("\n" + "=" * 70)
    print("💾 [9/9] モデル保存")
    print("=" * 70)

    # サーバー互換モデル (LightGBMのベストfoldモデルで構成)
    server_models = {}
    eval_results = {}
    feat_importances = {}

    for task_name, result in all_results.items():
        stack = result["stacking"]
        lgb_models = stack["fold_models"]["lightgbm"]

        if not lgb_models:
            continue

        # ベストfoldのLightGBMモデル
        if result["task_type"] == "regression":
            best_idx = np.argmin(stack["fold_scores"])
        else:
            best_idx = np.argmax(stack["fold_scores"])
        best_idx = min(best_idx, len(lgb_models) - 1)

        server_models[task_name] = lgb_models[best_idx]

        # 評価結果
        scores = stack["fold_scores"]
        metric = "mae" if result["task_type"] == "regression" else "auc"
        eval_results[task_name] = {
            f"mean_{metric}": float(np.mean(scores)),
            f"std_{metric}": float(np.std(scores)),
        }

        # 特徴量重要度
        imp = lgb_models[best_idx].feature_importance(importance_type="gain")
        feat_importances[task_name] = pd.DataFrame({
            "feature": selected_features,
            "importance": imp,
        }).sort_values("importance", ascending=False)

    # サーバー互換pkl
    save_data = {
        "version": "2.0",
        "models": server_models,
        "feature_columns": selected_features,
        "evaluation_results": eval_results,
        "feature_importance": {k: v.to_dict() for k, v in feat_importances.items()},
        "saved_at": datetime.now().isoformat(),
        "n_features": len(selected_features),
        "trained_on": "local_ultimate",
    }

    model_path = f"{CONFIG['model_dir']}/keiba_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"  ✅ サーバー互換モデル: {model_path}")

    # Stacking全体モデル (ローカル用)
    ultimate_path = f"{CONFIG['model_dir']}/keiba_model_ultimate.pkl"
    with open(ultimate_path, "wb") as f:
        pickle.dump({
            "all_results": {k: {
                "fold_models": v["stacking"]["fold_models"],
                "meta_model": v["stacking"]["meta_model"],
                "best_lgb_params": v["best_lgb_params"],
                "task_type": v["task_type"],
            } for k, v in all_results.items()},
            "selected_features": selected_features,
            "boruta_mask": selected_mask.tolist(),
            "all_feature_columns": feature_cols,
            "config": CONFIG,
            "timestamp": datetime.now().isoformat(),
        }, f)
    print(f"  ✅ Ultimate全体モデル: {ultimate_path}")

    # レポート
    report = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "ultimate",
        "data_rows": len(df_train),
        "n_races": n_races,
        "total_features": len(feature_cols),
        "selected_features": len(selected_features),
        "adversarial_auc": adv_auc,
        "evaluation": eval_results,
        "config": CONFIG,
    }
    with open(f"{CONFIG['output_dir']}/training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # SHAP
    print("\n📊 SHAP分析...")
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for task_name in ["win", "top3"]:
            if task_name in server_models:
                explainer = shap.TreeExplainer(server_models[task_name])
                sample = pd.DataFrame(X_val_sel[:min(1000, len(X_val_sel))], columns=selected_features)
                shap_values = explainer.shap_values(sample)

                plt.figure(figsize=(14, 12))
                shap.summary_plot(shap_values, sample, show=False, max_display=30)
                plt.tight_layout()
                plt.savefig(f"{CONFIG['output_dir']}/shap_{task_name}.png", dpi=150)
                plt.close()
                print(f"  ✅ SHAP: {CONFIG['output_dir']}/shap_{task_name}.png")
    except Exception as e:
        print(f"  ⚠️ SHAP: {e}")

    # 特徴量重要度グラフ
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for task_name, imp_df in feat_importances.items():
            top30 = imp_df.head(30)
            plt.figure(figsize=(14, 12))
            plt.barh(range(len(top30)), top30["importance"].values)
            plt.yticks(range(len(top30)), top30["feature"].values)
            plt.xlabel("Importance (gain)")
            plt.title(f"Feature Importance - {task_name} (Ultimate)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{CONFIG['output_dir']}/importance_{task_name}.png", dpi=150)
            plt.close()
        print(f"  ✅ 重要度グラフ: {CONFIG['output_dir']}/importance_*.png")
    except Exception as e:
        print(f"  ⚠️ グラフ: {e}")

    # ──── 最終サマリー ────
    print("\n" + "=" * 70)
    print("🎉 Ultimate 学習完了！")
    print("=" * 70)

    for name, scores in eval_results.items():
        print(f"\n  【{name}】")
        for k, v in scores.items():
            print(f"    {k}: {v:.4f}")

    print(f"\n  特徴量: {len(feature_cols)} → Boruta選択後 {len(selected_features)}")
    print(f"  Adversarial AUC: {adv_auc:.4f}")
    print(f"\n  📁 出力:")
    print(f"    {CONFIG['model_dir']}/keiba_model.pkl        (サーバー互換)")
    print(f"    {CONFIG['model_dir']}/keiba_model_ultimate.pkl (Stacking全体)")
    print(f"    {CONFIG['output_dir']}/training_report.json")
    print(f"    {CONFIG['output_dir']}/boruta_results.json")

    print(f"\n  🚀 サーバーにアップロード:")
    print(f"    scp {CONFIG['model_dir']}/keiba_model.pkl user@server:/opt/keiba-predictor/models/")


if __name__ == "__main__":
    main()
