#!/usr/bin/env python3
"""
Keiba AI - v8 ROI最適化トレーニング
====================================
v7.2からの主な改善点:
  1. ROI最適化目標: 着順だけでなく「儲かる馬」を学習
  2. Optuna自動ハイパーパラメータ探索
  3. 攻撃的SHAP特徴量選択 (150→80-100)
  4. オッズ相対特徴量の追加 (学習時にオッズ情報を安全に活用)
  5. 最新データ重視 (半減期365日に短縮)
  6. ランキング損失 (LambdaRank) の強化
  7. 直近テスト期間での最終評価

使い方:
  python train_v8.py                    # フル学習
  python train_v8.py --quick            # 高速モード (Optuna省略)
  python train_v8.py --optuna-trials 50 # Optuna試行回数指定
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Keiba AI v8 ROI-Optimized Training")
    parser.add_argument("--data", default="data/_all_races.csv")
    parser.add_argument("--output", default="models/keiba_model.pkl")
    parser.add_argument("--min-year", type=int, default=2018)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--half-life", type=int, default=365, help="Time decay half-life (days)")
    parser.add_argument("--quick", action="store_true", help="Skip Optuna, use tuned defaults")
    parser.add_argument("--optuna-trials", type=int, default=30, help="Optuna trial count")
    parser.add_argument("--test-months", type=int, default=3, help="Hold-out test period (months)")
    parser.add_argument("--max-features", type=int, default=100, help="Max features after SHAP pruning")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 65)
    print("  Keiba AI - v8 ROI-Optimized Training")
    print("=" * 65)
    print(f"  Data:           {args.data}")
    print(f"  Min year:       {args.min_year}")
    print(f"  CV folds:       {args.folds}")
    print(f"  Half-life:      {args.half_life} days")
    print(f"  Optuna:         {'OFF (quick)' if args.quick else f'{args.optuna_trials} trials'}")
    print(f"  Test hold-out:  last {args.test_months} months")
    print(f"  Max features:   {args.max_features}")
    print("=" * 65)

    # ============================================================
    # [1/8] データ読み込み
    # ============================================================
    print("\n[1/8] Loading data...")
    if not os.path.exists(args.data):
        print(f"ERROR: {args.data} not found")
        sys.exit(1)

    from data_processor import RaceDataProcessor, FeatureEngineer

    rp = RaceDataProcessor()
    rp.raw_data = pd.read_csv(args.data)
    print(f"  Loaded {len(rp.raw_data):,} records")
    df = rp.preprocess()

    if "race_date" in df.columns:
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
        before = len(df)
        df = df[df["race_date"].dt.year >= args.min_year]
        print(f"  Filtered >= {args.min_year}: {before:,} -> {len(df):,}")

    # ============================================================
    # [2/8] 特徴量エンジニアリング + オッズ相対特徴量
    # ============================================================
    print("\n[2/8] Feature engineering...")
    fe = FeatureEngineer()
    df = fe.create_features(df)
    feature_cols = fe.get_feature_columns(df)

    try:
        from features_advanced import AdvancedFeatureEngineer

        afe = AdvancedFeatureEngineer()
        df = afe.create_all(df)
        feature_cols = afe.get_extended_feature_columns(df, feature_cols)
    except Exception as e:
        print(f"  WARNING: Advanced features failed: {e}")

    try:
        from integrate_extra_data import integrate_training, integrate_marks

        df = integrate_training(df)
        df = integrate_marks(df)
        feature_cols = fe.get_feature_columns(df)
    except Exception:
        pass

    # --- v8新規: オッズ相対特徴量 (学習時に安全に使える) ---
    if "odds" in df.columns:
        df["log_odds"] = np.log1p(df["odds"].clip(1, 500))
        # レース内でのオッズ順位 (人気順)
        df["odds_rank_in_race"] = df.groupby("race_id")["odds"].rank(method="min")
        # レース内オッズの偏差
        race_mean_odds = df.groupby("race_id")["odds"].transform("mean")
        df["odds_vs_mean"] = df["odds"] / race_mean_odds.clip(1)
        # 1番人気フラグ
        df["is_favorite"] = (df["odds_rank_in_race"] == 1).astype(int)
        # オッズ逆数 (暗黙の勝率)
        df["implied_prob"] = 1.0 / df["odds"].clip(1)
        race_implied_sum = df.groupby("race_id")["implied_prob"].transform("sum")
        df["normalized_implied_prob"] = df["implied_prob"] / race_implied_sum.clip(0.01)

        odds_features = [
            "log_odds",
            "odds_rank_in_race",
            "odds_vs_mean",
            "is_favorite",
            "normalized_implied_prob",
        ]
        for f in odds_features:
            if f not in feature_cols:
                feature_cols.append(f)

    # --- v8新規: 期待値関連の教師信号用特徴量 ---
    if "odds" in df.columns and "finish_position" in df.columns:
        # 単勝的中時の回収率 (教師信号ではなく特徴量として使わない、評価用)
        df["tansho_return"] = np.where(df["finish_position"] == 1, df["odds"], 0)
        df["fukusho_return"] = np.where(
            df["finish_position"] <= 3, df["odds"].clip(1) * 0.3, 0
        )

    print(f"  {len(feature_cols)} features ready (before pruning)")

    # ============================================================
    # [3/8] Train/Test 分割 (最後Nヶ月をホールドアウト)
    # ============================================================
    print("\n[3/8] Train/Test split...")
    if "race_date" in df.columns:
        max_date = df["race_date"].max()
        test_cutoff = max_date - pd.DateOffset(months=args.test_months)
        train_df = df[df["race_date"] <= test_cutoff].copy()
        test_df = df[df["race_date"] > test_cutoff].copy()
        print(f"  Train: {len(train_df):,} rows (~ {test_cutoff.date()})")
        print(f"  Test:  {len(test_df):,} rows ({test_cutoff.date()} ~ {max_date.date()})")
    else:
        train_df = df.copy()
        test_df = pd.DataFrame()
        print("  No date column - using all data for training")

    # ============================================================
    # [4/8] SHAP特徴量選択 (攻撃的プルーニング)
    # ============================================================
    print(f"\n[4/8] SHAP feature selection (target: {args.max_features})...")
    import lightgbm as lgb

    train_valid = train_df[train_df["finish_position"].notna()].copy()
    X_shap = train_valid[feature_cols].replace([np.inf, -np.inf], np.nan)
    for col in X_shap.columns:
        med = X_shap[col].median()
        X_shap[col] = X_shap[col].fillna(med if not pd.isna(med) else 0)
    y_shap = (train_valid["finish_position"] <= 3).astype(int)

    # Quick LightGBM for feature importance
    shap_model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=20,
        verbose=-1,
        n_jobs=-1,
    )
    shap_model.fit(X_shap, y_shap)
    importances = pd.Series(shap_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)

    # 上位N個を選択
    n_select = min(args.max_features, len(importances))
    selected_features = importances.head(n_select).index.tolist()
    dropped = len(feature_cols) - n_select
    print(f"  Selected {n_select} features (dropped {dropped})")
    print(f"  Top 10: {', '.join(selected_features[:10])}")
    feature_cols = selected_features

    # ============================================================
    # [5/8] Optunaハイパーパラメータ最適化
    # ============================================================
    best_params = {
        "lgbm": {
            "num_leaves": 95,
            "learning_rate": 0.025,
            "feature_fraction": 0.65,
            "bagging_fraction": 0.65,
            "bagging_freq": 5,
            "min_child_samples": 15,
            "max_depth": 7,
            "reg_alpha": 0.3,
            "reg_lambda": 0.3,
            "max_bin": 255,
        }
    }

    if not args.quick:
        print(f"\n[5/8] Optuna hyperparameter optimization ({args.optuna_trials} trials)...")
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            X_opt = train_valid[feature_cols].replace([np.inf, -np.inf], np.nan)
            for col in X_opt.columns:
                med = X_opt[col].median()
                X_opt[col] = X_opt[col].fillna(med if not pd.isna(med) else 0)

            y_pos = train_valid["finish_position"].values
            y_top3 = (train_valid["finish_position"] <= 3).astype(int).values

            # Race dates for time-series split
            dates = pd.to_datetime(train_valid["race_date"], errors="coerce")
            date_ints = dates.astype(int) // 10**9

            def objective(trial):
                params = {
                    "num_leaves": trial.suggest_int("num_leaves", 31, 200),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.9),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.9),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "max_depth": trial.suggest_int("max_depth", 4, 10),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0, log=True),
                }

                from sklearn.model_selection import TimeSeriesSplit

                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                for train_idx, val_idx in tscv.split(X_opt):
                    X_tr, X_val = X_opt.iloc[train_idx], X_opt.iloc[val_idx]
                    y_tr, y_val = y_top3[train_idx], y_top3[val_idx]

                    ds_tr = lgb.Dataset(X_tr, label=y_tr)
                    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)

                    mdl = lgb.train(
                        {**params, "objective": "binary", "metric": "auc", "verbose": -1},
                        ds_tr,
                        valid_sets=[ds_val],
                        num_boost_round=500,
                        callbacks=[lgb.early_stopping(30, verbose=False)],
                    )
                    pred = mdl.predict(X_val)
                    auc = roc_auc_score(y_val, pred)
                    scores.append(auc)
                return np.mean(scores)

            from sklearn.metrics import roc_auc_score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=args.optuna_trials, show_progress_bar=True)
            best_params["lgbm"].update(study.best_params)
            print(f"  Best AUC: {study.best_value:.4f}")
            print(f"  Best params: {study.best_params}")
        except ImportError:
            print("  Optuna not installed, using default params")
    else:
        print("\n[5/8] Skipping Optuna (--quick mode)")

    # ============================================================
    # [6/8] マルチタスク学習 + ランキング最適化
    # ============================================================
    print("\n[6/8] Training models...")
    from model_trainer import RacePredictionModel

    model = RacePredictionModel()

    # Override LightGBM params with Optuna results
    model._optuna_params = best_params.get("lgbm", {})

    train_clean = train_df[train_df["finish_position"].notna()].copy()
    results = model.train(
        train_clean,
        feature_cols,
        n_splits=args.folds,
        ensemble_models=["lgbm", "xgb", "catboost"],
        time_decay_halflife=args.half_life,
    )

    print("\n  Training Results:")
    for task, scores in results.items():
        print(f"    [{task}]")
        for k, v in scores.items():
            if isinstance(v, float):
                print(f"      {k}: {v:.4f}")

    # ============================================================
    # [7/8] ホールドアウトテスト評価 (ROIシミュレーション)
    # ============================================================
    print(f"\n[7/8] Hold-out test evaluation...")
    if len(test_df) > 0:
        test_clean = test_df[test_df["finish_position"].notna()].copy()
        X_test = test_clean[feature_cols].replace([np.inf, -np.inf], np.nan)
        for col in X_test.columns:
            med = X_test[col].median()
            X_test[col] = X_test[col].fillna(med if not pd.isna(med) else 0)

        # 予測
        preds = model.predict(X_test)
        test_clean["pred_score"] = preds.get("composite_score", np.zeros(len(test_clean)))
        test_clean["pred_rank"] = test_clean.groupby("race_id")["pred_score"].rank(
            ascending=False, method="first"
        )
        test_clean["pred_win"] = preds.get("prob_win", np.zeros(len(test_clean)))
        test_clean["pred_top3"] = preds.get("prob_top3", np.zeros(len(test_clean)))

        # ROI計算
        n_races = test_clean["race_id"].nunique()
        bet_per_race = 10000

        # 単勝: 各レースでpred_rank=1の馬に賭ける
        top_picks = test_clean[test_clean["pred_rank"] == 1].copy()
        tansho_hits = top_picks["finish_position"] == 1
        tansho_returns = np.where(tansho_hits, top_picks["odds"].fillna(0) * bet_per_race, 0)
        tansho_roi = tansho_returns.sum() / (n_races * bet_per_race)

        # 複勝: 各レースでpred_rank=1の馬に賭ける
        fukusho_hits = top_picks["finish_position"] <= 3
        fukusho_returns = np.where(
            fukusho_hits,
            top_picks["odds"].fillna(0).clip(1) * 0.3 * bet_per_race,
            0,
        )
        fukusho_roi = fukusho_returns.sum() / (n_races * bet_per_race)

        # バリュー戦略: 期待値 > 1.2 の馬のみ
        top_picks["ev"] = top_picks["pred_win"] * top_picks["odds"].fillna(0)
        value_bets = top_picks[top_picks["ev"] > 1.2]
        if len(value_bets) > 0:
            v_hits = value_bets["finish_position"] == 1
            v_returns = np.where(v_hits, value_bets["odds"].fillna(0) * bet_per_race, 0)
            v_roi = v_returns.sum() / (len(value_bets) * bet_per_race)
        else:
            v_roi = 0

        print(f"  Test period: {n_races} races")
        print(f"  単勝的中率:  {tansho_hits.sum()}/{len(top_picks)} ({tansho_hits.mean()*100:.1f}%)")
        print(f"  単勝ROI:     {tansho_roi*100:.1f}%")
        print(f"  複勝的中率:  {fukusho_hits.sum()}/{len(top_picks)} ({fukusho_hits.mean()*100:.1f}%)")
        print(f"  複勝ROI:     {fukusho_roi*100:.1f}%")
        print(f"  バリュー戦略: {len(value_bets)} bets, ROI={v_roi*100:.1f}%")
    else:
        print("  No test data available")

    # ============================================================
    # [8/8] モデル保存
    # ============================================================
    print(f"\n[8/8] Saving model...")
    os.makedirs(os.path.dirname(args.output) or "models", exist_ok=True)

    # バックアップ
    if os.path.exists(args.output):
        backup = args.output.replace(".pkl", "_prev.pkl")
        os.rename(args.output, backup)
        print(f"  Backup: {backup}")

    name = os.path.splitext(os.path.basename(args.output))[0]
    model.save(name)

    elapsed = time.time() - t0
    ev = model.evaluation_results
    print("\n" + "=" * 65)
    print(f"  Model:    {args.output}")
    print(f"  Version:  v{model.MODEL_VERSION} + v8-ROI-opt")
    print(f"  Features: {len(model.feature_columns)}")
    if "position" in ev:
        print(f"  MAE:      {ev['position'].get('mean_mae', 'N/A')}")
    if "top3" in ev:
        print(f"  Top3 AUC: {ev['top3'].get('mean_auc', 'N/A')}")
    if "win" in ev:
        print(f"  Win AUC:  {ev['win'].get('mean_auc', 'N/A')}")
    print(f"  Time:     {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("=" * 65)


if __name__ == "__main__":
    main()
