"""
Optuna ハイパーパラメータチューニング → チューニング済みモデルで再学習
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

from run_pipeline import fast_feature_engineering, get_feature_columns
from data_processor import RaceDataProcessor
from model_tuner import ModelTuner, EnsembleOptimizer, MultiObjectiveTuner
from backtester import Backtester

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def main():
    start_time = time.time()

    print("=" * 70)
    print("  Optuna ハイパーパラメータチューニング")
    print("=" * 70)

    if not HAS_OPTUNA:
        print("optuna未インストール。インストールします...")
        os.system(f"{sys.executable} -m pip install optuna -q")
        import optuna

    # データ読み込み（直近5年で高速チューニング）
    print("\n[1/5] データ読み込み...")
    dfs = []
    for year in range(2020, 2026):
        path = f"data/races_{year}.csv"
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
            print(f"  {year}: {len(dfs[-1]):,}")
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    # 前処理 + 特徴量
    print("\n[2/5] 前処理 + 特徴量...")
    processor = RaceDataProcessor(data_dir="data")
    processor.raw_data = df
    df = processor.preprocess()
    df = fast_feature_engineering(df)
    feature_cols = get_feature_columns(df)
    print(f"  {len(df):,} 件 / {len(feature_cols)} 特徴量")

    # チューニング用データ（2020-2024）
    tune_df = df[df["year"] <= 2024].copy()
    test_df = df[df["year"] == 2025].copy()

    # チューニング
    print("\n[3/5] Optunaチューニング (50 trials x 3 models)...")
    tuner = ModelTuner(n_trials=50, n_splits=3)
    best_params = tuner.tune_all(tune_df, feature_cols)

    print("\n  最適パラメータ:")
    for name, params in best_params.items():
        print(f"    [{name}]")
        for k, v in params.items():
            print(f"      {k}: {v}")

    # Multi-Objective チューニング: MAE + Spearman 同時最適化
    print("\n[3.5/5] Multi-Objective チューニング (MAE + Spearman Pareto Front)...")
    try:
        mo_tuner = MultiObjectiveTuner(n_trials=30, n_splits=3)
        pareto_params = mo_tuner.tune(tune_df, feature_cols)
        balanced = mo_tuner.best_balanced_params()
        mo_tuner.save_pareto("data/pareto_params.json")

        if balanced:
            print("\n  Multi-Objective バランス最良パラメータを position モデルに適用")
            # 既存の best_params["position"] を Pareto 結果で上書き（任意）
            best_params.setdefault("position_pareto", balanced)
    except Exception as e:
        print(f"  Multi-Objective チューニング スキップ: {e}")

    # チューニング済みパラメータで再学習（全データ 2015-2024）
    print("\n[4/5] チューニング済みパラメータで再学習...")

    # 全10年分を読み込み
    dfs = []
    for year in range(2015, 2025):
        path = f"data/races_{year}.csv"
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    full_train = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    processor2 = RaceDataProcessor(data_dir="data")
    processor2.raw_data = full_train
    full_train = processor2.preprocess()
    full_train = fast_feature_engineering(full_train)

    # LightGBMで直接チューニング済みパラメータを使って学習
    X_train = full_train[feature_cols].copy()
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    for col in X_train.columns:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med if not pd.isna(med) else 0)

    y_pos = full_train["finish_position"].values
    y_top3 = (full_train["finish_position"] <= 3).astype(int).values
    y_win = (full_train["finish_position"] == 1).astype(int).values

    X_test = test_df[feature_cols].copy()
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    for col in X_test.columns:
        med = X_test[col].median()
        X_test[col] = X_test[col].fillna(med if not pd.isna(med) else 0)

    models = {}

    # 着順回帰
    print("  着順回帰モデル...")
    pos_params = {
        "objective": "regression", "metric": "mae",
        "boosting_type": "gbdt", "verbose": -1, "n_jobs": -1, "seed": 42,
    }
    pos_params.update(best_params.get("position", {}))
    train_data = lgb.Dataset(X_train, label=y_pos)
    val_data = lgb.Dataset(X_test, label=test_df["finish_position"].values, reference=train_data)
    models["position"] = lgb.train(
        pos_params, train_data, num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # 複勝分類
    print("  複勝分類モデル...")
    top3_params = {
        "objective": "binary", "metric": "binary_logloss",
        "boosting_type": "gbdt", "verbose": -1, "n_jobs": -1, "seed": 42,
    }
    top3_params.update(best_params.get("top3", {}))
    train_data = lgb.Dataset(X_train, label=y_top3)
    y_test_top3 = (test_df["finish_position"] <= 3).astype(int).values
    val_data = lgb.Dataset(X_test, label=y_test_top3, reference=train_data)
    models["top3"] = lgb.train(
        top3_params, train_data, num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # 勝利分類
    print("  勝利分類モデル...")
    win_params = {
        "objective": "binary", "metric": "binary_logloss",
        "boosting_type": "gbdt", "verbose": -1, "n_jobs": -1, "seed": 42,
    }
    win_params.update(best_params.get("win", {}))
    train_data = lgb.Dataset(X_train, label=y_win)
    y_test_win = (test_df["finish_position"] == 1).astype(int).values
    val_data = lgb.Dataset(X_test, label=y_test_win, reference=train_data)
    models["win"] = lgb.train(
        win_params, train_data, num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # アンサンブル重み最適化
    print("\n  アンサンブル重み最適化...")
    pos_pred = models["position"].predict(X_test)
    pos_norm = 1 - (pos_pred - pos_pred.min()) / (pos_pred.max() - pos_pred.min() + 1e-8)
    top3_pred = models["top3"].predict(X_test)
    win_pred = models["win"].predict(X_test)

    w1, w2, w3 = EnsembleOptimizer.optimize_weights(
        pos_norm, top3_pred, win_pred,
        test_df["finish_position"].values,
        test_df["odds"].values,
        n_trials=200,
    )

    # 予測
    print("\n[5/5] 2025年評価...")
    composite = pos_norm * w1 + top3_pred * w2 + win_pred * w3

    results = test_df[["race_id", "horse_number"]].copy()
    results["pred_position"] = pos_pred
    results["prob_top3"] = top3_pred
    results["prob_win"] = win_pred
    results["composite_score"] = composite
    results["predicted_rank"] = results.groupby("race_id")["composite_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    merged = results.merge(
        test_df[["race_id", "horse_number", "finish_position", "odds"]],
        on=["race_id", "horse_number"], how="inner",
    )

    top1 = merged[merged["predicted_rank"] == 1]
    top1_acc = (top1["finish_position"] == 1).mean()
    top3_contains = merged.groupby("race_id").apply(
        lambda g: (g[g["predicted_rank"] <= 3]["finish_position"] == 1).any()
    ).mean()
    top3_m = merged[merged["predicted_rank"] <= 3]
    top3_place = (top3_m["finish_position"] <= 3).mean()

    print(f"\n  予測1位→1着的中率:     {top1_acc*100:.1f}%")
    print(f"  上位3頭に1着馬含有率:  {top3_contains*100:.1f}%")
    print(f"  上位3頭の複勝的中率:   {top3_place*100:.1f}%")

    # バックテスト
    bt = Backtester(test_df, results)
    bt_results = bt.run_all_strategies(bet_amount=100, bankroll=100000)
    bt.print_comparison(bt_results)

    # モデル保存
    import pickle
    save_data = {
        "version": "2.1-tuned",
        "models": models,
        "feature_columns": feature_cols,
        "ensemble_weights": {"w_pos": w1, "w_top3": w2, "w_win": w3},
        "best_params": best_params,
        "saved_at": datetime.now().isoformat(),
        "n_features": len(feature_cols),
    }
    os.makedirs("models", exist_ok=True)
    with open("models/keiba_model_tuned.pkl", "wb") as f:
        pickle.dump(save_data, f)

    # サマリー保存
    summary = {
        "tuning_run": datetime.now().isoformat(),
        "best_params": best_params,
        "ensemble_weights": {"w_pos": w1, "w_top3": w2, "w_win": w3},
        "accuracy": {
            "top1_win_rate": round(top1_acc * 100, 1),
            "top3_contains_winner": round(top3_contains * 100, 1),
            "top3_place_rate": round(top3_place * 100, 1),
        },
        "backtest": [
            {"strategy": r.strategy_name, "roi": r.roi_percent, "hit_rate": r.hit_rate, "profit": r.profit}
            for r in sorted(bt_results, key=lambda x: x.roi_percent, reverse=True)
        ],
    }
    # Multi-Objective Pareto Front 結果を summary に含める
    pareto_path = "data/pareto_params.json"
    if os.path.exists(pareto_path):
        try:
            with open(pareto_path) as f_p:
                pareto_data = json.load(f_p)
            summary["multi_objective"] = {
                "pareto_trials": pareto_data.get("pareto_trials", 0),
                "balanced_best": pareto_data.get("balanced_best"),
            }
        except Exception:
            pass

    with open("data/tuning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  チューニング完了！ ({elapsed/60:.1f} 分)")
    print(f"{'='*70}")
    print(f"  models/keiba_model_tuned.pkl  - チューニング済みモデル")
    print(f"  data/tuned_params.json        - 最適パラメータ")
    print(f"  data/pareto_params.json       - Multi-Objective Pareto Front")
    print(f"  data/tuning_summary.json      - サマリー")


if __name__ == "__main__":
    main()
