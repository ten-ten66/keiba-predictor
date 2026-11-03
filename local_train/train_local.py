#!/usr/bin/env python3
"""
競馬予測AI - ローカル高度学習スクリプト
========================================
ローカルPCのフルスペックを活用した高精度学習。

使い方:
  python train_local.py                    ← 標準学習（直近10年）
  python train_local.py --full             ← 全データ学習（25年分）
  python train_local.py --advanced         ← 拡張特徴量 + ハイパーチューニング
  python train_local.py --full --advanced  ← 最高精度（全データ+拡張+チューニング）
  python train_local.py --upload           ← 学習後VPSに自動アップロード
"""

import sys
import os
import time
import argparse
import shutil
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# VPS設定
VPS_HOST = os.environ.get("KEIBA_VPS_HOST", "")
VPS_USER = os.environ.get("KEIBA_VPS_USER", "")
VPS_MODEL_PATH = "/opt/keiba-predictor/models/keiba_model.pkl"


def check_dependencies():
    """依存パッケージ確認"""
    required = {"pandas": "pandas", "numpy": "numpy", "lightgbm": "lightgbm", "sklearn": "scikit-learn"}
    optional = {"optuna": "optuna", "xgboost": "xgboost", "psutil": "psutil"}

    missing_req = []
    for pkg, pip_name in required.items():
        try:
            __import__(pkg)
        except ImportError:
            missing_req.append(pip_name)

    if missing_req:
        print(f"❌ 不足パッケージ: {', '.join(missing_req)}")
        print(f"   pip install {' '.join(missing_req)}")
        sys.exit(1)

    available_opt = {}
    for pkg, pip_name in optional.items():
        try:
            __import__(pkg)
            available_opt[pkg] = True
        except ImportError:
            available_opt[pkg] = False

    print("✅ 必須パッケージOK")
    if available_opt.get("optuna"):
        print("✅ Optuna（ハイパーチューニング）利用可能")
    else:
        print("⚠️  Optuna未インストール（pip install optuna でチューニング有効化）")
    if available_opt.get("xgboost"):
        print("✅ XGBoost 利用可能")

    return available_opt


def get_system_info():
    """システム情報取得"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_count()
        print(f"💻 CPU: {cpu}コア / メモリ: {mem.total / 1024**3:.1f}GB (空き {mem.available / 1024**3:.1f}GB)")
        return {"mem_gb": mem.total / 1024**3, "cpu_count": cpu}
    except ImportError:
        print("💻 システム情報: psutil未インストール（pip install psutil）")
        return {"mem_gb": 8, "cpu_count": 4}


def train_standard(years=10, use_full=False, n_splits=5):
    """標準学習"""
    import pandas as pd
    import datetime
    from data_processor import RaceDataProcessor, FeatureEngineer
    from model_trainer import RacePredictionModel

    data_dir = PROJECT_ROOT / "data"
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("📊 [1/4] データ読込")
    print("=" * 60)
    proc = RaceDataProcessor(data_dir=str(data_dir))
    df = proc.load_csv("_all_races.csv")
    df = proc.preprocess(df)
    print(f"   全データ: {len(df):,}行 / {df['race_id'].nunique():,}レース")

    if not use_full and "race_date" in df.columns:
        cutoff = datetime.datetime.now() - datetime.timedelta(days=365 * years)
        df = df[df["race_date"] >= cutoff].copy()
        print(f"   → 直近{years}年に絞り: {len(df):,}行")

    print("\n" + "=" * 60)
    print("🔧 [2/4] 特徴量生成")
    print("=" * 60)
    t0 = time.time()
    fe = FeatureEngineer()
    df = fe.create_features(df)
    feature_cols = fe.get_feature_columns(df)
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"   基本特徴量: {len(feature_cols)}個 ({time.time()-t0:.0f}秒)")

    print("\n" + "=" * 60)
    print(f"🤖 [3/4] モデル学習 ({len(df):,}行 × {len(feature_cols)}特徴量)")
    print("=" * 60)
    t0 = time.time()
    model = RacePredictionModel(model_dir=str(model_dir))
    scores = model.train(df, feature_cols, n_splits=n_splits)
    print(f"   ⏱ 学習時間: {time.time()-t0:.0f}秒")

    model_path = save_model(model, model_dir, scores)
    return model_path, scores, df, feature_cols


def train_advanced(years=10, use_full=False, n_splits=5, tune=True):
    """高度学習（拡張特徴量 + ハイパーチューニング）"""
    import pandas as pd
    import numpy as np
    import datetime
    from data_processor import RaceDataProcessor, FeatureEngineer
    from model_trainer import RacePredictionModel

    data_dir = PROJECT_ROOT / "data"
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    # 1. データ読込
    print("\n" + "=" * 60)
    print("📊 [1/5] データ読込")
    print("=" * 60)
    proc = RaceDataProcessor(data_dir=str(data_dir))
    df = proc.load_csv("_all_races.csv")
    df = proc.preprocess(df)
    print(f"   全データ: {len(df):,}行 / {df['race_id'].nunique():,}レース")

    if not use_full and "race_date" in df.columns:
        cutoff = datetime.datetime.now() - datetime.timedelta(days=365 * years)
        df = df[df["race_date"] >= cutoff].copy()
        print(f"   → 直近{years}年に絞り: {len(df):,}行")

    # 2. 基本特徴量
    print("\n" + "=" * 60)
    print("🔧 [2/5] 基本特徴量生成")
    print("=" * 60)
    t0 = time.time()
    fe = FeatureEngineer()
    df = fe.create_features(df)
    base_cols = fe.get_feature_columns(df)
    print(f"   基本特徴量: {len(base_cols)}個 ({time.time()-t0:.0f}秒)")

    # 3. 拡張特徴量
    print("\n" + "=" * 60)
    print("🔧 [3/5] 拡張特徴量生成（血統・調教・距離適性・コース適性）")
    print("=" * 60)
    t0 = time.time()
    try:
        from features_advanced import AdvancedFeatureEngineer
        adv = AdvancedFeatureEngineer()
        df = adv.create_all(df)
        feature_cols = adv.get_extended_feature_columns(df, base_cols)
        feature_cols = [c for c in feature_cols if c in df.columns]
        print(f"   拡張特徴量: {len(feature_cols)}個 ({time.time()-t0:.0f}秒)")
    except Exception as e:
        print(f"   ⚠️ 拡張特徴量失敗: {e}")
        print(f"   → 基本特徴量のみで続行")
        feature_cols = [c for c in base_cols if c in df.columns]

    gc.collect()

    # 4. ハイパーチューニング（Optuna）
    best_params = None
    if tune:
        print("\n" + "=" * 60)
        print("🎯 [4/5] ハイパーパラメータチューニング (Optuna)")
        print("=" * 60)
        best_params = run_optuna_tuning(df, feature_cols, n_trials=50)
    else:
        print("\n" + "=" * 60)
        print("⏭  [4/5] チューニングスキップ")
        print("=" * 60)

    # 5. 最終学習
    print("\n" + "=" * 60)
    print(f"🤖 [5/5] 最終モデル学習 ({len(df):,}行 × {len(feature_cols)}特徴量)")
    print("=" * 60)
    t0 = time.time()
    model = RacePredictionModel(model_dir=str(model_dir))

    # チューニング結果を適用
    if best_params:
        print(f"   チューニング済みパラメータ適用:")
        for k, v in best_params.items():
            print(f"     {k}: {v}")
        model.LGBM_PARAMS_REGRESSION.update(best_params)
        model.LGBM_PARAMS_BINARY.update({k: v for k, v in best_params.items()
                                          if k not in ("objective", "metric")})

    scores = model.train(df, feature_cols, n_splits=n_splits)
    print(f"   ⏱ 学習時間: {time.time()-t0:.0f}秒")

    model_path = save_model(model, model_dir, scores)
    return model_path, scores, df, feature_cols


def run_optuna_tuning(df, feature_cols, n_trials=50):
    """Optunaでハイパーパラメータ最適化"""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("   ⚠️ Optunaが未インストール。デフォルトパラメータで続行。")
        print("   pip install optuna でインストール可能")
        return None

    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        med = X[col].median()
        X[col] = X[col].fillna(med if not np.isnan(med) else 0)

    y = (df["finish_position"] <= 3).astype(int).values
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X))

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }

        aucs = []
        for train_idx, val_idx in splits:
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

            model = lgb.train(
                params, dtrain, num_boost_round=500,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
            )

            pred = model.predict(X_va)
            aucs.append(roc_auc_score(y_va, pred))

        return np.mean(aucs)

    print(f"   {n_trials}回の試行を実行中...")
    t0 = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"   ⏱ チューニング時間: {time.time()-t0:.0f}秒")
    print(f"   🏆 ベストAUC: {study.best_value:.4f}")
    print(f"   パラメータ:")
    for k, v in study.best_params.items():
        print(f"     {k}: {v}")

    # 最適パラメータを返す（objective/metricは含めない）
    best = study.best_params.copy()
    best["verbose"] = -1
    best["n_jobs"] = -1
    best["seed"] = 42
    return best


def save_model(model, model_dir, scores):
    """モデル保存"""
    print("\n" + "=" * 60)
    print("💾 モデル保存")
    print("=" * 60)

    model_path = model_dir / "keiba_model.pkl"
    if model_path.exists():
        backup = model_dir / "keiba_model_prev.pkl"
        shutil.copy2(model_path, backup)
        print(f"   旧モデルバックアップ: {backup}")

    model.save(str(model_path))
    size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"   ✅ 保存: {model_path} ({size_mb:.1f}MB)")
    return model_path


def upload_model(model_path):
    """VPSにアップロード"""
    import subprocess

    print("\n" + "=" * 60)
    print("🚀 VPSにアップロード")
    print("=" * 60)

    scp_cmd = f"scp {model_path} {VPS_USER}@{VPS_HOST}:{VPS_MODEL_PATH}"
    print(f"   $ {scp_cmd}")
    result = subprocess.run(scp_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   ❌ 失敗: {result.stderr}")
        print(f"   手動: scp {model_path} {VPS_USER}@{VPS_HOST}:{VPS_MODEL_PATH}")
        return False

    print("   ✅ アップロード完了")

    # APIサーバー再起動
    restart_cmd = (
        f'ssh {VPS_USER}@{VPS_HOST} '
        f'"pkill -f \'uvicorn api_server\'; sleep 2; '
        f'cd /opt/keiba-predictor && nohup venv/bin/uvicorn api_server:app '
        f'--host 0.0.0.0 --port 8000 >> /tmp/keiba_api.log 2>&1 &"'
    )
    print("   APIサーバー再起動中...")
    subprocess.run(restart_cmd, shell=True, capture_output=True)
    print(f"   ✅ 完了！ {os.environ.get('KEIBA_PUBLIC_URL', 'http://localhost:8000')}/keiba/ で確認")
    return True


def main():
    parser = argparse.ArgumentParser(description="競馬予測AI ローカル学習")
    parser.add_argument("--advanced", action="store_true",
                        help="拡張特徴量 + ハイパーチューニング（高精度）")
    parser.add_argument("--full", action="store_true",
                        help="全25年分のデータで学習（メモリ16GB以上推奨）")
    parser.add_argument("--years", type=int, default=10,
                        help="直近N年分で学習（デフォルト: 10）")
    parser.add_argument("--splits", type=int, default=5,
                        help="交差検証のfold数（デフォルト: 5）")
    parser.add_argument("--trials", type=int, default=50,
                        help="Optunaチューニング試行回数（デフォルト: 50）")
    parser.add_argument("--no-tune", action="store_true",
                        help="チューニングをスキップ")
    parser.add_argument("--upload", action="store_true",
                        help="学習後にVPSへ自動アップロード")
    args = parser.parse_args()

    print("=" * 60)
    print("🏇 競馬予測AI - ローカル学習")
    print("=" * 60)

    avail = check_dependencies()
    sys_info = get_system_info()
    mem_gb = sys_info["mem_gb"]

    # メモリ警告
    if args.full and mem_gb < 12:
        print(f"\n⚠️ 全データ学習にはメモリ12GB以上推奨（現在 {mem_gb:.1f}GB）")
        resp = input("続行しますか？ (y/N): ").strip().lower()
        if resp != 'y':
            sys.exit(0)

    if args.advanced and not avail.get("optuna") and not args.no_tune:
        print("\n⚠️ --advanced にはOptunaが必要です。--no-tune で続行するかpip install optuna")
        args.no_tune = True

    total_start = time.time()

    if args.advanced:
        model_path, scores, df, cols = train_advanced(
            years=args.years,
            use_full=args.full,
            n_splits=args.splits,
            tune=not args.no_tune,
        )
    else:
        model_path, scores, df, cols = train_standard(
            years=args.years,
            use_full=args.full,
            n_splits=args.splits,
        )

    total_time = time.time() - total_start

    # サマリー
    print("\n" + "=" * 60)
    print("📊 学習結果サマリー")
    print("=" * 60)
    print(f"   モード:     {'高度学習' if args.advanced else '標準学習'}")
    print(f"   データ範囲: {'全25年' if args.full else f'直近{args.years}年'}")
    print(f"   データ量:   {len(df):,}行")
    print(f"   特徴量数:   {len(cols)}個")
    print(f"   学習時間:   {total_time/60:.1f}分")
    print(f"   ─────────────────────────")

    for name, result in scores.items():
        if "mean_auc" in result:
            print(f"   {name}: AUC={result['mean_auc']:.4f} ± {result.get('std_auc',0):.4f}")
        if "mean_mae" in result:
            print(f"   {name}: MAE={result['mean_mae']:.3f} ± {result.get('std_mae',0):.3f}")

    print(f"   ─────────────────────────")
    print(f"   モデル: {model_path}")

    if args.upload:
        upload_model(model_path)
    else:
        print(f"\n   📤 VPSにアップロード:")
        print(f"   scp {model_path} {VPS_USER}@{VPS_HOST}:{VPS_MODEL_PATH}")

    print("\n✅ 完了！")


if __name__ == "__main__":
    main()
