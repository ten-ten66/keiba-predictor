#!/usr/bin/env python3
"""Local training script for keiba-predictor v8-optimized"""
import argparse, os, sys, time, json
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Keiba AI Model Training')
    parser.add_argument('--data', default='data/_all_races.csv')
    parser.add_argument('--output', default='models/keiba_model.pkl')
    parser.add_argument('--min-year', type=int, default=2018)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--blend', action='store_true', default=True, help='Use LGB+XGB+CatBoost blend')
    parser.add_argument('--no-blend', action='store_false', dest='blend')
    parser.add_argument('--half-life', type=int, default=365, help='Time decay half-life in days')
    parser.add_argument('--advanced-features', action='store_true', default=True)
    parser.add_argument('--no-advanced-features', action='store_false', dest='advanced_features')
    parser.add_argument('--smote', action='store_true', default=True, help='Apply SMOTE for win model')
    parser.add_argument('--no-smote', action='store_false', dest='smote')
    parser.add_argument('--ordinal', action='store_true', default=True, help='Train ordinal regression')
    parser.add_argument('--no-ordinal', action='store_false', dest='ordinal')
    parser.add_argument('--shap-prune', action='store_true', default=True, help='SHAP feature pruning')
    parser.add_argument('--no-shap-prune', action='store_false', dest='shap_prune')
    parser.add_argument('--adversarial', action='store_true', default=True, help='Adversarial validation')
    parser.add_argument('--no-adversarial', action='store_false', dest='adversarial')
    args = parser.parse_args()

    print("=" * 60)
    print("  Keiba AI - Model Training v8-optimized")
    print("=" * 60)
    model_types = ['lgbm', 'xgb', 'catboost'] if args.blend else ['lgbm']
    print(f"  Data:         {args.data}")
    print(f"  Models:       {model_types}")
    print(f"  Features:     {'Advanced' if args.advanced_features else 'Basic'}")
    print(f"  Min year:     {args.min_year}")
    print(f"  CV folds:     {args.folds}")
    print(f"  Time decay:   half-life {args.half_life} days")
    print(f"  SMOTE:        {'ON' if args.smote else 'OFF'}")
    print(f"  Ordinal:      {'ON' if args.ordinal else 'OFF'}")
    print(f"  SHAP prune:   {'ON' if args.shap_prune else 'OFF'}")
    print(f"  Adversarial:  {'ON' if args.adversarial else 'OFF'}")
    print(f"  Output:       {args.output}")
    print("=" * 60)

    import pandas as pd
    import numpy as np
    from data_processor import RaceDataProcessor, FeatureEngineer
    from model_trainer import RacePredictionModel

    # [1/6] Load data
    print("\n[1/6] Loading data...")
    if not os.path.exists(args.data):
        print(f"ERROR: {args.data} not found"); sys.exit(1)
    rp = RaceDataProcessor()
    rp.raw_data = pd.read_csv(args.data)
    print(f"  Loaded {len(rp.raw_data):,} records")
    df = rp.preprocess()

    # [2/6] Feature engineering
    # 注意 (2026-07-02): 特徴量は「全期間」で計算してから min_year で学習行を
    # 絞り込む。以前は先に行を削ってから特徴量を計算していたため、
    # 馬の通算成績・Elo などの履歴特徴量が min_year 以降だけで再スタートし、
    # 全履歴で特徴量を計算するサーブ時 (api_server) と分布がズレていた。
    print("\n[2/6] Feature engineering (全期間で履歴特徴量を計算)...")
    fe = FeatureEngineer()
    df = fe.create_features(df)
    feature_cols = fe.get_feature_columns(df)

    if args.advanced_features:
        try:
            from features_advanced import AdvancedFeatureEngineer
            afe = AdvancedFeatureEngineer()
            df = afe.create_all(df)
            feature_cols = afe.get_extended_feature_columns(df, feature_cols)
        except Exception as e:
            print(f"  WARNING: Advanced features failed: {e}")

    # Try integrating extra data
    try:
        from integrate_extra_data import integrate_training, integrate_marks
        df = integrate_training(df)
        df = integrate_marks(df)
        feature_cols = fe.get_feature_columns(df)
    except Exception as e:
        print(f"  Note: Extra data integration skipped: {e}")

    print(f"  {len(feature_cols)} features ready")

    # 学習対象行の絞り込み (特徴量計算後に実施)
    if 'race_date' in df.columns:
        df['race_date'] = pd.to_datetime(df['race_date'], errors='coerce')
        before = len(df)
        df = df[df['race_date'].dt.year >= args.min_year]
        print(f"  Filtered to >= {args.min_year}: {before:,} -> {len(df):,} rows")

    n_races = df['race_id'].nunique() if 'race_id' in df.columns else 0
    print(f"  {len(df):,} rows / {n_races:,} races")

    # [3/6] Time decay weights
    print("\n[3/6] Computing time decay weights...")
    sample_weights = None
    if 'race_date' in df.columns and args.half_life > 0:
        max_date = df['race_date'].max()
        days_ago = (max_date - df['race_date']).dt.days.fillna(0)
        decay = np.exp(-np.log(2) * days_ago / args.half_life)
        sample_weights = np.clip(decay, 0.05, 1.0)
        print(f"  Half-life: {args.half_life} days")
        print(f"  Weight range: {sample_weights.min():.4f} - {sample_weights.max():.4f}")

    # [4/6] Train
    print("\n[4/6] Training models...")
    model = RacePredictionModel()
    train_df = df[df['finish_position'].notna()].copy()

    train_kwargs = {
        'n_splits': args.folds,
        'time_decay_halflife': args.half_life if sample_weights is not None else None,
    }
    if args.blend:
        train_kwargs['ensemble_models'] = model_types

    results = model.train(train_df, feature_cols, **train_kwargs)

    # Print results
    print("\n" + "=" * 60)
    print("  Training Results")
    print("=" * 60)
    for task, scores in results.items():
        print(f"  [{task}]")
        for k, v in scores.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")

    # [5/6] Save
    print(f"\n[5/6] Saving model...")
    os.makedirs(os.path.dirname(args.output) or 'models', exist_ok=True)
    name = os.path.splitext(os.path.basename(args.output))[0]
    model.save(name)

    # [6/6] Summary
    print(f"\n[6/6] Done!")
    ev = model.evaluation_results
    print("=" * 60)
    print(f"  Model:    {args.output}")
    print(f"  Version:  v{model.MODEL_VERSION}")
    print(f"  Races:    {n_races:,}")
    print(f"  Features: {len(model.feature_columns)}")
    if 'position' in ev:
        print(f"  MAE:      {ev['position'].get('mean_mae', 'N/A'):.4f}")
    if 'top3' in ev:
        print(f"  Top3 AUC: {ev['top3'].get('mean_auc', 'N/A'):.4f}")
    if 'win' in ev:
        print(f"  Win AUC:  {ev['win'].get('mean_auc', 'N/A'):.4f}")
    if 'ranking' in ev:
        print(f"  NDCG:     {ev['ranking'].get('mean_ndcg', 'N/A')}")
    print("=" * 60)

if __name__ == '__main__':
    main()
