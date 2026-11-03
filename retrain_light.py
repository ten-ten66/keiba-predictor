"""
メモリ節約版モデル再学習
直近N年のデータのみ使用してOOMを回避
"""
import os, sys, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

MIN_YEAR = 2016  # 直近10年分

print("=" * 60)
print("🔄 メモリ節約版 再学習")
print("=" * 60)

# 1. データ読込（直近のみ）
print("\n[1/5] データ読込...")
df = pd.read_csv("data/_all_races.csv", low_memory=False)
print(f"  全データ: {len(df):,} 行")

df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
df = df[df["race_date"].dt.year >= MIN_YEAR].copy()
print(f"  {MIN_YEAR}年以降: {len(df):,} 行")
gc.collect()

# 2. 前処理
from data_processor import RaceDataProcessor, FeatureEngineer
proc = RaceDataProcessor()
proc.raw_data = df
df = proc.preprocess(df)
n_races = df["race_id"].nunique() if "race_id" in df.columns else 0
print(f"  前処理後: {len(df):,} 行 / {n_races:,} レース")
gc.collect()

# 3. 特徴量生成
print("\n[2/5] 特徴量生成...")
fe = FeatureEngineer()
df = fe.create_features(df)
gc.collect()

from features_advanced import AdvancedFeatureEngineer
afe = AdvancedFeatureEngineer()
df = afe.create_all(df)
cols = afe.get_extended_feature_columns(df, [])
print(f"  特徴量: {len(cols)} 個")
gc.collect()

# 4. 学習
print("\n[3/5] モデル学習...")
from model_trainer import RacePredictionModel
model = RacePredictionModel(model_dir="models")
model.train(df, cols, n_splits=5)
gc.collect()

# 5. 検証 & 保存
print("\n[4/5] モデル検証...")
new_auc = 0
for name, scores in model.evaluation_results.items():
    if "mean_auc" in scores:
        new_auc = max(new_auc, scores["mean_auc"])
        print(f"  {name}: AUC={scores['mean_auc']:.4f}")

print(f"\n[5/5] モデル保存...")
# バックアップ
import shutil
from pathlib import Path
model_dir = Path("models")
current = model_dir / "keiba_model.pkl"
if current.exists():
    shutil.copy2(current, model_dir / "keiba_model_prev.pkl")

model.save("keiba_model")

# auto_learn state更新
from datetime import datetime
from auto_learn import AutoLearnState
state = AutoLearnState()
version = datetime.now().strftime("v%Y%m%d_%H%M%S")
state.mark_retrained(version, new_auc, n_races)

print(f"\n✅ 完了: {version}")
print(f"  AUC: {new_auc:.4f}")
print(f"  データ: {n_races:,} レース / {len(cols)} 特徴量")
print(f"  期間: {MIN_YEAR}年〜")
