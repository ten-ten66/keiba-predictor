"""軽量精度テスト - API経由で50レースを予測して精度検証"""
import os, sys, json, random, requests
sys.path.insert(0, ".")
import pandas as pd
import numpy as np

# 2025年のレースから50件サンプル
df = pd.read_csv("data/_all_races.csv", dtype=str, usecols=["race_id","horse_number","finish_position","race_date"])
df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
df25 = df[df["race_date"].dt.year == 2025].copy()
df25["finish_position"] = pd.to_numeric(df25["finish_position"], errors="coerce")
df25 = df25.dropna(subset=["finish_position"])

race_sizes = df25.groupby("race_id").size()
valid = race_sizes[race_sizes >= 8].index.tolist()
random.seed(42)
sample = random.sample(valid, min(50, len(valid)))

# ログイン
s = requests.Session()
r = s.post("http://localhost:8000/api/auth/login", json={"username":"admin","password":os.environ["KEIBA_ADMIN_PASSWORD"]})
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

top1_correct = 0
top3_winner = 0
top3_overlap_total = 0
tested = 0

print(f"Testing {len(sample)} races...")
for i, rid in enumerate(sample):
    try:
        r = s.post("http://localhost:8000/api/predict", json={"race_id": rid}, headers=headers, timeout=30)
        if r.status_code != 200:
            continue
        preds = r.json().get("predictions", [])
        if len(preds) < 3:
            continue
        
        # Actual results
        race_df = df25[df25["race_id"] == rid]
        actual_winner = race_df[race_df["finish_position"] == 1]["horse_number"].values
        actual_top3 = set(race_df[race_df["finish_position"] <= 3]["horse_number"].astype(str).values)
        
        if len(actual_winner) == 0 or len(actual_top3) < 3:
            continue
        
        actual_winner_num = str(actual_winner[0])
        pred_top1 = str(preds[0]["horse_number"])
        pred_top3 = set(str(p["horse_number"]) for p in preds[:3])
        
        tested += 1
        if pred_top1 == actual_winner_num:
            top1_correct += 1
        if actual_winner_num in pred_top3:
            top3_winner += 1
        overlap = len(pred_top3 & actual_top3)
        top3_overlap_total += overlap
        
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(sample)} tested...")
    except Exception as e:
        continue

print(f"\n{'='*50}")
print(f"  精度テスト結果 ({tested}レース)")
print(f"{'='*50}")
print(f"  1着的中率:        {top1_correct}/{tested} = {top1_correct/tested*100:.1f}%")
print(f"  Top3に1着馬:      {top3_winner}/{tested} = {top3_winner/tested*100:.1f}%")
print(f"  Top3一致数平均:   {top3_overlap_total/tested:.2f}/3")
print(f"  Top3一致率:       {top3_overlap_total/(tested*3)*100:.1f}%")

# ランダム比較（ベースライン）
avg_field = race_sizes[race_sizes >= 8].mean()
random_top1 = 1/avg_field * 100
random_top3 = min(3/avg_field, 1) * 100 * 3 / avg_field  # 実際はハイパー幾何分布
print(f"\n  参考: ランダム1着的中率 ≈ {random_top1:.1f}% (平均{avg_field:.0f}頭)")
