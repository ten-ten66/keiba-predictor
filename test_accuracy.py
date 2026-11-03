"""
Prediction Accuracy Test
========================
Tests the trained model against ~100 random 2025 races.

Metrics:
  1. Top-1 accuracy: predicted 1st == actual 1st
  2. Top-3 accuracy: actual winner is in predicted top 3
  3. Top-3 hit rate: how many of predicted top 3 were in actual top 3
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_trainer import RacePredictionModel
from data_processor import RaceDataProcessor, FeatureEngineer


def main():
    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 65)
    print("  Prediction Accuracy Test")
    print("=" * 65)

    csv_path = Path("data/_all_races.csv")
    print(f"\nLoading data from {csv_path} ...")

    processor = RaceDataProcessor(data_dir="data")
    df = processor.load_csv("_all_races.csv")

    # ------------------------------------------------------------------
    # 2. Preprocess + feature engineering (on 2019+ data for speed)
    # ------------------------------------------------------------------
    print("\nPreprocessing ...")
    df = processor.preprocess(df)

    # Trim to 2019+ for speed (enough history for rolling features + Elo)
    if "race_date" in df.columns:
        df = df[df["race_date"] >= "2019-01-01"].copy()
        df = df.reset_index(drop=True)
        print(f"  Trimmed to 2019+: {len(df)} rows")

    print("\nFeature engineering ...")
    fe = FeatureEngineer()
    df = fe.create_features(df)
    feature_cols = fe.get_feature_columns(df)
    print(f"  Feature columns: {len(feature_cols)}")

    # ------------------------------------------------------------------
    # 3. Load trained model
    # ------------------------------------------------------------------
    print("\nLoading model ...")
    model = RacePredictionModel(model_dir="models")
    model.load("keiba_model")

    # Align feature columns: use the model's stored feature list
    # (some columns may be missing in this dataset or vice-versa)
    model_features = set(model.feature_columns)
    data_features = set(feature_cols)
    common_features = sorted(model_features & data_features)

    missing_in_data = model_features - data_features
    extra_in_data = data_features - model_features

    if missing_in_data:
        print(f"\n  WARNING: {len(missing_in_data)} model features missing in data:")
        for f in sorted(missing_in_data)[:10]:
            print(f"    - {f}")
        # Add missing columns filled with 0
        for f in missing_in_data:
            df[f] = 0.0

    if extra_in_data:
        print(f"\n  INFO: {len(extra_in_data)} data features not used by model (ignored)")

    # ------------------------------------------------------------------
    # 4. Filter 2025 races
    # ------------------------------------------------------------------
    if "race_date" not in df.columns:
        print("ERROR: race_date column missing after preprocessing")
        sys.exit(1)

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df_2025 = df[df["race_date"].dt.year == 2025].copy()

    if df_2025.empty:
        print("ERROR: No 2025 race data found. Checking available years ...")
        years = df["race_date"].dt.year.dropna().unique()
        print(f"  Available years: {sorted(years)}")
        # Fall back to latest year
        latest_year = int(sorted(years)[-1])
        print(f"  Falling back to year {latest_year}")
        df_2025 = df[df["race_date"].dt.year == latest_year].copy()

    race_ids_2025 = df_2025["race_id"].unique()
    print(f"\n  Found {len(race_ids_2025)} races in test year with {len(df_2025)} entries")

    # Filter races with at least 5 runners (meaningful race)
    race_sizes = df_2025.groupby("race_id").size()
    valid_race_ids = race_sizes[race_sizes >= 5].index.values
    print(f"  Races with >= 5 runners: {len(valid_race_ids)}")

    # ------------------------------------------------------------------
    # 5. Sample ~100 races
    # ------------------------------------------------------------------
    n_sample = min(100, len(valid_race_ids))
    sampled_ids = np.random.choice(valid_race_ids, size=n_sample, replace=False)
    df_test = df_2025[df_2025["race_id"].isin(sampled_ids)].copy()
    print(f"\n  Sampled {n_sample} races ({len(df_test)} entries) for testing")

    # ------------------------------------------------------------------
    # 6. Run predictions
    # ------------------------------------------------------------------
    print("\nRunning predictions ...")
    predictions = model.predict(df_test)

    # Merge actual finish_position back
    actuals = df_test[["race_id", "horse_number", "finish_position"]].copy()
    merged = predictions.merge(actuals, on=["race_id", "horse_number"], how="inner")

    # ------------------------------------------------------------------
    # 7. Compute accuracy metrics
    # ------------------------------------------------------------------
    top1_correct = 0
    top3_winner_found = 0
    top3_overlap_total = 0
    top3_overlap_count = 0
    total_races = 0

    for race_id, race in merged.groupby("race_id"):
        race_sorted = race.sort_values("predicted_rank")

        # Actual results
        actual_winner = race.loc[race["finish_position"] == 1]
        actual_top3 = set(race.loc[race["finish_position"] <= 3, "horse_number"].values)

        # Predicted results
        predicted_top1 = race_sorted.iloc[0]
        predicted_top3 = set(race_sorted.head(3)["horse_number"].values)

        if actual_winner.empty or len(actual_top3) < 3:
            continue  # Skip races with incomplete results

        total_races += 1

        # Metric 1: Top-1 accuracy
        if predicted_top1["finish_position"] == 1:
            top1_correct += 1

        # Metric 2: Top-3 accuracy (winner in predicted top 3)
        actual_winner_number = actual_winner.iloc[0]["horse_number"]
        if actual_winner_number in predicted_top3:
            top3_winner_found += 1

        # Metric 3: Top-3 hit rate (overlap of predicted top 3 and actual top 3)
        overlap = len(predicted_top3 & actual_top3)
        top3_overlap_total += overlap
        top3_overlap_count += 1

    # ------------------------------------------------------------------
    # 8. Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    if total_races == 0:
        print("  No valid races to evaluate!")
        sys.exit(1)

    top1_acc = top1_correct / total_races * 100
    top3_acc = top3_winner_found / total_races * 100
    top3_hit = top3_overlap_total / (top3_overlap_count * 3) * 100 if top3_overlap_count else 0

    avg_overlap = top3_overlap_total / top3_overlap_count if top3_overlap_count else 0

    print(f"\n  Races evaluated:  {total_races}")
    print(f"\n  [1] Top-1 Accuracy (pred 1st == actual 1st)")
    print(f"      {top1_correct} / {total_races} = {top1_acc:.1f}%")

    print(f"\n  [2] Top-3 Accuracy (actual winner in predicted top 3)")
    print(f"      {top3_winner_found} / {total_races} = {top3_acc:.1f}%")

    print(f"\n  [3] Top-3 Hit Rate (predicted top 3 vs actual top 3)")
    print(f"      Avg overlap: {avg_overlap:.2f} / 3 = {top3_hit:.1f}%")

    # Baseline: random chance
    print("\n  --- Baseline (random chance) ---")
    # For a race with N horses:
    # P(top1 correct) = 1/N
    # P(winner in top3) = 3/N
    # E(overlap of top3) = 3*3/N = 9/N
    avg_runners = merged.groupby("race_id").size().mean()
    print(f"  Avg runners per race: {avg_runners:.1f}")
    print(f"  Random Top-1:         {1/avg_runners*100:.1f}%")
    print(f"  Random Top-3 winner:  {3/avg_runners*100:.1f}%")
    print(f"  Random Top-3 overlap: {3/avg_runners*100:.1f}%")

    # Additional: breakdown by popularity
    print("\n  --- Top-1 breakdown (predicted 1st's actual finish) ---")
    top1_rows = []
    for race_id, race in merged.groupby("race_id"):
        race_sorted = race.sort_values("predicted_rank")
        top1_rows.append(race_sorted.iloc[0])
    if top1_rows:
        top1_df = pd.DataFrame(top1_rows)
        pos_dist = top1_df["finish_position"].describe()
        print(f"      Mean actual position: {pos_dist['mean']:.2f}")
        print(f"      Median actual position: {pos_dist['50%']:.1f}")
        print(f"      Finished 1st:  {(top1_df['finish_position'] == 1).sum()}")
        print(f"      Finished top3: {(top1_df['finish_position'] <= 3).sum()}")
        print(f"      Finished top5: {(top1_df['finish_position'] <= 5).sum()}")

    print("\n" + "=" * 65)
    print("  Test complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
