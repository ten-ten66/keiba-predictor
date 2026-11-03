#!/usr/bin/env python3
"""
Local Backtest Script
=====================
Walk-forward backtest using the trained model.
Trains on 2001-2024, tests on 2025.
Requires: data/_all_races.csv, data/payouts.csv, models/keiba_model.pkl

Usage:
    python backtest_local.py
    python backtest_local.py --test-year 2024
    python backtest_local.py --detailed
"""
import argparse
import os
import sys
import warnings
import gc

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Local Backtest")
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    parser.add_argument("--model", default="models/keiba_model.pkl")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  Walk-Forward Backtest (test year: {args.test_year})")
    print("=" * 70)

    from model_trainer import RacePredictionModel
    from data_processor import RaceDataProcessor, FeatureEngineer

    # 1. Load model
    print("\n[1/4] Loading model...")
    m = RacePredictionModel()
    name = os.path.splitext(os.path.basename(args.model))[0]
    m.load(name)
    print(f"  v{m.MODEL_VERSION}, {len(m.feature_columns)} features")

    # 2. Load data (test year + 1 year before for Elo warmup)
    print("\n[2/4] Loading data...")
    df = pd.read_csv(
        "data/_all_races.csv",
        dtype={"race_id": str, "horse_id": str, "jockey_id": str, "trainer_id": str},
        low_memory=False,
    )
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    warmup_start = args.test_year - 2
    df = df[df["race_date"].dt.year >= warmup_start].copy()
    print(f"  {len(df):,} rows ({warmup_start}-{args.test_year})")

    # 3. Feature engineering
    print("\n[3/4] Feature engineering...")
    proc = RaceDataProcessor()
    proc.raw_data = df
    df_proc = proc.preprocess()
    del df
    gc.collect()

    fe = FeatureEngineer()
    df_proc = fe.create_features(df_proc)

    # Advanced features
    try:
        from features_advanced import AdvancedFeatureEngineer
        afe = AdvancedFeatureEngineer()
        df_proc = afe.create_all(df_proc)
    except Exception as e:
        print(f"  Advanced features skipped: {e}")

    # Split test year
    df_test = df_proc[df_proc["race_date"].dt.year == args.test_year].copy()
    n_races = df_test["race_id"].nunique()
    print(f"  Test: {len(df_test):,} rows, {n_races} races")

    if len(df_test) == 0:
        print("ERROR: No test data")
        sys.exit(1)

    # 4. Predict & Evaluate
    print("\n[4/4] Predicting & evaluating...")
    preds = m.predict(df_test)

    # Attach actual results
    preds["odds"] = df_test["odds"].values
    preds["finish_position"] = df_test["finish_position"].values
    preds["popularity"] = df_test["popularity"].values if "popularity" in df_test.columns else 0
    preds["ev"] = preds["prob_win"] * preds["odds"]

    # Load payouts
    payouts_dict = {}
    try:
        pay_df = pd.read_csv("data/payouts.csv", dtype={"race_id": str})
        for _, row in pay_df.iterrows():
            rid = row["race_id"]
            if rid not in payouts_dict:
                payouts_dict[rid] = {}
            bt = row["bet_type"]
            if bt not in payouts_dict[rid]:
                payouts_dict[rid][bt] = []
            payouts_dict[rid][bt].append({
                "combination": str(row.get("combination", "")),
                "payout": float(row.get("payout", 0)),
            })
        print(f"  Payouts: {len(payouts_dict)} races")
    except Exception as e:
        print(f"  Payouts not loaded: {e}")

    # ===============================
    # Results
    # ===============================
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # --- prob_win distribution ---
    print("\n--- prob_win distribution ---")
    for q in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {q}%ile: {preds['prob_win'].quantile(q / 100):.4f}")

    # --- Calibration ---
    print("\n--- prob_win calibration ---")
    preds["prob_bin"] = pd.cut(
        preds["prob_win"],
        bins=[0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
    )
    for bl, g in preds.groupby("prob_bin", observed=True):
        actual = (g["finish_position"] == 1).mean()
        predicted = g["prob_win"].mean()
        n = len(g)
        if n > 20:
            ratio = actual / predicted if predicted > 0 else 0
            print(
                f"  {str(bl):15s}: pred={predicted:.4f} actual={actual:.4f} "
                f"ratio={ratio:.2f} n={n}"
            )

    # --- EV thresholds ---
    print("\n--- EV threshold analysis (tansho) ---")
    print(f"  {'Thresh':>6s} {'ROI':>8s} {'WinRate':>8s} {'Bets':>6s} {'Wins':>5s} {'AvgOdds':>8s}")
    print("  " + "-" * 48)
    for t in [0.8, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
        b = preds[preds["ev"] > t]
        if len(b) == 0:
            continue
        w = (b["finish_position"] == 1).sum()
        tb = len(b) * 100
        tr = b[b["finish_position"] == 1]["odds"].sum() * 100
        roi = (tr / tb - 1) * 100
        wr = w / len(b) * 100
        ao = b[b["finish_position"] == 1]["odds"].mean() if w > 0 else 0
        print(f"  EV>{t:<4} {roi:>+7.1f}% {wr:>7.1f}% {len(b):>6} {w:>5} {ao:>7.1f}")

    # --- Predicted rank ---
    print("\n--- predicted_rank tansho ---")
    for r in [1, 2, 3, 4, 5]:
        b = preds[preds["predicted_rank"] == r]
        if len(b) == 0:
            continue
        w = (b["finish_position"] == 1).sum()
        tb, tr = len(b) * 100, b[b["finish_position"] == 1]["odds"].sum() * 100
        roi = (tr / tb - 1) * 100 if tb > 0 else 0
        wr = w / len(b) * 100 if len(b) > 0 else 0
        print(f"  rank={r}: ROI={roi:+.1f}% win={wr:.1f}% n={len(b)} wins={w}")

    # --- Fukusho (place) ---
    print("\n--- predicted_rank fukusho (top3) ---")
    for r in [1, 2, 3]:
        b = preds[preds["predicted_rank"] == r]
        h = (b["finish_position"] <= 3).sum()
        hr = h / len(b) * 100 if len(b) > 0 else 0
        # Fukusho payout
        total_return = 0
        for _, row in b[b["finish_position"] <= 3].iterrows():
            rid = row["race_id"]
            hn = int(row["horse_number"])
            if rid in payouts_dict and "複勝" in payouts_dict[rid]:
                for p in payouts_dict[rid]["複勝"]:
                    if str(hn) in str(p["combination"]):
                        total_return += p["payout"]
                        break
        tb = len(b) * 100
        roi = (total_return / tb - 1) * 100 if tb > 0 else 0
        print(f"  rank={r}: hit={hr:.1f}% ROI={roi:+.1f}% n={len(b)} hits={h}")

    # --- Combined strategies ---
    print("\n--- Combined strategies ---")

    # Strategy: rank=1 + EV>1.2
    b = preds[(preds["predicted_rank"] == 1) & (preds["ev"] > 1.2)]
    if len(b) > 0:
        w = (b["finish_position"] == 1).sum()
        tr = b[b["finish_position"] == 1]["odds"].sum() * 100
        roi = (tr / (len(b) * 100) - 1) * 100
        print(f"  rank=1 & EV>1.2: ROI={roi:+.1f}% win={w / len(b) * 100:.1f}% n={len(b)}")

    # Strategy: rank=1 + EV>1.3
    b = preds[(preds["predicted_rank"] == 1) & (preds["ev"] > 1.3)]
    if len(b) > 0:
        w = (b["finish_position"] == 1).sum()
        tr = b[b["finish_position"] == 1]["odds"].sum() * 100
        roi = (tr / (len(b) * 100) - 1) * 100
        print(f"  rank=1 & EV>1.3: ROI={roi:+.1f}% win={w / len(b) * 100:.1f}% n={len(b)}")

    # Strategy: rank<=2 & EV>1.5
    b = preds[(preds["predicted_rank"] <= 2) & (preds["ev"] > 1.5)]
    if len(b) > 0:
        w = (b["finish_position"] == 1).sum()
        tr = b[b["finish_position"] == 1]["odds"].sum() * 100
        roi = (tr / (len(b) * 100) - 1) * 100
        print(f"  rank<=2 & EV>1.5: ROI={roi:+.1f}% win={w / len(b) * 100:.1f}% n={len(b)}")

    # Strategy: prob_win > median & odds > 5
    med = preds["prob_win"].median()
    b = preds[(preds["prob_win"] > med) & (preds["odds"] > 5)]
    if len(b) > 0:
        w = (b["finish_position"] == 1).sum()
        tr = b[b["finish_position"] == 1]["odds"].sum() * 100
        roi = (tr / (len(b) * 100) - 1) * 100
        print(f"  prob_win>median & odds>5: ROI={roi:+.1f}% win={w / len(b) * 100:.1f}% n={len(b)}")

    # Strategy: composite_score top in race & odds >= 3
    top1 = preds[preds["predicted_rank"] == 1]
    b = top1[top1["odds"] >= 3]
    if len(b) > 0:
        w = (b["finish_position"] == 1).sum()
        tr = b[b["finish_position"] == 1]["odds"].sum() * 100
        roi = (tr / (len(b) * 100) - 1) * 100
        print(f"  rank=1 & odds>=3: ROI={roi:+.1f}% win={w / len(b) * 100:.1f}% n={len(b)}")

    # Strategy: rank=1 & NOT 1番人気 (contrarian)
    b = preds[(preds["predicted_rank"] == 1) & (preds["popularity"] != 1)]
    if len(b) > 0:
        w = (b["finish_position"] == 1).sum()
        tr = b[b["finish_position"] == 1]["odds"].sum() * 100
        roi = (tr / (len(b) * 100) - 1) * 100
        print(f"  rank=1 & NOT 1番人気: ROI={roi:+.1f}% win={w / len(b) * 100:.1f}% n={len(b)}")

    # Strategy: ev_std (uncertainty) high & EV > 1.3
    if "ev_std" in preds.columns:
        ev_std_med = preds["ev_std"].median()
        b = preds[(preds["ev_std"] > ev_std_med) & (preds["ev"] > 1.3)]
        if len(b) > 0:
            w = (b["finish_position"] == 1).sum()
            tr = b[b["finish_position"] == 1]["odds"].sum() * 100
            roi = (tr / (len(b) * 100) - 1) * 100
            print(f"  high_uncertainty & EV>1.3: ROI={roi:+.1f}% win={w / len(b) * 100:.1f}% n={len(b)}")

    if args.detailed:
        # --- Pop x Rank cross ---
        print("\n--- Popularity x Predicted Rank ---")
        print(f"  {'Pop':>3s} {'Rank':>4s} {'WinRate':>8s} {'ROI':>8s} {'N':>5s}")
        for pop in [1, 2, 3, 4, 5]:
            for rank in [1, 2, 3]:
                sub = preds[
                    (preds["popularity"] == pop) & (preds["predicted_rank"] == rank)
                ]
                if len(sub) < 10:
                    continue
                w = (sub["finish_position"] == 1).sum()
                wr = w / len(sub) * 100
                tr = sub[sub["finish_position"] == 1]["odds"].sum() * 100
                roi = (tr / (len(sub) * 100) - 1) * 100
                print(f"  {pop:>3} {rank:>4} {wr:>7.1f}% {roi:>+7.1f}% {len(sub):>5}")

        # --- Monthly breakdown ---
        print("\n--- Monthly ROI (rank=1 tansho) ---")
        top1 = preds[preds["predicted_rank"] == 1].copy()
        top1["month"] = pd.to_datetime(
            df_test.loc[top1.index, "race_date"] if "race_date" in df_test.columns else "2025-01-01"
        ).dt.strftime("%Y-%m")
        for month, g in top1.groupby("month"):
            w = (g["finish_position"] == 1).sum()
            tr = g[g["finish_position"] == 1]["odds"].sum() * 100
            roi = (tr / (len(g) * 100) - 1) * 100 if len(g) > 0 else 0
            print(f"  {month}: ROI={roi:+.1f}% win={w}/{len(g)} ({w / len(g) * 100:.1f}%)")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
