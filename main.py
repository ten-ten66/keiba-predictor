"""
競馬予想AI - メインパイプライン
================================
全処理を一括実行するエントリーポイント。

使い方:
  # サンプルデータでテスト
  python main.py --demo

  # 自前のCSVデータで学習
  python main.py --data data/my_races.csv

  # 学習済みモデルで予測
  python main.py --predict --model models/keiba_model.pkl --data data/next_race.csv
"""

import argparse
import sys
import os
import json

# スクリプトディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processor import RaceDataProcessor, FeatureEngineer, SampleDataGenerator
from model_trainer import RacePredictionModel, export_results_json


def run_demo():
    """デモ実行: サンプルデータで全パイプラインをテスト"""
    print("🏇" * 20)
    print("  競馬予想AI - デモモード")
    print("🏇" * 20)

    # データ生成
    print("\n[1/5] サンプルデータ生成中...")
    gen = SampleDataGenerator()
    raw_df = gen.generate(n_races=1000)
    os.makedirs("data", exist_ok=True)
    raw_df.to_csv("data/sample_races.csv", index=False)

    # 前処理
    print("\n[2/5] データ前処理中...")
    processor = RaceDataProcessor()
    processor.raw_data = raw_df
    df = processor.preprocess()

    # 特徴量生成
    print("\n[3/5] 特徴量エンジニアリング中...")
    fe = FeatureEngineer()
    df = fe.create_features(df)
    feature_cols = fe.get_feature_columns(df)

    # 学習・評価分割（最後の200レース分をテスト用）
    race_ids = df["race_id"].unique()
    split_idx = len(race_ids) - 200
    train_races = set(race_ids[:split_idx])
    test_races = set(race_ids[split_idx:])

    train_df = df[df["race_id"].isin(train_races)]
    test_df = df[df["race_id"].isin(test_races)]

    print(f"   学習データ: {len(train_df)} 件 ({len(train_races)} レース)")
    print(f"   テストデータ: {len(test_df)} 件 ({len(test_races)} レース)")

    # 学習
    print("\n[4/5] モデル学習中...")
    model = RacePredictionModel()
    model.train(train_df, feature_cols)

    # 予測 & 評価
    print("\n[5/5] テストデータで予測中...")
    predictions = model.predict(test_df)

    # 的中率計算
    merged = predictions.merge(
        test_df[["race_id", "horse_number", "finish_position"]],
        on=["race_id", "horse_number"],
    )

    # 予測1位が実際に1着だった割合
    top1_preds = merged[merged["predicted_rank"] == 1]
    top1_accuracy = (top1_preds["finish_position"] == 1).mean()

    # 予測上位3頭に実際の1着が含まれる割合
    top3_preds = merged[merged["predicted_rank"] <= 3]
    top3_accuracy = top3_preds.groupby("race_id").apply(
        lambda x: (x["finish_position"] == 1).any()
    ).mean()

    # 予測上位3頭が実際に3着以内に入る割合
    top3_in_top3 = (top3_preds["finish_position"] <= 3).mean()

    print("\n" + "=" * 60)
    print("📊 テスト結果")
    print("=" * 60)
    print(f"  予測1位→1着的中率:       {top1_accuracy*100:.1f}%")
    print(f"  上位3頭に1着馬含有率:    {top3_accuracy*100:.1f}%")
    print(f"  上位3頭の複勝的中率:     {top3_in_top3*100:.1f}%")

    # 回収率シミュレーション
    print("\n💰 回収率シミュレーション:")
    for strategy in ["top1_win", "top3_wide", "value_bet"]:
        result = model.simulate_betting(predictions, test_df, strategy=strategy)
        print(f"\n  【{result['strategy']}】")
        print(f"    投資額: ¥{result['total_bet_amount']:,.0f}")
        print(f"    回収額: ¥{result['total_return']:,.0f}")
        print(f"    回収率: {result['roi_percent']}%")
        print(f"    的中率: {result['win_rate_percent']}%")

    # モデル保存
    os.makedirs("models", exist_ok=True)
    model.save("demo_model")
    export_results_json(predictions, "data/predictions.json")

    # UI用サマリー生成
    summary = {
        "model_info": {
            "trained_at": "demo",
            "train_records": len(train_df),
            "test_records": len(test_df),
            "n_features": len(feature_cols),
        },
        "accuracy": {
            "top1_win_rate": round(top1_accuracy * 100, 1),
            "top3_contains_winner": round(top3_accuracy * 100, 1),
            "top3_place_rate": round(top3_in_top3 * 100, 1),
        },
        "evaluation": model.evaluation_results,
        "top_features": {
            name: model.get_top_features(name, 15).to_dict("records")
            for name in ["position", "top3", "win"]
        },
    }
    with open("data/model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ デモ完了！")
    print("   📁 data/predictions.json    - 予測結果")
    print("   📁 data/model_summary.json  - モデルサマリー")
    print("   📁 models/demo_model.pkl    - 学習済みモデル")


def run_train(data_path: str, model_name: str = "keiba_model"):
    """自前データで学習"""
    print(f"📂 データ読み込み: {data_path}")
    processor = RaceDataProcessor(data_dir=os.path.dirname(data_path))
    df = processor.load_csv(os.path.basename(data_path))
    df = processor.preprocess(df)

    fe = FeatureEngineer()
    df = fe.create_features(df)
    feature_cols = fe.get_feature_columns(df)

    model = RacePredictionModel()
    model.train(df, feature_cols)
    model.save(model_name)

    print(f"\n✅ 学習完了: models/{model_name}.pkl")


def run_predict(data_path: str, model_name: str = "keiba_model"):
    """学習済みモデルで予測"""
    model = RacePredictionModel()
    model.load(model_name)

    processor = RaceDataProcessor(data_dir=os.path.dirname(data_path))
    df = processor.load_csv(os.path.basename(data_path))
    df = processor.preprocess(df)

    fe = FeatureEngineer()
    df = fe.create_features(df)

    predictions = model.predict(df)
    export_results_json(predictions, "data/predictions.json")

    print("\n🏇 予測結果:")
    for race_id, group in predictions.groupby("race_id"):
        print(f"\n  【{race_id}】")
        for _, row in group.head(5).iterrows():
            name = row.get("horse_name", f"馬{int(row['horse_number'])}")
            print(
                f"    {row['predicted_rank']:2d}位: {name} "
                f"(勝率{row.get('prob_win', 0)*100:.1f}% / "
                f"複勝{row.get('prob_top3', 0)*100:.1f}%)"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競馬予想AI")
    parser.add_argument("--demo", action="store_true", help="サンプルデータでデモ実行")
    parser.add_argument("--train", action="store_true", help="モデル学習")
    parser.add_argument("--predict", action="store_true", help="予測実行")
    parser.add_argument("--data", type=str, default="data/sample_races.csv", help="入力データパス")
    parser.add_argument("--model", type=str, default="keiba_model", help="モデル名")

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.train:
        run_train(args.data, args.model)
    elif args.predict:
        run_predict(args.data, args.model)
    else:
        # デフォルトはデモ
        run_demo()
