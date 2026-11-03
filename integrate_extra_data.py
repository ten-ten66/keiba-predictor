"""
追加データ（血統・調教・払戻・予想印）をメインCSVに統合するスクリプト。
統合後に再学習をトリガーする。

Usage:
    python integrate_extra_data.py [--retrain]
"""

import os
import sys
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = os.environ.get("KEIBA_DATA_DIR", "data")
MAIN_CSV = os.path.join(DATA_DIR, "_all_races.csv")

# 予想印スコア定義（◎=5, ○=4, ▲=3, △=2, ×=1）
MARK_SCORE_MAP = {"◎": 5, "○": 4, "▲": 3, "△": 2, "×": 1, "☆": 2}


def integrate_bloodline(df: pd.DataFrame) -> pd.DataFrame:
    """血統データを統合（horse_id でJOIN）"""
    bl_path = os.path.join(DATA_DIR, "bloodline.csv")
    if not os.path.exists(bl_path):
        log.info("血統CSV未検出、スキップ")
        return df

    bl = pd.read_csv(bl_path, dtype=str)
    log.info(f"血統データ: {len(bl)} 頭")

    if "horse_id" not in df.columns or "horse_id" not in bl.columns:
        log.warning("horse_id カラムなし、スキップ")
        return df

    # 既存の血統カラムを削除（再統合のため）
    bl_cols = ["sire", "dam", "sire_sire", "dam_sire"]
    for c in bl_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    bl_dedup = bl.drop_duplicates(subset=["horse_id"], keep="first")
    before = len(df)
    df = df.merge(bl_dedup, on="horse_id", how="left")
    matched = df["sire"].notna().sum() if "sire" in df.columns else 0
    log.info(f"血統統合完了: {matched:,}/{before:,} 行にマッチ ({matched/before*100:.1f}%)")

    return df


def integrate_training(df: pd.DataFrame) -> pd.DataFrame:
    """調教データを統合（race_id + horse_number または horse_id でJOIN）。
    training_eval_score を数値特徴量として追加する。

    優先マッチキー:
      1. race_id + horse_number （両方ある場合）
      2. race_id + horse_id    （horse_number が training.csv に無い場合）
    NaN補完: training_eval_score の欠損は 0（スコア不明）で埋める。
    """
    tr_path = os.path.join(DATA_DIR, "training.csv")
    if not os.path.exists(tr_path):
        log.info("調教CSV未検出、スキップ")
        return df

    tr = pd.read_csv(tr_path)
    log.info(f"調教データ: {len(tr):,} 件")

    if "race_id" not in df.columns:
        log.warning("race_id カラムなし、スキップ")
        return df

    # 型統一
    df["race_id"] = df["race_id"].astype(str)
    tr["race_id"] = tr["race_id"].astype(str)

    # 既存の調教カラムを削除（再統合のため）
    tr_cols = ["training_evaluation", "training_eval_score", "training_comment"]
    for c in tr_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # --- キー選択: race_id + horse_number を優先 ---
    use_horse_number = (
        "horse_number" in df.columns
        and "horse_number" in tr.columns
        and tr["horse_number"].notna().sum() > 0
    )

    if use_horse_number:
        df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce").fillna(0).astype(int)
        tr["horse_number"] = pd.to_numeric(tr["horse_number"], errors="coerce").fillna(0).astype(int)

        keep_cols = ["race_id", "horse_number", "training_evaluation", "training_eval_score"]
        keep_cols = [c for c in keep_cols if c in tr.columns]
        tr_merge = tr[keep_cols].drop_duplicates(subset=["race_id", "horse_number"], keep="first")

        before = len(df)
        df = df.merge(tr_merge, on=["race_id", "horse_number"], how="left")
        matched_primary = df["training_eval_score"].notna().sum()
        log.info(
            f"調教統合 (race_id+horse_number): "
            f"{matched_primary:,}/{before:,} 行 ({matched_primary/before*100:.1f}%)"
        )

        # --- フォールバック: horse_id で未マッチ行を補完 ---
        unmatched_mask = df["training_eval_score"].isna()
        if unmatched_mask.sum() > 0 and "horse_id" in df.columns and "horse_id" in tr.columns:
            df["horse_id"] = df["horse_id"].astype(str)
            tr["horse_id"] = tr["horse_id"].astype(str)

            keep_cols2 = ["race_id", "horse_id", "training_evaluation", "training_eval_score"]
            keep_cols2 = [c for c in keep_cols2 if c in tr.columns]
            tr_fallback = tr[keep_cols2].drop_duplicates(subset=["race_id", "horse_id"], keep="first")

            # 未マッチ行だけ補完
            df_unmatched = df.loc[unmatched_mask, ["race_id", "horse_id"]].merge(
                tr_fallback, on=["race_id", "horse_id"], how="left"
            )
            for col in ["training_evaluation", "training_eval_score"]:
                if col in df_unmatched.columns:
                    df.loc[unmatched_mask, col] = df_unmatched[col].values

            matched_fallback = df["training_eval_score"].notna().sum() - matched_primary
            log.info(
                f"調教統合 フォールバック (horse_id): "
                f"+{matched_fallback:,} 行追加"
            )

    elif "horse_id" in df.columns and "horse_id" in tr.columns:
        df["horse_id"] = df["horse_id"].astype(str)
        tr["horse_id"] = tr["horse_id"].astype(str)

        keep_cols = ["race_id", "horse_id", "training_evaluation", "training_eval_score"]
        keep_cols = [c for c in keep_cols if c in tr.columns]
        tr_merge = tr[keep_cols].drop_duplicates(subset=["race_id", "horse_id"], keep="first")

        before = len(df)
        df = df.merge(tr_merge, on=["race_id", "horse_id"], how="left")
        matched = df["training_eval_score"].notna().sum()
        log.info(
            f"調教統合 (race_id+horse_id): "
            f"{matched:,}/{before:,} 行 ({matched/before*100:.1f}%)"
        )
    else:
        log.warning("マッチキーが不足（horse_number / horse_id）、スキップ")
        return df

    # NaN補完: スコア不明 → 0、評価不明 → 空文字
    if "training_eval_score" in df.columns:
        df["training_eval_score"] = (
            pd.to_numeric(df["training_eval_score"], errors="coerce").fillna(0).astype(int)
        )
    if "training_evaluation" in df.columns:
        df["training_evaluation"] = df["training_evaluation"].fillna("")

    total_matched = df["training_eval_score"].gt(0).sum()
    log.info(f"調教統合完了: eval_score > 0 が {total_matched:,} 行")

    return df


def integrate_marks(df: pd.DataFrame) -> pd.DataFrame:
    """予想印データを統合（marks.csv が存在する場合のみ）。
    記者印をスコア化: ◎=5, ○=4, ▲=3, △=2, ×=1
    marks.csv が無ければスキップ（パイプラインを壊さない）。

    追加カラム:
      mark_honmei  : ◎ を付けた記者数
      mark_taikou  : ○ を付けた記者数
      mark_tanana  : ▲ を付けた記者数
      mark_renka   : △ を付けた記者数
      mark_score   : 重み付き合計スコア
      mark_score_norm : mark_score / mark_total_reporters（正規化）
    """
    marks_path = os.path.join(DATA_DIR, "marks.csv")
    if not os.path.exists(marks_path):
        log.info("予想印CSV未検出（marks.csv）、スキップ")
        return df

    marks = pd.read_csv(marks_path)
    log.info(f"予想印データ: {len(marks):,} 件")

    if "race_id" not in df.columns or "horse_number" not in df.columns:
        log.warning("race_id/horse_number カラムなし、予想印スキップ")
        return df

    if "race_id" not in marks.columns or "horse_number" not in marks.columns:
        log.warning("marks.csv に race_id/horse_number がない、スキップ")
        return df

    # 型統一
    df["race_id"] = df["race_id"].astype(str)
    marks["race_id"] = marks["race_id"].astype(str)
    df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce").fillna(0).astype(int)
    marks["horse_number"] = pd.to_numeric(marks["horse_number"], errors="coerce").fillna(0).astype(int)

    # 既存マークカラムを削除
    mark_cols_existing = [
        "mark_honmei", "mark_taikou", "mark_tanana", "mark_renka",
        "mark_hoshi", "mark_score", "mark_score_norm", "mark_total_reporters",
    ]
    for c in mark_cols_existing:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # marks.csv に mark_score がなければ再計算
    if "mark_score" not in marks.columns:
        log.info("mark_score を再計算中 (◎=5, ○=4, ▲=3, △=2, ×=1)...")
        score = 0
        for col, weight in [
            ("mark_honmei", 5), ("mark_taikou", 4), ("mark_tanana", 3),
            ("mark_renka", 2), ("mark_hoshi", 2),
        ]:
            if col in marks.columns:
                score = score + pd.to_numeric(marks[col], errors="coerce").fillna(0) * weight
        marks["mark_score"] = score

    # 正規化スコア（reporter 数で除算）
    if "mark_total_reporters" in marks.columns:
        reporters = pd.to_numeric(marks["mark_total_reporters"], errors="coerce").replace(0, np.nan)
        marks["mark_score_norm"] = marks["mark_score"] / reporters
    else:
        marks["mark_score_norm"] = marks["mark_score"]

    # マージ対象カラムを確定
    mark_merge_cols = ["race_id", "horse_number"]
    for c in [
        "mark_honmei", "mark_taikou", "mark_tanana", "mark_renka",
        "mark_hoshi", "mark_score", "mark_score_norm", "mark_total_reporters",
    ]:
        if c in marks.columns:
            mark_merge_cols.append(c)

    marks_merge = marks[mark_merge_cols].drop_duplicates(
        subset=["race_id", "horse_number"], keep="first"
    )

    before = len(df)
    df = df.merge(marks_merge, on=["race_id", "horse_number"], how="left")
    matched = df["mark_score"].notna().sum() if "mark_score" in df.columns else 0
    log.info(f"予想印統合完了: {matched:,}/{before:,} 行にマッチ ({matched/before*100:.1f}%)")

    # NaN補完: 印なし → 0
    for c in ["mark_honmei", "mark_taikou", "mark_tanana", "mark_renka",
              "mark_hoshi", "mark_score", "mark_score_norm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df


def integrate_all(retrain: bool = False):
    """全データ統合"""
    log.info("=" * 60)
    log.info("追加データ統合開始")
    log.info("=" * 60)

    if not os.path.exists(MAIN_CSV):
        log.error(f"メインCSV未検出: {MAIN_CSV}")
        return

    log.info(f"メインCSV読込: {MAIN_CSV}")
    df = pd.read_csv(MAIN_CSV)
    log.info(f"元データ: {len(df):,} 行, {len(df.columns)} カラム")

    # 血統統合
    df = integrate_bloodline(df)

    # 調教統合（training_eval_score を数値特徴量として追加）
    df = integrate_training(df)

    # 予想印統合（marks.csv が存在する場合のみ）
    df = integrate_marks(df)

    # 保存
    out_path = os.path.join(DATA_DIR, "_all_races_enriched.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    log.info(f"統合CSV保存: {out_path} ({len(df):,} 行, {len(df.columns)} カラム)")

    # auto_learn 用に main_csv を更新
    import shutil
    backup = MAIN_CSV + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(MAIN_CSV, backup)
    log.info(f"バックアップ: {backup}")

    shutil.move(out_path, MAIN_CSV)
    log.info(f"メインCSV更新完了: {MAIN_CSV}")

    # カラム一覧
    log.info(f"最終カラム ({len(df.columns)}): {sorted(df.columns.tolist())}")

    if retrain:
        log.info("")
        log.info("=" * 60)
        log.info("モデル再学習開始")
        log.info("=" * 60)
        from auto_learn import retrain_model
        version = retrain_model(force=True)
        if version:
            log.info(f"再学習完了: {version}")
        else:
            log.warning("再学習失敗またはスキップ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="追加データ統合")
    parser.add_argument("--retrain", action="store_true", help="統合後に再学習")
    args = parser.parse_args()
    integrate_all(retrain=args.retrain)
