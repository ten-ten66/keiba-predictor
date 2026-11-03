"""予測時の履歴コンテキスト構築ヘルパー。

背景 (2026-07-02 修正):
  従来は「対象馬自身の行のみ」を context にしていたため、過去レースに
  対象馬が1頭しか存在せず、レース内で計算される特徴量が全て縮退していた:
    - Elo/Glicko: update_race() が参加者2頭未満で一切発火せず全馬1500固定
    - combined_elo(_rank): trainer_elo の NaN 連鎖で全馬 NaN (重要度1位の特徴量)
    - n_runners: 過去レースが「1頭立て」扱いになり正規化系ラグ特徴を汚染
    - daily_inner_bias / venue系: レース内の内外枠平均が計算不能
  学習時は全レース全馬で計算しているため、深刻な train/serve 不整合だった。

  本ヘルパーは「対象馬が出走した全過去レースの"全出走馬"の行」を返すことで
  学習時と同じレース単位の文脈を復元する。未来情報は含まれない
  (他馬の行も過去レースの確定結果であり、予測時点で入手可能な情報)。
"""

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

# フルフィールド展開の安全上限。超える場合は従来方式 (対象馬の行のみ) に
# フォールバックする。16頭×50走×16頭 ≈ 13k 行が通常上限なので余裕を持たせる。
DEFAULT_MAX_CONTEXT_ROWS = 60_000

# 予測対象レース側でマスクする「結果」カラム (リーク防止)
LEAK_COLS = ["finish_position", "finish_time", "prize_money", "passing_order", "last_3f"]


def build_full_field_context(
    db_df: pd.DataFrame,
    horse_ids: List,
    max_rows: int = DEFAULT_MAX_CONTEXT_ROWS,
) -> Optional[pd.DataFrame]:
    """対象馬の過去レースを「全出走馬の行ごと」取り出す。

    horse_id の照合は文字列に正規化して行う (スクレイプ由来のstr IDと
    CSV由来のint IDの型不一致で context が静かに空になる事故の防止)。

    Args:
        db_df: 全レースデータ (race_id / horse_id 列必須)
        horse_ids: 予測対象レースの出走馬 horse_id リスト
        max_rows: フルフィールド展開の行数上限 (超過時は対象馬の行のみへ縮退)

    Returns:
        context DataFrame (copy)。horse_ids が空なら None。
    """
    if not horse_ids or "horse_id" not in db_df.columns or "race_id" not in db_df.columns:
        return None

    id_set = {str(h) for h in horse_ids if pd.notna(h)}
    own_rows = db_df[db_df["horse_id"].astype(str).isin(id_set)]
    if own_rows.empty:
        return own_rows.copy()

    ctx_race_ids = own_rows["race_id"].unique()
    full_field = db_df[db_df["race_id"].isin(ctx_race_ids)]

    if len(full_field) > max_rows:
        # 異常に大きい場合は旧挙動 (対象馬のみ) に安全フォールバック
        return own_rows.copy()
    return full_field.copy()


_BLOODLINE_CACHE: Optional[pd.DataFrame] = None


def _load_bloodline() -> Optional[pd.DataFrame]:
    """data/bloodline.csv (horse_id→sire/dam/sire_sire/dam_sire) を遅延ロード・キャッシュ。"""
    global _BLOODLINE_CACHE
    if _BLOODLINE_CACHE is not None:
        return _BLOODLINE_CACHE
    data_dir = os.environ.get("KEIBA_DATA_DIR", "data")
    path = Path(data_dir) / "bloodline.csv"
    if not path.exists():
        return None
    try:
        bl = pd.read_csv(path, dtype=str).drop_duplicates(subset=["horse_id"], keep="first")
        _BLOODLINE_CACHE = bl
    except Exception:
        return None
    return _BLOODLINE_CACHE


def _merge_bloodline_into_target(target_df: pd.DataFrame) -> pd.DataFrame:
    """スクレイプ直後の出馬表に血統カラムをマージする (無ければそのまま)。

    DB由来のcontext行は起動時に血統マージ済みだが、出馬表の新規行には
    sire/dam が無く、血統統計特徴量が対象レースだけNaNになるのを防ぐ。
    """
    if "horse_id" not in target_df.columns:
        return target_df
    if "sire" in target_df.columns and target_df["sire"].notna().any():
        return target_df  # マージ済み
    bl = _load_bloodline()
    if bl is None:
        return target_df
    out = target_df.copy()
    out["horse_id"] = out["horse_id"].astype(str)
    drop = [c for c in ["sire", "dam", "sire_sire", "dam_sire"] if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    return out.merge(bl, on="horse_id", how="left")


def featurize_with_full_context(
    target_df: pd.DataFrame,
    db_df: pd.DataFrame,
    max_rows: int = DEFAULT_MAX_CONTEXT_ROWS,
) -> Optional[pd.DataFrame]:
    """出馬表 (1レース分の生行) を本パイプラインで特徴量化する。

    /api/predict/{race_id} (GET/POST) と同一の経路:
      full-field context + 対象行 → リークマスク → preprocess
      → FeatureEngineer → AdvancedFeatureEngineer → 対象レース抽出

    upcoming/precache 経路は従来 enrich_with_history() の粗い近似
    (疑似Elo等の数個) + 欠損0埋めで予測しており、220特徴量の大半が
    死んでいた。この関数で学習時と同じ特徴量分布に揃える。

    Returns:
        対象レースの featured DataFrame。前提が満たせない場合は None
        (呼び出し側で従来経路にフォールバックすること)。
    """
    if target_df is None or target_df.empty:
        return None
    if "race_id" not in target_df.columns or "horse_id" not in target_df.columns:
        return None

    # 遅延import (このモジュールを軽量に保つ)
    from data_processor import RaceDataProcessor, FeatureEngineer

    race_id = str(target_df["race_id"].iloc[0])
    target = _merge_bloodline_into_target(target_df)

    horse_ids = target["horse_id"].dropna().unique().tolist()
    ctx = build_full_field_context(db_df, horse_ids, max_rows=max_rows)
    if ctx is not None and not ctx.empty:
        combined = pd.concat([ctx, target], ignore_index=True)
        # 対象レースがDBにも存在する場合はスクレイプ行 (最新) を優先
        combined = combined.drop_duplicates(subset=["race_id", "horse_number"], keep="last")
    else:
        combined = target.copy()

    # リーク防止: 対象レースの結果カラムをマスク
    target_mask = combined["race_id"].astype(str) == race_id
    for col in LEAK_COLS:
        if col in combined.columns:
            combined.loc[target_mask, col] = None

    proc = RaceDataProcessor()
    proc.raw_data = combined
    processed = proc.preprocess()
    featured = FeatureEngineer().create_features(processed)
    try:
        from features_advanced import AdvancedFeatureEngineer
        featured = AdvancedFeatureEngineer().create_all(featured)
    except Exception:
        pass  # 拡張特徴量は任意 (モデル側で不足分は補完される)

    result = featured[featured["race_id"].astype(str) == race_id]
    return result if not result.empty else None
