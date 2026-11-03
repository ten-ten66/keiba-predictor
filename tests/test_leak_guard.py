"""
Leak-guard tests
================
予測パイプラインが「対象レースの結果列」を履歴マージ時にNullマスクしているか検証する。

Regression history:
  過去にGET /api/predict/{race_id} が履歴マージを怠って finish_position 等を
  使った特徴量を0埋めするバグがあった。さらにその後、mask処理自体が漏れると
  モデルが対象レースの結果を直接読んでしまう深刻なリークが起きる。
  本テストは
    (a) 既知のリーク列リストが api_server.py の両予測ルートに残っていること
    (b) mask logic を synthetic data に適用したとき target レースだけが Null 化されること
  を確認する。
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
API_SERVER_PATH = REPO_ROOT / "api_server.py"

# 結果列（predict 対象レースでは未知であるべき列）
EXPECTED_LEAK_COLS = {
    "finish_position",
    "finish_time",
    "prize_money",
    "passing_order",
    "last_3f",
}


def _extract_leak_col_literals(source: str) -> list[list[str]]:
    """`leak_cols = [...]` の右辺リテラルを全て抽出。"""
    pattern = re.compile(r"leak_cols\s*=\s*\[([^\]]*)\]")
    hits: list[list[str]] = []
    for m in pattern.finditer(source):
        items = re.findall(r'"([^"]+)"', m.group(1))
        hits.append(items)
    return hits


class TestLeakColumnListConsistency:
    """api_server.py 内の leak_cols 定義がすべて期待セットと一致することを保証する。"""

    def test_api_server_has_leak_cols(self):
        src = API_SERVER_PATH.read_text(encoding="utf-8")
        assert "leak_cols" in src, "api_server.py に leak_cols 定義が見当たらない"

    def test_all_leak_col_lists_match_expected(self):
        src = API_SERVER_PATH.read_text(encoding="utf-8")
        occurrences = _extract_leak_col_literals(src)
        assert len(occurrences) >= 2, (
            f"leak_cols の定義箇所が {len(occurrences)} 箇所しかない。"
            "GET/POST 両予測ルートに定義されているべき。"
        )
        for idx, cols in enumerate(occurrences):
            got = set(cols)
            assert got == EXPECTED_LEAK_COLS, (
                f"leak_cols[{idx}] が期待と異なる: {got} vs {EXPECTED_LEAK_COLS}"
            )

    def test_leak_cols_not_silently_removed(self):
        """重要な結果系列が誤って leak_cols から外されるのを防ぐ。"""
        src = API_SERVER_PATH.read_text(encoding="utf-8")
        for col in EXPECTED_LEAK_COLS:
            assert f'"{col}"' in src, f"{col} が api_server.py のどこにも現れない"


class TestLeakMaskBehavior:
    """対象レース行だけが Null 化され、履歴行は温存されるという不変式をテストする。"""

    @pytest.fixture
    def combined_df(self) -> pd.DataFrame:
        # 馬 H1/H2 × race_1 (過去) / race_2 (target) / race_3 (過去)
        rows = []
        for race_id, race_date, fp1, fp2 in [
            ("race_past_1", "2024-01-01", 1, 2),
            ("race_target", "2024-02-01", 3, 5),   # これを Null 化したい
            ("race_past_2", "2024-01-15", 2, 1),
        ]:
            for h, pos in [("H1", fp1), ("H2", fp2)]:
                rows.append({
                    "race_id": race_id,
                    "race_date": race_date,
                    "horse_id": h,
                    "horse_number": int(h[1]),
                    "odds": 3.5,
                    "finish_position": float(pos),
                    "finish_time": 60.0 + pos,
                    "prize_money": 1000 * (10 - pos),
                    "passing_order": f"{pos}-{pos}-{pos}",
                    "last_3f": 34.0 + pos * 0.1,
                })
        return pd.DataFrame(rows)

    def _apply_mask(self, df: pd.DataFrame, target_race_id: str) -> pd.DataFrame:
        """api_server.py のマスク処理と同一ロジック。"""
        df = df.copy()
        target_mask = df["race_id"] == target_race_id
        for col in EXPECTED_LEAK_COLS:
            if col in df.columns:
                df.loc[target_mask, col] = None
        return df

    def test_target_rows_are_nulled(self, combined_df):
        out = self._apply_mask(combined_df, "race_target")
        tgt = out[out["race_id"] == "race_target"]
        for col in EXPECTED_LEAK_COLS:
            assert tgt[col].isna().all(), f"{col} が対象レース行で Null 化されていない"

    def test_context_rows_are_preserved(self, combined_df):
        out = self._apply_mask(combined_df, "race_target")
        ctx = out[out["race_id"] != "race_target"]
        for col in EXPECTED_LEAK_COLS:
            assert ctx[col].notna().all(), f"{col} が履歴行で失われた"
        # 数値一致も確認
        before = combined_df[combined_df["race_id"] != "race_target"].reset_index(drop=True)
        after = ctx.reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after, check_dtype=False)

    def test_non_leak_cols_are_untouched(self, combined_df):
        out = self._apply_mask(combined_df, "race_target")
        tgt = out[out["race_id"] == "race_target"]
        # odds や horse_number 等、対象レースでも既知であるべき列は不変
        assert tgt["odds"].notna().all()
        assert tgt["horse_number"].notna().all()
        assert (tgt["horse_number"] == combined_df.loc[combined_df["race_id"] == "race_target", "horse_number"].values).all()

    def test_missing_columns_dont_crash(self, combined_df):
        """いずれかの結果列がそもそも df に無い場合でも例外にならない。"""
        partial = combined_df.drop(columns=["passing_order", "last_3f"])
        out = self._apply_mask(partial, "race_target")
        tgt = out[out["race_id"] == "race_target"]
        assert tgt["finish_position"].isna().all()
        assert tgt["finish_time"].isna().all()
        assert tgt["prize_money"].isna().all()
