"""
競馬予想AI - ペース分析・脚質分類モジュール
=============================================
馬の脚質（逃げ/先行/差し/追込）を過去通過順から分類し、
レースのペース（ハイ/ミドル/スロー）を予測して
勝率を調整する。

脚質分類:
  逃げ (Front-runner):  コーナー1で1-2番手
  先行 (Stalker):       コーナー1で3-5番手
  差し (Closer):        コーナー1で6-10番手
  追込 (Deep closer):   コーナー1で11番手以降

ペース予測:
  ハイペース:   逃げ馬3頭以上 → 差し/追込有利
  ミドルペース: 逃げ馬2頭     → 展開次第
  スローペース: 逃げ馬0-1頭   → 逃げ/先行有利
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PaceAnalyzer:
    """ペース分析・脚質分類エンジン"""

    # 脚質分類の閾値（コーナー1通過順）
    STYLE_THRESHOLDS = {
        "逃げ": (1, 2),
        "先行": (3, 5),
        "差し": (6, 10),
        "追込": (11, 99),
    }

    # ペース別の勝率調整係数
    PACE_ADJUSTMENTS = {
        "ハイペース": {
            "逃げ": -0.12,
            "先行": -0.05,
            "差し": +0.10,
            "追込": +0.15,
        },
        "ミドルペース": {
            "逃げ": 0.0,
            "先行": +0.02,
            "差し": +0.02,
            "追込": -0.02,
        },
        "スローペース": {
            "逃げ": +0.12,
            "先行": +0.10,
            "差し": -0.05,
            "追込": -0.15,
        },
    }

    def __init__(self, history_df: Optional[pd.DataFrame] = None):
        """
        Args:
            history_df: 過去レース結果DataFrame。
                        必須カラム: horse_id, passing_order
                        任意カラム: last_3f, finish_position
        """
        self.history = history_df
        self._style_cache: Dict[str, Dict] = {}

    def set_history(self, df: pd.DataFrame):
        """履歴データを設定/更新"""
        self.history = df
        self._style_cache.clear()

    # ----------------------------------------------------------
    # 通過順パース
    # ----------------------------------------------------------

    @staticmethod
    def parse_passing_order(passing_str) -> List[int]:
        """
        通過順文字列をパースして整数リストにする。

        Examples:
            "3-2"       -> [3, 2]
            "5-4-3-2"   -> [5, 4, 3, 2]
            "1-1-1-1"   -> [1, 1, 1, 1]
            NaN / ""    -> []
        """
        if pd.isna(passing_str) or str(passing_str).strip() == "":
            return []
        try:
            parts = str(passing_str).strip().split("-")
            return [int(p) for p in parts if p.strip().isdigit()]
        except (ValueError, AttributeError):
            return []

    @staticmethod
    def get_first_corner_position(passing_str) -> Optional[int]:
        """通過順から最初のコーナー通過順位を取得"""
        positions = PaceAnalyzer.parse_passing_order(passing_str)
        return positions[0] if positions else None

    # ----------------------------------------------------------
    # 脚質分類
    # ----------------------------------------------------------

    def classify_running_style(self, horse_id: str, min_races: int = 3) -> Dict:
        """
        馬の脚質を過去のコーナー通過順位から分類する。

        Args:
            horse_id: 馬ID
            min_races: 最低レース数（これ未満はデータ不足扱い）

        Returns:
            {
                "style": "差し",
                "style_en": "Closer",
                "avg_corner1_pos": 7.2,
                "median_corner1_pos": 7.0,
                "race_count": 15,
                "style_distribution": {"逃げ": 1, "先行": 3, "差し": 8, "追込": 3},
                "confidence": "high",
                "last_3f_avg": 34.8,
            }
        """
        # キャッシュチェック
        cache_key = str(horse_id)
        if cache_key in self._style_cache:
            return self._style_cache[cache_key]

        result = self._compute_style(cache_key, min_races)
        self._style_cache[cache_key] = result
        return result

    def _compute_style(self, horse_id: str, min_races: int) -> Dict:
        """脚質計算の内部実装"""
        if self.history is None or "passing_order" not in self.history.columns:
            return self._unknown_style()

        horse_data = self.history[self.history["horse_id"] == horse_id]
        if horse_data.empty:
            return self._unknown_style()

        # コーナー1通過順を集計
        corner1_positions = []
        for _, row in horse_data.iterrows():
            pos = self.get_first_corner_position(row.get("passing_order"))
            if pos is not None:
                corner1_positions.append(pos)

        if len(corner1_positions) < min_races:
            return self._unknown_style(race_count=len(corner1_positions))

        avg_pos = np.mean(corner1_positions)
        median_pos = np.median(corner1_positions)

        # 脚質分布を計算
        distribution = {"逃げ": 0, "先行": 0, "差し": 0, "追込": 0}
        for pos in corner1_positions:
            for style, (lo, hi) in self.STYLE_THRESHOLDS.items():
                if lo <= pos <= hi:
                    distribution[style] += 1
                    break

        # 最頻脚質を決定（平均値ベース）
        style = self._classify_by_position(avg_pos)

        # 信頼度: レース数に基づく
        n = len(corner1_positions)
        if n >= 10:
            confidence = "high"
        elif n >= 5:
            confidence = "medium"
        else:
            confidence = "low"

        # last_3f平均
        last_3f_avg = None
        if "last_3f" in horse_data.columns:
            l3f = pd.to_numeric(horse_data["last_3f"], errors="coerce").dropna()
            if len(l3f) > 0:
                last_3f_avg = round(float(l3f.mean()), 1)

        style_en_map = {"逃げ": "Front-runner", "先行": "Stalker", "差し": "Closer", "追込": "Deep closer"}

        return {
            "style": style,
            "style_en": style_en_map.get(style, "Unknown"),
            "avg_corner1_pos": round(float(avg_pos), 1),
            "median_corner1_pos": round(float(median_pos), 1),
            "race_count": n,
            "style_distribution": distribution,
            "confidence": confidence,
            "last_3f_avg": last_3f_avg,
        }

    def _classify_by_position(self, avg_pos: float) -> str:
        """平均通過順から脚質を分類"""
        for style, (lo, hi) in self.STYLE_THRESHOLDS.items():
            if lo <= avg_pos <= hi:
                return style
        if avg_pos < 1:
            return "逃げ"
        return "追込"

    @staticmethod
    def _unknown_style(race_count: int = 0) -> Dict:
        """データ不足時のデフォルト返却"""
        return {
            "style": "不明",
            "style_en": "Unknown",
            "avg_corner1_pos": None,
            "median_corner1_pos": None,
            "race_count": race_count,
            "style_distribution": {"逃げ": 0, "先行": 0, "差し": 0, "追込": 0},
            "confidence": "none",
            "last_3f_avg": None,
        }

    # ----------------------------------------------------------
    # バッチ分類（レース全馬一括）
    # ----------------------------------------------------------

    def classify_race_field(self, horse_ids: List[str]) -> Dict[str, Dict]:
        """
        レース出走馬全頭の脚質を一括分類。

        Args:
            horse_ids: 出走馬のhorse_idリスト

        Returns:
            {horse_id: style_info, ...}
        """
        return {hid: self.classify_running_style(hid) for hid in horse_ids}

    # ----------------------------------------------------------
    # ペース予測
    # ----------------------------------------------------------

    def predict_race_pace(self, horse_ids: List[str]) -> Dict:
        """
        出走馬の脚質構成からレースペースを予測する。

        Args:
            horse_ids: 出走馬のhorse_idリスト

        Returns:
            {
                "predicted_pace": "ハイペース",
                "predicted_pace_en": "Fast",
                "front_runners": 4,
                "stalkers": 3,
                "closers": 5,
                "deep_closers": 2,
                "unknown": 0,
                "pace_advantage": "差し・追込有利",
                "pace_score": 0.72,   # 0=スロー, 0.5=ミドル, 1=ハイ
                "confidence": "medium",
            }
        """
        styles = self.classify_race_field(horse_ids)

        # 脚質カウント
        counts = {"逃げ": 0, "先行": 0, "差し": 0, "追込": 0, "不明": 0}
        known_count = 0
        for hid, info in styles.items():
            s = info["style"]
            if s in counts:
                counts[s] += 1
            else:
                counts["不明"] += 1
            if s != "不明":
                known_count += 1

        front_runners = counts["逃げ"]
        stalkers = counts["先行"]

        # ペーススコア計算
        # 逃げ馬が多いほどハイペース
        if known_count > 0:
            pace_score = min(1.0, (front_runners * 0.35 + stalkers * 0.10) / max(1, known_count / 3))
        else:
            pace_score = 0.5  # データなし → ミドル

        pace_score = round(min(1.0, max(0.0, pace_score)), 2)

        # ペース判定
        if front_runners >= 3:
            predicted_pace = "ハイペース"
            predicted_pace_en = "Fast"
        elif front_runners == 2 and stalkers >= 3:
            predicted_pace = "ハイペース"
            predicted_pace_en = "Fast"
        elif front_runners <= 1 and stalkers <= 2:
            predicted_pace = "スローペース"
            predicted_pace_en = "Slow"
        else:
            predicted_pace = "ミドルペース"
            predicted_pace_en = "Medium"

        # ペース有利脚質
        if predicted_pace == "ハイペース":
            pace_advantage = "差し・追込有利"
        elif predicted_pace == "スローペース":
            pace_advantage = "逃げ・先行有利"
        else:
            pace_advantage = "展開次第"

        # 信頼度
        unknown_ratio = counts["不明"] / max(1, len(horse_ids))
        if unknown_ratio > 0.5:
            confidence = "low"
        elif unknown_ratio > 0.2:
            confidence = "medium"
        else:
            confidence = "high"

        return {
            "predicted_pace": predicted_pace,
            "predicted_pace_en": predicted_pace_en,
            "front_runners": front_runners,
            "stalkers": stalkers,
            "closers": counts["差し"],
            "deep_closers": counts["追込"],
            "unknown": counts["不明"],
            "pace_advantage": pace_advantage,
            "pace_score": pace_score,
            "confidence": confidence,
        }

    # ----------------------------------------------------------
    # 勝率調整
    # ----------------------------------------------------------

    def adjust_predictions(
        self,
        predictions: List[Dict],
        horse_styles: Dict[str, Dict],
        pace_info: Dict,
    ) -> List[Dict]:
        """
        ペース予測に基づいて勝率を調整する。

        Args:
            predictions: 予測結果リスト
                [{"horse_number": 1, "horse_id": "...", "prob_win": 0.2, "prob_top3": 0.5, ...}, ...]
            horse_styles: classify_race_field()の結果
            pace_info: predict_race_pace()の結果

        Returns:
            調整済みの予測リスト（元リストは変更しない）
        """
        predicted_pace = pace_info.get("predicted_pace", "ミドルペース")
        adjustments = self.PACE_ADJUSTMENTS.get(predicted_pace, self.PACE_ADJUSTMENTS["ミドルペース"])

        adjusted = []
        for pred in predictions:
            p = dict(pred)  # コピー
            horse_id = str(p.get("horse_id", ""))
            style_info = horse_styles.get(horse_id, self._unknown_style())
            style = style_info["style"]

            # 脚質情報を付加
            p["running_style"] = style
            p["running_style_en"] = style_info.get("style_en", "Unknown")

            # 不明の場合は調整しない
            if style == "不明" or style_info["confidence"] == "none":
                p["pace_adjustment"] = 0.0
                p["pace_adjustment_pct"] = "+0%"
                adjusted.append(p)
                continue

            # 調整係数
            adj = adjustments.get(style, 0.0)

            # 信頼度による調整幅スケーリング
            if style_info["confidence"] == "low":
                adj *= 0.5
            elif style_info["confidence"] == "medium":
                adj *= 0.75

            # ペース信頼度による追加スケーリング
            if pace_info.get("confidence") == "low":
                adj *= 0.5
            elif pace_info.get("confidence") == "medium":
                adj *= 0.75

            # 確率調整（上限/下限ガード付き）
            original_win = p.get("prob_win", 0)
            original_top3 = p.get("prob_top3", 0)

            p["prob_win"] = max(0.01, min(0.95, original_win * (1 + adj)))
            p["prob_top3"] = max(0.02, min(0.98, original_top3 * (1 + adj * 0.7)))

            p["pace_adjustment"] = round(adj, 4)
            p["pace_adjustment_pct"] = f"{adj*100:+.0f}%"

            adjusted.append(p)

        # 確率を再正規化（全馬のprob_winの合計が1.0になるように）
        total_win = sum(p["prob_win"] for p in adjusted)
        total_top3 = sum(p["prob_top3"] for p in adjusted)
        if total_win > 0:
            for p in adjusted:
                p["prob_win"] = round(p["prob_win"] / total_win, 4)
        if total_top3 > 0:
            # top3は合計3.0が理想
            target_top3 = min(3.0, total_top3)
            for p in adjusted:
                p["prob_top3"] = round(p["prob_top3"] / total_top3 * target_top3, 4)

        return adjusted

    # ----------------------------------------------------------
    # 統合分析（1レース分を一括処理）
    # ----------------------------------------------------------

    def analyze_race(
        self,
        race_df: pd.DataFrame,
        predictions: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        レース1件の完全なペース分析を実行する。

        Args:
            race_df: レースの出走表DataFrame（horse_id列が必要）
            predictions: 予測結果リスト（あれば調整も行う）

        Returns:
            {
                "pace_analysis": { ... },
                "horse_styles": { horse_id: { ... }, ... },
                "adjusted_predictions": [ ... ] or None,
            }
        """
        if "horse_id" not in race_df.columns:
            return {
                "pace_analysis": {
                    "predicted_pace": "不明",
                    "predicted_pace_en": "Unknown",
                    "front_runners": 0,
                    "stalkers": 0,
                    "closers": 0,
                    "deep_closers": 0,
                    "unknown": len(race_df),
                    "pace_advantage": "データ不足",
                    "pace_score": 0.5,
                    "confidence": "none",
                },
                "horse_styles": {},
                "adjusted_predictions": predictions,
            }

        horse_ids = race_df["horse_id"].astype(str).unique().tolist()

        # 脚質分類
        styles = self.classify_race_field(horse_ids)

        # ペース予測
        pace_info = self.predict_race_pace(horse_ids)

        # 予測調整
        adjusted = None
        if predictions is not None:
            adjusted = self.adjust_predictions(predictions, styles, pace_info)

        return {
            "pace_analysis": pace_info,
            "horse_styles": styles,
            "adjusted_predictions": adjusted,
        }
