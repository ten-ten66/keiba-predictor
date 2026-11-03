"""
Shadow model evaluation
========================
本番モデル (model) とシャドウモデル (shadow_model) の予測を同じ特徴量で比較し、
JSONL にログするライブラリ。本番昇格前の新モデル評価用。

設計方針:
  - シャドウ予測は **失敗しても本番の応答に影響しない** (try/except で握りつぶす)
  - 重い全馬予測は避け、ログには集約済み指標だけを保存
  - `data/shadow_eval/YYYYMMDD.jsonl` に追記
  - 比較指標:
      - rank_spearman  : 予測着順の順位相関
      - kendall_tau    : 同上 (カテゴリカル)
      - top1_agreement : 1番手予測が一致したか
      - top3_overlap   : トップ3の重なり (0/1/2/3)
      - prob_kl        : prob_win 分布のKL divergence (対称)
      - latency_ms     : シャドウの追加レイテンシ
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("shadow_eval")


def _rank_spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    # 簡易: tie は average rank
    def avg_ranks(vs: list[float]) -> list[float]:
        idx = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and vs[idx[j + 1]] == vs[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[idx[k]] = avg
            i = j + 1
        return ranks

    rx = avg_ranks(x)
    ry = avg_ranks(y)
    n = len(x)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _symmetric_kl(p: list[float], q: list[float], eps: float = 1e-9) -> float:
    if len(p) != len(q) or not p:
        return 0.0
    sp = sum(p) or 1.0
    sq = sum(q) or 1.0
    p = [max(eps, x / sp) for x in p]
    q = [max(eps, x / sq) for x in q]
    kl_pq = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
    kl_qp = sum(qi * math.log(qi / pi) for pi, qi in zip(p, q))
    return (kl_pq + kl_qp) / 2.0


def compare_predictions(
    primary_df: pd.DataFrame, shadow_df: pd.DataFrame, key: str = "horse_number"
) -> dict[str, Any]:
    """同じレースに対する2モデルの予測DFを突き合わせて集約指標を返す。"""
    if primary_df is None or shadow_df is None or primary_df.empty or shadow_df.empty:
        return {"error": "empty predictions"}
    # key で inner join
    merged = primary_df.merge(shadow_df, on=key, suffixes=("_p", "_s"))
    if merged.empty:
        return {"error": "no overlap on key"}

    p_rank = merged.get("predicted_rank_p")
    s_rank = merged.get("predicted_rank_s")
    p_prob = merged.get("prob_win_p")
    s_prob = merged.get("prob_win_s")

    out: dict[str, Any] = {"n_horses": int(len(merged))}

    if p_rank is not None and s_rank is not None:
        out["rank_spearman"] = round(_rank_spearman(p_rank.tolist(), s_rank.tolist()), 4)
        # Top-1 agreement
        p_top1 = merged.iloc[p_rank.argmin()][key]
        s_top1 = merged.iloc[s_rank.argmin()][key]
        out["top1_agreement"] = bool(p_top1 == s_top1)
        # Top-3 overlap
        p_top3 = set(merged.sort_values("predicted_rank_p").head(3)[key].tolist())
        s_top3 = set(merged.sort_values("predicted_rank_s").head(3)[key].tolist())
        out["top3_overlap"] = len(p_top3 & s_top3)

    if p_prob is not None and s_prob is not None:
        out["prob_kl"] = round(_symmetric_kl(p_prob.tolist(), s_prob.tolist()), 4)
        out["prob_max_abs_diff"] = round(float((p_prob - s_prob).abs().max()), 4)

    return out


class ShadowLogger:
    """シャドウ評価結果を JSONL に追記する軽量ロガー。"""

    def __init__(self, data_dir: Path | str, max_records_per_day: int = 500):
        self.data_dir = Path(data_dir) / "shadow_eval"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_records_per_day = max_records_per_day

    def _today_path(self) -> Path:
        from datetime import datetime as _dt
        return self.data_dir / f"{_dt.now().strftime('%Y%m%d')}.jsonl"

    def log(self, race_id: str, metrics: dict[str, Any], extra: Optional[dict] = None) -> None:
        rec = {
            "ts": time.time(),
            "race_id": race_id,
            "metrics": metrics,
        }
        if extra:
            rec["extra"] = extra
        path = self._today_path()
        # 日次上限で古い行を保持しない (単純に書き続ける → ローテーションは外部)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("shadow log write failed: %s", e)

    def summarize(self, days: int = 7) -> dict[str, Any]:
        """直近N日の集約統計を返す。"""
        from datetime import datetime as _dt, timedelta as _td
        recs: list[dict] = []
        today = _dt.now().date()
        for i in range(days):
            d = today - _td(days=i)
            p = self.data_dir / f"{d.strftime('%Y%m%d')}.jsonl"
            if not p.exists():
                continue
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        recs.append(json.loads(line))
            except Exception:
                continue
        if not recs:
            return {"count": 0, "days": days, "message": "no shadow logs"}

        def _avg(key: str) -> float:
            vals = [r["metrics"].get(key) for r in recs if isinstance(r.get("metrics", {}).get(key), (int, float))]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        def _rate(key: str) -> float:
            vals = [r["metrics"].get(key) for r in recs if r.get("metrics", {}).get(key) is not None]
            if not vals:
                return 0.0
            true_ct = sum(1 for v in vals if v)
            return round(true_ct / len(vals), 4)

        return {
            "count": len(recs),
            "days": days,
            "avg_rank_spearman": _avg("rank_spearman"),
            "avg_prob_kl": _avg("prob_kl"),
            "avg_top3_overlap": _avg("top3_overlap"),
            "top1_agreement_rate": _rate("top1_agreement"),
            "first_ts": min(r.get("ts", 0) for r in recs),
            "last_ts": max(r.get("ts", 0) for r in recs),
        }
