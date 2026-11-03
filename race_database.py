"""
競馬予想AI - レースデータベース & 検索エンジン
================================================
過去のレース結果・馬・騎手・競馬場を横断検索し、
詳細データをJSON形式で出力する。

機能:
  1. 馬名検索 → 全出走履歴・成績サマリー
  2. レース名検索 → 過去の同名レース結果一覧
  3. 競馬場検索 → 場別の統計・傾向
  4. レースカレンダー → 月別の開催一覧
  5. 複合検索（馬×距離、騎手×場 etc.）
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict


class RaceDatabase:
    """過去レースの検索・分析データベース"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.df: Optional[pd.DataFrame] = None
        self._horse_cache: Dict[str, pd.DataFrame] = {}
        self._race_cache: Dict[str, pd.DataFrame] = {}

    def load(self, csv_path: str = None):
        """CSVデータを読み込み"""
        if csv_path:
            path = Path(csv_path)
        else:
            # 最新の結合CSVを探す
            csvs = sorted(self.data_dir.glob("races_*.csv"), reverse=True)
            if not csvs:
                raise FileNotFoundError("CSVファイルが見つかりません")
            path = csvs[0]

        self.df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        # ID系カラムを文字列に統一
        for col in ["race_id", "horse_id", "jockey_id", "trainer_id"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)
        if "race_date" in self.df.columns:
            self.df["race_date"] = pd.to_datetime(self.df["race_date"], errors="coerce", format="mixed")
            self.df = self.df.sort_values("race_date")

        print(f"✅ {len(self.df):,} レコード / {self.df['race_id'].nunique():,} レース読み込み")
        return self

    # ==========================================================
    # 1. 馬の検索・詳細
    # ==========================================================

    def search_horse(self, query: str, limit: int = 20) -> List[Dict]:
        """馬名で部分一致検索"""
        if self.df is None or not query or not query.strip():
            return []

        query = query.strip()
        mask = self.df["horse_name"].str.contains(query, case=False, na=False)
        matches = self.df[mask].drop_duplicates("horse_id").head(limit)

        results = []
        for _, row in matches.iterrows():
            horse_data = self.get_horse_detail(row["horse_id"])
            if horse_data:
                results.append({
                    "horse_id": row["horse_id"],
                    "horse_name": row["horse_name"],
                    "summary": horse_data["summary"],
                })
        return results

    def get_horse_detail(self, horse_id: str) -> Optional[Dict]:
        """馬の全詳細データ"""
        if self.df is None:
            return None

        # horse_idの型を合わせて検索
        try:
            horse_id_typed = type(self.df["horse_id"].iloc[0])(horse_id)
        except (ValueError, TypeError, IndexError):
            horse_id_typed = horse_id
        hdf = self.df[self.df["horse_id"] == horse_id_typed].sort_values("race_date")
        if hdf.empty:
            return None

        name = hdf.iloc[-1]["horse_name"]
        total = len(hdf)
        wins = (hdf["finish_position"] == 1).sum()
        seconds = (hdf["finish_position"] == 2).sum()
        thirds = (hdf["finish_position"] == 3).sum()
        out = total - wins - seconds - thirds

        # 賞金合計
        prize_total = hdf["prize_money"].sum() if "prize_money" in hdf.columns else 0

        # コース別成績
        course_stats = {}
        if "course_type" in hdf.columns:
            for ct, grp in hdf.groupby("course_type"):
                ct_total = len(grp)
                ct_wins = (grp["finish_position"] == 1).sum()
                ct_top3 = (grp["finish_position"] <= 3).sum()
                course_stats[ct] = {
                    "races": ct_total,
                    "record": f"{ct_wins}-{(grp['finish_position']==2).sum()}-{(grp['finish_position']==3).sum()}-{ct_total-ct_top3}",
                    "win_rate": round(ct_wins / ct_total * 100, 1) if ct_total > 0 else 0,
                    "top3_rate": round(ct_top3 / ct_total * 100, 1) if ct_total > 0 else 0,
                }

        # 距離別成績
        dist_stats = {}
        if "distance" in hdf.columns:
            for dist, grp in hdf.groupby("distance"):
                d_total = len(grp)
                d_wins = (grp["finish_position"] == 1).sum()
                d_top3 = (grp["finish_position"] <= 3).sum()
                dist_stats[int(dist)] = {
                    "races": d_total,
                    "win_rate": round(d_wins / d_total * 100, 1) if d_total > 0 else 0,
                    "top3_rate": round(d_top3 / d_total * 100, 1) if d_total > 0 else 0,
                    "avg_pos": round(grp["finish_position"].mean(), 2),
                }

        # 場別成績
        venue_stats = {}
        if "venue" in hdf.columns:
            for venue, grp in hdf.groupby("venue"):
                v_total = len(grp)
                v_wins = (grp["finish_position"] == 1).sum()
                v_top3 = (grp["finish_position"] <= 3).sum()
                venue_stats[venue] = {
                    "races": v_total,
                    "win_rate": round(v_wins / v_total * 100, 1) if v_total > 0 else 0,
                    "top3_rate": round(v_top3 / v_total * 100, 1) if v_total > 0 else 0,
                }

        # 騎手別成績
        jockey_stats = {}
        if "jockey_name" in hdf.columns:
            for jname, grp in hdf.groupby("jockey_name"):
                j_total = len(grp)
                j_wins = (grp["finish_position"] == 1).sum()
                jockey_stats[jname] = {
                    "races": j_total,
                    "wins": int(j_wins),
                    "win_rate": round(j_wins / j_total * 100, 1) if j_total > 0 else 0,
                }

        # 全出走履歴
        race_history = []
        for _, row in hdf.iterrows():
            entry = {
                "date": row["race_date"].strftime("%Y-%m-%d") if pd.notna(row["race_date"]) else "",
                "race_name": row.get("race_name", ""),
                "venue": row.get("venue", ""),
                "course": row.get("course_type", ""),
                "distance": int(row.get("distance", 0)),
                "track": row.get("track_condition", ""),
                "position": int(row["finish_position"]),
                "odds": row.get("odds", 0),
                "jockey": row.get("jockey_name", ""),
                "time": row.get("finish_time", ""),
                "last_3f": row.get("last_3f", 0),
                "prize": row.get("prize_money", 0),
            }
            race_history.append(entry)

        return {
            "horse_id": horse_id,
            "horse_name": name,
            "summary": {
                "total_races": total,
                "record": f"{wins}-{seconds}-{thirds}-{out}",
                "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                "top3_rate": round((wins + seconds + thirds) / total * 100, 1) if total > 0 else 0,
                "avg_position": round(hdf["finish_position"].mean(), 2),
                "prize_total": round(prize_total, 1),
            },
            "course_stats": course_stats,
            "distance_stats": dict(sorted(dist_stats.items())),
            "venue_stats": venue_stats,
            "jockey_stats": dict(sorted(jockey_stats.items(), key=lambda x: x[1]["wins"], reverse=True)),
            "race_history": list(reversed(race_history)),  # 新しい順
        }

    # ==========================================================
    # 2. レース検索・詳細
    # ==========================================================

    def search_race(self, query: str, limit: int = 20) -> List[Dict]:
        """レース名で部分一致検索"""
        if self.df is None or not query or not query.strip():
            return []

        query = query.strip()
        mask = self.df["race_name"].str.contains(query, case=False, na=False)
        race_ids = self.df[mask]["race_id"].unique()[:limit]

        results = []
        for rid in race_ids:
            detail = self.get_race_detail(rid)
            if detail:
                results.append({
                    "race_id": rid,
                    "race_name": detail["race_name"],
                    "date": detail["date"],
                    "venue": detail["venue"],
                    "course": detail["course"],
                    "n_horses": detail["n_horses"],
                    "winner": detail["results"][0]["horse_name"] if detail["results"] else "",
                })
        return results

    def get_race_detail(self, race_id: str) -> Optional[Dict]:
        """レースの全詳細データ"""
        if self.df is None:
            return None

        rdf = self.df[self.df["race_id"] == race_id].sort_values("finish_position")
        if rdf.empty:
            return None

        first_row = rdf.iloc[0]

        results = []
        for _, row in rdf.iterrows():
            results.append({
                "position": int(row["finish_position"]),
                "horse_number": int(row.get("horse_number", 0)),
                "horse_name": row.get("horse_name", ""),
                "horse_id": row.get("horse_id", ""),
                "jockey": row.get("jockey_name", ""),
                "odds": row.get("odds", 0),
                "popularity": int(row.get("popularity", 0)),
                "time": row.get("finish_time", ""),
                "last_3f": row.get("last_3f", 0),
                "weight": row.get("weight", 0),
                "horse_weight": int(row.get("horse_weight", 0)),
                "passing": row.get("passing_order", ""),
            })

        return {
            "race_id": race_id,
            "race_name": first_row.get("race_name", ""),
            "date": first_row["race_date"].strftime("%Y-%m-%d") if pd.notna(first_row.get("race_date")) else "",
            "venue": first_row.get("venue", ""),
            "course": f"{first_row.get('course_type', '')}{int(first_row.get('distance', 0))}m",
            "distance": int(first_row.get("distance", 0)),
            "track_condition": first_row.get("track_condition", ""),
            "weather": first_row.get("weather", ""),
            "n_horses": len(rdf),
            "results": results,
        }

    # ==========================================================
    # 3. 競馬場検索・統計
    # ==========================================================

    def get_venue_stats(self, venue: str) -> Optional[Dict]:
        """競馬場の統計データ"""
        if self.df is None or "venue" not in self.df.columns:
            return None

        vdf = self.df[self.df["venue"] == venue]
        if vdf.empty:
            return None

        total_races = vdf["race_id"].nunique()
        total_records = len(vdf)

        # コース別レース数
        course_dist = {}
        if "course_type" in vdf.columns and "distance" in vdf.columns:
            for (ct, dist), grp in vdf.groupby(["course_type", "distance"]):
                key = f"{ct}{int(dist)}m"
                course_dist[key] = grp["race_id"].nunique()

        # 馬場状態分布
        track_dist = {}
        if "track_condition" in vdf.columns:
            for tc, grp in vdf.groupby("track_condition"):
                track_dist[tc] = grp["race_id"].nunique()

        # 枠番有利不利
        frame_stats = {}
        if "frame_number" in vdf.columns:
            for fn, grp in vdf.groupby("frame_number"):
                f_total = len(grp)
                f_wins = (grp["finish_position"] == 1).sum()
                frame_stats[int(fn)] = {
                    "win_rate": round(f_wins / f_total * 100, 2) if f_total > 0 else 0,
                    "avg_pos": round(grp["finish_position"].mean(), 2),
                }

        # 人気別勝率
        popularity_stats = {}
        if "popularity" in vdf.columns:
            for pop, grp in vdf.groupby("popularity"):
                pop = int(pop)
                if pop < 1 or pop > 18:
                    continue
                p_total = len(grp)
                p_wins = int((grp["finish_position"] == 1).sum())
                p_top3 = int((grp["finish_position"] <= 3).sum())
                popularity_stats[pop] = {
                    "races": p_total,
                    "wins": p_wins,
                    "win_rate": round(p_wins / p_total * 100, 1) if p_total > 0 else 0,
                    "top3_rate": round(p_top3 / p_total * 100, 1) if p_total > 0 else 0,
                }

        # 馬場状態別勝率（1番人気の勝率を馬場状態ごとに）
        track_fav_stats = {}
        if "track_condition" in vdf.columns and "popularity" in vdf.columns:
            fav_df = vdf[vdf["popularity"] == 1]
            for tc, grp in fav_df.groupby("track_condition"):
                tc_total = len(grp)
                tc_wins = int((grp["finish_position"] == 1).sum())
                track_fav_stats[tc] = {
                    "races": tc_total,
                    "fav_win_rate": round(tc_wins / tc_total * 100, 1) if tc_total > 0 else 0,
                }

        # 平均配当（単勝）
        avg_odds_winner = 0
        fav_win_rate = 0
        winners = vdf[vdf["finish_position"] == 1]
        if not winners.empty and "odds" in vdf.columns:
            avg_odds_winner = round(float(winners["odds"].mean()), 1)
        if "popularity" in vdf.columns:
            fav_winners = vdf[(vdf["popularity"] == 1)]
            fav_total = len(fav_winners)
            fav_wins = (fav_winners["finish_position"] == 1).sum()
            fav_win_rate = round(fav_wins / fav_total * 100, 1) if fav_total > 0 else 0

        # 距離別傾向（逃げ有利度）
        distance_pace = {}
        if "distance" in vdf.columns and "horse_number" in vdf.columns:
            for dist, grp in vdf.groupby("distance"):
                dist_key = f"{int(dist)}m"
                d_winners = grp[grp["finish_position"] == 1]
                if len(d_winners) < 3:
                    continue
                avg_winner_num = round(float(d_winners["horse_number"].mean()), 1)
                d_total_races = grp["race_id"].nunique()
                distance_pace[dist_key] = {
                    "races": d_total_races,
                    "avg_winner_number": avg_winner_num,
                }

        # 騎手リーディング
        jockey_ranking = []
        if "jockey_name" in vdf.columns:
            for jname, grp in vdf.groupby("jockey_name"):
                j_total = len(grp)
                j_wins = (grp["finish_position"] == 1).sum()
                if j_wins >= 2:
                    jockey_ranking.append({
                        "jockey": jname,
                        "races": j_total,
                        "wins": int(j_wins),
                        "win_rate": round(j_wins / j_total * 100, 1),
                    })
            jockey_ranking.sort(key=lambda x: x["wins"], reverse=True)

        # 最近のレース
        recent_races = []
        recent_ids = vdf.drop_duplicates("race_id").sort_values("race_date", ascending=False)["race_id"].head(10)
        for rid in recent_ids:
            rdf = vdf[vdf["race_id"] == rid]
            first = rdf.iloc[0]
            winner = rdf[rdf["finish_position"] == 1]
            recent_races.append({
                "race_id": rid,
                "date": first["race_date"].strftime("%Y-%m-%d") if pd.notna(first.get("race_date")) else "",
                "race_name": first.get("race_name", ""),
                "course": f"{first.get('course_type', '')}{int(first.get('distance', 0))}m",
                "winner": winner.iloc[0]["horse_name"] if not winner.empty else "",
                "winner_odds": winner.iloc[0].get("odds", 0) if not winner.empty else 0,
            })

        return {
            "venue": venue,
            "total_races": total_races,
            "total_records": total_records,
            "avg_odds_winner": avg_odds_winner,
            "fav_win_rate": fav_win_rate,
            "course_distribution": dict(sorted(course_dist.items())),
            "track_distribution": track_dist,
            "track_fav_stats": track_fav_stats,
            "frame_stats": frame_stats,
            "popularity_stats": dict(sorted(popularity_stats.items())),
            "distance_pace": dict(sorted(distance_pace.items())),
            "jockey_ranking": jockey_ranking[:15],
            "recent_races": recent_races,
        }

    def list_venues(self) -> List[str]:
        """全競馬場リスト"""
        if self.df is None or "venue" not in self.df.columns:
            return []
        return sorted(self.df["venue"].dropna().unique().tolist())

    # ==========================================================
    # 4. レースカレンダー
    # ==========================================================

    def get_calendar(self, year: int, month: int) -> List[Dict]:
        """指定月のレースカレンダー"""
        if self.df is None:
            return []

        mask = (self.df["race_date"].dt.year == year) & (self.df["race_date"].dt.month == month)
        mdf = self.df[mask]

        calendar = []
        for date, day_group in mdf.groupby(self.df["race_date"].dt.date):
            venues = {}
            for venue, v_group in day_group.groupby("venue"):
                race_ids = v_group["race_id"].unique()
                races = []
                for rid in race_ids:
                    r = v_group[v_group["race_id"] == rid].iloc[0]
                    winner = v_group[(v_group["race_id"] == rid) & (v_group["finish_position"] == 1)]
                    rname = str(r.get("race_name", ""))
                    # Detect grade from race name
                    import re
                    grade = ""
                    if re.search(r'\(GI\)|\(G1\)', rname):
                        grade = "G1"
                    elif re.search(r'\(GII\)|\(G2\)', rname):
                        grade = "G2"
                    elif re.search(r'\(GIII\)|\(G3\)', rname):
                        grade = "G3"
                    elif re.search(r'\(L\)', rname):
                        grade = "L"
                    elif re.search(r'\(OP\)', rname):
                        grade = "OP"
                    races.append({
                        "race_id": rid,
                        "race_name": rname,
                        "course": f"{r.get('course_type', '')}{int(r.get('distance', 0))}m",
                        "n_horses": len(v_group[v_group["race_id"] == rid]),
                        "winner": winner.iloc[0]["horse_name"] if not winner.empty else "",
                        "grade": grade,
                    })
                venues[venue] = {"race_count": len(races), "races": races}

            calendar.append({
                "date": str(date),
                "day_of_week": date.strftime("%a"),
                "venues": venues,
                "total_races": day_group["race_id"].nunique(),
            })

        return calendar

    def get_calendar_summary(self, year: int) -> List[Dict]:
        """年間カレンダーサマリー（月別レース数）"""
        if self.df is None:
            return []

        mask = self.df["race_date"].dt.year == year
        ydf = self.df[mask]

        months = []
        for m in range(1, 13):
            m_mask = ydf["race_date"].dt.month == m
            m_df = ydf[m_mask]
            months.append({
                "month": m,
                "total_races": m_df["race_id"].nunique(),
                "total_records": len(m_df),
                "race_days": m_df["race_date"].dt.date.nunique(),
            })
        return months

    # ==========================================================
    # 5. 統合検索
    # ==========================================================

    def search(self, query: str, limit: int = 10) -> Dict:
        """統合検索: 馬名・レース名・競馬場を横断検索"""
        results = {
            "horses": self.search_horse(query, limit),
            "races": self.search_race(query, limit),
            "venues": [],
        }

        # 競馬場名マッチ
        for v in self.list_venues():
            if query in v:
                stats = self.get_venue_stats(v)
                if stats:
                    results["venues"].append({
                        "venue": v,
                        "total_races": stats["total_races"],
                    })

        return results

    # ==========================================================
    # 6. JSON出力
    # ==========================================================

    def export_horse_json(self, horse_id: str, output_path: str = None) -> str:
        """馬の詳細をJSONファイルに出力"""
        detail = self.get_horse_detail(horse_id)
        if not detail:
            return ""

        if output_path is None:
            output_path = str(self.data_dir / f"horse_{horse_id}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)
        return output_path

    def export_dashboard_data(self, output_path: str = None) -> str:
        """ダッシュボード用の全データをJSON出力"""
        if output_path is None:
            output_path = str(self.data_dir / "db_dashboard.json")

        data = {
            "venues": self.list_venues(),
            "total_races": self.df["race_id"].nunique() if self.df is not None else 0,
            "total_horses": self.df["horse_id"].nunique() if self.df is not None else 0,
            "date_range": {
                "start": self.df["race_date"].min().strftime("%Y-%m-%d") if self.df is not None else "",
                "end": self.df["race_date"].max().strftime("%Y-%m-%d") if self.df is not None else "",
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return output_path


# ==========================================================
# テスト
# ==========================================================

if __name__ == "__main__":
    print("=" * 50)
    print("🏇 レースデータベース テスト")
    print("=" * 50)

    db = RaceDatabase(data_dir="data")

    # サンプルデータ生成
    from generate_large_data import LargeScaleSampleGenerator
    import os
    os.makedirs("data", exist_ok=True)

    if not Path("data/races_2024.csv").exists():
        gen = LargeScaleSampleGenerator(seed=42)
        gen.generate(start_year=2024, end_year=2024, output_dir="data")

    db.load("data/races_2024.csv")

    # 1. 馬名検索
    print("\n🐴 馬名検索: 'テスト馬1'")
    horses = db.search_horse("テスト馬1", limit=3)
    for h in horses:
        s = h["summary"]
        print(f"  {h['horse_name']}: {s['record']} (勝率{s['win_rate']}%)")

    # 2. 馬の詳細
    if horses:
        detail = db.get_horse_detail(horses[0]["horse_id"])
        if detail:
            print(f"\n  詳細: {detail['horse_name']}")
            print(f"  成績: {detail['summary']['record']}")
            print(f"  コース別: {detail['course_stats']}")
            print(f"  出走数: {len(detail['race_history'])} レース")

    # 3. レース検索
    print("\n🏇 レース検索: 'スプリング'")
    races = db.search_race("スプリング", limit=3)
    for r in races:
        print(f"  {r['date']} {r['race_name']} ({r['venue']}) → {r['winner']}")

    # 4. 競馬場統計
    venues = db.list_venues()
    if venues:
        print(f"\n🏟️ 競馬場: {venues}")
        stats = db.get_venue_stats(venues[0])
        if stats:
            print(f"  {stats['venue']}: {stats['total_races']} レース")
            print(f"  枠番別: {stats['frame_stats']}")

    # 5. カレンダー
    print("\n📅 カレンダー: 2024年7月")
    cal = db.get_calendar(2024, 7)
    for day in cal[:3]:
        venue_names = list(day["venues"].keys())
        print(f"  {day['date']} ({day['day_of_week']}): {day['total_races']}R @ {', '.join(venue_names)}")

    # 6. 統合検索
    print("\n🔍 統合検索: '札幌'")
    sr = db.search("札幌")
    print(f"  馬: {len(sr['horses'])}件 / レース: {len(sr['races'])}件 / 競馬場: {len(sr['venues'])}件")
