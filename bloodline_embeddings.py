"""
Bloodline Phenotype Embeddings
==============================
血統データからサイアー（父）・ダムサイアー（母父）の
表現型ベクトルを構築し、馬の条件適性を推定する。
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Embedding dimensions
# Distance bins (4) x 2 metrics = 8
# Surface (2) x 2 metrics = 4
# Track condition (4) x 1 metric = 4
# Age curve (4) x 1 metric = 4
# Stamina index (1) + early speed index (1) + wet aptitude (1) + dirt aptitude (1) = 4
SIRE_VEC_DIM = 24
COMBINED_VEC_DIM = SIRE_VEC_DIM * 2  # sire + dam_sire


class BloodlineEmbedder:
    """Build and query bloodline phenotype embeddings."""

    def __init__(self, history_df=None, bloodline_df=None):
        self.history = history_df
        self.bloodline = bloodline_df
        self.sire_vectors = {}
        self.dam_sire_vectors = {}
        self._sire_stats = {}
        self._dam_sire_stats = {}

    @classmethod
    def load_from_files(cls, races_csv=None, bloodline_csv=None):
        """Load data from CSV files and create an embedder."""
        if races_csv is None:
            races_csv = os.path.join(DATA_DIR, "_all_races.csv")
        if bloodline_csv is None:
            bloodline_csv = os.path.join(DATA_DIR, "bloodline.csv")

        history_df = None
        bloodline_df = None

        if os.path.exists(races_csv):
            history_df = pd.read_csv(races_csv, dtype={"horse_id": str})
            logger.info(f"Loaded {len(history_df)} race records")
        else:
            logger.warning(f"Race history not found: {races_csv}")

        if os.path.exists(bloodline_csv):
            bloodline_df = pd.read_csv(bloodline_csv, dtype={"horse_id": str})
            logger.info(f"Loaded {len(bloodline_df)} bloodline records")
        else:
            logger.warning(f"Bloodline data not found: {bloodline_csv}")

        return cls(history_df, bloodline_df)

    def build_embeddings(self, min_offspring=10):
        """
        Build phenotype vectors for each sire/dam_sire based on offspring performance.

        Parameters
        ----------
        min_offspring : int
            Minimum number of race records for a sire to generate an embedding.
        """
        if self.history is None or self.bloodline is None:
            logger.error("Cannot build embeddings: missing data")
            return

        # Ensure horse_id is string for merge
        self.history["horse_id"] = self.history["horse_id"].astype(str)
        self.bloodline["horse_id"] = self.bloodline["horse_id"].astype(str)

        # Merge race history with bloodline
        merged = self.history.merge(
            self.bloodline[["horse_id", "sire", "dam_sire"]],
            on="horse_id",
            how="left",
        )

        # Build sire vectors
        logger.info("Building sire embeddings...")
        self.sire_vectors, self._sire_stats = self._build_vectors(
            merged, "sire", min_offspring
        )
        logger.info(f"Built {len(self.sire_vectors)} sire embeddings")

        # Build dam_sire vectors (smaller sample sizes, use lower threshold)
        logger.info("Building dam_sire embeddings...")
        dam_min = max(5, min_offspring // 2)
        self.dam_sire_vectors, self._dam_sire_stats = self._build_vectors(
            merged, "dam_sire", dam_min
        )
        logger.info(f"Built {len(self.dam_sire_vectors)} dam_sire embeddings")

    def _build_vectors(self, merged, col, min_records):
        """Build phenotype vectors for a given column (sire or dam_sire)."""
        vectors = {}
        stats = {}

        for name in merged[col].dropna().unique():
            sire_data = merged[merged[col] == name]
            if len(sire_data) < min_records:
                continue

            vec = []
            stat = {"name": name, "record_count": len(sire_data)}

            # --- Performance by distance (4 bins x 2 metrics = 8) ---
            distance_bins = [
                (1000, 1400, "sprint"),
                (1400, 1800, "mile"),
                (1800, 2200, "intermediate"),
                (2200, 3600, "long"),
            ]
            for lo, hi, label in distance_bins:
                subset = sire_data[
                    (sire_data["distance"] >= lo) & (sire_data["distance"] < hi)
                ]
                if len(subset) > 3:
                    avg_pos = subset["finish_position"].mean()
                    top3_rate = (subset["finish_position"] <= 3).mean()
                else:
                    avg_pos = 8.0
                    top3_rate = 0.2
                vec.append(avg_pos)
                vec.append(top3_rate)
                stat[f"{label}_avg_pos"] = round(avg_pos, 2)
                stat[f"{label}_top3_rate"] = round(top3_rate, 3)

            # --- Performance by surface (2 surfaces x 2 metrics = 4) ---
            for surface in ["芝", "ダート"]:
                subset = sire_data[sire_data["course_type"] == surface]
                if len(subset) > 3:
                    avg_pos = subset["finish_position"].mean()
                    win_rate = (subset["finish_position"] <= 1).mean()
                else:
                    avg_pos = 8.0
                    win_rate = 0.1
                vec.append(avg_pos)
                vec.append(win_rate)
                label = "turf" if surface == "芝" else "dirt"
                stat[f"{label}_avg_pos"] = round(avg_pos, 2)
                stat[f"{label}_win_rate"] = round(win_rate, 3)

            # --- Performance by track condition (4 conditions x 1 metric = 4) ---
            for cond in ["良", "稍重", "重", "不良"]:
                subset = sire_data[sire_data["track_condition"] == cond]
                if len(subset) > 3:
                    top3_rate = (subset["finish_position"] <= 3).mean()
                else:
                    top3_rate = 0.2
                vec.append(top3_rate)

            # --- Age curve (4 ages x 1 metric = 4) ---
            for age in [2, 3, 4, 5]:
                subset = sire_data[sire_data["age"] == age]
                if len(subset) > 3:
                    avg_pos = subset["finish_position"].mean()
                else:
                    avg_pos = 8.0
                vec.append(avg_pos)

            # --- Derived indices (4) ---
            # Stamina index: long distance top3 rate vs sprint
            sprint_sub = sire_data[sire_data["distance"] < 1400]
            long_sub = sire_data[sire_data["distance"] >= 2200]
            sprint_rate = (sprint_sub["finish_position"] <= 3).mean() if len(sprint_sub) > 3 else 0.2
            long_rate = (long_sub["finish_position"] <= 3).mean() if len(long_sub) > 3 else 0.2
            stamina_idx = long_rate - sprint_rate  # positive = stamina type
            vec.append(stamina_idx)

            # Early speed index: average passing_order position (lower = faster start)
            # passing_order is like "3-2-1" -- take first number
            early_speeds = []
            if "passing_order" in sire_data.columns:
                for po in sire_data["passing_order"].dropna():
                    try:
                        first_pos = int(str(po).split("-")[0])
                        early_speeds.append(first_pos)
                    except (ValueError, IndexError):
                        pass
            early_speed_idx = np.mean(early_speeds) if early_speeds else 8.0
            vec.append(early_speed_idx)

            # Wet aptitude: heavy/bad track top3 rate vs good track
            good_sub = sire_data[sire_data["track_condition"] == "良"]
            wet_sub = sire_data[sire_data["track_condition"].isin(["重", "不良"])]
            good_rate = (good_sub["finish_position"] <= 3).mean() if len(good_sub) > 3 else 0.2
            wet_rate = (wet_sub["finish_position"] <= 3).mean() if len(wet_sub) > 3 else 0.2
            wet_aptitude = wet_rate - good_rate  # positive = better on wet
            vec.append(wet_aptitude)

            # Dirt aptitude: dirt win rate vs turf win rate
            turf_sub = sire_data[sire_data["course_type"] == "芝"]
            dirt_sub = sire_data[sire_data["course_type"] == "ダート"]
            turf_wr = (turf_sub["finish_position"] <= 1).mean() if len(turf_sub) > 3 else 0.1
            dirt_wr = (dirt_sub["finish_position"] <= 1).mean() if len(dirt_sub) > 3 else 0.1
            dirt_aptitude = dirt_wr - turf_wr
            vec.append(dirt_aptitude)

            stat["stamina_index"] = round(stamina_idx, 3)
            stat["early_speed_index"] = round(early_speed_idx, 2)
            stat["wet_aptitude"] = round(wet_aptitude, 3)
            stat["dirt_aptitude"] = round(dirt_aptitude, 3)

            vectors[name] = np.array(vec, dtype=np.float32)
            stats[name] = stat

        return vectors, stats

    def get_horse_embedding(self, horse_id):
        """
        Get combined sire + dam_sire embedding for a horse.

        Returns
        -------
        np.ndarray
            48-dimensional vector (24 sire + 24 dam_sire)
        """
        horse_id = str(horse_id)
        row = self.bloodline[self.bloodline["horse_id"] == horse_id] if self.bloodline is not None else pd.DataFrame()
        if row.empty:
            return np.zeros(COMBINED_VEC_DIM, dtype=np.float32)

        sire = row.iloc[0].get("sire", "")
        dam_sire = row.iloc[0].get("dam_sire", "")

        sire_vec = self.sire_vectors.get(sire, np.zeros(SIRE_VEC_DIM, dtype=np.float32))
        ds_vec = self.dam_sire_vectors.get(dam_sire, np.zeros(SIRE_VEC_DIM, dtype=np.float32))

        return np.concatenate([sire_vec, ds_vec])

    def get_horse_profile(self, horse_id):
        """
        Get human-readable bloodline profile for a horse.

        Returns
        -------
        dict
            Bloodline profile with phenotype strengths.
        """
        horse_id = str(horse_id)
        row = self.bloodline[self.bloodline["horse_id"] == horse_id] if self.bloodline is not None else pd.DataFrame()
        if row.empty:
            return {"horse_id": horse_id, "found": False, "message": "血統データなし"}

        r = row.iloc[0]
        sire = r.get("sire", "")
        dam_sire = r.get("dam_sire", "")
        dam = r.get("dam", "")
        sire_sire = r.get("sire_sire", "")

        profile = {
            "horse_id": horse_id,
            "found": True,
            "pedigree": {
                "sire": sire,
                "dam": dam,
                "sire_sire": sire_sire,
                "dam_sire": dam_sire,
            },
            "sire_profile": self._sire_stats.get(sire, {"name": sire, "record_count": 0}),
            "dam_sire_profile": self._dam_sire_stats.get(dam_sire, {"name": dam_sire, "record_count": 0}),
            "phenotype_strengths": [],
        }

        # Determine strengths from sire stats
        sire_stat = self._sire_stats.get(sire, {})
        strengths = []

        if sire_stat.get("stamina_index", 0) > 0.05:
            strengths.append("スタミナ型（長距離向き）")
        elif sire_stat.get("stamina_index", 0) < -0.05:
            strengths.append("スピード型（短距離向き）")

        if sire_stat.get("wet_aptitude", 0) > 0.03:
            strengths.append("重馬場巧者")
        elif sire_stat.get("wet_aptitude", 0) < -0.03:
            strengths.append("良馬場向き")

        if sire_stat.get("dirt_aptitude", 0) > 0.03:
            strengths.append("ダート適性高")
        elif sire_stat.get("dirt_aptitude", 0) < -0.03:
            strengths.append("芝適性高")

        if sire_stat.get("early_speed_index", 8) < 5:
            strengths.append("先行力あり")
        elif sire_stat.get("early_speed_index", 8) > 9:
            strengths.append("追込脚質傾向")

        # Best distance
        best_dist = None
        best_rate = 0
        for dist_label in ["sprint", "mile", "intermediate", "long"]:
            rate = sire_stat.get(f"{dist_label}_top3_rate", 0)
            if rate > best_rate:
                best_rate = rate
                best_dist = dist_label

        dist_labels_jp = {
            "sprint": "短距離（~1400m）",
            "mile": "マイル（1400~1800m）",
            "intermediate": "中距離（1800~2200m）",
            "long": "長距離（2200m~）",
        }
        if best_dist:
            strengths.append(f"得意距離: {dist_labels_jp.get(best_dist, best_dist)}")

        profile["phenotype_strengths"] = strengths

        return profile

    def predict_first_time_conditions(self, horse_id, distance, surface, condition):
        """
        Predict performance in conditions the horse hasn't tried before,
        using bloodline embeddings.

        Returns
        -------
        dict
            Estimated performance metrics.
        """
        embed = self.get_horse_embedding(horse_id)
        if np.all(embed == 0):
            return {"horse_id": str(horse_id), "prediction": "insufficient_data"}

        sire_vec = embed[:SIRE_VEC_DIM]
        ds_vec = embed[SIRE_VEC_DIM:]

        # Determine distance bin index
        dist_idx = 0
        if distance >= 2200:
            dist_idx = 3
        elif distance >= 1800:
            dist_idx = 2
        elif distance >= 1400:
            dist_idx = 1

        # Surface index offset (0=turf, 1=dirt)
        surf_idx = 1 if surface == "ダート" else 0

        # Condition index
        cond_map = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
        cond_idx = cond_map.get(condition, 0)

        # Extract relevant metrics from sire vector
        # Distance: indices 0-7 (4 bins x 2 metrics)
        sire_dist_avg = sire_vec[dist_idx * 2]
        sire_dist_top3 = sire_vec[dist_idx * 2 + 1]

        # Surface: indices 8-11 (2 surfaces x 2 metrics)
        sire_surf_avg = sire_vec[8 + surf_idx * 2]
        sire_surf_win = sire_vec[8 + surf_idx * 2 + 1]

        # Track condition: indices 12-15
        sire_cond_top3 = sire_vec[12 + cond_idx]

        # Same for dam_sire
        ds_dist_avg = ds_vec[dist_idx * 2] if len(ds_vec) > dist_idx * 2 else 8.0
        ds_dist_top3 = ds_vec[dist_idx * 2 + 1] if len(ds_vec) > dist_idx * 2 + 1 else 0.2

        # Weighted combination (sire 60%, dam_sire 40%)
        est_avg_pos = sire_dist_avg * 0.6 + ds_dist_avg * 0.4
        est_top3_rate = sire_dist_top3 * 0.6 + ds_dist_top3 * 0.4

        return {
            "horse_id": str(horse_id),
            "conditions": {
                "distance": distance,
                "surface": surface,
                "track_condition": condition,
            },
            "estimated_avg_position": round(float(est_avg_pos), 2),
            "estimated_top3_rate": round(float(est_top3_rate), 3),
            "sire_surface_win_rate": round(float(sire_surf_win), 3),
            "sire_condition_top3_rate": round(float(sire_cond_top3), 3),
            "confidence": "medium" if sire_dist_avg != 8.0 else "low",
        }

    def save(self, path=None):
        """Save embeddings to pickle."""
        if path is None:
            path = os.path.join(DATA_DIR, "bloodline_embeddings.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "sire_vectors": self.sire_vectors,
            "dam_sire_vectors": self.dam_sire_vectors,
            "sire_stats": self._sire_stats,
            "dam_sire_stats": self._dam_sire_stats,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved bloodline embeddings to {path} "
                     f"({len(self.sire_vectors)} sires, {len(self.dam_sire_vectors)} dam_sires)")

    def load(self, path=None):
        """Load embeddings from pickle."""
        if path is None:
            path = os.path.join(DATA_DIR, "bloodline_embeddings.pkl")
        if not os.path.exists(path):
            logger.warning(f"Embeddings file not found: {path}")
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.sire_vectors = data.get("sire_vectors", {})
        self.dam_sire_vectors = data.get("dam_sire_vectors", {})
        self._sire_stats = data.get("sire_stats", {})
        self._dam_sire_stats = data.get("dam_sire_stats", {})
        logger.info(f"Loaded bloodline embeddings: "
                     f"{len(self.sire_vectors)} sires, {len(self.dam_sire_vectors)} dam_sires")
        return True


# ---------- Singleton / lazy init ----------

_embedder = None


def get_embedder():
    """Get or create the global BloodlineEmbedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = BloodlineEmbedder()
        pkl_path = os.path.join(DATA_DIR, "bloodline_embeddings.pkl")
        if os.path.exists(pkl_path):
            _embedder.load(pkl_path)
            # Load bloodline CSV for profile lookups
            bl_csv = os.path.join(DATA_DIR, "bloodline.csv")
            if os.path.exists(bl_csv):
                _embedder.bloodline = pd.read_csv(bl_csv, dtype={"horse_id": str})
        else:
            logger.info("No cached embeddings found. Building from scratch...")
            _embedder = BloodlineEmbedder.load_from_files()
            if _embedder.history is not None and _embedder.bloodline is not None:
                _embedder.build_embeddings()
                _embedder.save(pkl_path)
    return _embedder


def build_and_save():
    """CLI helper: build embeddings from data files and save to pickle."""
    embedder = BloodlineEmbedder.load_from_files()
    if embedder.history is not None and embedder.bloodline is not None:
        embedder.build_embeddings()
        embedder.save()
        print(f"Done. {len(embedder.sire_vectors)} sire embeddings, "
              f"{len(embedder.dam_sire_vectors)} dam_sire embeddings saved.")
    else:
        print("Error: missing data files.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_and_save()
