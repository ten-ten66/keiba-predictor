"""
SQLite-based query layer for keiba data.
=========================================
Converts the large CSV (_all_races.csv) into a SQLite database
with indexed columns for fast lookups, replacing full-DataFrame scans
for search, horse stats, jockey stats, etc.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

DB_PATH = Path("data/keiba.db")


def init_db():
    """Create SQLite database from CSV if not exists or CSV is newer."""
    csv_path = Path("data/_all_races.csv")

    if not csv_path.exists():
        logger.warning("CSV file not found: %s", csv_path)
        return

    if DB_PATH.exists() and DB_PATH.stat().st_mtime > csv_path.stat().st_mtime:
        logger.info("SQLite database is up to date: %s", DB_PATH)
        return  # DB is up to date

    logger.info("Building SQLite database from CSV...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Ensure ID columns are strings
    for col in ["race_id", "horse_id", "jockey_id", "trainer_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    conn = sqlite3.connect(str(DB_PATH))
    df.to_sql("races", conn, if_exists="replace", index=False)

    # Create indexes for fast queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_horse_id ON races(horse_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jockey_id ON races(jockey_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trainer_id ON races(trainer_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_race_id ON races(race_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_race_date ON races(race_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_venue ON races(venue)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_horse_name ON races(horse_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_race_name ON races(race_name)")
    conn.commit()
    conn.close()
    size_mb = DB_PATH.stat().st_size / 1024 / 1024
    logger.info("SQLite database created: %s (%.1f MB)", DB_PATH, size_mb)


def _get_conn() -> sqlite3.Connection:
    """Get a SQLite connection with row_factory set."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str, params=None) -> List[Dict]:
    """Execute a SQL query and return results as list of dicts."""
    conn = _get_conn()
    try:
        cursor = conn.execute(sql, params or [])
        results = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
    return results


def query_df(sql: str, params=None) -> pd.DataFrame:
    """Execute a SQL query and return as DataFrame."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    return df


def db_available() -> bool:
    """Check if SQLite DB exists and is usable."""
    return DB_PATH.exists() and DB_PATH.stat().st_size > 0


# ============================================================
# High-level query functions for API endpoints
# ============================================================

def search_horses(query_str: str, limit: int = 20) -> List[Dict]:
    """Search horses by name (partial match) using SQLite."""
    if not db_available():
        return []
    sql = """
        SELECT DISTINCT horse_id, horse_name
        FROM races
        WHERE horse_name LIKE ?
        LIMIT ?
    """
    return query(sql, [f"%{query_str}%", limit])


def search_races(query_str: str, limit: int = 20) -> List[Dict]:
    """Search races by name (partial match) using SQLite."""
    if not db_available():
        return []
    sql = """
        SELECT race_id, race_name, race_date, venue,
               COUNT(*) as n_horses
        FROM races
        WHERE race_name LIKE ?
        GROUP BY race_id
        ORDER BY race_date DESC
        LIMIT ?
    """
    return query(sql, [f"%{query_str}%", limit])


def get_horse_races(horse_id: str) -> pd.DataFrame:
    """Get all races for a specific horse as DataFrame."""
    if not db_available():
        return pd.DataFrame()
    return query_df(
        "SELECT * FROM races WHERE horse_id = ? ORDER BY race_date",
        [str(horse_id)]
    )


def get_jockey_races(jockey_id: str) -> pd.DataFrame:
    """Get all races for a specific jockey as DataFrame."""
    if not db_available():
        return pd.DataFrame()
    return query_df(
        "SELECT * FROM races WHERE jockey_id = ? ORDER BY race_date",
        [str(jockey_id)]
    )


def get_trainer_races(trainer_id: str) -> pd.DataFrame:
    """Get all races for a specific trainer as DataFrame."""
    if not db_available():
        return pd.DataFrame()
    return query_df(
        "SELECT * FROM races WHERE trainer_id = ? ORDER BY race_date",
        [str(trainer_id)]
    )


def get_venue_races(venue: str) -> pd.DataFrame:
    """Get all races at a specific venue as DataFrame."""
    if not db_available():
        return pd.DataFrame()
    return query_df(
        "SELECT * FROM races WHERE venue = ? ORDER BY race_date",
        [venue]
    )


def get_venues() -> List[str]:
    """Get list of all venues."""
    if not db_available():
        return []
    rows = query("SELECT DISTINCT venue FROM races WHERE venue IS NOT NULL ORDER BY venue")
    return [r["venue"] for r in rows]


def get_jockey_ranking_data(min_rides: int = 100, limit: int = 50) -> List[Dict]:
    """Get jockey ranking data using SQL aggregation."""
    if not db_available():
        return []
    sql = """
        SELECT
            jockey_id,
            MAX(jockey_name) as jockey_name,
            COUNT(*) as total_rides,
            SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) as top3,
            AVG(CASE WHEN prize_money IS NOT NULL THEN prize_money ELSE 0 END) as avg_prize
        FROM races
        WHERE jockey_id IS NOT NULL AND finish_position IS NOT NULL
        GROUP BY jockey_id
        HAVING COUNT(*) >= ?
        ORDER BY CAST(SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) DESC
        LIMIT ?
    """
    return query(sql, [min_rides, limit])


def get_trainer_ranking_data(min_rides: int = 50, limit: int = 50) -> List[Dict]:
    """Get trainer ranking data using SQL aggregation."""
    if not db_available():
        return []
    sql = """
        SELECT
            trainer_id,
            MAX(trainer_name) as trainer_name,
            COUNT(*) as total_rides,
            SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) as top3,
            AVG(CASE WHEN prize_money IS NOT NULL THEN prize_money ELSE 0 END) as avg_prize
        FROM races
        WHERE trainer_id IS NOT NULL AND finish_position IS NOT NULL
        GROUP BY trainer_id
        HAVING COUNT(*) >= ?
        ORDER BY CAST(SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) DESC
        LIMIT ?
    """
    return query(sql, [min_rides, limit])
