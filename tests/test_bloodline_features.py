"""Unit tests for bloodline feature generation."""
import numpy as np
import pandas as pd
import pytest

from data_processor import FeatureEngineer


@pytest.fixture
def sample_df():
    # 3 horses, 2 races each, with lineage info
    rows = []
    horses = [
        ("H1", "Deep Impact", "Wind In Her Hair", "Sunday Silence", "Alzao"),
        ("H2", "Deep Impact", "Princess Olivia", "Sunday Silence", "Lomitas"),
        ("H3", "Kingmambo", "Miesque", "Mr. Prospector", "Nureyev"),
    ]
    for race_idx, race_date in enumerate(pd.date_range("2024-01-01", periods=4)):
        for pos, (hid, sire, dam, sire_sire, dam_sire) in enumerate(horses, start=1):
            rows.append({
                "race_id": f"R{race_idx:04d}",
                "race_date": race_date,
                "horse_id": hid,
                "horse_number": pos,
                "finish_position": (pos + race_idx) % 3 + 1,  # rotating winners
                "sire": sire,
                "dam": dam,
                "sire_sire": sire_sire,
                "dam_sire": dam_sire,
            })
    df = pd.DataFrame(rows)
    df["finish_position"] = df["finish_position"].astype(float)
    return df


def test_bloodline_features_created(sample_df):
    fe = FeatureEngineer()
    out = fe._god_bloodline(sample_df.sort_values(["race_date", "horse_number"]).reset_index(drop=True))
    expected = [
        "sire_win_rate", "sire_top3_rate", "sire_avg_pos", "sire_std_pos", "sire_n_runners",
        "dam_win_rate", "dam_top3_rate", "dam_avg_pos", "dam_std_pos", "dam_n_runners",
        "sire_sire_win_rate", "sire_sire_avg_pos",
        "dam_sire_win_rate", "dam_sire_top3_rate", "dam_sire_avg_pos",
    ]
    for col in expected:
        assert col in out.columns, f"missing {col}"


def test_no_future_leakage(sample_df):
    """Row N's win_rate must reflect only rows < N (no peeking at current/future)."""
    fe = FeatureEngineer()
    df = sample_df.sort_values(["race_date", "horse_number"]).reset_index(drop=True)
    out = fe._god_bloodline(df)
    # First occurrence of each sire: n_runners (counts prior observations) must be 0
    first_of_sire = out.groupby("sire").head(1)
    assert (first_of_sire["sire_n_runners"].fillna(0) == 0).all()


def test_missing_columns_dont_crash(sample_df):
    """If lineage columns are missing, method returns without error."""
    fe = FeatureEngineer()
    df = sample_df.drop(columns=["sire_sire", "dam_sire"])
    out = fe._god_bloodline(df.sort_values(["race_date", "horse_number"]).reset_index(drop=True))
    # sire/dam features still created
    assert "sire_win_rate" in out.columns
    # sire_sire/dam_sire features not created
    assert "sire_sire_win_rate" not in out.columns
    assert "dam_sire_win_rate" not in out.columns


def test_all_nan_lineage_skipped(sample_df):
    """If a lineage column exists but is all NaN, features are not created."""
    fe = FeatureEngineer()
    df = sample_df.copy()
    df["sire_sire"] = np.nan
    out = fe._god_bloodline(df.sort_values(["race_date", "horse_number"]).reset_index(drop=True))
    assert "sire_sire_win_rate" not in out.columns
