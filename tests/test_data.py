from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup2026.data import (
    load_fifa_rankings_timeseries,
    load_rating_lookups,
    validate_dataset,
)
from worldcup2026.constants import REQUIRED_DATASET_FILES
from worldcup2026.team_names import normalize_team_name


DATASET = Path("Dataset")


def test_required_dataset_files_exist():
    validation = validate_dataset(DATASET)
    assert validation.ok
    assert validation.file_count >= len(REQUIRED_DATASET_FILES)


def test_team_aliases_cover_2026_rating_names():
    assert normalize_team_name("USA") == "United States"
    assert normalize_team_name("IR Iran") == "Iran"
    assert normalize_team_name("Korea Republic") == "South Korea"
    assert normalize_team_name("Türkiye") == "Turkey"
    assert normalize_team_name("Congo DR") == "DR Congo"
    assert normalize_team_name("Cabo Verde") == "Cape Verde"


def test_fifa_rankings_are_deduplicated():
    fifa = load_fifa_rankings_timeseries(DATASET)
    assert not fifa.duplicated(["rating_date", "team_norm"]).any()


def test_rating_asof_lookup_does_not_use_future_rows():
    fifa_lookup, elo_lookup = load_rating_lookups(DATASET)
    match_date = pd.Timestamp("2018-06-14")
    fifa = fifa_lookup.lookup("Brazil", match_date)
    elo = elo_lookup.lookup("Brazil", match_date)
    assert fifa["fifa_rank_missing"] == 0.0
    assert elo["elo_rating_missing"] == 0.0
