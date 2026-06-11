from __future__ import annotations

from pathlib import Path

from worldcup2026.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    augment_neutral_home_away_swap,
    build_split,
    build_2026_match_features,
    build_training_features,
)


DATASET = Path("Dataset")


def test_training_features_build_and_use_mens_matches():
    features = build_training_features(DATASET)
    assert len(features) == 964
    assert set(features["label"]).issubset({"home_win", "draw", "away_win"})
    assert {"home_elo_rating", "away_elo_rating", "squad_size_diff"}.issubset(features.columns)
    assert "stage_type" in CATEGORICAL_FEATURES
    assert "stage_name" not in CATEGORICAL_FEATURES
    assert "group_name" not in CATEGORICAL_FEATURES
    assert "group_stage" not in NUMERIC_FEATURES
    assert "knockout_stage" not in NUMERIC_FEATURES
    assert not any("manager" in feature for feature in NUMERIC_FEATURES)
    assert not any("manager" in column for column in features.columns)
    assert set(features["stage_type"]).issubset({"group", "knockout", "unknown"})


def test_split_contains_each_tournament_in_train_and_test():
    features = build_training_features(DATASET)
    train_idx, test_idx = build_split(features, seed=42, test_size=0.2)
    train_tournaments = set(features.iloc[train_idx]["tournament_id"])
    test_tournaments = set(features.iloc[test_idx]["tournament_id"])
    assert train_tournaments == test_tournaments == set(features["tournament_id"])


def test_neutral_home_away_swap_augmentation_flips_orientation():
    features = build_training_features(DATASET)
    neutral = features[(features["home_host"] == 0.0) & (features["away_host"] == 0.0)].head(1)
    swapped = augment_neutral_home_away_swap(neutral)
    assert len(swapped) == 1
    assert swapped.iloc[0]["home_team"] == neutral.iloc[0]["away_team"]
    assert swapped.iloc[0]["away_team"] == neutral.iloc[0]["home_team"]
    assert swapped.iloc[0]["home_goals"] == neutral.iloc[0]["away_goals"]
    assert swapped.iloc[0]["away_goals"] == neutral.iloc[0]["home_goals"]


def test_2026_stage_type_uses_group_or_knockout_only():
    group = build_2026_match_features(
        DATASET, team="Mexico", opponent="South Africa", match_number=1
    )
    knockout = build_2026_match_features(
        DATASET, team="Brazil", opponent="Morocco", match_number=97
    )
    assert group.iloc[0]["stage_type"] == "group"
    assert knockout.iloc[0]["stage_type"] == "knockout"
