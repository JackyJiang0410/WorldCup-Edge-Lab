from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import unicodedata

import numpy as np
import pandas as pd

from .constants import REQUIRED_DATASET_FILES
from .team_names import normalize_team_name


@dataclass(slots=True)
class DatasetValidation:
    dataset_dir: Path
    file_count: int
    row_count: int
    missing_files: list[str]
    present_files: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_files

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "file_count": self.file_count,
            "row_count": self.row_count,
            "missing_files": self.missing_files,
            "present_files": self.present_files,
            "ok": self.ok,
        }


def read_csv(dataset_dir: str | Path, filename: str) -> pd.DataFrame:
    path = Path(dataset_dir) / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def validate_dataset(dataset_dir: str | Path) -> DatasetValidation:
    base = Path(dataset_dir)
    files = sorted(path.name for path in base.glob("*.csv"))
    missing = sorted(REQUIRED_DATASET_FILES - set(files))
    row_count = 0
    for name in files:
        with (base / name).open(newline="", encoding="utf-8-sig") as handle:
            row_count += max(sum(1 for _ in handle) - 1, 0)
    return DatasetValidation(base, len(files), row_count, missing, files)


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().eq("TRUE")


def parse_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", np.nan), errors="coerce")


def parse_date(series: pd.Series, *, dayfirst: bool = False) -> pd.Series:
    return pd.to_datetime(series.replace("", np.nan), errors="coerce", dayfirst=dayfirst)


def normalize_person_key(*parts: str) -> str:
    text = " ".join(part for part in parts if part).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text).split())


def load_country_name_map(dataset_dir: str | Path) -> dict[str, str]:
    path = Path(dataset_dir) / "countries_names.csv"
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.reader(handle)
        next(rows, None)
        for row in rows:
            if len(row) < 2:
                continue
            mapping[normalize_team_name(row[0])] = normalize_team_name(row[1])
    return mapping


def load_all_matches(dataset_dir: str | Path) -> pd.DataFrame:
    matches = read_csv(dataset_dir, "all_matches.csv")
    country_map = load_country_name_map(dataset_dir)
    matches["match_dt"] = parse_date(matches["date"])
    matches["home_team_norm"] = matches["home_team"].map(normalize_team_name).replace(country_map)
    matches["away_team_norm"] = matches["away_team"].map(normalize_team_name).replace(country_map)
    matches["country_norm"] = matches["country"].map(normalize_team_name).replace(country_map)
    matches["home_score_num"] = parse_number(matches["home_score"])
    matches["away_score_num"] = parse_number(matches["away_score"])
    matches["neutral_bool"] = matches["neutral"].astype(str).str.lower().eq("true")
    return matches.sort_values(["match_dt", "home_team_norm", "away_team_norm"]).reset_index(drop=True)


def load_tournaments(dataset_dir: str | Path) -> pd.DataFrame:
    tournaments = read_csv(dataset_dir, "tournaments_curated.csv")
    tournaments["year_num"] = parse_number(tournaments["year"]).astype("Int64")
    tournaments["men_bool"] = parse_bool(tournaments["men"])
    tournaments["women_bool"] = parse_bool(tournaments["women"])
    tournaments["start_dt"] = parse_date(tournaments["start_date"])
    tournaments["end_dt"] = parse_date(tournaments["end_date"])
    return tournaments


def load_matches_with_tournaments(dataset_dir: str | Path) -> pd.DataFrame:
    matches = read_csv(dataset_dir, "matches_curated.csv")
    tournaments = load_tournaments(dataset_dir)[
        ["tournament_id", "year_num", "men_bool", "women_bool", "host_country", "start_dt"]
    ]
    merged = matches.merge(tournaments, on="tournament_id", how="left")
    merged["match_dt"] = parse_date(merged["match_date"])
    numeric_cols = [
        "home_team_score",
        "away_team_score",
        "home_team_score_penalties",
        "away_team_score_penalties",
    ]
    for col in numeric_cols:
        if col in merged:
            merged[col] = parse_number(merged[col])
    for col in ("extra_time", "penalty_shootout", "group_stage", "knockout_stage"):
        if col in merged:
            merged[f"{col}_bool"] = parse_bool(merged[col])
    for col in ("home_team_name", "away_team_name", "host_country"):
        merged[f"{col}_norm"] = merged[col].map(normalize_team_name)
    return merged


def load_mens_matches(dataset_dir: str | Path) -> pd.DataFrame:
    matches = load_matches_with_tournaments(dataset_dir)
    mens = matches[matches["men_bool"]].copy()
    return mens.sort_values(["match_dt", "match_id"]).reset_index(drop=True)


def load_fifa_rankings_timeseries(dataset_dir: str | Path) -> pd.DataFrame:
    fifa = read_csv(dataset_dir, "fifa_rankings_timeseries.csv")
    fifa["rating_date"] = parse_date(fifa["rank_date"])
    fifa["team_norm"] = fifa["country_full"].map(normalize_team_name)
    fifa["fifa_rank"] = parse_number(fifa["rank"])
    fifa["fifa_points"] = parse_number(fifa["total_points"])
    fifa = fifa.sort_values(
        ["rating_date", "team_norm", "fifa_rank", "fifa_points"],
        na_position="last",
        kind="mergesort",
    )
    fifa = fifa.drop_duplicates(["rating_date", "team_norm"], keep="first")
    return fifa[["rating_date", "team_norm", "fifa_rank", "fifa_points"]].reset_index(drop=True)


def load_elo_ratings_timeseries(dataset_dir: str | Path) -> pd.DataFrame:
    elo = read_csv(dataset_dir, "elo_ratings_timeseries.csv")
    elo["rating_date"] = parse_date(elo["snapshot_date"])
    elo["team_norm"] = elo["country"].map(normalize_team_name)
    elo["elo_rank"] = parse_number(elo["rank"])
    elo["elo_rating"] = parse_number(elo["rating"])
    elo["elo_rank_avg"] = parse_number(elo["rank_avg"])
    elo["elo_rating_avg"] = parse_number(elo["rating_avg"])
    return elo[
        ["rating_date", "team_norm", "elo_rank", "elo_rating", "elo_rank_avg", "elo_rating_avg"]
    ].sort_values(["team_norm", "rating_date"]).reset_index(drop=True)


def load_fifa_rankings_current(dataset_dir: str | Path) -> pd.DataFrame:
    fifa = read_csv(dataset_dir, "fifa_rankings_current.csv")
    fifa["team_norm"] = fifa["team"].map(normalize_team_name)
    fifa["fifa_rank"] = parse_number(fifa["rank"])
    fifa["fifa_points"] = parse_number(fifa["points"])
    return fifa[["team_norm", "fifa_rank", "fifa_points"]].drop_duplicates("team_norm")


def load_elo_ratings_current(dataset_dir: str | Path) -> pd.DataFrame:
    elo = read_csv(dataset_dir, "elo_ratings_current.csv")
    elo["team_norm"] = elo["team"].map(normalize_team_name)
    elo["elo_rank"] = parse_number(elo["rank"])
    elo["elo_rating"] = parse_number(elo["rating"])
    elo["elo_rank_avg"] = parse_number(elo["average_rank"])
    elo["elo_rating_avg"] = parse_number(elo["average_rating"])
    return elo[
        ["team_norm", "elo_rank", "elo_rating", "elo_rank_avg", "elo_rating_avg"]
    ].drop_duplicates("team_norm")


class AsOfRatingLookup:
    def __init__(self, ratings: pd.DataFrame, value_columns: list[str]):
        self.value_columns = value_columns
        self._by_team: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
        for team, group in ratings.dropna(subset=["rating_date"]).groupby("team_norm"):
            ordered = group.sort_values("rating_date").reset_index(drop=True)
            dates = ordered["rating_date"].values.astype("datetime64[ns]")
            self._by_team[str(team)] = (dates, ordered[value_columns])

    def lookup(self, team: str, match_date: pd.Timestamp) -> dict[str, float]:
        team_norm = normalize_team_name(team)
        default = {col: np.nan for col in self.value_columns}
        default.update({f"{col}_missing": 1.0 for col in self.value_columns})
        if team_norm not in self._by_team or pd.isna(match_date):
            return default
        dates, values = self._by_team[team_norm]
        pos = int(np.searchsorted(dates, np.datetime64(match_date), side="right") - 1)
        if pos < 0:
            return default
        row = values.iloc[pos]
        result = {col: float(row[col]) if pd.notna(row[col]) else np.nan for col in self.value_columns}
        result.update({f"{col}_missing": 0.0 if pd.notna(row[col]) else 1.0 for col in self.value_columns})
        return result


def load_rating_lookups(dataset_dir: str | Path) -> tuple[AsOfRatingLookup, AsOfRatingLookup]:
    fifa_lookup = AsOfRatingLookup(
        load_fifa_rankings_timeseries(dataset_dir), ["fifa_rank", "fifa_points"]
    )
    elo_lookup = AsOfRatingLookup(
        load_elo_ratings_timeseries(dataset_dir),
        ["elo_rank", "elo_rating", "elo_rank_avg", "elo_rating_avg"],
    )
    return fifa_lookup, elo_lookup


def build_historical_squad_features(dataset_dir: str | Path) -> pd.DataFrame:
    squads = read_csv(dataset_dir, "squads_curated.csv")
    players = read_csv(dataset_dir, "players_curated.csv")[["player_id", "birth_date"]]
    tournaments = load_tournaments(dataset_dir)[
        ["tournament_id", "start_dt", "year_num", "men_bool"]
    ]
    merged = squads.merge(players, on="player_id", how="left").merge(tournaments, on="tournament_id")
    merged = merged[merged["men_bool"]].copy()
    merged["birth_dt"] = parse_date(merged["birth_date"])
    merged["age"] = (merged["start_dt"] - merged["birth_dt"]).dt.days / 365.25
    merged["team_norm"] = merged["team_name"].map(normalize_team_name)
    for pos in ("GK", "DF", "MF", "FW"):
        merged[f"pos_{pos.lower()}"] = merged["squad_position_code"].eq(pos).astype(float)

    prior_squads = merged[["tournament_id", "player_id", "start_dt"]].drop_duplicates().sort_values(
        ["player_id", "start_dt", "tournament_id"]
    )
    prior_squads["prior_wc_squad_tournaments"] = (
        prior_squads.groupby("player_id").cumcount().astype(float)
    )

    appearances = read_csv(dataset_dir, "player_appearances_curated.csv")
    appearances = appearances.merge(tournaments, on="tournament_id", how="left")
    appearances = appearances[appearances["men_bool"]].copy()
    app_by_tournament = appearances.groupby(["player_id", "tournament_id"], as_index=False).agg(
        apps=("match_id", "count"),
        starts=("starter", lambda values: values.astype(str).str.upper().eq("TRUE").sum()),
    )
    app_by_tournament = app_by_tournament.merge(
        tournaments[["tournament_id", "start_dt"]], on="tournament_id", how="left"
    ).sort_values(["player_id", "start_dt", "tournament_id"])
    app_by_tournament["prior_wc_apps"] = (
        app_by_tournament.groupby("player_id")["apps"].cumsum() - app_by_tournament["apps"]
    )
    app_by_tournament["prior_wc_starts"] = (
        app_by_tournament.groupby("player_id")["starts"].cumsum() - app_by_tournament["starts"]
    )

    goals = read_csv(dataset_dir, "goals_curated.csv")
    goals = goals.merge(tournaments, on="tournament_id", how="left")
    goals = goals[goals["men_bool"] & goals["own_goal"].astype(str).str.upper().ne("TRUE")].copy()
    goals_by_tournament = goals.groupby(["player_id", "tournament_id"], as_index=False).agg(
        goals=("goal_id", "count")
    )
    goals_by_tournament = goals_by_tournament.merge(
        tournaments[["tournament_id", "start_dt"]], on="tournament_id", how="left"
    ).sort_values(["player_id", "start_dt", "tournament_id"])
    goals_by_tournament["prior_wc_goals"] = (
        goals_by_tournament.groupby("player_id")["goals"].cumsum() - goals_by_tournament["goals"]
    )

    merged = merged.merge(
        prior_squads[["tournament_id", "player_id", "prior_wc_squad_tournaments"]],
        on=["tournament_id", "player_id"],
        how="left",
    )
    merged = merged.merge(
        app_by_tournament[["tournament_id", "player_id", "prior_wc_apps", "prior_wc_starts"]],
        on=["tournament_id", "player_id"],
        how="left",
    )
    merged = merged.merge(
        goals_by_tournament[["tournament_id", "player_id", "prior_wc_goals"]],
        on=["tournament_id", "player_id"],
        how="left",
    )
    for col in ("prior_wc_squad_tournaments", "prior_wc_apps", "prior_wc_starts", "prior_wc_goals"):
        merged[col] = parse_number(merged[col]).fillna(0.0)
    merged["age_u23"] = (merged["age"] < 23).astype(float)
    merged["age_peak"] = ((merged["age"] >= 24) & (merged["age"] <= 29)).astype(float)
    merged["age_30plus"] = (merged["age"] >= 30).astype(float)
    merged["has_prior_wc_squad"] = (merged["prior_wc_squad_tournaments"] > 0).astype(float)
    merged["has_prior_wc_apps"] = (merged["prior_wc_apps"] > 0).astype(float)
    merged["has_prior_wc_goals"] = (merged["prior_wc_goals"] > 0).astype(float)
    grouped = merged.groupby(["tournament_id", "team_norm"], as_index=False).agg(
        squad_size=("player_id", "nunique"),
        squad_avg_age=("age", "mean"),
        squad_age_std=("age", "std"),
        squad_u23_share=("age_u23", "mean"),
        squad_peak_age_share=("age_peak", "mean"),
        squad_30plus_share=("age_30plus", "mean"),
        squad_gk_share=("pos_gk", "mean"),
        squad_df_share=("pos_df", "mean"),
        squad_mf_share=("pos_mf", "mean"),
        squad_fw_share=("pos_fw", "mean"),
        squad_prior_wc_squad_mean=("prior_wc_squad_tournaments", "mean"),
        squad_prior_wc_squad_max=("prior_wc_squad_tournaments", "max"),
        squad_prior_wc_squad_share=("has_prior_wc_squad", "mean"),
        squad_prior_wc_apps_mean=("prior_wc_apps", "mean"),
        squad_prior_wc_apps_max=("prior_wc_apps", "max"),
        squad_prior_wc_apps_share=("has_prior_wc_apps", "mean"),
        squad_prior_wc_starts_mean=("prior_wc_starts", "mean"),
        squad_prior_wc_goals_mean=("prior_wc_goals", "mean"),
        squad_prior_wc_goals_max=("prior_wc_goals", "max"),
        squad_prior_wc_goals_share=("has_prior_wc_goals", "mean"),
    )
    grouped["squad_age_std"] = grouped["squad_age_std"].fillna(0.0)
    return grouped


def _player_prior_totals(dataset_dir: str | Path, as_of_date: pd.Timestamp) -> pd.DataFrame:
    tournaments = load_tournaments(dataset_dir)[["tournament_id", "start_dt", "men_bool"]]
    squads = read_csv(dataset_dir, "squads_curated.csv").merge(tournaments, on="tournament_id")
    squads = squads[squads["men_bool"] & (squads["start_dt"] < as_of_date)].copy()
    squad_totals = squads.groupby("player_id", as_index=False).agg(
        prior_wc_squad_tournaments=("tournament_id", "nunique")
    )

    appearances = read_csv(dataset_dir, "player_appearances_curated.csv").merge(
        tournaments, on="tournament_id"
    )
    appearances = appearances[
        appearances["men_bool"] & (appearances["start_dt"] < as_of_date)
    ].copy()
    app_totals = appearances.groupby("player_id", as_index=False).agg(
        prior_wc_apps=("match_id", "count"),
        prior_wc_starts=("starter", lambda values: values.astype(str).str.upper().eq("TRUE").sum()),
    )

    goals = read_csv(dataset_dir, "goals_curated.csv").merge(tournaments, on="tournament_id")
    goals = goals[
        goals["men_bool"]
        & (goals["start_dt"] < as_of_date)
        & goals["own_goal"].astype(str).str.upper().ne("TRUE")
    ].copy()
    goal_totals = goals.groupby("player_id", as_index=False).agg(prior_wc_goals=("goal_id", "count"))
    return squad_totals.merge(app_totals, on="player_id", how="outer").merge(
        goal_totals, on="player_id", how="outer"
    )


def build_2026_squad_features(dataset_dir: str | Path) -> pd.DataFrame:
    squads = read_csv(dataset_dir, "world_cup_2026_squads.csv")
    ref_date = pd.Timestamp("2026-06-11")
    squads["team_norm"] = squads["national_team"].map(normalize_team_name)
    squads["birth_dt"] = parse_date(squads["dob"], dayfirst=True)
    squads["age"] = (ref_date - squads["birth_dt"]).dt.days / 365.25
    squads["person_key"] = [
        normalize_person_key(first, last)
        for first, last in zip(squads["first_names"], squads["last_names"], strict=False)
    ]

    players = read_csv(dataset_dir, "players_curated.csv")
    players["birth_dt"] = parse_date(players["birth_date"])
    players["person_key"] = [
        normalize_person_key(given, family)
        for given, family in zip(players["given_name"], players["family_name"], strict=False)
    ]
    players = players[players["male"].astype(str).str.upper().eq("TRUE")].copy()
    players = players.drop_duplicates(["person_key", "birth_dt"], keep="first")
    squads = squads.merge(
        players[["player_id", "person_key", "birth_dt"]],
        on=["person_key", "birth_dt"],
        how="left",
        suffixes=("", "_curated"),
    )
    prior_totals = _player_prior_totals(dataset_dir, ref_date)
    squads = squads.merge(prior_totals, on="player_id", how="left")
    for col in ("prior_wc_squad_tournaments", "prior_wc_apps", "prior_wc_starts", "prior_wc_goals"):
        squads[col] = parse_number(squads[col]).fillna(0.0)
    for pos in ("GK", "DF", "MF", "FW"):
        squads[f"pos_{pos.lower()}"] = squads["pos"].eq(pos).astype(float)
    squads["age_u23"] = (squads["age"] < 23).astype(float)
    squads["age_peak"] = ((squads["age"] >= 24) & (squads["age"] <= 29)).astype(float)
    squads["age_30plus"] = (squads["age"] >= 30).astype(float)
    squads["has_prior_wc_squad"] = (squads["prior_wc_squad_tournaments"] > 0).astype(float)
    squads["has_prior_wc_apps"] = (squads["prior_wc_apps"] > 0).astype(float)
    squads["has_prior_wc_goals"] = (squads["prior_wc_goals"] > 0).astype(float)
    grouped = squads.groupby("team_norm", as_index=False).agg(
        squad_size=("player_name", "nunique"),
        squad_avg_age=("age", "mean"),
        squad_age_std=("age", "std"),
        squad_u23_share=("age_u23", "mean"),
        squad_peak_age_share=("age_peak", "mean"),
        squad_30plus_share=("age_30plus", "mean"),
        squad_gk_share=("pos_gk", "mean"),
        squad_df_share=("pos_df", "mean"),
        squad_mf_share=("pos_mf", "mean"),
        squad_fw_share=("pos_fw", "mean"),
        squad_prior_wc_squad_mean=("prior_wc_squad_tournaments", "mean"),
        squad_prior_wc_squad_max=("prior_wc_squad_tournaments", "max"),
        squad_prior_wc_squad_share=("has_prior_wc_squad", "mean"),
        squad_prior_wc_apps_mean=("prior_wc_apps", "mean"),
        squad_prior_wc_apps_max=("prior_wc_apps", "max"),
        squad_prior_wc_apps_share=("has_prior_wc_apps", "mean"),
        squad_prior_wc_starts_mean=("prior_wc_starts", "mean"),
        squad_prior_wc_goals_mean=("prior_wc_goals", "mean"),
        squad_prior_wc_goals_max=("prior_wc_goals", "max"),
        squad_prior_wc_goals_share=("has_prior_wc_goals", "mean"),
    )
    grouped["squad_age_std"] = grouped["squad_age_std"].fillna(0.0)
    return grouped


def build_historical_manager_features(dataset_dir: str | Path) -> pd.DataFrame:
    managers = read_csv(dataset_dir, "manager_appointments_curated.csv")
    managers["team_norm"] = managers["team_name"].map(normalize_team_name)
    managers["manager_country_norm"] = managers["country_name"].map(normalize_team_name)
    managers["manager_same_country"] = (
        managers["team_norm"].eq(managers["manager_country_norm"]).astype(float)
    )
    return managers.groupby(["tournament_id", "team_norm"], as_index=False).agg(
        manager_same_country=("manager_same_country", "max")
    )


def build_2026_manager_features(dataset_dir: str | Path) -> pd.DataFrame:
    managers = read_csv(dataset_dir, "world_cup_2026_managers.csv")
    managers["team_norm"] = managers["national_team"].map(normalize_team_name)
    managers["manager_country_norm"] = managers["nationality"].map(normalize_team_name)
    managers["manager_same_country"] = (
        managers["team_norm"].eq(managers["manager_country_norm"]).astype(float)
    )
    return managers[["team_norm", "manager_same_country"]].drop_duplicates("team_norm")


def load_2026_schedule(dataset_dir: str | Path) -> pd.DataFrame:
    schedule = read_csv(dataset_dir, "world_cup_2026_schedule.csv")
    schedule["match_number"] = parse_number(schedule["Match Number"]).astype("Int64")
    schedule["match_dt"] = parse_date(schedule["Date"], dayfirst=True)
    schedule["home_team_norm"] = schedule["Home Team"].map(normalize_team_name)
    schedule["away_team_norm"] = schedule["Away Team"].map(normalize_team_name)
    return schedule
