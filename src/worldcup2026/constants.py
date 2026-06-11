from __future__ import annotations

from pathlib import Path

LABEL_TO_ID = {"home_win": 0, "draw": 1, "away_win": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

DEFAULT_DATASET_DIR = Path("Dataset")
DEFAULT_ARTIFACT_DIR = Path("artifacts/v1")
DEFAULT_REPORT_DIR = Path("reports")

REQUIRED_DATASET_FILES = {
    "award_winners_curated.csv",
    "awards_curated.csv",
    "all_matches.csv",
    "bookings_curated.csv",
    "confederations_curated.csv",
    "countries_names.csv",
    "elo_ratings_current.csv",
    "elo_ratings_timeseries.csv",
    "fifa_rankings_current.csv",
    "fifa_rankings_timeseries.csv",
    "goals_curated.csv",
    "group_standings_curated.csv",
    "groups_curated.csv",
    "host_countries_curated.csv",
    "manager_appearances_curated.csv",
    "manager_appointments_curated.csv",
    "managers_curated.csv",
    "matches_curated.csv",
    "penalty_kicks_curated.csv",
    "player_appearances_curated.csv",
    "players_curated.csv",
    "qualified_teams_curated.csv",
    "referee_appearances_curated.csv",
    "referee_appointments_curated.csv",
    "referees_curated.csv",
    "squads_curated.csv",
    "stadiums_curated.csv",
    "substitutions_curated.csv",
    "team_appearances_curated.csv",
    "teams_curated.csv",
    "tournament_stages_curated.csv",
    "tournament_standings_curated.csv",
    "tournaments_curated.csv",
    "world_cup_2026_managers.csv",
    "world_cup_2026_schedule.csv",
    "world_cup_2026_squads.csv",
    "world_cup_goals.csv",
    "world_cup_matches.csv",
}

PROHIBITED_TRADE_TEAMS = {"argentina", "spain"}

TEAM_ALIASES = {
    "usa": "United States",
    "united states of america": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "ir iran": "Iran",
    "iran": "Iran",
    "korea republic": "South Korea",
    "south korea": "South Korea",
    "turkiye": "Turkey",
    "turkey": "Turkey",
    "congo dr": "DR Congo",
    "dr congo": "DR Congo",
    "d.r. congo": "DR Congo",
    "cabo verde": "Cape Verde",
    "cape verde": "Cape Verde",
    "cape verde islands": "Cape Verde",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "czechia": "Czech Republic",
    "czech republic": "Czech Republic",
    "england": "England",
    "scotland": "Scotland",
    "wales": "Wales",
    "northern ireland": "Northern Ireland",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "curaçao": "Curacao",
    "curacao": "Curacao",
}
