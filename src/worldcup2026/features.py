from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import LABEL_TO_ID
from .data import (
    build_2026_squad_features,
    build_historical_squad_features,
    load_all_matches,
    load_2026_schedule,
    load_elo_ratings_current,
    load_fifa_rankings_current,
    load_mens_matches,
    load_rating_lookups,
    read_csv,
)
from .team_names import normalize_team_name


NUMERIC_FEATURES = [
    "year_num",
    "era_team_goal_rate",
    "home_host",
    "away_host",
    "host_diff",
    "played_diff",
    "played_total",
    "experience_log_diff",
    "experience_log_total",
    "win_rate_diff",
    "draw_rate_diff",
    "loss_rate_diff",
    "gf_avg_diff",
    "ga_avg_diff",
    "gd_avg_diff",
    "gf_avg_pair_avg",
    "ga_avg_pair_avg",
    "gd_avg_pair_avg",
    "h2h_played_total",
    "h2h_win_rate_diff",
    "h2h_draw_rate_avg",
    "all_recent_form_5_diff",
    "all_recent_form_10_diff",
    "all_recent_form_20_diff",
    "all_recent_form_10_pair_avg",
    "all_recent_form_weighted_diff",
    "all_recent_gf_10_diff",
    "all_recent_ga_10_diff",
    "all_recent_gd_10_diff",
    "all_recent_gd_weighted_diff",
    "all_recent_competitive_form_diff",
    "all_recent_friendly_form_diff",
    "all_h2h_played",
    "all_h2h_win_rate_diff",
    "all_h2h_draw_rate",
    "home_fifa_rank",
    "away_fifa_rank",
    "fifa_rank_diff",
    "fifa_rank_pair_avg",
    "home_fifa_points",
    "away_fifa_points",
    "fifa_points_diff",
    "fifa_points_pair_avg",
    "fifa_missing_any",
    "home_elo_rank",
    "away_elo_rank",
    "elo_rank_diff",
    "elo_rank_pair_avg",
    "home_elo_rating",
    "away_elo_rating",
    "elo_rating_diff",
    "elo_rating_pair_avg",
    "home_elo_rating_avg",
    "away_elo_rating_avg",
    "elo_rating_avg_diff",
    "elo_rating_avg_pair_avg",
    "elo_missing_any",
    "home_squad_avg_age",
    "away_squad_avg_age",
    "squad_avg_age_diff",
    "squad_avg_age_pair_avg",
    "squad_age_std_diff",
    "squad_age_std_pair_avg",
    "squad_u23_share_diff",
    "squad_u23_share_pair_avg",
    "squad_peak_age_share_diff",
    "squad_peak_age_share_pair_avg",
    "squad_30plus_share_diff",
    "squad_30plus_share_pair_avg",
    "squad_gk_share_diff",
    "squad_gk_share_pair_avg",
    "squad_df_share_diff",
    "squad_df_share_pair_avg",
    "squad_mf_share_diff",
    "squad_mf_share_pair_avg",
    "squad_fw_share_diff",
    "squad_fw_share_pair_avg",
    "squad_prior_wc_squad_mean_diff",
    "squad_prior_wc_squad_mean_pair_avg",
    "squad_prior_wc_squad_share_diff",
    "squad_prior_wc_squad_share_pair_avg",
    "squad_prior_wc_apps_mean_diff",
    "squad_prior_wc_apps_mean_pair_avg",
    "squad_prior_wc_apps_share_diff",
    "squad_prior_wc_apps_share_pair_avg",
    "squad_prior_wc_starts_mean_diff",
    "squad_prior_wc_starts_mean_pair_avg",
    "squad_prior_wc_goals_mean_diff",
    "squad_prior_wc_goals_mean_pair_avg",
]

CATEGORICAL_FEATURES = [
    "stage_type",
    "home_confederation",
    "away_confederation",
    "venue_country",
]

SQUAD_FEATURE_COLUMNS = [
    "squad_size",
    "squad_avg_age",
    "squad_age_std",
    "squad_u23_share",
    "squad_peak_age_share",
    "squad_30plus_share",
    "squad_gk_share",
    "squad_df_share",
    "squad_mf_share",
    "squad_fw_share",
    "squad_prior_wc_squad_mean",
    "squad_prior_wc_squad_max",
    "squad_prior_wc_squad_share",
    "squad_prior_wc_apps_mean",
    "squad_prior_wc_apps_max",
    "squad_prior_wc_apps_share",
    "squad_prior_wc_starts_mean",
    "squad_prior_wc_goals_mean",
    "squad_prior_wc_goals_max",
    "squad_prior_wc_goals_share",
]


@dataclass(slots=True)
class FeatureSchema:
    numeric_features: list[str]
    categorical_features: list[str]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    numeric_fill: dict[str, float]
    category_maps: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FeatureSchema":
        return cls(
            numeric_features=list(raw["numeric_features"]),
            categorical_features=list(raw["categorical_features"]),
            numeric_means={k: float(v) for k, v in raw["numeric_means"].items()},
            numeric_stds={k: float(v) for k, v in raw["numeric_stds"].items()},
            numeric_fill={k: float(v) for k, v in raw["numeric_fill"].items()},
            category_maps={
                feature: {str(k): int(v) for k, v in mapping.items()}
                for feature, mapping in raw["category_maps"].items()
            },
        )


def fit_feature_schema(df: pd.DataFrame) -> FeatureSchema:
    numeric_fill: dict[str, float] = {}
    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}
    for feature in NUMERIC_FEATURES:
        values = pd.to_numeric(df.get(feature, np.nan), errors="coerce")
        fill = float(values.median()) if values.notna().any() else 0.0
        filled = values.fillna(fill)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        numeric_fill[feature] = fill
        numeric_means[feature] = mean
        numeric_stds[feature] = std if std > 1e-8 else 1.0

    category_maps: dict[str, dict[str, int]] = {}
    for feature in CATEGORICAL_FEATURES:
        values = df.get(feature, pd.Series(["unknown"] * len(df))).fillna("unknown").astype(str)
        categories = sorted(set(values) | {"unknown"})
        category_maps[feature] = {"__UNK__": 0, **{cat: idx + 1 for idx, cat in enumerate(categories)}}

    return FeatureSchema(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        numeric_fill=numeric_fill,
        category_maps=category_maps,
    )


def transform_features(df: pd.DataFrame, schema: FeatureSchema) -> tuple[np.ndarray, np.ndarray]:
    numeric = []
    for feature in schema.numeric_features:
        values = pd.to_numeric(df.get(feature, np.nan), errors="coerce")
        filled = values.fillna(schema.numeric_fill[feature])
        scaled = (filled - schema.numeric_means[feature]) / schema.numeric_stds[feature]
        numeric.append(scaled.to_numpy(dtype=np.float32))
    numeric_array = np.vstack(numeric).T.astype(np.float32)

    categorical = []
    for feature in schema.categorical_features:
        mapping = schema.category_maps[feature]
        values = df.get(feature, pd.Series(["unknown"] * len(df))).fillna("unknown").astype(str)
        categorical.append(values.map(lambda x: mapping.get(x, 0)).to_numpy(dtype=np.int64))
    categorical_array = np.vstack(categorical).T.astype(np.int64)
    return numeric_array, categorical_array


def category_cardinalities(schema: FeatureSchema) -> list[int]:
    return [max(mapping.values()) + 1 for mapping in schema.category_maps.values()]


def _empty_stats() -> dict[str, float]:
    return {
        "played": 0.0,
        "wins": 0.0,
        "draws": 0.0,
        "losses": 0.0,
        "gf": 0.0,
        "ga": 0.0,
    }


def _team_stat_features(stats: dict[str, float], prefix: str) -> dict[str, float]:
    played = stats["played"]
    denom = max(played, 1.0)
    return {
        f"{prefix}_played": played,
        f"{prefix}_win_rate": stats["wins"] / denom,
        f"{prefix}_draw_rate": stats["draws"] / denom,
        f"{prefix}_loss_rate": stats["losses"] / denom,
        f"{prefix}_gf_avg": stats["gf"] / denom,
        f"{prefix}_ga_avg": stats["ga"] / denom,
        f"{prefix}_gd_avg": (stats["gf"] - stats["ga"]) / denom,
        f"{prefix}_experience_log": float(np.log1p(played)),
    }


def _h2h_features(h2h: dict[tuple[str, str], dict[str, float]], team: str, opp: str, prefix: str):
    stats = h2h.get((team, opp), _empty_stats())
    played = stats["played"]
    denom = max(played, 1.0)
    return {
        f"{prefix}_h2h_played": played,
        f"{prefix}_h2h_win_rate": stats["wins"] / denom,
        f"{prefix}_h2h_draw_rate": stats["draws"] / denom,
    }


def _num(record: dict[str, Any], key: str) -> float:
    value = record.get(key, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _pair_avg(record: dict[str, Any], left: str, right: str) -> float:
    values = np.array([_num(record, left), _num(record, right)], dtype=float)
    if np.isnan(values).all():
        return np.nan
    return float(np.nanmean(values))


def _safe_nanmean(values: list[float]) -> float:
    array = np.array(values, dtype=float)
    if np.isnan(array).all():
        return np.nan
    return float(np.nanmean(array))


def _pair_diff(record: dict[str, Any], left: str, right: str) -> float:
    return _num(record, left) - _num(record, right)


def _add_pairwise_engineered_features(record: dict[str, Any]) -> None:
    pair_specs = [
        ("host", "home_host", "away_host"),
        ("played", "home_played", "away_played"),
        ("experience_log", "home_experience_log", "away_experience_log"),
        ("win_rate", "home_win_rate", "away_win_rate"),
        ("draw_rate", "home_draw_rate", "away_draw_rate"),
        ("loss_rate", "home_loss_rate", "away_loss_rate"),
        ("gf_avg", "home_gf_avg", "away_gf_avg"),
        ("ga_avg", "home_ga_avg", "away_ga_avg"),
        ("gd_avg", "home_gd_avg", "away_gd_avg"),
        ("h2h_win_rate", "home_h2h_win_rate", "away_h2h_win_rate"),
        ("squad_gk_share", "home_squad_gk_share", "away_squad_gk_share"),
        ("squad_df_share", "home_squad_df_share", "away_squad_df_share"),
        ("squad_mf_share", "home_squad_mf_share", "away_squad_mf_share"),
        ("squad_fw_share", "home_squad_fw_share", "away_squad_fw_share"),
        ("squad_age_std", "home_squad_age_std", "away_squad_age_std"),
        ("squad_u23_share", "home_squad_u23_share", "away_squad_u23_share"),
        ("squad_peak_age_share", "home_squad_peak_age_share", "away_squad_peak_age_share"),
        ("squad_30plus_share", "home_squad_30plus_share", "away_squad_30plus_share"),
        (
            "squad_prior_wc_squad_mean",
            "home_squad_prior_wc_squad_mean",
            "away_squad_prior_wc_squad_mean",
        ),
        (
            "squad_prior_wc_squad_share",
            "home_squad_prior_wc_squad_share",
            "away_squad_prior_wc_squad_share",
        ),
        (
            "squad_prior_wc_apps_mean",
            "home_squad_prior_wc_apps_mean",
            "away_squad_prior_wc_apps_mean",
        ),
        (
            "squad_prior_wc_apps_share",
            "home_squad_prior_wc_apps_share",
            "away_squad_prior_wc_apps_share",
        ),
        (
            "squad_prior_wc_starts_mean",
            "home_squad_prior_wc_starts_mean",
            "away_squad_prior_wc_starts_mean",
        ),
        (
            "squad_prior_wc_goals_mean",
            "home_squad_prior_wc_goals_mean",
            "away_squad_prior_wc_goals_mean",
        ),
    ]
    for name, home_key, away_key in pair_specs:
        record[f"{name}_diff"] = _pair_diff(record, home_key, away_key)

    total_specs = [
        ("played_total", "home_played", "away_played"),
        ("experience_log_total", "home_experience_log", "away_experience_log"),
        ("h2h_played_total", "home_h2h_played", "away_h2h_played"),
    ]
    for name, home_key, away_key in total_specs:
        record[name] = _num(record, home_key) + _num(record, away_key)

    avg_specs = [
        ("gf_avg_pair_avg", "home_gf_avg", "away_gf_avg"),
        ("ga_avg_pair_avg", "home_ga_avg", "away_ga_avg"),
        ("gd_avg_pair_avg", "home_gd_avg", "away_gd_avg"),
        ("h2h_draw_rate_avg", "home_h2h_draw_rate", "away_h2h_draw_rate"),
        ("fifa_rank_pair_avg", "home_fifa_rank", "away_fifa_rank"),
        ("fifa_points_pair_avg", "home_fifa_points", "away_fifa_points"),
        ("elo_rank_pair_avg", "home_elo_rank", "away_elo_rank"),
        ("elo_rating_pair_avg", "home_elo_rating", "away_elo_rating"),
        ("elo_rating_avg_pair_avg", "home_elo_rating_avg", "away_elo_rating_avg"),
        ("squad_size_pair_avg", "home_squad_size", "away_squad_size"),
        ("squad_avg_age_pair_avg", "home_squad_avg_age", "away_squad_avg_age"),
        ("squad_gk_share_pair_avg", "home_squad_gk_share", "away_squad_gk_share"),
        ("squad_df_share_pair_avg", "home_squad_df_share", "away_squad_df_share"),
        ("squad_mf_share_pair_avg", "home_squad_mf_share", "away_squad_mf_share"),
        ("squad_fw_share_pair_avg", "home_squad_fw_share", "away_squad_fw_share"),
        ("squad_age_std_pair_avg", "home_squad_age_std", "away_squad_age_std"),
        ("squad_u23_share_pair_avg", "home_squad_u23_share", "away_squad_u23_share"),
        (
            "squad_peak_age_share_pair_avg",
            "home_squad_peak_age_share",
            "away_squad_peak_age_share",
        ),
        (
            "squad_30plus_share_pair_avg",
            "home_squad_30plus_share",
            "away_squad_30plus_share",
        ),
        (
            "squad_prior_wc_squad_mean_pair_avg",
            "home_squad_prior_wc_squad_mean",
            "away_squad_prior_wc_squad_mean",
        ),
        (
            "squad_prior_wc_squad_share_pair_avg",
            "home_squad_prior_wc_squad_share",
            "away_squad_prior_wc_squad_share",
        ),
        (
            "squad_prior_wc_apps_mean_pair_avg",
            "home_squad_prior_wc_apps_mean",
            "away_squad_prior_wc_apps_mean",
        ),
        (
            "squad_prior_wc_apps_share_pair_avg",
            "home_squad_prior_wc_apps_share",
            "away_squad_prior_wc_apps_share",
        ),
        (
            "squad_prior_wc_starts_mean_pair_avg",
            "home_squad_prior_wc_starts_mean",
            "away_squad_prior_wc_starts_mean",
        ),
        (
            "squad_prior_wc_goals_mean_pair_avg",
            "home_squad_prior_wc_goals_mean",
            "away_squad_prior_wc_goals_mean",
        ),
    ]
    for name, home_key, away_key in avg_specs:
        record[name] = _pair_avg(record, home_key, away_key)

    record["fifa_missing_any"] = max(
        _num(record, "home_fifa_rank_missing"), _num(record, "away_fifa_rank_missing")
    )
    record["elo_missing_any"] = max(
        _num(record, "home_elo_rating_missing"), _num(record, "away_elo_rating_missing")
    )


def _label_for_match(row: pd.Series) -> str:
    if bool(row.get("extra_time_bool", False)) or bool(row.get("penalty_shootout_bool", False)):
        return "draw"
    home = float(row["home_team_score"])
    away = float(row["away_team_score"])
    if home > away:
        return "home_win"
    if away > home:
        return "away_win"
    return "draw"


def _update_stats(
    stats: dict[str, dict[str, float]],
    h2h: dict[tuple[str, str], dict[str, float]],
    home: str,
    away: str,
    home_goals: float,
    away_goals: float,
) -> None:
    for team in (home, away):
        stats.setdefault(team, _empty_stats())
    for key in ((home, away), (away, home)):
        h2h.setdefault(key, _empty_stats())

    home_result = "draws"
    away_result = "draws"
    if home_goals > away_goals:
        home_result = "wins"
        away_result = "losses"
    elif away_goals > home_goals:
        home_result = "losses"
        away_result = "wins"

    stats[home]["played"] += 1
    stats[home][home_result] += 1
    stats[home]["gf"] += home_goals
    stats[home]["ga"] += away_goals

    stats[away]["played"] += 1
    stats[away][away_result] += 1
    stats[away]["gf"] += away_goals
    stats[away]["ga"] += home_goals

    h2h[(home, away)]["played"] += 1
    h2h[(home, away)][home_result] += 1
    h2h[(home, away)]["gf"] += home_goals
    h2h[(home, away)]["ga"] += away_goals

    h2h[(away, home)]["played"] += 1
    h2h[(away, home)][away_result] += 1
    h2h[(away, home)]["gf"] += away_goals
    h2h[(away, home)]["ga"] += home_goals


def _host_flag(team: str, host_country: str) -> float:
    if not host_country:
        return 0.0
    host_norm = normalize_team_name(host_country)
    if team == host_norm:
        return 1.0
    if "," in host_country:
        hosts = {normalize_team_name(part.strip()) for part in host_country.split(",")}
        return 1.0 if team in hosts else 0.0
    return 0.0


def _stage_type(row: pd.Series) -> str:
    if bool(row.get("group_stage_bool", False)):
        return "group"
    if bool(row.get("knockout_stage_bool", False)):
        return "knockout"
    return "unknown"


def _competition_weight(tournament: str) -> float:
    name = str(tournament).lower()
    if name == "world cup":
        return 1.0
    if "friendly" in name:
        return 0.2
    if "world cup" in name and "qual" in name:
        return 0.65
    if "qual" in name:
        return 0.45
    if "nations league" in name:
        return 0.5
    if any(
        marker in name
        for marker in (
            "european championship",
            "copa america",
            "african nations",
            "asian cup",
            "concacaf championship",
            "gold cup",
            "championship",
            "cup",
        )
    ):
        return 0.75
    return 0.35


@dataclass(slots=True)
class AllMatchHistory:
    rows: pd.DataFrame
    by_team: dict[str, pd.DataFrame]


def _build_all_match_history(dataset_dir: str | Path) -> AllMatchHistory:
    matches = load_all_matches(dataset_dir)
    matches = matches.dropna(subset=["match_dt", "home_score_num", "away_score_num"]).copy()
    matches["competition_weight"] = matches["tournament"].map(_competition_weight)
    matches["friendly_bool"] = matches["tournament"].astype(str).str.lower().str.contains("friendly")

    home_rows = pd.DataFrame(
        {
            "match_dt": matches["match_dt"],
            "team": matches["home_team_norm"],
            "opponent": matches["away_team_norm"],
            "gf": matches["home_score_num"].astype(float),
            "ga": matches["away_score_num"].astype(float),
            "tournament": matches["tournament"],
            "competition_weight": matches["competition_weight"],
            "friendly_bool": matches["friendly_bool"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "match_dt": matches["match_dt"],
            "team": matches["away_team_norm"],
            "opponent": matches["home_team_norm"],
            "gf": matches["away_score_num"].astype(float),
            "ga": matches["home_score_num"].astype(float),
            "tournament": matches["tournament"],
            "competition_weight": matches["competition_weight"],
            "friendly_bool": matches["friendly_bool"],
        }
    )
    rows = pd.concat([home_rows, away_rows], ignore_index=True)
    rows["result_score"] = np.select(
        [rows["gf"] > rows["ga"], rows["gf"].eq(rows["ga"])],
        [1.0, 0.5],
        default=0.0,
    )
    rows["gd"] = rows["gf"] - rows["ga"]
    rows = rows.sort_values(["team", "match_dt", "opponent"]).reset_index(drop=True)
    by_team = {str(team): group.reset_index(drop=True) for team, group in rows.groupby("team")}
    return AllMatchHistory(rows=rows, by_team=by_team)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    clean = pd.DataFrame({"value": values, "weight": weights}).dropna()
    clean = clean[clean["weight"] > 0]
    if clean.empty:
        return np.nan
    return float(np.average(clean["value"], weights=clean["weight"]))


def _team_all_match_summary(
    history: AllMatchHistory,
    team: str,
    *,
    as_of: pd.Timestamp,
    opponent: str | None = None,
) -> dict[str, float]:
    rows = history.by_team.get(team)
    if rows is None or pd.isna(as_of):
        return {}
    prior = rows[rows["match_dt"] < as_of].copy()
    if opponent is not None:
        prior = prior[prior["opponent"] == opponent].copy()
    if prior.empty:
        return {}

    prior["days_ago"] = (as_of - prior["match_dt"]).dt.days.clip(lower=0)
    prior["time_weight"] = 0.5 ** (prior["days_ago"] / (365.25 * 4.0))
    prior["combined_weight"] = prior["competition_weight"] * prior["time_weight"]

    result: dict[str, float] = {}
    for n in (5, 10, 20):
        recent = prior.tail(n)
        result[f"form_{n}"] = _weighted_mean(recent["result_score"], recent["competition_weight"])
    recent_10 = prior.tail(10)
    result["gf_10"] = _weighted_mean(recent_10["gf"], recent_10["competition_weight"])
    result["ga_10"] = _weighted_mean(recent_10["ga"], recent_10["competition_weight"])
    result["gd_10"] = _weighted_mean(recent_10["gd"], recent_10["competition_weight"])
    result["form_weighted"] = _weighted_mean(prior["result_score"], prior["combined_weight"])
    result["gd_weighted"] = _weighted_mean(prior["gd"], prior["combined_weight"])

    competitive = prior[~prior["friendly_bool"]]
    friendly = prior[prior["friendly_bool"]]
    result["competitive_form"] = _weighted_mean(
        competitive["result_score"], competitive["combined_weight"]
    )
    result["friendly_form"] = _weighted_mean(friendly["result_score"], friendly["combined_weight"])
    result["played"] = float(len(prior))
    result["win_rate"] = float((prior["result_score"] == 1.0).mean())
    result["draw_rate"] = float((prior["result_score"] == 0.5).mean())
    return result


def _add_all_match_features(
    record: dict[str, Any],
    history: AllMatchHistory,
    home: str,
    away: str,
    match_dt: pd.Timestamp,
) -> None:
    home_summary = _team_all_match_summary(history, home, as_of=match_dt)
    away_summary = _team_all_match_summary(history, away, as_of=match_dt)

    for key in ("form_5", "form_10", "form_20", "form_weighted"):
        record[f"all_recent_{key}_diff"] = home_summary.get(key, np.nan) - away_summary.get(
            key, np.nan
        )
    record["all_recent_form_10_pair_avg"] = _safe_nanmean(
        [home_summary.get("form_10", np.nan), away_summary.get("form_10", np.nan)]
    )
    for key in ("gf_10", "ga_10", "gd_10", "gd_weighted"):
        record[f"all_recent_{key}_diff"] = home_summary.get(key, np.nan) - away_summary.get(
            key, np.nan
        )
    record["all_recent_competitive_form_diff"] = home_summary.get(
        "competitive_form", np.nan
    ) - away_summary.get("competitive_form", np.nan)
    record["all_recent_friendly_form_diff"] = home_summary.get(
        "friendly_form", np.nan
    ) - away_summary.get("friendly_form", np.nan)

    home_h2h = _team_all_match_summary(history, home, as_of=match_dt, opponent=away)
    away_h2h = _team_all_match_summary(history, away, as_of=match_dt, opponent=home)
    record["all_h2h_played"] = max(home_h2h.get("played", 0.0), away_h2h.get("played", 0.0))
    record["all_h2h_win_rate_diff"] = home_h2h.get("win_rate", np.nan) - away_h2h.get(
        "win_rate", np.nan
    )
    record["all_h2h_draw_rate"] = _safe_nanmean(
        [home_h2h.get("draw_rate", np.nan), away_h2h.get("draw_rate", np.nan)]
    )


def _prior_era_goal_rates(matches: pd.DataFrame) -> dict[str, float]:
    by_tournament = (
        matches.assign(total_goals=matches["home_team_score"] + matches["away_team_score"])
        .groupby(["tournament_id", "year_num"], as_index=False)
        .agg(total_goals=("total_goals", "sum"), matches=("match_id", "count"))
        .sort_values(["year_num", "tournament_id"])
    )
    total_goals = 0.0
    total_team_matches = 0.0
    fallback = 1.35
    rates: dict[str, float] = {}
    for _, row in by_tournament.iterrows():
        tournament_id = str(row["tournament_id"])
        rates[tournament_id] = (
            total_goals / total_team_matches if total_team_matches > 0 else fallback
        )
        total_goals += float(row["total_goals"])
        total_team_matches += float(row["matches"]) * 2.0
    return rates


def _load_team_confederations(dataset_dir: str | Path) -> dict[str, str]:
    teams = read_csv(dataset_dir, "teams_curated.csv")
    teams["team_norm"] = teams["team_name"].map(normalize_team_name)
    return dict(zip(teams["team_norm"], teams["confederation_code"]))


def _row_squad_features(
    table: pd.DataFrame, tournament_id: str | None, team: str, prefix: str
) -> dict[str, float]:
    if "tournament_id" in table.columns and tournament_id is not None:
        match = table[(table["tournament_id"] == tournament_id) & (table["team_norm"] == team)]
    else:
        match = table[table["team_norm"] == team]
    if match.empty:
        return {f"{prefix}_{feature}": np.nan for feature in SQUAD_FEATURE_COLUMNS}
    row = match.iloc[0]
    return {
        f"{prefix}_{feature}": float(row[feature]) if feature in row and pd.notna(row[feature]) else np.nan
        for feature in SQUAD_FEATURE_COLUMNS
    }


def build_training_features(dataset_dir: str | Path) -> pd.DataFrame:
    matches = load_mens_matches(dataset_dir)
    fifa_lookup, elo_lookup = load_rating_lookups(dataset_dir)
    squad_features = build_historical_squad_features(dataset_dir)
    confeds = _load_team_confederations(dataset_dir)
    era_goal_rates = _prior_era_goal_rates(matches)
    all_match_history = _build_all_match_history(dataset_dir)

    stats: dict[str, dict[str, float]] = {}
    h2h: dict[tuple[str, str], dict[str, float]] = {}
    rows: list[dict[str, Any]] = []

    for _, row in matches.iterrows():
        home = normalize_team_name(row["home_team_name"])
        away = normalize_team_name(row["away_team_name"])
        stats.setdefault(home, _empty_stats())
        stats.setdefault(away, _empty_stats())

        record: dict[str, Any] = {
            "match_id": row["match_id"],
            "tournament_id": row["tournament_id"],
            "year_num": float(row["year_num"]),
            "match_date": row["match_date"],
            "match_dt": row["match_dt"],
            "home_team": home,
            "away_team": away,
            "stage_type": _stage_type(row),
            "venue_country": normalize_team_name(row.get("country_name") or "unknown"),
            "home_confederation": confeds.get(home, "unknown"),
            "away_confederation": confeds.get(away, "unknown"),
            "era_team_goal_rate": era_goal_rates.get(str(row["tournament_id"]), 1.35),
            "home_host": _host_flag(home, row.get("host_country_norm", "")),
            "away_host": _host_flag(away, row.get("host_country_norm", "")),
            "home_goals": float(row["home_team_score"]),
            "away_goals": float(row["away_team_score"]),
        }
        record.update(_team_stat_features(stats[home], "home"))
        record.update(_team_stat_features(stats[away], "away"))
        record.update(_h2h_features(h2h, home, away, "home"))
        record.update(_h2h_features(h2h, away, home, "away"))
        _add_all_match_features(record, all_match_history, home, away, row["match_dt"])

        for prefix, team in (("home", home), ("away", away)):
            fifa = fifa_lookup.lookup(team, row["match_dt"])
            elo = elo_lookup.lookup(team, row["match_dt"])
            record.update({f"{prefix}_{key}": value for key, value in fifa.items()})
            record.update({f"{prefix}_{key}": value for key, value in elo.items()})
            record.update(_row_squad_features(squad_features, row["tournament_id"], team, prefix))

        record["fifa_rank_diff"] = record["home_fifa_rank"] - record["away_fifa_rank"]
        record["fifa_points_diff"] = record["home_fifa_points"] - record["away_fifa_points"]
        record["elo_rank_diff"] = record["home_elo_rank"] - record["away_elo_rank"]
        record["elo_rating_diff"] = record["home_elo_rating"] - record["away_elo_rating"]
        record["elo_rating_avg_diff"] = record["home_elo_rating_avg"] - record["away_elo_rating_avg"]
        record["squad_size_diff"] = record["home_squad_size"] - record["away_squad_size"]
        record["squad_avg_age_diff"] = record["home_squad_avg_age"] - record["away_squad_avg_age"]
        _add_pairwise_engineered_features(record)
        record["label"] = _label_for_match(row)
        record["label_id"] = LABEL_TO_ID[record["label"]]
        rows.append(record)

        _update_stats(
            stats,
            h2h,
            home,
            away,
            float(row["home_team_score"]),
            float(row["away_team_score"]),
        )

    return pd.DataFrame(rows)


def _label_from_scores(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "home_win"
    if away_goals > home_goals:
        return "away_win"
    return "draw"


def _external_sample_weight(
    match_dt: pd.Timestamp,
    tournament: str,
    *,
    base_weight: float,
    half_life_years: float,
) -> float:
    reference_date = pd.Timestamp("2026-06-11")
    days_ago = max(float((reference_date - match_dt).days), 0.0) if pd.notna(match_dt) else 3652.5
    recency = 0.5 ** (days_ago / (365.25 * max(half_life_years, 0.1)))
    return float(base_weight * _competition_weight(tournament) * recency)


def build_external_training_features(
    dataset_dir: str | Path,
    *,
    start_year: int,
    max_rows: int,
    base_weight: float,
    half_life_years: float,
) -> pd.DataFrame:
    matches = load_all_matches(dataset_dir)
    matches = matches.dropna(subset=["match_dt", "home_score_num", "away_score_num"]).copy()
    matches = matches[matches["match_dt"].dt.year >= start_year].copy()
    matches = matches[matches["tournament"].astype(str).ne("World Cup")].copy()
    matches = matches.sort_values("match_dt").tail(max_rows).copy() if max_rows > 0 else matches
    if matches.empty:
        return pd.DataFrame()

    fifa_lookup, elo_lookup = load_rating_lookups(dataset_dir)
    confeds = _load_team_confederations(dataset_dir)
    all_match_history = _build_all_match_history(dataset_dir)
    era_goal_rate = era_goal_rate_for_2026(dataset_dir)
    rows: list[dict[str, Any]] = []

    empty_stats = _empty_stats()
    for idx, row in matches.iterrows():
        home = str(row["home_team_norm"])
        away = str(row["away_team_norm"])
        match_dt = row["match_dt"]
        home_goals = float(row["home_score_num"])
        away_goals = float(row["away_score_num"])
        tournament = str(row["tournament"])
        is_friendly = "friendly" in tournament.lower()
        country = str(row.get("country_norm") or "unknown")

        record: dict[str, Any] = {
            "match_id": f"all_matches_{idx}",
            "tournament_id": f"external_{int(match_dt.year)}",
            "year_num": float(match_dt.year),
            "match_date": str(row["date"]),
            "match_dt": match_dt,
            "home_team": home,
            "away_team": away,
            "stage_type": "external_friendly" if is_friendly else "external_competitive",
            "venue_country": country,
            "home_confederation": confeds.get(home, "unknown"),
            "away_confederation": confeds.get(away, "unknown"),
            "era_team_goal_rate": era_goal_rate,
            "home_host": 1.0 if (not bool(row["neutral_bool"]) and home == country) else 0.0,
            "away_host": 1.0 if (not bool(row["neutral_bool"]) and away == country) else 0.0,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "sample_weight": _external_sample_weight(
                match_dt,
                tournament,
                base_weight=base_weight,
                half_life_years=half_life_years,
            ),
            "source": "all_matches",
        }
        record.update(_team_stat_features(empty_stats, "home"))
        record.update(_team_stat_features(empty_stats, "away"))
        record.update(_h2h_features({}, home, away, "home"))
        record.update(_h2h_features({}, away, home, "away"))
        _add_all_match_features(record, all_match_history, home, away, match_dt)

        for prefix, team in (("home", home), ("away", away)):
            fifa = fifa_lookup.lookup(team, match_dt)
            elo = elo_lookup.lookup(team, match_dt)
            record.update({f"{prefix}_{key}": value for key, value in fifa.items()})
            record.update({f"{prefix}_{key}": value for key, value in elo.items()})
            for feature in SQUAD_FEATURE_COLUMNS:
                record[f"{prefix}_{feature}"] = np.nan

        record["fifa_rank_diff"] = record["home_fifa_rank"] - record["away_fifa_rank"]
        record["fifa_points_diff"] = record["home_fifa_points"] - record["away_fifa_points"]
        record["elo_rank_diff"] = record["home_elo_rank"] - record["away_elo_rank"]
        record["elo_rating_diff"] = record["home_elo_rating"] - record["away_elo_rating"]
        record["elo_rating_avg_diff"] = record["home_elo_rating_avg"] - record["away_elo_rating_avg"]
        record["squad_size_diff"] = record["home_squad_size"] - record["away_squad_size"]
        record["squad_avg_age_diff"] = record["home_squad_avg_age"] - record["away_squad_avg_age"]
        _add_pairwise_engineered_features(record)
        record["label"] = _label_from_scores(home_goals, away_goals)
        record["label_id"] = LABEL_TO_ID[record["label"]]
        rows.append(record)

    return pd.DataFrame(rows)


def add_recency_weights(df: pd.DataFrame, *, half_life_tournaments: float) -> pd.DataFrame:
    weighted = df.copy()
    if half_life_tournaments <= 0:
        weighted["sample_weight"] = 1.0
        return weighted
    tournaments = (
        weighted[["tournament_id", "year_num"]]
        .drop_duplicates()
        .sort_values(["year_num", "tournament_id"])
        .reset_index(drop=True)
    )
    order = {row.tournament_id: idx for idx, row in tournaments.iterrows()}
    latest = max(order.values()) if order else 0
    half_life = max(float(half_life_tournaments), 0.1)
    weights = weighted["tournament_id"].map(
        lambda tournament_id: 0.5 ** ((latest - order[tournament_id]) / half_life)
    )
    weighted["sample_weight"] = weights / float(weights.mean())
    return weighted


def augment_neutral_home_away_swap(df: pd.DataFrame) -> pd.DataFrame:
    neutral = df[
        (pd.to_numeric(df.get("home_host", 0.0), errors="coerce").fillna(0.0) == 0.0)
        & (pd.to_numeric(df.get("away_host", 0.0), errors="coerce").fillna(0.0) == 0.0)
    ].copy()
    if neutral.empty:
        return neutral

    swapped = neutral.copy()
    columns = list(swapped.columns)
    for column in columns:
        if column.startswith("home_"):
            opposite = "away_" + column.removeprefix("home_")
            if opposite in swapped.columns:
                swapped[column] = neutral[opposite]
        elif column.startswith("away_"):
            opposite = "home_" + column.removeprefix("away_")
            if opposite in swapped.columns:
                swapped[column] = neutral[opposite]

    for column in columns:
        if column.endswith("_diff") and column in swapped.columns:
            swapped[column] = -pd.to_numeric(neutral[column], errors="coerce")

    if {"home_goals", "away_goals"}.issubset(swapped.columns):
        swapped["home_goals"] = neutral["away_goals"]
        swapped["away_goals"] = neutral["home_goals"]

    swapped = swapped.copy()
    label_swap = {"home_win": "away_win", "away_win": "home_win", "draw": "draw"}
    swapped["label"] = neutral["label"].map(label_swap)
    swapped["label_id"] = swapped["label"].map(LABEL_TO_ID)
    swapped["augmented_swap"] = 1.0
    return swapped.reset_index(drop=True)


def era_goal_rate_for_2026(dataset_dir: str | Path) -> float:
    matches = load_mens_matches(dataset_dir)
    if matches.empty:
        return 1.35
    total_goals = float((matches["home_team_score"] + matches["away_team_score"]).sum())
    total_team_matches = float(len(matches) * 2)
    return total_goals / total_team_matches if total_team_matches else 1.35


def build_split(df: pd.DataFrame, *, seed: int, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []

    for _, tournament in df.groupby("tournament_id", sort=False):
        tournament_train: list[int] = []
        tournament_test: list[int] = []
        for _, class_rows in tournament.groupby("label_id", sort=False):
            indices = class_rows.index.to_numpy(copy=True)
            rng.shuffle(indices)
            if len(indices) <= 1:
                tournament_train.extend(indices.tolist())
                continue
            n_test = max(1, int(round(len(indices) * test_size)))
            n_test = min(n_test, len(indices) - 1)
            tournament_test.extend(indices[:n_test].tolist())
            tournament_train.extend(indices[n_test:].tolist())

        if not tournament_test and len(tournament_train) > 1:
            moved = tournament_train.pop()
            tournament_test.append(moved)
        if not tournament_train and len(tournament_test) > 1:
            moved = tournament_test.pop()
            tournament_train.append(moved)

        train_idx.extend(tournament_train)
        test_idx.extend(tournament_test)

    return np.array(sorted(train_idx), dtype=np.int64), np.array(sorted(test_idx), dtype=np.int64)


def _final_history_state(dataset_dir: str | Path) -> tuple[dict[str, dict[str, float]], dict]:
    matches = load_mens_matches(dataset_dir)
    stats: dict[str, dict[str, float]] = {}
    h2h: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in matches.iterrows():
        home = normalize_team_name(row["home_team_name"])
        away = normalize_team_name(row["away_team_name"])
        _update_stats(
            stats,
            h2h,
            home,
            away,
            float(row["home_team_score"]),
            float(row["away_team_score"]),
        )
    return stats, h2h


def _current_rating_map(dataset_dir: str | Path) -> dict[str, dict[str, float]]:
    fifa = load_fifa_rankings_current(dataset_dir)
    elo = load_elo_ratings_current(dataset_dir)
    merged = fifa.merge(elo, on="team_norm", how="outer")
    result: dict[str, dict[str, float]] = {}
    for _, row in merged.iterrows():
        result[str(row["team_norm"])] = {
            "fifa_rank": float(row["fifa_rank"]) if pd.notna(row.get("fifa_rank")) else np.nan,
            "fifa_points": float(row["fifa_points"]) if pd.notna(row.get("fifa_points")) else np.nan,
            "elo_rank": float(row["elo_rank"]) if pd.notna(row.get("elo_rank")) else np.nan,
            "elo_rating": float(row["elo_rating"]) if pd.notna(row.get("elo_rating")) else np.nan,
            "elo_rank_avg": float(row["elo_rank_avg"]) if pd.notna(row.get("elo_rank_avg")) else np.nan,
            "elo_rating_avg": float(row["elo_rating_avg"])
            if pd.notna(row.get("elo_rating_avg"))
            else np.nan,
        }
    return result


def build_2026_match_features(
    dataset_dir: str | Path,
    *,
    team: str,
    opponent: str,
    match_number: int | None,
) -> pd.DataFrame:
    home = normalize_team_name(team)
    away = normalize_team_name(opponent)
    schedule = load_2026_schedule(dataset_dir)
    schedule_row = None
    if match_number is not None:
        match = schedule[schedule["match_number"] == match_number]
        if not match.empty:
            schedule_row = match.iloc[0]
            scheduled_home = normalize_team_name(schedule_row["Home Team"])
            scheduled_away = normalize_team_name(schedule_row["Away Team"])
            if {scheduled_home, scheduled_away} == {home, away}:
                home, away = scheduled_home, scheduled_away

    stats, h2h = _final_history_state(dataset_dir)
    for item in (home, away):
        stats.setdefault(item, _empty_stats())
    squad_features = build_2026_squad_features(dataset_dir)
    confeds = _load_team_confederations(dataset_dir)
    ratings = _current_rating_map(dataset_dir)
    all_match_history = _build_all_match_history(dataset_dir)

    match_dt = pd.Timestamp("2026-06-11")
    stage_type = "group"
    venue_country = "unknown"
    if schedule_row is not None:
        match_dt = schedule_row["match_dt"]
        raw_group = schedule_row.get("Group") or ""
        stage_type = "group" if raw_group else "knockout"
        venue_country = "unknown"

    record: dict[str, Any] = {
        "year_num": 2026.0,
        "match_dt": match_dt,
        "home_team": home,
        "away_team": away,
        "stage_type": stage_type,
        "venue_country": venue_country,
        "home_confederation": confeds.get(home, "unknown"),
        "away_confederation": confeds.get(away, "unknown"),
        "era_team_goal_rate": era_goal_rate_for_2026(dataset_dir),
        "home_host": 1.0 if home in {"Canada", "Mexico", "United States"} else 0.0,
        "away_host": 1.0 if away in {"Canada", "Mexico", "United States"} else 0.0,
    }
    record.update(_team_stat_features(stats[home], "home"))
    record.update(_team_stat_features(stats[away], "away"))
    record.update(_h2h_features(h2h, home, away, "home"))
    record.update(_h2h_features(h2h, away, home, "away"))
    _add_all_match_features(record, all_match_history, home, away, match_dt)

    for prefix, item in (("home", home), ("away", away)):
        rating = ratings.get(item, {})
        for key in ("fifa_rank", "fifa_points", "elo_rank", "elo_rating", "elo_rank_avg", "elo_rating_avg"):
            record[f"{prefix}_{key}"] = rating.get(key, np.nan)
        record.update(_row_squad_features(squad_features, None, item, prefix))

    record["fifa_rank_diff"] = record["home_fifa_rank"] - record["away_fifa_rank"]
    record["fifa_points_diff"] = record["home_fifa_points"] - record["away_fifa_points"]
    record["elo_rank_diff"] = record["home_elo_rank"] - record["away_elo_rank"]
    record["elo_rating_diff"] = record["home_elo_rating"] - record["away_elo_rating"]
    record["elo_rating_avg_diff"] = record["home_elo_rating_avg"] - record["away_elo_rating_avg"]
    record["squad_size_diff"] = record["home_squad_size"] - record["away_squad_size"]
    record["squad_avg_age_diff"] = record["home_squad_avg_age"] - record["away_squad_avg_age"]
    _add_pairwise_engineered_features(record)
    return pd.DataFrame([record])
