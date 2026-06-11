from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .features import FeatureSchema, build_2026_match_features, transform_features
from .simulation import SimulationResult, relative_counts, simulate_scoreline
from .team_names import normalize_team_name
from .training import load_model_from_artifact


@dataclass(slots=True)
class MatchPrediction:
    home_team: str
    away_team: str
    requested_team: str
    requested_team_is_home: bool
    class_probabilities: dict[str, float]
    expected_goals: dict[str, float]
    simulation: SimulationResult
    relative_simulation_counts: dict[str, int]

    @property
    def requested_team_win_probability(self) -> float:
        if self.requested_team_is_home:
            return self.simulation.frequencies["home_win"]
        return self.simulation.frequencies["away_win"]

    @property
    def relative_predicted_result(self) -> str:
        if self.simulation.predicted_result == "draw":
            return "draw"
        if self.simulation.predicted_result == "home_win":
            return "team_win" if self.requested_team_is_home else "opponent_win"
        return "opponent_win" if self.requested_team_is_home else "team_win"


def predict_2026_match(
    *,
    model_path: str | Path,
    dataset_dir: str | Path,
    team: str,
    opponent: str,
    match_number: int | None,
    simulation_runs: int = 1000,
    seed: int = 42,
) -> MatchPrediction:
    model, artifact = load_model_from_artifact(model_path)
    schema = FeatureSchema.from_dict(artifact["feature_schema"])
    row = build_2026_match_features(
        dataset_dir, team=team, opponent=opponent, match_number=match_number
    )
    numeric, categorical = transform_features(row, schema)
    with torch.no_grad():
        logits, goal_multipliers = model(
            torch.tensor(numeric, dtype=torch.float32),
            torch.tensor(categorical, dtype=torch.long),
        )
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        goals = goal_multipliers.cpu().numpy()[0]
        if artifact.get("goal_output") == "era_scaled_multipliers":
            goal_scale = float(row.iloc[0]["era_team_goal_rate"])
            goals = goals * goal_scale

    simulation = simulate_scoreline(
        float(goals[0]), float(goals[1]), runs=simulation_runs, seed=seed
    )
    home_team = normalize_team_name(str(row.iloc[0]["home_team"]))
    away_team = normalize_team_name(str(row.iloc[0]["away_team"]))
    requested = normalize_team_name(team)
    requested_is_home = requested == home_team
    if requested not in {home_team, away_team}:
        # Snapshot can omit orientation; default to the team/opponent order passed by the user.
        requested_is_home = True

    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
        requested_team=requested,
        requested_team_is_home=requested_is_home,
        class_probabilities={
            "home_win": float(probs[0]),
            "draw": float(probs[1]),
            "away_win": float(probs[2]),
        },
        expected_goals={"home": float(goals[0]), "away": float(goals[1])},
        simulation=simulation,
        relative_simulation_counts=relative_counts(
            simulation.counts, requested_team_is_home=requested_is_home
        ),
    )
