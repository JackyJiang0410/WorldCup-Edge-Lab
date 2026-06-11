from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SimulationResult:
    counts: dict[str, int]
    frequencies: dict[str, float]
    predicted_result: str


def simulate_scoreline(
    home_expected_goals: float,
    away_expected_goals: float,
    *,
    runs: int,
    seed: int,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    home_goals = rng.poisson(max(home_expected_goals, 0.03), size=runs)
    away_goals = rng.poisson(max(away_expected_goals, 0.03), size=runs)
    counts = {
        "home_win": int(np.sum(home_goals > away_goals)),
        "draw": int(np.sum(home_goals == away_goals)),
        "away_win": int(np.sum(home_goals < away_goals)),
    }
    frequencies = {key: value / runs for key, value in counts.items()}
    predicted_result = max(counts.items(), key=lambda item: item[1])[0]
    return SimulationResult(counts=counts, frequencies=frequencies, predicted_result=predicted_result)


def relative_counts(counts: dict[str, int], *, requested_team_is_home: bool) -> dict[str, int]:
    if requested_team_is_home:
        return {
            "team_win": counts["home_win"],
            "draw": counts["draw"],
            "opponent_win": counts["away_win"],
        }
    return {
        "team_win": counts["away_win"],
        "draw": counts["draw"],
        "opponent_win": counts["home_win"],
    }

