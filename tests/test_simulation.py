from __future__ import annotations

from worldcup2026.simulation import simulate_scoreline


def test_simulation_is_deterministic_with_fixed_seed():
    first = simulate_scoreline(1.7, 0.9, runs=1000, seed=42)
    second = simulate_scoreline(1.7, 0.9, runs=1000, seed=42)
    assert first.counts == second.counts
    assert sum(first.counts.values()) == 1000
    assert first.predicted_result in {"home_win", "draw", "away_win"}

