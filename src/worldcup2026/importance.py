from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import TrainConfig
from .features import FeatureSchema, build_training_features
from .training import (
    _frame_arrays,
    _predict_arrays,
    classification_metrics,
    composite_score,
    load_model_from_artifact,
    simulation_probabilities,
)


def _feature_groups(features: list[str]) -> dict[str, list[str]]:
    groups = {
        "all_matches_recent": [f for f in features if f.startswith("all_")],
        "world_cup_history": [
            f
            for f in features
            if any(
                token in f
                for token in (
                    "played",
                    "experience_log",
                    "win_rate",
                    "draw_rate",
                    "loss_rate",
                    "gf_avg",
                    "ga_avg",
                    "gd_avg",
                    "h2h_",
                )
            )
            and not f.startswith("all_")
        ],
        "fifa": [f for f in features if f.startswith("fifa_") or "_fifa_" in f],
        "elo": [f for f in features if f.startswith("elo_") or "_elo_" in f],
        "squad": [f for f in features if f.startswith("squad_") or "_squad_" in f],
        "host": [f for f in features if "host" in f],
        "era_year": [f for f in features if f in {"year_num", "era_team_goal_rate"}],
        "categorical_context": [
            f for f in features if f in {"stage_type", "home_confederation", "away_confederation", "venue_country"}
        ],
    }
    return {name: cols for name, cols in groups.items() if cols}


def _evaluate_frame(model, artifact: dict[str, Any], frame: pd.DataFrame) -> dict[str, float]:
    config = TrainConfig(
        **{k: v for k, v in artifact["train_config"].items() if k in TrainConfig.__dataclass_fields__}
    )
    schema = FeatureSchema.from_dict(artifact["feature_schema"])
    numeric, categorical, labels, _, goal_scales, _ = _frame_arrays(
        frame,
        schema,
        era_goal_scaling=artifact.get("goal_output") == "era_scaled_multipliers",
    )
    _, expected_goals = _predict_arrays(
        model,
        numeric,
        categorical,
        goal_scales=goal_scales if artifact.get("goal_output") == "era_scaled_multipliers" else None,
        batch_size=config.batch_size,
        device=torch.device("cpu"),
    )
    sim_probs = simulation_probabilities(expected_goals, runs=config.simulation_runs, seed=config.seed)
    metrics = classification_metrics(sim_probs, labels)
    return {**metrics, "score": composite_score(metrics)}


def permutation_importance(
    model_path: str | Path,
    *,
    output_path: str | Path = "reports/feature_importance.json",
    seed: int = 42,
) -> dict[str, Any]:
    model, artifact = load_model_from_artifact(model_path)
    config = TrainConfig(
        **{k: v for k, v in artifact["train_config"].items() if k in TrainConfig.__dataclass_fields__}
    )
    config.dataset_dir = Path(config.dataset_dir)
    schema = FeatureSchema.from_dict(artifact["feature_schema"])
    features = build_training_features(config.dataset_dir)
    test_idx = np.array(artifact["split"]["test_idx"], dtype=np.int64)
    frame = features.iloc[test_idx].copy().reset_index(drop=True)
    baseline = _evaluate_frame(model, artifact, frame)
    rng = np.random.default_rng(seed)

    feature_results: list[dict[str, Any]] = []
    for feature in schema.numeric_features + schema.categorical_features:
        if feature not in frame.columns:
            continue
        shuffled = frame.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        metrics = _evaluate_frame(model, artifact, shuffled)
        feature_results.append(
            {
                "feature": feature,
                "delta_score": baseline["score"] - metrics["score"],
                "delta_accuracy": baseline["accuracy"] - metrics["accuracy"],
                "delta_log_loss": metrics["log_loss"] - baseline["log_loss"],
                "delta_brier": metrics["brier"] - baseline["brier"],
                "permuted": metrics,
            }
        )

    group_results: list[dict[str, Any]] = []
    groups = _feature_groups(schema.numeric_features + schema.categorical_features)
    for group, columns in groups.items():
        present = [col for col in columns if col in frame.columns]
        if not present:
            continue
        shuffled = frame.copy()
        permutation = rng.permutation(len(shuffled))
        shuffled.loc[:, present] = shuffled.loc[permutation, present].to_numpy()
        metrics = _evaluate_frame(model, artifact, shuffled)
        group_results.append(
            {
                "group": group,
                "feature_count": len(present),
                "features": present,
                "delta_score": baseline["score"] - metrics["score"],
                "delta_accuracy": baseline["accuracy"] - metrics["accuracy"],
                "delta_log_loss": metrics["log_loss"] - baseline["log_loss"],
                "delta_brier": metrics["brier"] - baseline["brier"],
                "permuted": metrics,
            }
        )

    report = {
        "model_path": str(model_path),
        "baseline": baseline,
        "features": sorted(feature_results, key=lambda item: item["delta_score"], reverse=True),
        "groups": sorted(group_results, key=lambda item: item["delta_score"], reverse=True),
        "note": "Positive delta_score means performance worsened when permuted; negative means the feature/group may be noisy or redundant.",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"output_path": str(output), **report}
