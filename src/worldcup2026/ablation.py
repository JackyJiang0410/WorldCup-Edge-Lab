from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import TrainConfig
from .training import train_pipeline


def _ablation_variants(config: TrainConfig) -> dict[str, dict[str, Any]]:
    recency_half_life = (
        config.recency_half_life_tournaments
        if config.recency_half_life_tournaments > 0
        else 6.0
    )
    class_loss_weight = config.class_loss_weight if config.class_loss_weight > 0 else 0.25
    calibration_loss_weight = (
        config.calibration_loss_weight if config.calibration_loss_weight > 0 else 0.05
    )
    all_on = {
        "home_away_swap_augmentation": True,
        "recency_half_life_tournaments": recency_half_life,
        "era_goal_scaling": True,
        "class_loss_weight": class_loss_weight,
        "calibration_loss_weight": calibration_loss_weight,
    }
    return {
        "baseline": all_on,
        "no_swap": {**all_on, "home_away_swap_augmentation": False},
        "no_recency": {**all_on, "recency_half_life_tournaments": 0.0},
        "no_era_scaling": {**all_on, "era_goal_scaling": False},
        "no_aux_class": {**all_on, "class_loss_weight": 0.0, "calibration_loss_weight": 0.0},
    }


def _variant_config(config: TrainConfig, name: str, overrides: dict[str, Any]) -> TrainConfig:
    artifact_dir = Path("artifacts") / "ablations" / name
    report_dir = Path("reports") / "ablations" / name
    return replace(
        config,
        artifact_dir=artifact_dir,
        report_dir=report_dir,
        optuna_trials=0,
        **overrides,
    )


def _compact_result(name: str, config: TrainConfig, result: dict[str, Any]) -> dict[str, Any]:
    best = result["best"]
    return {
        "name": name,
        "model_path": result["model_path"],
        "metrics_path": result["metrics_path"],
        "training_rows": result.get("training_rows", {}),
        "config": {
            "home_away_swap_augmentation": config.home_away_swap_augmentation,
            "recency_half_life_tournaments": config.recency_half_life_tournaments,
            "era_goal_scaling": config.era_goal_scaling,
            "class_loss_weight": config.class_loss_weight,
            "calibration_loss_weight": config.calibration_loss_weight,
            "optuna_trials": config.optuna_trials,
        },
        "best": {
            "seed": best["seed"],
            "best_epoch": best["best_epoch"],
            "stopped_epoch": best["stopped_epoch"],
            "score": best["score"],
            "test": best["test"],
            "aux_class_test": best["aux_class_test"],
            "test_goal_mae": best["test_goal_mae"],
        },
    }


def run_ablation_pipeline(config: TrainConfig) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for name, overrides in _ablation_variants(config).items():
        variant_config = _variant_config(config, name, overrides)
        result = train_pipeline(variant_config)
        results.append(_compact_result(name, variant_config, result))

    ranked = sorted(results, key=lambda item: item["best"]["score"], reverse=True)
    report = {
        "selection_metric": "accuracy - 0.08 * log_loss - 0.15 * brier",
        "note": "Ablations disable Optuna to isolate each switch under the same base config.",
        "best_variant": ranked[0]["name"],
        "ranked": ranked,
        "results": results,
    }
    report_path = Path("reports") / "ablation_v1.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"report_path": str(report_path), **report}
