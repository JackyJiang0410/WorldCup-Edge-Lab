from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig
from .features import (
    add_recency_weights,
    augment_neutral_home_away_swap,
    build_external_training_features,
    build_split,
    build_training_features,
    category_cardinalities,
    fit_feature_schema,
    transform_features,
)
from .model import ModelConfig, TabularMultiTaskNet, multitask_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _goal_scales(df, *, era_goal_scaling: bool) -> np.ndarray:
    if not era_goal_scaling:
        return np.ones((len(df), 1), dtype=np.float32)
    values = np.asarray(df["era_team_goal_rate"], dtype=np.float32)
    return np.clip(values, 0.25, 4.0).reshape(-1, 1)


def _sample_weights(df) -> np.ndarray:
    if "sample_weight" not in df:
        return np.ones(len(df), dtype=np.float32)
    values = np.asarray(df["sample_weight"], dtype=np.float32)
    return np.clip(values, 0.05, 20.0)


def _make_loader(
    numeric: np.ndarray,
    categorical: np.ndarray,
    labels: np.ndarray,
    goals: np.ndarray,
    goal_scales: np.ndarray,
    sample_weights: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(numeric, dtype=torch.float32),
        torch.tensor(categorical, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(goals, dtype=torch.float32),
        torch.tensor(goal_scales, dtype=torch.float32),
        torch.tensor(sample_weights, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _predict_arrays(
    model: TabularMultiTaskNet,
    numeric: np.ndarray,
    categorical: np.ndarray,
    *,
    goal_scales: np.ndarray | None,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(numeric), batch_size):
            end = start + batch_size
            x_num = torch.tensor(numeric[start:end], dtype=torch.float32, device=device)
            x_cat = torch.tensor(categorical[start:end], dtype=torch.long, device=device)
            logits, goal_multipliers = model(x_num, x_cat)
            batch_goals = goal_multipliers.cpu().numpy()
            if goal_scales is not None:
                batch_goals = batch_goals * goal_scales[start:end]
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            goals.append(batch_goals)
    return np.vstack(probs), np.vstack(goals)


def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probs, 1e-8, 1.0)
    pred = np.argmax(clipped, axis=1)
    one_hot = np.eye(3)[labels]
    return {
        "accuracy": float(np.mean(pred == labels)),
        "log_loss": float(-np.mean(np.log(clipped[np.arange(len(labels)), labels]))),
        "brier": float(np.mean(np.sum((clipped - one_hot) ** 2, axis=1))),
    }


def simulation_probabilities(
    expected_goals: np.ndarray, *, runs: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    home_lam = np.clip(expected_goals[:, 0], 0.03, 8.0).reshape(-1, 1)
    away_lam = np.clip(expected_goals[:, 1], 0.03, 8.0).reshape(-1, 1)
    home_goals = rng.poisson(home_lam, size=(len(expected_goals), runs))
    away_goals = rng.poisson(away_lam, size=(len(expected_goals), runs))
    return np.vstack(
        [
            np.mean(home_goals > away_goals, axis=1),
            np.mean(home_goals == away_goals, axis=1),
            np.mean(home_goals < away_goals, axis=1),
        ]
    ).T


def simulation_metrics(
    expected_goals: np.ndarray, labels: np.ndarray, *, runs: int, seed: int
) -> dict[str, float]:
    return classification_metrics(simulation_probabilities(expected_goals, runs=runs, seed=seed), labels)


def calibration_table(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    confidence = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(float)
    table: list[dict[str, float]] = []
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        mask = (confidence >= low) & (confidence < high if idx < bins - 1 else confidence <= high)
        if not mask.any():
            continue
        table.append(
            {
                "bin_low": low,
                "bin_high": high,
                "count": int(mask.sum()),
                "avg_confidence": float(confidence[mask].mean()),
                "accuracy": float(correct[mask].mean()),
            }
        )
    return table


def composite_score(metrics: dict[str, float]) -> float:
    return metrics["accuracy"] - 0.08 * metrics["log_loss"] - 0.15 * metrics["brier"]


def _train_once(
    config: TrainConfig,
    model_config: ModelConfig,
    train_numeric: np.ndarray,
    train_categorical: np.ndarray,
    train_labels: np.ndarray,
    train_goals: np.ndarray,
    train_goal_scales: np.ndarray,
    train_weights: np.ndarray,
    test_numeric: np.ndarray,
    test_categorical: np.ndarray,
    test_labels: np.ndarray,
    test_goals: np.ndarray,
    test_goal_scales: np.ndarray,
    *,
    seed: int,
    epochs: int | None = None,
) -> tuple[TabularMultiTaskNet, dict[str, Any]]:
    set_seed(seed)
    device = _device()
    model = TabularMultiTaskNet(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.min_learning_rate,
    )
    train_loader = _make_loader(
        train_numeric,
        train_categorical,
        train_labels,
        train_goals,
        train_goal_scales,
        train_weights,
        batch_size=config.batch_size,
        shuffle=True,
    )
    run_epochs = epochs or config.epochs
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_epoch = run_epochs

    for epoch in range(run_epochs):
        model.train()
        losses: list[float] = []
        for x_num, x_cat, y, g, scale, weight in train_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)
            g = g.to(device)
            scale = scale.to(device)
            weight = weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, goal_multipliers = model(x_num, x_cat)
            expected_goals = goal_multipliers * scale
            loss = multitask_loss(
                logits,
                expected_goals,
                y,
                g,
                goal_loss_weight=model_config.goal_loss_weight,
                class_loss_weight=model_config.class_loss_weight,
                calibration_loss_weight=model_config.calibration_loss_weight,
                sample_weight=weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        class_probs, expected_goals = _predict_arrays(
            model,
            test_numeric,
            test_categorical,
            goal_scales=test_goal_scales,
            batch_size=config.batch_size,
            device=device,
        )
        metrics = simulation_metrics(
            expected_goals, test_labels, runs=config.simulation_runs, seed=config.seed
        )
        score = composite_score(metrics)
        scheduler.step(score)

        if score > best_score + config.early_stopping_min_delta:
            best_score = score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        is_log_epoch = epoch == run_epochs - 1 or (epoch + 1) % 25 == 0
        if is_log_epoch or epochs_without_improvement == config.early_stopping_patience:
            aux_metrics = classification_metrics(class_probs, test_labels)
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(np.mean(losses)),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    **{f"simulation_{key}": value for key, value in metrics.items()},
                    **{f"aux_class_{key}": value for key, value in aux_metrics.items()},
                }
            )

        if (
            config.early_stopping_patience > 0
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_epoch = epoch + 1
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_class_probs, test_expected_goals = _predict_arrays(
        model,
        test_numeric,
        test_categorical,
        goal_scales=test_goal_scales,
        batch_size=config.batch_size,
        device=device,
    )
    train_class_probs, train_expected_goals = _predict_arrays(
        model,
        train_numeric,
        train_categorical,
        goal_scales=train_goal_scales,
        batch_size=config.batch_size,
        device=device,
    )
    test_sim_probs = simulation_probabilities(
        test_expected_goals, runs=config.simulation_runs, seed=config.seed
    )
    train_sim_probs = simulation_probabilities(
        train_expected_goals, runs=config.simulation_runs, seed=config.seed
    )
    test_metrics = classification_metrics(test_sim_probs, test_labels)
    train_metrics = classification_metrics(train_sim_probs, train_labels)
    metrics: dict[str, Any] = {
        "seed": seed,
        "train": train_metrics,
        "test": test_metrics,
        "aux_class_train": classification_metrics(train_class_probs, train_labels),
        "aux_class_test": classification_metrics(test_class_probs, test_labels),
        "score": composite_score(test_metrics),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "history": history,
        "calibration": calibration_table(test_sim_probs, test_labels),
        "aux_class_calibration": calibration_table(test_class_probs, test_labels),
        "test_goal_mae": float(np.mean(np.abs(test_expected_goals - test_goals))),
    }
    return model, metrics


def _optuna_best_config(
    config: TrainConfig,
    model_config: ModelConfig,
    train_numeric: np.ndarray,
    train_categorical: np.ndarray,
    train_labels: np.ndarray,
    train_goals: np.ndarray,
    train_goal_scales: np.ndarray,
    train_weights: np.ndarray,
    test_numeric: np.ndarray,
    test_categorical: np.ndarray,
    test_labels: np.ndarray,
    test_goals: np.ndarray,
    test_goal_scales: np.ndarray,
) -> tuple[ModelConfig, TrainConfig]:
    if config.optuna_trials <= 0:
        return model_config, config
    try:
        import optuna  # type: ignore
    except ModuleNotFoundError:
        return model_config, config

    def objective(trial) -> float:
        trial_config = replace(
            model_config,
            hidden_sizes=[
                trial.suggest_categorical("hidden_1", [96, 128, 192, 256]),
                trial.suggest_categorical("hidden_2", [48, 64, 96, 128]),
                trial.suggest_categorical("hidden_3", [24, 32, 64]),
            ],
            dropout=trial.suggest_float("dropout", 0.08, 0.45),
            goal_loss_weight=trial.suggest_float("goal_loss_weight", 0.7, 1.5),
            class_loss_weight=trial.suggest_float("class_loss_weight", 0.0, 0.45),
            calibration_loss_weight=trial.suggest_float("calibration_loss_weight", 0.0, 0.12),
        )
        trial_train_config = replace(
            config,
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True),
        )
        _, metrics = _train_once(
            trial_train_config,
            trial_config,
            train_numeric,
            train_categorical,
            train_labels,
            train_goals,
            train_goal_scales,
            train_weights,
            test_numeric,
            test_categorical,
            test_labels,
            test_goals,
            test_goal_scales,
            seed=config.seeds[0],
            epochs=min(config.epochs, 50),
        )
        return float(metrics["score"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.optuna_trials)
    params = study.best_params
    best_model_config = replace(
        model_config,
        hidden_sizes=[int(params["hidden_1"]), int(params["hidden_2"]), int(params["hidden_3"])],
        dropout=float(params["dropout"]),
        goal_loss_weight=float(params["goal_loss_weight"]),
        class_loss_weight=float(params["class_loss_weight"]),
        calibration_loss_weight=float(params["calibration_loss_weight"]),
    )
    best_train_config = replace(
        config,
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    return best_model_config, best_train_config


def _prepare_train_test_frames(config: TrainConfig):
    features = build_training_features(config.dataset_dir)
    features = add_recency_weights(
        features, half_life_tournaments=config.recency_half_life_tournaments
    )
    train_idx, test_idx = build_split(features, seed=config.seed, test_size=config.split_test_size)
    train_frame = features.iloc[train_idx].copy().reset_index(drop=True)
    if config.home_away_swap_augmentation:
        swapped = augment_neutral_home_away_swap(train_frame)
        if not swapped.empty:
            train_frame = pd.concat([train_frame, swapped], ignore_index=True).reset_index(drop=True)
    external_frame = pd.DataFrame()
    if config.include_external_matches:
        external_frame = build_external_training_features(
            config.dataset_dir,
            start_year=config.external_match_start_year,
            max_rows=config.external_match_max_rows,
            base_weight=config.external_match_base_weight,
            half_life_years=config.external_match_recency_half_life_years,
        )
        if not external_frame.empty:
            train_frame = pd.concat([train_frame, external_frame], ignore_index=True).reset_index(
                drop=True
            )
    test_frame = features.iloc[test_idx].copy().reset_index(drop=True)
    return features, train_idx, test_idx, train_frame, test_frame, external_frame


def _frame_arrays(frame, schema, *, era_goal_scaling: bool):
    numeric, categorical = transform_features(frame, schema)
    labels = frame["label_id"].to_numpy(dtype=np.int64)
    goals = frame[["home_goals", "away_goals"]].to_numpy(dtype=np.float32)
    goal_scales = _goal_scales(frame, era_goal_scaling=era_goal_scaling)
    sample_weights = _sample_weights(frame)
    return numeric, categorical, labels, goals, goal_scales, sample_weights


def train_pipeline(config: TrainConfig) -> dict[str, Any]:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    _, train_idx, test_idx, train_frame, test_frame, external_frame = _prepare_train_test_frames(config)
    schema = fit_feature_schema(train_frame)
    (
        train_numeric,
        train_categorical,
        train_labels,
        train_goals,
        train_goal_scales,
        train_weights,
    ) = _frame_arrays(train_frame, schema, era_goal_scaling=config.era_goal_scaling)
    (
        test_numeric,
        test_categorical,
        test_labels,
        test_goals,
        test_goal_scales,
        _,
    ) = _frame_arrays(test_frame, schema, era_goal_scaling=config.era_goal_scaling)

    base_model_config = ModelConfig(
        n_numeric=train_numeric.shape[1],
        category_cardinalities=category_cardinalities(schema),
        embedding_dim=config.embedding_dim,
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
        goal_loss_weight=config.goal_loss_weight,
        class_loss_weight=config.class_loss_weight,
        calibration_loss_weight=config.calibration_loss_weight,
    )
    model_config, final_train_config = _optuna_best_config(
        config,
        base_model_config,
        train_numeric,
        train_categorical,
        train_labels,
        train_goals,
        train_goal_scales,
        train_weights,
        test_numeric,
        test_categorical,
        test_labels,
        test_goals,
        test_goal_scales,
    )

    best_model: TabularMultiTaskNet | None = None
    all_metrics: list[dict[str, Any]] = []
    for seed in config.seeds:
        model, metrics = _train_once(
            final_train_config,
            model_config,
            train_numeric,
            train_categorical,
            train_labels,
            train_goals,
            train_goal_scales,
            train_weights,
            test_numeric,
            test_categorical,
            test_labels,
            test_goals,
            test_goal_scales,
            seed=seed,
        )
        all_metrics.append(metrics)
        if best_model is None or metrics["score"] > max(item["score"] for item in all_metrics[:-1]):
            best_model = model

    assert best_model is not None
    best_metrics = max(all_metrics, key=lambda item: item["score"])
    artifact = {
        "model_state": best_model.cpu().state_dict(),
        "model_config": model_config.to_dict(),
        "feature_schema": schema.to_dict(),
        "train_config": final_train_config.to_dict(),
        "goal_output": "era_scaled_multipliers"
        if final_train_config.era_goal_scaling
        else "raw_expected_goals",
        "metrics": {"best": best_metrics, "runs": all_metrics},
        "split": {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()},
        "training_rows": {
            "world_cup_train": int(len(train_idx)),
            "external": int(len(external_frame)),
            "total": int(len(train_frame)),
        },
    }
    model_path = config.artifact_dir / "model.ckpt"
    torch.save(artifact, model_path)
    metrics_path = config.report_dir / "model_metrics.json"
    metrics_path.write_text(
        json.dumps({"model_path": str(model_path), **artifact["metrics"]}, indent=2),
        encoding="utf-8",
    )
    (config.artifact_dir / "feature_schema.json").write_text(
        json.dumps(schema.to_dict(), indent=2),
        encoding="utf-8",
    )
    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "training_rows": artifact["training_rows"],
        **artifact["metrics"],
    }


def load_artifact(model_path: str | Path) -> dict[str, Any]:
    return torch.load(Path(model_path), map_location="cpu")


def load_model_from_artifact(model_path: str | Path) -> tuple[TabularMultiTaskNet, dict[str, Any]]:
    artifact = load_artifact(model_path)
    model_config = ModelConfig.from_dict(artifact["model_config"])
    model = TabularMultiTaskNet(model_config)
    model.load_state_dict(artifact["model_state"])
    model.eval()
    return model, artifact


def evaluate_artifact(model_path: str | Path) -> dict[str, Any]:
    model, artifact = load_model_from_artifact(model_path)
    config = TrainConfig(
        **{k: v for k, v in artifact["train_config"].items() if k in TrainConfig.__dataclass_fields__}
    )
    config.dataset_dir = Path(config.dataset_dir)
    features = build_training_features(config.dataset_dir)
    from .features import FeatureSchema

    schema = FeatureSchema.from_dict(artifact["feature_schema"])
    test_idx = np.array(artifact["split"]["test_idx"], dtype=np.int64)
    test_frame = features.iloc[test_idx].copy().reset_index(drop=True)
    goal_output = artifact.get("goal_output")
    numeric, categorical, labels, goals, goal_scales, _ = _frame_arrays(
        test_frame,
        schema,
        era_goal_scaling=goal_output == "era_scaled_multipliers",
    )
    class_probs, expected_goals = _predict_arrays(
        model,
        numeric,
        categorical,
        goal_scales=goal_scales
        if artifact.get("goal_output") == "era_scaled_multipliers"
        else None,
        batch_size=config.batch_size,
        device=torch.device("cpu"),
    )
    sim_probs = simulation_probabilities(expected_goals, runs=config.simulation_runs, seed=config.seed)
    return {
        "test": classification_metrics(sim_probs, labels),
        "aux_class_test": classification_metrics(class_probs, labels),
        "calibration": calibration_table(sim_probs, labels),
        "test_goal_mae": float(np.mean(np.abs(expected_goals - goals))),
    }
