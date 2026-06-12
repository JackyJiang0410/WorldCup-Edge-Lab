from __future__ import annotations

import json
import random
import sys
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


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal"
    return "CPU"


def _progress_bar(current: int, total: int, *, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int(round(width * min(current, total) / total))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _write_progress(
    message: str,
    *,
    end: str = "\n",
    stream=sys.stderr,
) -> None:
    stream.write(message + end)
    stream.flush()


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


def exact_scoreline_probabilities(expected_goals: np.ndarray, *, max_goals: int = 15) -> np.ndarray:
    max_goals = max(3, int(max_goals))
    goal_values = np.arange(max_goals + 1)
    home_lam = np.clip(expected_goals[:, 0], 0.03, 8.0).astype(np.float64)
    away_lam = np.clip(expected_goals[:, 1], 0.03, 8.0).astype(np.float64)

    home_pmf = np.empty((len(expected_goals), max_goals + 1), dtype=np.float64)
    away_pmf = np.empty((len(expected_goals), max_goals + 1), dtype=np.float64)
    home_pmf[:, 0] = np.exp(-home_lam)
    away_pmf[:, 0] = np.exp(-away_lam)
    for goals in range(1, max_goals + 1):
        home_pmf[:, goals] = home_pmf[:, goals - 1] * home_lam / goals
        away_pmf[:, goals] = away_pmf[:, goals - 1] * away_lam / goals

    score_matrix = home_pmf[:, :, None] * away_pmf[:, None, :]
    home_win = score_matrix[:, goal_values[:, None] > goal_values[None, :]].sum(axis=1)
    draw = score_matrix[:, goal_values[:, None] == goal_values[None, :]].sum(axis=1)
    away_win = score_matrix[:, goal_values[:, None] < goal_values[None, :]].sum(axis=1)
    probs = np.vstack([home_win, draw, away_win]).T
    return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)


def _blend_probabilities(
    score_probs: np.ndarray, class_probs: np.ndarray, *, class_weight: float
) -> np.ndarray:
    weight = float(np.clip(class_weight, 0.0, 1.0))
    probs = (1.0 - weight) * score_probs + weight * class_probs
    return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)


def _probabilities_by_source(
    class_probs: np.ndarray,
    expected_goals: np.ndarray,
    *,
    config: TrainConfig,
) -> dict[str, np.ndarray]:
    exact_probs = exact_scoreline_probabilities(
        expected_goals, max_goals=config.exact_score_max_goals
    )
    return {
        "exact_score": exact_probs,
        "direct_class": class_probs,
        "blend": _blend_probabilities(
            exact_probs,
            class_probs,
            class_weight=config.class_probability_blend_weight,
        ),
    }


def _selected_probabilities(sources: dict[str, np.ndarray], source: str) -> np.ndarray:
    if source not in sources:
        allowed = ", ".join(sorted(sources))
        raise ValueError(f"Unknown result_probability_source {source!r}; expected one of: {allowed}")
    return sources[source]


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


def per_class_calibration_table(
    probs: np.ndarray, labels: np.ndarray, bins: int = 10
) -> dict[str, list[dict[str, float]]]:
    names = ("home_win", "draw", "away_win")
    table: dict[str, list[dict[str, float]]] = {}
    for class_idx, name in enumerate(names):
        class_probs = probs[:, class_idx]
        observed = (labels == class_idx).astype(float)
        rows: list[dict[str, float]] = []
        for idx in range(bins):
            low = idx / bins
            high = (idx + 1) / bins
            mask = (class_probs >= low) & (
                class_probs < high if idx < bins - 1 else class_probs <= high
            )
            if not mask.any():
                continue
            rows.append(
                {
                    "bin_low": low,
                    "bin_high": high,
                    "count": int(mask.sum()),
                    "avg_probability": float(class_probs[mask].mean()),
                    "observed_rate": float(observed[mask].mean()),
                }
            )
        table[name] = rows
    return table


def draw_diagnostics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    draw_mask = labels == 1
    return {
        "actual_draw_rate": float(draw_mask.mean()),
        "avg_predicted_draw_probability": float(probs[:, 1].mean()),
        "draw_brier": float(np.mean((probs[:, 1] - draw_mask.astype(float)) ** 2)),
        "draw_log_loss": float(-np.mean(np.log(np.clip(probs[draw_mask, 1], 1e-8, 1.0))))
        if draw_mask.any()
        else 0.0,
    }


def composite_score(metrics: dict[str, float]) -> float:
    return -metrics["log_loss"] - 0.75 * metrics["brier"]


def _temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-3)
    scaled = np.power(np.clip(probs, 1e-8, 1.0), 1.0 / temperature)
    return scaled / np.clip(scaled.sum(axis=1, keepdims=True), 1e-8, None)


def _fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.concatenate(
        [
            np.linspace(0.55, 1.45, 19),
            np.linspace(1.5, 3.0, 16),
        ]
    )
    best_temperature = 1.0
    best_log_loss = float("inf")
    for temperature in candidates:
        metrics = classification_metrics(_temperature_scale(probs, float(temperature)), labels)
        if metrics["log_loss"] < best_log_loss:
            best_log_loss = metrics["log_loss"]
            best_temperature = float(temperature)
    return best_temperature


def _source_metrics(
    sources: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    train_sources: dict[str, np.ndarray] | None = None,
    train_labels: np.ndarray | None = None,
    temperature_calibration: bool,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, probs in sources.items():
        raw_metrics = classification_metrics(probs, labels)
        metrics[name] = {
            **raw_metrics,
            "score": composite_score(raw_metrics),
            "draw": draw_diagnostics(probs, labels),
        }
        if temperature_calibration and train_sources is not None and train_labels is not None:
            temperature = _fit_temperature(train_sources[name], train_labels)
            calibrated = _temperature_scale(probs, temperature)
            calibrated_metrics = classification_metrics(calibrated, labels)
            metrics[name]["temperature"] = temperature
            metrics[name]["calibrated"] = {
                **calibrated_metrics,
                "score": composite_score(calibrated_metrics),
                "draw": draw_diagnostics(calibrated, labels),
            }
    return metrics


def _maybe_calibrated_selected(
    probs: np.ndarray,
    *,
    train_probs: np.ndarray,
    train_labels: np.ndarray,
    temperature_calibration: bool,
) -> tuple[np.ndarray, float]:
    if not temperature_calibration:
        return probs, 1.0
    temperature = _fit_temperature(train_probs, train_labels)
    return _temperature_scale(probs, temperature), temperature


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
    run_label: str | None = None,
    show_progress: bool = False,
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
    progress_stream = sys.stderr
    progress_is_tty = progress_stream.isatty()

    if show_progress and run_label is not None:
        _write_progress(f"{run_label}: seed={seed}, device={device.type} ({_device_name(device)})")

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
        sources = _probabilities_by_source(
            class_probs,
            expected_goals,
            config=config,
        )
        metrics = classification_metrics(
            _selected_probabilities(sources, config.result_probability_source),
            test_labels,
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
                    **{f"selected_{key}": value for key, value in metrics.items()},
                    **{f"aux_class_{key}": value for key, value in aux_metrics.items()},
                }
            )

        if show_progress and run_label is not None:
            progress_message = (
                f"{run_label} "
                f"{_progress_bar(epoch + 1, run_epochs)} "
                f"epoch {epoch + 1}/{run_epochs} "
                f"loss={float(np.mean(losses)):.4f} "
                f"score={score:.4f} "
                f"best={best_score:.4f} "
                f"lr={float(optimizer.param_groups[0]['lr']):.2g} "
                f"patience={epochs_without_improvement}/{config.early_stopping_patience}"
            )
            if progress_is_tty:
                _write_progress("\r" + progress_message, end="")
            elif epoch == 0 or (epoch + 1) % 10 == 0 or epoch == run_epochs - 1:
                _write_progress(progress_message)

        if (
            config.early_stopping_patience > 0
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_epoch = epoch + 1
            break

    if show_progress and run_label is not None and progress_is_tty:
        _write_progress("")

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
    test_sources = _probabilities_by_source(
        test_class_probs,
        test_expected_goals,
        config=config,
    )
    train_sources = _probabilities_by_source(
        train_class_probs,
        train_expected_goals,
        config=config,
    )
    test_selected_raw = _selected_probabilities(test_sources, config.result_probability_source)
    train_selected_raw = _selected_probabilities(train_sources, config.result_probability_source)
    test_selected, selected_temperature = _maybe_calibrated_selected(
        test_selected_raw,
        train_probs=train_selected_raw,
        train_labels=train_labels,
        temperature_calibration=config.temperature_calibration,
    )
    train_selected = (
        _temperature_scale(train_selected_raw, selected_temperature)
        if config.temperature_calibration
        else train_selected_raw
    )
    test_metrics = classification_metrics(test_selected, test_labels)
    train_metrics = classification_metrics(train_selected, train_labels)
    metrics: dict[str, Any] = {
        "seed": seed,
        "result_probability_source": config.result_probability_source,
        "selected_temperature": selected_temperature,
        "train": train_metrics,
        "test": test_metrics,
        "probability_sources": _source_metrics(
            test_sources,
            test_labels,
            train_sources=train_sources,
            train_labels=train_labels,
            temperature_calibration=config.temperature_calibration,
        ),
        "aux_class_train": classification_metrics(train_class_probs, train_labels),
        "aux_class_test": classification_metrics(test_class_probs, test_labels),
        "score": composite_score(test_metrics),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "history": history,
        "calibration": calibration_table(test_selected, test_labels),
        "per_class_calibration": per_class_calibration_table(test_selected, test_labels),
        "draw": draw_diagnostics(test_selected, test_labels),
        "aux_class_calibration": calibration_table(test_class_probs, test_labels),
        "test_goal_mae": float(np.mean(np.abs(test_expected_goals - test_goals))),
        "monte_carlo_simulation_test": classification_metrics(test_sim_probs, test_labels),
        "monte_carlo_simulation_train": classification_metrics(train_sim_probs, train_labels),
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
    device = _device()
    _write_progress(
        "Training on "
        f"{device.type} ({_device_name(device)}); "
        f"world_cup_train={len(train_idx)}, "
        f"external={len(external_frame)}, "
        f"total={len(train_frame)}, "
        f"seeds={config.seeds}"
    )
    for seed_index, seed in enumerate(config.seeds, start=1):
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
            run_label=f"seed {seed_index}/{len(config.seeds)}",
            show_progress=True,
        )
        all_metrics.append(metrics)
        _write_progress(
            f"seed {seed_index}/{len(config.seeds)} complete: "
            f"best_epoch={metrics['best_epoch']}, "
            f"stopped_epoch={metrics['stopped_epoch']}, "
            f"test_accuracy={metrics['test']['accuracy']:.4f}, "
            f"log_loss={metrics['test']['log_loss']:.4f}, "
            f"brier={metrics['test']['brier']:.4f}, "
            f"score={metrics['score']:.4f}"
        )
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
        "result_probability_source": final_train_config.result_probability_source,
        "selected_temperature": best_metrics["selected_temperature"],
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
    sources = _probabilities_by_source(class_probs, expected_goals, config=config)
    selected = _selected_probabilities(
        sources,
        artifact.get("result_probability_source", config.result_probability_source),
    )
    if artifact.get("selected_temperature") is not None:
        selected = _temperature_scale(selected, float(artifact["selected_temperature"]))
    return {
        "test": classification_metrics(selected, labels),
        "probability_sources": _source_metrics(
            sources,
            labels,
            temperature_calibration=False,
        ),
        "aux_class_test": classification_metrics(class_probs, labels),
        "calibration": calibration_table(selected, labels),
        "per_class_calibration": per_class_calibration_table(selected, labels),
        "draw": draw_diagnostics(selected, labels),
        "test_goal_mae": float(np.mean(np.abs(expected_goals - goals))),
        "monte_carlo_simulation_test": classification_metrics(sim_probs, labels),
    }
