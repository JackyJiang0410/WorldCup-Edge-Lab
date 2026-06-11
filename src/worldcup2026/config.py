from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .constants import DEFAULT_ARTIFACT_DIR, DEFAULT_DATASET_DIR, DEFAULT_REPORT_DIR


@dataclass(slots=True)
class TrainConfig:
    dataset_dir: Path = DEFAULT_DATASET_DIR
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    report_dir: Path = DEFAULT_REPORT_DIR
    seed: int = 42
    split_test_size: float = 0.2
    simulation_runs: int = 1000
    epochs: int = 120
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 0.0005
    batch_size: int = 64
    learning_rate: float = 0.001
    lr_patience: int = 4
    lr_factor: float = 0.5
    min_learning_rate: float = 0.00001
    weight_decay: float = 0.0001
    dropout: float = 0.15
    hidden_sizes: list[int] = field(default_factory=lambda: [192, 128, 64])
    embedding_dim: int = 8
    goal_loss_weight: float = 1.0
    class_loss_weight: float = 0.25
    calibration_loss_weight: float = 0.05
    era_goal_scaling: bool = True
    recency_half_life_tournaments: float = 6.0
    home_away_swap_augmentation: bool = False
    include_external_matches: bool = True
    external_match_start_year: int = 2010
    external_match_max_rows: int = 4000
    external_match_base_weight: float = 0.05
    external_match_recency_half_life_years: float = 8.0
    seeds: list[int] = field(default_factory=lambda: [11, 23, 37])
    optuna_trials: int = 8

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["dataset_dir"] = str(self.dataset_dir)
        raw["artifact_dir"] = str(self.artifact_dir)
        raw["report_dir"] = str(self.report_dir)
        return raw


def _coerce_config_value(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        if value in {"", "null", "None"}:
            return None
        if value in {"true", "True"}:
            return True
        if value in {"false", "False"}:
            return False
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        values[key.strip()] = _coerce_config_value(value)
    return values


def load_config(path: str | Path | None = None) -> TrainConfig:
    if path is None:
        return TrainConfig()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except ModuleNotFoundError:
        raw = _load_simple_yaml(config_path)

    allowed = set(TrainConfig.__dataclass_fields__)
    clean = {key: _coerce_config_value(value) for key, value in raw.items() if key in allowed}
    for path_key in ("dataset_dir", "artifact_dir", "report_dir"):
        if path_key in clean and clean[path_key] is not None:
            clean[path_key] = Path(clean[path_key])
    return TrainConfig(**clean)
