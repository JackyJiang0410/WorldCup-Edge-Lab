from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class ModelConfig:
    n_numeric: int
    category_cardinalities: list[int]
    embedding_dim: int
    hidden_sizes: list[int]
    dropout: float
    goal_loss_weight: float
    class_loss_weight: float
    calibration_loss_weight: float

    def to_dict(self) -> dict:
        return {
            "n_numeric": self.n_numeric,
            "category_cardinalities": self.category_cardinalities,
            "embedding_dim": self.embedding_dim,
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "goal_loss_weight": self.goal_loss_weight,
            "class_loss_weight": self.class_loss_weight,
            "calibration_loss_weight": self.calibration_loss_weight,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        return cls(
            n_numeric=int(raw["n_numeric"]),
            category_cardinalities=[int(x) for x in raw["category_cardinalities"]],
            embedding_dim=int(raw["embedding_dim"]),
            hidden_sizes=[int(x) for x in raw["hidden_sizes"]],
            dropout=float(raw["dropout"]),
            goal_loss_weight=float(raw.get("goal_loss_weight", 1.0)),
            class_loss_weight=float(raw.get("class_loss_weight", 0.25)),
            calibration_loss_weight=float(raw.get("calibration_loss_weight", 0.05)),
        )


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.relu(x + self.layers(x)))


class TabularMultiTaskNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, min(config.embedding_dim, max(2, (cardinality + 1) // 2)))
                for cardinality in config.category_cardinalities
            ]
        )
        embedding_width = sum(embedding.embedding_dim for embedding in self.embeddings)
        input_width = config.n_numeric + embedding_width
        layers: list[nn.Module] = []
        previous = input_width
        for hidden in config.hidden_sizes:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(config.dropout)])
            layers.append(ResidualBlock(hidden, config.dropout))
            previous = hidden
        self.backbone = nn.Sequential(*layers)
        self.class_head = nn.Linear(previous, 3)
        self.goal_head = nn.Sequential(nn.Linear(previous, 2), nn.Softplus())

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.embeddings:
            embedded = [
                embedding(categorical[:, idx].clamp(min=0, max=embedding.num_embeddings - 1))
                for idx, embedding in enumerate(self.embeddings)
            ]
            x = torch.cat([numeric, *embedded], dim=1)
        else:
            x = numeric
        hidden = self.backbone(x)
        logits = self.class_head(hidden)
        expected_goals = self.goal_head(hidden).clamp(min=0.03, max=8.0)
        return logits, expected_goals


def multitask_loss(
    logits: torch.Tensor,
    expected_goals: torch.Tensor,
    labels: torch.Tensor,
    goals: torch.Tensor,
    *,
    goal_loss_weight: float,
    class_loss_weight: float,
    calibration_loss_weight: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    per_class_loss = F.cross_entropy(logits, labels, reduction="none")
    per_goal_loss = F.poisson_nll_loss(expected_goals, goals, log_input=False, reduction="none").mean(dim=1)
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=3).float()
    per_brier = torch.sum((probs - one_hot) ** 2, dim=1)
    if sample_weight is None:
        class_loss = per_class_loss.mean()
        goal_loss = per_goal_loss.mean()
        brier = per_brier.mean()
    else:
        weights = sample_weight / sample_weight.mean().clamp_min(1e-8)
        class_loss = torch.mean(per_class_loss * weights)
        goal_loss = torch.mean(per_goal_loss * weights)
        brier = torch.mean(per_brier * weights)
    return goal_loss_weight * goal_loss + class_loss_weight * class_loss + calibration_loss_weight * brier


def build_lightning_module(config: ModelConfig, learning_rate: float, weight_decay: float):
    """Create a LightningModule lazily so ordinary inference avoids Lightning import side effects."""
    try:
        import lightning.pytorch as pl  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional install state
        raise RuntimeError("Lightning is not installed. Install project dependencies first.") from exc

    class LightningTabularModel(pl.LightningModule):  # type: ignore[misc]
        def __init__(self):
            super().__init__()
            self.save_hyperparameters(config.to_dict())
            self.model = TabularMultiTaskNet(config)
            self.config_obj = config
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay

        def forward(self, numeric: torch.Tensor, categorical: torch.Tensor):
            return self.model(numeric, categorical)

        def training_step(self, batch, batch_idx):
            numeric, categorical, labels, goals = batch
            logits, expected_goals = self.model(numeric, categorical)
            loss = multitask_loss(
                logits,
                expected_goals,
                labels,
                goals,
                goal_loss_weight=self.config_obj.goal_loss_weight,
                class_loss_weight=self.config_obj.class_loss_weight,
                calibration_loss_weight=self.config_obj.calibration_loss_weight,
            )
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

    return LightningTabularModel()
