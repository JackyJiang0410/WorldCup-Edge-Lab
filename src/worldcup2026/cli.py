from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import TrainConfig, load_config
from .data import validate_dataset
from .schemas import model_to_dict


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _cmd_validate_data(args: argparse.Namespace) -> None:
    validation = validate_dataset(args.dataset)
    _print_json(validation.to_dict())
    if not validation.ok:
        raise SystemExit(1)


def _cmd_train(args: argparse.Namespace) -> None:
    from .training import train_pipeline

    config = load_config(args.config)
    result = train_pipeline(config)
    _print_json(result)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from .training import evaluate_artifact

    result = evaluate_artifact(args.model)
    _print_json(result)


def _cmd_ablate(args: argparse.Namespace) -> None:
    from .ablation import run_ablation_pipeline

    config = load_config(args.config)
    result = run_ablation_pipeline(config)
    _print_json(result)


def _cmd_feature_importance(args: argparse.Namespace) -> None:
    from .importance import permutation_importance

    result = permutation_importance(args.model, output_path=args.output, seed=args.seed)
    _print_json(result)


def _cmd_recommend(args: argparse.Namespace) -> None:
    from .trading import recommend_from_dict

    config = load_config(args.config) if args.config else TrainConfig(dataset_dir=Path(args.dataset))
    payload = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    recommendation = recommend_from_dict(
        payload,
        model_path=args.model,
        dataset_dir=config.dataset_dir if args.dataset is None else args.dataset,
        config=config,
    )
    _print_json(model_to_dict(recommendation))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="worldcup2026")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-data")
    validate_parser.add_argument("--dataset", default="Dataset")
    validate_parser.set_defaults(func=_cmd_validate_data)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", default="configs/v1.yaml")
    train_parser.set_defaults(func=_cmd_train)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    ablate_parser = subparsers.add_parser("ablate")
    ablate_parser.add_argument("--config", default="configs/v1.yaml")
    ablate_parser.set_defaults(func=_cmd_ablate)

    importance_parser = subparsers.add_parser("feature-importance")
    importance_parser.add_argument("--model", required=True)
    importance_parser.add_argument("--output", default="reports/feature_importance.json")
    importance_parser.add_argument("--seed", type=int, default=42)
    importance_parser.set_defaults(func=_cmd_feature_importance)

    recommend_parser = subparsers.add_parser("recommend")
    recommend_parser.add_argument("--model", required=True)
    recommend_parser.add_argument("--snapshot", required=True)
    recommend_parser.add_argument("--dataset", default=None)
    recommend_parser.add_argument("--config", default="configs/v1.yaml")
    recommend_parser.set_defaults(func=_cmd_recommend)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
