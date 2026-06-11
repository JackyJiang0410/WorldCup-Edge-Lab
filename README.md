# WorldCup Edge Lab

WorldCup Edge Lab is a research toolkit for World Cup match prediction and prediction-market decision support. It trains calibrated pre-match models, simulates scorelines, and compares model probabilities with user-supplied market prices after fees and exposure limits.

This repository is for research, backtesting, and decision support. It is not an automated trading system and should not be treated as financial advice.

## What It Does

- Builds leakage-safe World Cup match features from historical CSV data.
- Uses FIFA ranking and Elo rating snapshots with as-of joins.
- Adds broader international match history from `all_matches.csv` for recent-form features and low-weight external training rows.
- Trains a PyTorch tabular expected-goals model with an auxiliary outcome head.
- Simulates each match 1,000 times from predicted expected goals.
- Evaluates accuracy, log loss, Brier score, calibration, and feature importance.
- Produces deterministic `team_regulation_win` market recommendations from a JSON snapshot.

## Project Layout

```text
Dataset/                  source CSV inputs
configs/v1.yaml           default training config
src/worldcup2026/         package source
tests/                    data, feature, simulation, and trading tests
artifacts/                trained model artifacts
reports/                  metrics and feature-importance outputs
pyproject.toml            package and dependency metadata
```

## Install

From the repository root:

```bash
pip3 install -e '.[dev]'
```

If the console script is not on your PATH, use:

```bash
python3 -m worldcup2026.cli --help
```

## Common Commands

Validate the dataset:

```bash
worldcup2026 validate-data
```

Train the V1 model:

```bash
worldcup2026 train --config configs/v1.yaml
```

Evaluate a trained model:

```bash
worldcup2026 evaluate --model artifacts/v1/model.ckpt
```

Compute permutation feature importance:

```bash
worldcup2026 feature-importance --model artifacts/v1/model.ckpt
```

Run a recommendation from a market snapshot:

```bash
worldcup2026 recommend --model artifacts/v1/model.ckpt --snapshot market_snapshot.json
```

## Market Snapshot Example

```json
{
  "phase": "pre",
  "market_type": "team_regulation_win",
  "contract_side": "yes",
  "match_number": 7,
  "team": "Brazil",
  "opponent": "Morocco",
  "q_market": 0.60,
  "fee_estimate": 0.012,
  "bankroll": 500,
  "match_exposure": 0,
  "team_exposure": 0,
  "total_open_exposure": 0,
  "liquidity_cap": 50
}
```

The recommendation output includes the decision, simulated result counts, model probability, effective market price, edge, threshold, stake, contracts, and reason.

## Current V1 Scope

V1 supports:

- Men's World Cup pre-match modeling.
- Regulation-time `home_win`, `draw`, and `away_win` outcomes.
- `team_regulation_win` market recommendations.
- `phase = pre`.
- 1,000-run scoreline simulation per match.
- Quarter-Kelly sizing with exposure caps.

V1 does not yet support:

- Standalone draw-contract trading.
- Half-time or post-match futures recommendations.
- Historical market backtests without supplied market snapshots.
- Automated trading execution.

## Data Notes

The model uses source CSVs in `Dataset/`. Source files should be treated as immutable inputs. Derived features, metrics, and model artifacts are written outside `Dataset/`.

Important modeling rules:

- Historical World Cup test rows come from held-out `matches_curated.csv` matches only.
- Non-World-Cup rows from `all_matches.csv` may be used only as low-weight training rows.
- Rating time series are joined by latest rating date on or before the match date.
- Manager-derived features are intentionally excluded from V1 model input.
- Matches involving Argentina or Spain are kept in simulations but excluded from trade recommendations by default.

## Current Model Inputs

The trained V1 artifact currently uses 89 numeric features and 4 categorical features. The full schema is stored in:

```text
artifacts/v1/model.ckpt
```

To inspect it:

```bash
python3 - <<'PY'
from worldcup2026.training import load_artifact

schema = load_artifact("artifacts/v1/model.ckpt")["feature_schema"]
print(schema["numeric_features"])
print(schema["categorical_features"])
PY
```

## Verification

Run:

```bash
python3 -m ruff check .
python3 -m pytest -q
```

The project is intentionally deterministic where practical: fixed seeds are used for train/test splits and scoreline simulation tests.

## Risk Disclaimer

Prediction markets involve risk. Model probabilities can be wrong, calibration can drift, liquidity can change, and fees can erase apparent edge. Use this project for research and validation, not as a standalone real-money trading authority.
