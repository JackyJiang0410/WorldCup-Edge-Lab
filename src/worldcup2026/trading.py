from __future__ import annotations

from pathlib import Path

from .config import TrainConfig
from .predict import predict_2026_match
from .schemas import MarketSnapshot, Recommendation
from .team_names import is_prohibited_trade_team


PRE_MATCH_THRESHOLD = 0.03
PRE_MATCH_PHASE_CAP = 0.02
MATCH_CAP = 0.04
TEAM_CAP = 0.06
TOTAL_OPEN_CAP = 0.20


def _stake_size(
    *,
    bankroll: float,
    p_model: float,
    q_eff: float,
    phase_exposure: float,
    match_exposure: float,
    team_exposure: float,
    total_open_exposure: float,
    liquidity_cap: float,
) -> tuple[float, int, dict[str, float]]:
    if q_eff >= 1.0 or p_model <= q_eff:
        return 0.0, 0, {}
    full_kelly = (p_model - q_eff) / (1.0 - q_eff)
    fractional_kelly = max(0.0, 0.25 * full_kelly)
    limits = {
        "kelly": bankroll * fractional_kelly,
        "phase_cap": max(0.0, bankroll * PRE_MATCH_PHASE_CAP - phase_exposure),
        "match_cap": max(0.0, bankroll * MATCH_CAP - match_exposure),
        "team_cap": max(0.0, bankroll * TEAM_CAP - team_exposure),
        "total_open_cap": max(0.0, bankroll * TOTAL_OPEN_CAP - total_open_exposure),
        "liquidity_cap": max(0.0, liquidity_cap),
    }
    stake = min(limits.values())
    contracts = int(stake // q_eff) if q_eff > 0 else 0
    return round(float(stake), 2), contracts, limits


def recommend_trade(
    snapshot: MarketSnapshot,
    *,
    model_path: str | Path,
    dataset_dir: str | Path,
    simulation_runs: int = 1000,
    seed: int = 42,
) -> Recommendation:
    if is_prohibited_trade_team(snapshot.team) or is_prohibited_trade_team(snapshot.opponent):
        return Recommendation(
            decision="no_trade",
            predicted_result="excluded_match",
            simulation_counts={"team_win": 0, "draw": 0, "opponent_win": 0},
            p_model=0.0,
            q_market=snapshot.q_market,
            q_eff=min(1.0, snapshot.q_market + snapshot.fee_estimate),
            edge=0.0,
            threshold=PRE_MATCH_THRESHOLD,
            stake_usd=0.0,
            contracts=0,
            reason="match involves Argentina or Spain",
        )

    prediction = predict_2026_match(
        model_path=model_path,
        dataset_dir=dataset_dir,
        team=snapshot.team,
        opponent=snapshot.opponent,
        match_number=snapshot.match_number,
        simulation_runs=simulation_runs,
        seed=seed,
    )
    team_win = prediction.requested_team_win_probability
    p_model = team_win if snapshot.contract_side == "yes" else 1.0 - team_win
    q_eff = min(1.0, snapshot.q_market + snapshot.fee_estimate)
    edge = p_model - q_eff

    if edge <= PRE_MATCH_THRESHOLD:
        return Recommendation(
            decision="no_trade",
            predicted_result=prediction.relative_predicted_result,
            simulation_counts=prediction.relative_simulation_counts,
            p_model=round(float(p_model), 6),
            q_market=snapshot.q_market,
            q_eff=round(float(q_eff), 6),
            edge=round(float(edge), 6),
            threshold=PRE_MATCH_THRESHOLD,
            stake_usd=0.0,
            contracts=0,
            reason="edge does not clear pre-match threshold",
        )

    stake, contracts, _ = _stake_size(
        bankroll=snapshot.bankroll,
        p_model=p_model,
        q_eff=q_eff,
        phase_exposure=0.0,
        match_exposure=snapshot.match_exposure,
        team_exposure=snapshot.team_exposure,
        total_open_exposure=snapshot.total_open_exposure,
        liquidity_cap=snapshot.liquidity_cap,
    )
    if stake <= 1.0 or contracts <= 0:
        return Recommendation(
            decision="no_trade",
            predicted_result=prediction.relative_predicted_result,
            simulation_counts=prediction.relative_simulation_counts,
            p_model=round(float(p_model), 6),
            q_market=snapshot.q_market,
            q_eff=round(float(q_eff), 6),
            edge=round(float(edge), 6),
            threshold=PRE_MATCH_THRESHOLD,
            stake_usd=0.0,
            contracts=0,
            reason="stake is too small after Kelly and exposure caps",
        )

    return Recommendation(
        decision="buy_yes" if snapshot.contract_side == "yes" else "buy_no",
        predicted_result=prediction.relative_predicted_result,
        simulation_counts=prediction.relative_simulation_counts,
        p_model=round(float(p_model), 6),
        q_market=snapshot.q_market,
        q_eff=round(float(q_eff), 6),
        edge=round(float(edge), 6),
        threshold=PRE_MATCH_THRESHOLD,
        stake_usd=stake,
        contracts=contracts,
        reason="edge clears threshold and exposure caps",
    )


def recommend_from_dict(
    payload: dict,
    *,
    model_path: str | Path,
    dataset_dir: str | Path,
    config: TrainConfig | None = None,
) -> Recommendation:
    snapshot = MarketSnapshot(**payload)
    simulation_runs = config.simulation_runs if config else 1000
    seed = config.seed if config else 42
    return recommend_trade(
        snapshot,
        model_path=model_path,
        dataset_dir=dataset_dir,
        simulation_runs=simulation_runs,
        seed=seed,
    )

