from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MarketSnapshot(BaseModel):
    phase: Literal["pre"]
    market_type: Literal["team_regulation_win"]
    contract_side: Literal["yes", "no"]
    match_number: int | None = None
    team: str
    opponent: str
    q_market: float = Field(ge=0.0, le=1.0)
    fee_estimate: float = Field(default=0.0, ge=0.0)
    bankroll: float = Field(gt=0.0)
    match_exposure: float = Field(default=0.0, ge=0.0)
    team_exposure: float = Field(default=0.0, ge=0.0)
    total_open_exposure: float = Field(default=0.0, ge=0.0)
    liquidity_cap: float = Field(default=0.0, ge=0.0)


class Recommendation(BaseModel):
    decision: Literal["buy_yes", "buy_no", "no_trade"]
    predicted_result: str
    simulation_counts: dict[str, int]
    p_model: float
    q_market: float
    q_eff: float
    edge: float
    threshold: float
    stake_usd: float
    contracts: int
    reason: str


def model_to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

