from __future__ import annotations

from worldcup2026.schemas import MarketSnapshot
from worldcup2026.trading import _stake_size, recommend_trade


def test_argentina_spain_exclusion_does_not_require_model():
    snapshot = MarketSnapshot(
        phase="pre",
        market_type="team_regulation_win",
        contract_side="yes",
        match_number=1,
        team="Argentina",
        opponent="Brazil",
        q_market=0.55,
        fee_estimate=0.01,
        bankroll=500,
        liquidity_cap=50,
    )
    recommendation = recommend_trade(snapshot, model_path="missing.ckpt", dataset_dir="Dataset")
    assert recommendation.decision == "no_trade"
    assert "Argentina or Spain" in recommendation.reason


def test_kelly_stake_respects_caps():
    stake, contracts, limits = _stake_size(
        bankroll=500,
        p_model=0.66,
        q_eff=0.612,
        phase_exposure=0,
        match_exposure=0,
        team_exposure=0,
        total_open_exposure=0,
        liquidity_cap=50,
    )
    assert stake == 10.0
    assert contracts == 16
    assert limits["phase_cap"] == 10.0

