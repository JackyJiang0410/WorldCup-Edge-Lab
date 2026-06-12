# Match 1: Mexico vs South Africa

## Model Input

- match_number: 1
- round_number: 1
- date: 11/06/2026 19:00
- group: Group A
- location: Mexico City Stadium
- home_team: Mexico
- away_team: South Africa
- model_artifact: artifacts/experiments/direct_old_hparams/model.ckpt
- config: configs/v1.yaml
- bankroll: 500
- max_per_game_usd: 100

## Prediction

- requested_team: Mexico
- requested_team_alias_from_input: MEX
- opponent_alias_from_input: RSA
- predicted_result: team_win
- p_model_mexico_yes: 0.775602
- prediction_error: 

## Market Input

- market_type: team_regulation_win
- contract_side: yes
- team: MEX
- opponent: RSA
- q_market: 0.72
- fee_estimate: 0.01
- q_eff: 0.73
- liquidity_cap: 100
- match_exposure: 0
- team_exposure: 0
- total_open_exposure: 0

## Recommendation

- trade_excluded: no
- decision: buy_yes
- recommended_bet: Mexico YES
- p_model: 0.775602
- edge: 0.045602
- threshold: 0.03
- stake_usd: 10.0
- contracts: 13
- reason: edge clears threshold and exposure caps
- note: With a 100 max per game, this is the only positive-edge side from the supplied prices. The recorded stake is the configured recommendation before manually raising caps.

## Actual Trading

- MEX limit buy: filled
- contracts_bought: 98
- buy_price: 0.72
- cash_outflow_usd: 71.94
- canceled_orders: MEX wins 1st half limit buy

## Final Result And Profit

- final_score: Mexico won
- bet_result: won
- payout_usd: 98.00
- realized_pnl_usd: 26.06
- notes: Realized PnL = 98.00 payout - 71.94 cash outflow.
