# Match 2: Korea Republic vs Czechia

## Model Input

- match_number: 2
- round_number: 1
- date: 12/06/2026 02:00
- group: Group A
- location: Guadalajara Stadium
- home_team: Korea Republic
- away_team: Czechia
- model_artifact: artifacts/v1/model.ckpt
- config: configs/v1.yaml
- bankroll: 500
- max_per_game_usd: 100

## Prediction

- requested_team: South Korea
- requested_team_alias_from_input: KOR
- opponent_alias_from_input: CZE
- canonical_opponent: Czech Republic / Czechia
- predicted_result: draw
- p_south_korea_yes: 0.352933
- p_draw_yes: 0.3844
- p_czech_yes: 0.2626
- prediction_error: Earlier raw KOR/CZE output was not reliable because of code-alias mapping.

## Market Input

- market_type: team_regulation_win
- contract_side: yes
- team: KOR
- opponent: CZE
- q_market: 0.37
- fee_estimate: 0.01
- q_eff: 0.38
- liquidity_cap: 100
- match_exposure: 0
- team_exposure: 0
- total_open_exposure: 0

## Recommendation

- trade_excluded: no
- decision: no_trade
- evaluated_side: South Korea YES
- p_model: 0.352933
- edge: -0.027067
- threshold: 0.03
- stake_usd: 0.0
- contracts: 0
- reason: edge does not clear pre-match threshold
- draw_yes_max_q_market_to_clear_threshold: 0.3444
- czech_yes_max_q_market_to_clear_threshold: 0.2226
- note: No draw or Czech YES bet can be recommended until their market prices are supplied.

## Actual Trading

- Tie limit buy: filled
- tie_contracts_bought: 83
- tie_buy_price: 0.32
- tie_cash_outflow_usd: 28.22
- Tie limit sell: filled
- tie_contracts_sold: 83
- tie_sell_price: 0.08
- tie_cash_inflow_usd: 5.19
- tie_realized_pnl_usd: -23.03
- KOR limit buy: filled
- kor_contracts_bought: 162
- kor_buy_price: 0.89
- kor_cash_outflow_usd: 147.39
- midgame_note: Bought Korea Republic at halftime / midgame.
- screenshot_mark_to_market_note: Screen showed KOR position +16.20 near settlement at 99%.

## Final Result And Profit

- final_score: Korea Republic won
- bet_result: KOR won; Tie lost after being sold
- payout_usd: 162.00
- realized_pnl_usd: -8.42
- notes: KOR PnL = 162.00 payout - 147.39 cash outflow = 14.61. Tie PnL = 5.19 sell inflow - 28.22 buy outflow = -23.03. Total match PnL = -8.42.
