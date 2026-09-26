# Cookie Cats A/B Test — Summary Report

**Question:** does moving the game's first gate from level 30 to level 40 change player engagement and retention?

**Sample size (after cleaning):** 90,188 players (44,699 gate_30 / 45,489 gate_40)

## Key Findings

- **Day-1 retention:** 44.8% (gate_30) vs. 44.2% (gate_40) — no statistically significant difference (p=0.074).
- **Day-7 retention:** 19.0% (gate_30) vs. 18.2% (gate_40) — a real, statistically significant difference (p=0.0016). Moving the gate to level 40 **reduces** week-long retention.
- **Rounds played:** median 17 (gate_30) vs. 16 (gate_40) — no statistically significant difference (p=0.051). The gate position doesn't change how much players play overall, only whether they come back a week later.

## Recommendation

- **Do not move the gate to level 40.** The only metric that moved with statistical confidence — 7-day retention — moved in the wrong direction. There's no offsetting gain in short-term engagement to justify the change.

## Where the Effect Comes From (SQL segmentation)

Slicing retention by engagement bucket (see `sql/queries.sql`) shows the drop isn't spread evenly across all players:

| Engagement bucket | gate_30 retention_7 | gate_40 retention_7 |
|---|---|---|
| 0 rounds (never started) | 0.8% | 0.6% |
| 1-10 rounds (light) | 1.9% | 1.9% |
| 11-50 rounds (moderate) | 12.0% | 10.8% |
| 51-200 rounds (heavy) | 47.6% | 44.7% |
| 200+ rounds (whale) | 84.9% | 85.0% |

- **Activation is unaffected** — a near-identical share of players in both groups never play a single round, so the gate isn't scaring people off before they start.
- **Power users (top 10% by rounds played) retain equally well either way** — once someone is hooked, gate position stops mattering.
- **The drop concentrates in the moderate and heavy engagement segments** (11-200 rounds) — exactly the players who are engaged enough to hit the gate but not yet die-hard fans — consistent with the level-40 gate blocking that group before they build a long-term habit.

## Predictive Model: What Predicts a Player Returning?

- A Random Forest model predicts 7-day retention with ROC-AUC = 0.89 (vs. a baseline retention rate of 18.6%).
- The strongest predictor is **sum_gamerounds** (importance=0.86) — how much a player plays in the first 14 days matters far more for retention than which gate version they saw.
- **Actionable takeaway:** rather than tuning gate placement, product efforts to boost early game-rounds-played (e.g. onboarding, early rewards) are likely to have more impact on long-term retention.