# EXP-310-final: feedback loop (champion-challenger) final run

Anonymized e-commerce transactions released by Vesta for the IEEE-CIS competition. Champion: the frozen B5 configuration (five seeds); challengers: the same configuration retrained on matured labels. Deployment from day 121, warm-up 121-150, evaluation on test days 151-182 (92,427 rows, 3,213 frauds), label delays L = 0 / 7 / 30 / 60, 5 seeds, 1,000-sample bootstrap. Pre-registered in D210 with amendments D211, D214, D215 (tooling), D216 (tooling); results recorded as-is (D218).

Run `experiments/runs/EXP-310-final/20261020-174032-191246`; parity before the simulation: the five champion models reproduced the stored EXP-200-final score_B5 on every test row (max absolute difference 0.0, tolerance 1e-6) and matched their frozen booster hashes.

## Hypotheses

| Hypothesis | Policy | Estimate | 95% CI | p | Holm p | Read-out |
|---|---|---|---|---|---|---|
| H-FB1: PR-AUC gain over P0 at L=30 (Holm over P1/P2/P3) | P1 | +0.00049 | -0.00027 to +0.00123 | 0.230 | 0.690 | not supported |
| H-FB1: PR-AUC gain over P0 at L=30 (Holm over P1/P2/P3) | P2 | +0.00000 | +0.00000 to +0.00000 | 1.000 | 1.000 | not supported |
| H-FB1: PR-AUC gain over P0 at L=30 (Holm over P1/P2/P3) | P3 | +0.00016 | -0.00056 to +0.00087 | 0.656 | 1.000 | not supported |
| H-FB1 overall | | | | | | **not supported** |
| H-FB2: cost change vs P0 at L=30 (E12 cost) | P1 | -598 | -1,120 to +5 | 0.052 | | no cost increase |
| H-FB2: cost change vs P0 at L=30 (E12 cost) | P2 | +0 | +0 to +0 | 1.000 | | no cost increase |
| H-FB2: cost change vs P0 at L=30 (E12 cost) | P3 | -391 | -909 to +162 | 0.174 | | no cost increase |
| H-FB2 overall (no cost increase; not a cost reduction) | | | | | | **supported** |
| H-FB3: gain does not grow with delay (descriptive, no test) | P1 | L=0: +0.0491 / L=7: +0.0220 / L=30: +0.0005 / L=60: +0.0000 | | | | **supported** |
| H-FB4: gated minus ungated P1 cost at L=30 (decided on the point estimate) | P1 | -415 | -1,282 to +531 | 0.410 | | **supported** |

H-FB2 is supported in the pre-registered sense: no policy's cost-difference CI lies above 0, so there is **no cost increase** versus P0. It does **not** show a cost reduction: P1's CI reaches +5 and P3's crosses 0. H-FB4 is supported on the point estimate only; its CI includes 0, and ungated P1 has a slightly higher PR-AUC at L=30 (0.6506 vs 0.6499).

## Gain by delay (P1 minus P0 PR-AUC, test days 151-182)

| L | P0 PR-AUC | P1 PR-AUC | Gain | 95% CI | p |
|---|---|---|---|---|---|
| 0 | 0.8100 | 0.8591 | +0.0491 | +0.0440 to +0.0544 | 0.000 |
| 7 | 0.7031 | 0.7251 | +0.0220 | +0.0184 to +0.0258 | 0.000 |
| 30 | 0.6494 | 0.6499 | +0.0005 | -0.0003 to +0.0012 | 0.230 |
| 60 | 0.6208 | 0.6208 | +0.0000 | +0.0000 to +0.0000 | 1.000 |

P0's PR-AUC is not comparable across delays (D212 point 4): the champion's three label-derived inputs follow each arm's delay. Policies are compared within a delay only.

## All policies by delay (PR-AUC / E12 cost, seed mean)

| L | P0 | P1 | P2 | P3 | P1_ungated | P1_monthly |
|---|---|---|---|---|---|---|
| 0 | 0.8100 / 67,208 | 0.8591 / 55,894 | 0.8100 / 67,208 | 0.8591 / 55,894 | 0.8678 / 58,737 | 0.8626 / 56,500 |
| 7 | 0.7031 / 76,834 | 0.7251 / 75,712 | 0.7031 / 76,834 | 0.7236 / 76,080 | 0.7273 / 75,584 | 0.7101 / 75,435 |
| 30 | 0.6494 / 86,185 | 0.6499 / 85,586 | 0.6494 / 86,185 | 0.6496 / 85,794 | 0.6506 / 86,001 | 0.6489 / 86,028 |
| 60 | 0.6208 / 92,969 | 0.6208 / 92,969 | 0.6208 / 92,969 | 0.6208 / 92,969 | 0.6208 / 92,969 | 0.6208 / 92,969 |

## Promotions and rejections (sum over 5 seeds)

| L | Policy | Attempts | Accepted | Gain below margin / CI | Cost higher | Insufficient evidence | Alarms |
|---|---|---|---|---|---|---|---|
| 0 | P1 | 40 | 11 | 4 | 25 | 0 | 0 |
| 0 | P3 | 40 | 11 | 4 | 25 | 0 | 19 |
| 0 | P1_monthly | 10 | 8 | 0 | 2 | 0 | 0 |
| 0 | P1_ungated | 40 | 40 | 0 | 0 | 0 | 0 |
| 0 | P2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | P1 | 40 | 17 | 14 | 4 | 5 | 0 |
| 7 | P3 | 25 | 11 | 8 | 6 | 0 | 15 |
| 7 | P1_monthly | 10 | 5 | 0 | 5 | 0 | 0 |
| 7 | P1_ungated | 40 | 35 | 0 | 0 | 5 | 0 |
| 7 | P2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 30 | P1 | 40 | 4 | 10 | 6 | 20 | 0 |
| 30 | P3 | 18 | 3 | 10 | 5 | 0 | 5 |
| 30 | P1_monthly | 10 | 4 | 0 | 1 | 5 | 0 |
| 30 | P1_ungated | 40 | 20 | 0 | 0 | 20 | 0 |
| 30 | P2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 60 | P1 | 40 | 0 | 0 | 0 | 40 | 0 |
| 60 | P3 | 0 | 0 | 0 | 0 | 0 | 5 |
| 60 | P1_monthly | 10 | 0 | 0 | 0 | 10 | 0 |
| 60 | P1_ungated | 40 | 0 | 0 | 0 | 40 | 0 |
| 60 | P2 | 0 | 0 | 0 | 0 | 0 | 0 |

P1_ungated promotes every challenger it can fit (its 'accepted' column counts those).

## Notes

- **P2 = P0 at every delay.** The PSI monitor never exceeded 0.2, so P2 never attempted a challenger and served the champion throughout, exactly as the design-run diagnostic predicted (D212 point 6).
- **P3 = P1 at L=0** (PR-AUC 0.8591, cost 55,894 for both): an ADWIN alarm was pending at every weekly slot (alarms fired soon after each promotion, and a pending alarm is retried at each slot until a challenger is accepted, at most 6 times), so P3 attempted on exactly P1's days in 5 of 5 seeds and reached the same decisions. At L=7 and L=30 the two differ.
- **L=60 is a structural null** (D212 point 3): every attempt was 'insufficient evidence' (fewer than 50 matured frauds in the gate window), so every policy served the champion. P3's alarms at L=60 came on day(s) 182, after the last weekly slot (177), so P3 never attempted.
- **In-sample champion cost in early gates (D214).** The champion's E12 thresholds were chosen on days 147-150. Attempts whose gate window includes those days compare the challenger with a champion whose cost there is in-sample: L=0: 0 of 40 P1 attempts (0 accepted); L=7: 1 of 40 P1 attempts (1 accepted); L=30: 0 of 40 P1 attempts (0 accepted); L=60: 0 of 40 P1 attempts (0 accepted). Reported as a limitation.
- **Refused first start (D217).** Run `20261020-171045-498585` pushed its start row and was then refused by the loader's token check (a `python -m` double-import bug) before any test-period row was read: no parity, no simulation, no results. It is marked aborted on record (FINAL-ABORTED). After the fix and a logged re-freeze, the run above was started with a logged re-run reason naming D217.

## Design run (EXP-310-dev) vs final

The design run used training days only (champion fitted on days 1-56, evaluated on 61-120), the old label curtain (its_time + L <= t) and a 28-day monthly cadence; its numbers are not hypothesis evidence (D212).

| L | Design P0 | Design gain (P1 - P0) | Design CI | Final P0 | Final gain | Final CI | Design P1 promotions | Final P1 promotions |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.7920 | +0.0638 | +0.0594 to +0.0679 | 0.8100 | +0.0491 | +0.0440 to +0.0544 | 21 | 11 |
| 7 | 0.6842 | +0.0213 | +0.0185 to +0.0238 | 0.7031 | +0.0220 | +0.0184 to +0.0258 | 17 | 17 |
| 30 | 0.6333 | +0.0016 | +0.0004 to +0.0027 | 0.6494 | +0.0005 | -0.0003 to +0.0012 | 12 | 4 |
| 60 | 0.6057 | +0.0000 | +0.0000 to +0.0000 | 0.6208 | +0.0000 | +0.0000 to +0.0000 | 0 | 0 |

The pattern carries over: a clear gain at L=0 and L=7, a small one at L=30 (smaller in the final run, CI including 0) and none at L=60.
