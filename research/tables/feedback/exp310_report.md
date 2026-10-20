# EXP-310-final: feedback loop (champion-challenger) final run

Anonymized e-commerce transactions released by Vesta for the IEEE-CIS competition. Champion: the frozen B5 configuration (five seeds); challengers: the same configuration retrained on matured labels. Deployment from day 121, warm-up 121-150, evaluation on test days 151-182 (92,427 rows, 3,213 frauds), label delays L = 0 / 7 / 30 / 60, 5 seeds, 1,000-sample bootstrap. Pre-registered in D210 with amendments D211 and D214, plus D215 and D216 (tooling only); results recorded as-is (D218).

Run `experiments/runs/EXP-310-final/20261020-174032-191246`. Parity before the simulation: the five champion models reproduced the stored EXP-200-final score_B5 on all 92,427 test rows (max absolute difference 0.0, tolerance 1e-06) and matched their frozen booster hashes.

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

H-FB2 is supported in the pre-registered sense: no policy's cost-difference CI lies above 0, so there is **no cost increase** versus P0. It does **not** show a cost reduction (P1's CI reaches +5; P2 identical to P0; P3's CI reaches +162). H-FB3 is partly true by construction: the L=60 gain is 0 because no gate decision was possible there (see the structural limits), and the L=30 gain's CI includes 0, so only the L=0 > L=7 > L=30 ordering carries information. H-FB4 is supported on the point estimate only; its CI includes 0, and ungated P1 has a slightly higher PR-AUC at L=30 (0.6506 vs 0.6499).

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

## Structural limits of the 32-day test period (pre-registered, section 7)

At a weekly slot on day d, a label is known only if its transaction time + L is before d, so the gate window (the 7 most recent days of known labels) covers roughly days d - L - 7 to d - L. Rows the champion trained on (days 1-120) are removed from it, so a gate decision needs at least d - L > 120, and enough window days after day 120 to hold >= 50 frauds. With weekly slots on days 128, 135, 142, 149, 156, 163, 170, 177 (last slot 177):

- L=0: P1 gate decisions on 8 of 8 slots per seed (128, 135, 142, 149, 156, 163, 170, 177); 0 of 40 P1 attempts 'insufficient evidence'.
- L=7: P1 gate decisions on 7 of 8 slots per seed (135, 142, 149, 156, 163, 170, 177); 5 of 40 P1 attempts 'insufficient evidence'.
- L=30: P1 gate decisions on 4 of 8 slots per seed (156, 163, 170, 177); 20 of 40 P1 attempts 'insufficient evidence'.
- L=60: P1 gate decisions on 0 of 8 slots per seed (none); 40 of 40 P1 attempts 'insufficient evidence' (the last slot 177 minus 60 is 117, inside the champion's training days: no gate decision is possible in the span).

These are limits of the span, not of the policies: a longer test period would allow more gate decisions at L=30 and any at L=60.

## Notes

- **P2 = P0 at every delay.** The PSI monitor never exceeded 0.2 (no P2 alarm), so P2 never attempted a challenger and served the champion throughout, as in the design-run diagnostic (D212 point 6).
- **P3 = P1 at L=0** (PR-AUC 0.8591, cost 55,894 for both): an ADWIN alarm was pending at every weekly slot (a pending alarm is retried at each slot until a challenger is accepted, at most 6 times), so P3 attempted on exactly P1's days in 5 of 5 seeds and reached the same decisions. At L = 7, 30 the two differ.
- **L=60 is a structural null** (D212 point 3): every attempt of every policy was 'insufficient evidence', so every policy served the champion. P3's alarms at L=60 came on day(s) 182, after the last weekly slot (177), so P3 never attempted.

## Limitations

- **Champion head start (pre-registered).** The champion was trained on days 1-120 with every label, including labels a real deployment at day 121 could not yet have had under delay L (days 91-120 at L = 30), and its E12 thresholds and P2 reference were chosen once on calibrate_fused (days 147-150; D211 c). This favours P0.
- **Second use of the test period (D211 a).** Days 151-182 were already used by the stage-1 (EXP-200-final) and stage-2 final runs; EXP-310-final is a second use of those days under a new pre-registration. No B5 setting was chosen using test results.
- **In-sample champion cost in early gates (D214).** The champion's E12 thresholds were chosen on days 147-150; a gate window containing those days compares the challenger with a champion whose cost there is in-sample. Gated attempts with such an overlap (from the events): P1_monthly at L=0: 5 of 10 attempts (5 accepted); P1 at L=7: 1 of 40 attempts (1 accepted); P3 at L=7: 1 of 25 attempts (1 accepted); P1_monthly at L=30: 5 of 10 attempts (4 accepted). P1 and P3 have no overlap at L=30, so H-FB1 and H-FB2 are unaffected.
- **Structural limits of the span**, as above (L=60 null, at most 4 gate decisions per seed at L=30).

## Record trail

- The first final start was refused before any test-period row was read by the loader's token check (a `python -m` double-import bug, D217): no parity, no simulation, no results; it is marked aborted on record. After the fix and a logged re-freeze, the run above was started with a logged re-run reason naming D217.
- `93f9730` freeze record and FREEZE row (frozen at `845f51b`)
- `967e8a8` FINAL-STARTED for run `20261020-171045-498585`
- `9de52d0` FINAL-ABORTED for run `20261020-171045-498585`
- `7a08ecc` freeze record and FREEZE row (frozen at `c0b7bea`)
- `1adc724` FINAL-STARTED for run `20261020-174032-191246`
- FINAL-RERUN row in research/decisions.md (--final re-run after `20261020-171045-498585`), reason: D217: the first start 20261020-171045-498585 was refused by the loader token check (python -m double-import bug) before any test data was read; no parity, no results

## Design run (EXP-310-dev) vs final

The design run used training days only (champion fitted on days 1-56, evaluated on 61-120), the old label curtain (its_time + L <= t) and a 28-day monthly cadence; its numbers are not hypothesis evidence (D212). D212 point 5: the stored frequency encoders were fitted on days 1-120, so the design champion (days 1-56) saw mildly forward-looking frequency counts; the final run deploys at day 121 and is not affected.

| L | Design P0 | Design gain (P1 - P0) | Design CI | Final P0 | Final gain | Final CI | Design P1 promotions | Final P1 promotions |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.7920 | +0.0638 | +0.0594 to +0.0679 | 0.8100 | +0.0491 | +0.0440 to +0.0544 | 21 | 11 |
| 7 | 0.6842 | +0.0213 | +0.0185 to +0.0238 | 0.7031 | +0.0220 | +0.0184 to +0.0258 | 17 | 17 |
| 30 | 0.6333 | +0.0016 | +0.0004 to +0.0027 | 0.6494 | +0.0005 | -0.0003 to +0.0012 | 12 | 4 |
| 60 | 0.6057 | +0.0000 | +0.0000 to +0.0000 | 0.6208 | +0.0000 | +0.0000 to +0.0000 | 0 | 0 |
