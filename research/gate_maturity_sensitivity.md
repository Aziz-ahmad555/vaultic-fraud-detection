# E10 with label-mature gate folds (POST-HOC / EXPLORATORY)

**POST-HOC / EXPLORATORY (D141; disclosure D140).** Not pre-registered; every E10 number in research/tables/final_e10.md stands as reported. Generated 2026-10-11 by `python -m vaultic.fusion.gate_maturity_sensitivity`.

The three cross-fitted gate folds are rebuilt with 30-day label maturity (each fold's view models train only on labels released by the start of its predicted block); the first fold (1-30 -> 31-60) has no released labels under maturity and is left out (D147); the fixed fold is unchanged, so its gate rows on validation days 128-143 keep their immature view scores (the sensitivity removes the freshness advantage for the cross-fitted folds only) and the test-period view scores equal the final run's (max |diff| 0.0e+00). Gate rows (fit + tune): 294,118 (11,616 frauds) pre-registered, 212,734 (8,417 frauds) with mature folds. The gate methods are refitted with their D95 settings (not re-tuned), seeds 0-4. Test period of final run `20261011-002400-202994`: 92,427 rows, 3,213 frauds; 1,000 paired bootstrap resamples; p unadjusted (exploratory).

| method | test PR-AUC, pre-registered | test PR-AUC, mature gate folds [95% CI] | difference [95% CI] | p |
|---|---|---|---|---|
| MVAF | 0.6219 | 0.6301 [0.6129, 0.6468] | +0.0082 [+0.0049, +0.0116] | < 0.001 |
| F1 | 0.5858 | 0.5858 [0.5660, 0.6051] | +0.0000 [+0.0000, +0.0000] | 1.000 |
| F2 | 0.6122 | 0.6127 [0.5948, 0.6301] | +0.0005 [-0.0006, +0.0015] | 0.328 |
| F3 | 0.4229 | 0.4341 [0.4156, 0.4526] | +0.0112 [+0.0101, +0.0123] | < 0.001 |
| F4 | 0.6206 | 0.6109 [0.5921, 0.6299] | -0.0097 [-0.0152, -0.0045] | < 0.001 |
| F5 | 0.6247 | 0.6297 [0.6120, 0.6470] | +0.0051 [+0.0019, +0.0081] | < 0.001 |
| F6 | 0.6264 | 0.6303 [0.6126, 0.6469] | +0.0040 [+0.0015, +0.0065] | 0.002 |
| F7 | 0.6248 | 0.6298 [0.6114, 0.6468] | +0.0050 [+0.0028, +0.0070] | < 0.001 |
| B5 | 0.6532 | does not use the gate; unchanged | | |
| F0 | 0.6527 | does not use the gate; unchanged | | |

## Reading (added by hand, not generated)

POST-HOC / EXPLORATORY (D141, D147); every pre-registered E10 number stands as reported.

- **The gate methods gain a little, tree stacking loses a little.** With label-mature gate folds,
  MVAF rises by +0.008 (0.6219 -> 0.6301) and the other gates by +0.004 to +0.005 (F5 0.6297, F6
  0.6303, F7 0.6298). F4 (LightGBM stacking) falls by -0.010 (0.6206 -> 0.6109); F2 does not move
  and F1 cannot (no training). F3 gains +0.011 but stays collapsed (0.434; the cause is the
  unbounded history count, D118, not label maturity).
- **Direction of the D140 bias.** Training the gate on better-informed view scores than deployment
  provides did not favour MVAF: removing that advantage for the cross-fitted folds helps it
  slightly. The disclosure stands, but it does not explain MVAF trailing the strongest models.
- **Only the cross-fitted folds are made label-mature.** The fixed fold's gate rows on validation
  days 128-143 keep their immature view scores (they are 14% of the gate rows and 83% of the gate's
  inner tuning rows, D148), so part of the freshness advantage remains; and the first cross-fitted
  fold (1-30 -> 31-60) has no released labels under maturity and is dropped (D147: 212,734 gate
  rows instead of 294,118).
- **The overall picture does not change.** Even with mature folds MVAF (0.630) stays below B5
  (0.653) and F0 (0.653), and is level with F5-F7 (0.630). Against F4 the two scores now differ by
  about +0.019 in MVAF's favour, but this sensitivity compares each method only with its own
  pre-registered score; no MVAF - F4 test was run, and the pre-registered E10 comparison (+0.0013,
  p 0.68) stands.
- **Caveats.** Two things change at once: label maturity of the cross-fitted folds and fewer gate
  rows (fold 0 dropped, above). The gate settings were not re-tuned, there is a single post-hoc run
  on the test period, and p-values are unadjusted.

