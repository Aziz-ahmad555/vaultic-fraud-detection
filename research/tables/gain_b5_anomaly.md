# Phase 6: B5 + anomaly scores over B5 (frozen B5 hyperparameters)

Generated 2026-10-09 by `python -m vaultic.reports.gain` from development runs (**validation period only**). Gain = new − base PR-AUC, mean over seeds; paired bootstrap (1000 resamples, same rows for both runs). Groups by the uid's number of earlier transactions (point-in-time).

- base run: `20261008-211052-888539` (EXP-009)
- new run: `20261009-133329-203930` (EXP-108-frozen)

| group | rows | frauds | base PR-AUC | new PR-AUC | gain | 95% CI | p |
|---|---|---|---|---|---|---|---|
| all | 60602 | 2114 | 0.6831 | 0.6771 | -0.0060 | [-0.0087, -0.0034] | 0.000 |
| >= 5 past | 20766 | 743 | 0.8568 | 0.8524 | -0.0044 | [-0.0073, -0.0018] | 0.000 |
| 1-4 past | 21071 | 783 | 0.5996 | 0.5888 | -0.0108 | [-0.0161, -0.0057] | 0.000 |
| cold start | 18765 | 588 | 0.5464 | 0.5471 | +0.0007 | [-0.0058, +0.0070] | 0.830 |
