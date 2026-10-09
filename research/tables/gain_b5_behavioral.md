# Phase 3: B5 + behavioral features over B5 (frozen B5 hyperparameters)

Generated 2026-10-09 by `python -m vaultic.reports.gain` from development runs (**validation period only**). Gain = new − base PR-AUC, mean over seeds; paired bootstrap (1000 resamples, same rows for both runs). Groups by the uid's number of earlier transactions (point-in-time).

- base run: `20261008-211052-888539` (EXP-009)
- new run: `20261009-105027-836010` (EXP-103-frozen)

| group | rows | frauds | base PR-AUC | new PR-AUC | gain | 95% CI | p |
|---|---|---|---|---|---|---|---|
| all | 60602 | 2114 | 0.6831 | 0.6769 | -0.0063 | [-0.0089, -0.0037] | 0.000 |
| >= 5 past | 20766 | 743 | 0.8568 | 0.8564 | -0.0005 | [-0.0033, +0.0026] | 0.738 |
| 1-4 past | 21071 | 783 | 0.5996 | 0.5912 | -0.0084 | [-0.0133, -0.0036] | 0.000 |
| cold start | 18765 | 588 | 0.5464 | 0.5378 | -0.0085 | [-0.0146, -0.0026] | 0.004 |
