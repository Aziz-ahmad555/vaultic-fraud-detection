# Phase 4: B5 + graph features (setting C) over B5 (frozen B5 hyperparameters)

Generated 2026-10-09 by `python -m vaultic.reports.gain` from development runs (**validation period only**). Gain = new − base PR-AUC, mean over seeds; paired bootstrap (1000 resamples, same rows for both runs). Groups by the uid's number of earlier transactions (point-in-time).

- base run: `20261008-211052-888539` (EXP-009)
- new run: `20261009-123145-610994` (EXP-105-frozen)

| group | rows | frauds | base PR-AUC | new PR-AUC | gain | 95% CI | p |
|---|---|---|---|---|---|---|---|
| all | 60602 | 2114 | 0.6831 | 0.6642 | -0.0189 | [-0.0235, -0.0141] | 0.000 |
| >= 5 past | 20766 | 743 | 0.8568 | 0.8506 | -0.0063 | [-0.0112, -0.0012] | 0.020 |
| 1-4 past | 21071 | 783 | 0.5996 | 0.5773 | -0.0223 | [-0.0309, -0.0138] | 0.000 |
| cold start | 18765 | 588 | 0.5464 | 0.5277 | -0.0186 | [-0.0289, -0.0087] | 0.000 |
