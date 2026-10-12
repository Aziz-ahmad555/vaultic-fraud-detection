# Emerging fraud: B5 trained without ProductCD R frauds vs full B5, on ProductCD R validation rows

Generated 2026-10-09 by `python -m vaultic.reports.gain` from development runs (**validation period only**). Gain = new − base PR-AUC, mean over seeds; paired bootstrap (1000 resamples, same rows for both runs). Groups by the uid's number of earlier transactions (point-in-time).

- base run: `20261008-211052-888539` (EXP-009)
- new run: `20261009-112357-306004` (EXP-009-holdout-R)

| group | rows | frauds | base PR-AUC | new PR-AUC | gain | 95% CI | p |
|---|---|---|---|---|---|---|---|
| all | 2424 | 93 | 0.7953 | 0.5648 | -0.2305 | [-0.3065, -0.1629] | 0.000 |
| >= 5 past | 99 | 2 | 0.5529 | 0.1637 | -0.3891 | [-0.8833, +0.1328] | 0.520 |
| 1-4 past | 691 | 31 | 0.8362 | 0.5806 | -0.2556 | [-0.3827, -0.1326] | 0.000 |
| cold start | 1634 | 60 | 0.7914 | 0.5799 | -0.2115 | [-0.2919, -0.1408] | 0.000 |
