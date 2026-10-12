# Emerging fraud: B5 + anomaly vs B5, both trained without ProductCD R frauds, on ProductCD R validation rows

Generated 2026-10-09 by `python -m vaultic.reports.gain` from development runs (**validation period only**). Gain = new − base PR-AUC, mean over seeds; paired bootstrap (1000 resamples, same rows for both runs). Groups by the uid's number of earlier transactions (point-in-time).

- base run: `20261009-112357-306004` (EXP-009-holdout-R)
- new run: `20261009-142427-959567` (EXP-108-holdout-R)

| group | rows | frauds | base PR-AUC | new PR-AUC | gain | 95% CI | p |
|---|---|---|---|---|---|---|---|
| all | 2424 | 93 | 0.5648 | 0.5539 | -0.0109 | [-0.0349, +0.0109] | 0.338 |
| >= 5 past | 99 | 2 | 0.1637 | 0.0867 | -0.0771 | [-0.2640, -0.0188] | 0.010 |
| 1-4 past | 691 | 31 | 0.5806 | 0.5847 | +0.0041 | [-0.0358, +0.0442] | 0.782 |
| cold start | 1634 | 60 | 0.5799 | 0.5609 | -0.0189 | [-0.0466, +0.0058] | 0.146 |
