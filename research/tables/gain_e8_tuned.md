# E8 (B5 + anomaly features, tuned with B5's budget) over B5

Generated 2026-10-10 by `python -m vaultic.reports.gain` from development runs (**validation period only**). Gain = new − base PR-AUC, mean over seeds; paired bootstrap (1000 resamples, same rows for both runs). Groups by the uid's number of earlier transactions (point-in-time).

- base run: `20261008-211052-888539` (EXP-009)
- new run: `20261010-191717-729992` (EXP-108-tuned)

| group | rows | frauds | base PR-AUC | new PR-AUC | gain | 95% CI | p |
|---|---|---|---|---|---|---|---|
| all | 60602 | 2114 | 0.6831 | 0.6782 | -0.0049 | [-0.0079, -0.0019] | 0.004 |
| >= 5 past | 20766 | 743 | 0.8568 | 0.8500 | -0.0069 | [-0.0100, -0.0040] | 0.000 |
| 1-4 past | 21071 | 783 | 0.5996 | 0.5922 | -0.0075 | [-0.0135, -0.0015] | 0.008 |
| cold start | 18765 | 588 | 0.5464 | 0.5496 | +0.0032 | [-0.0028, +0.0091] | 0.272 |
| has_identity yes | 10835 | 974 | 0.7706 | 0.7679 | -0.0027 | [-0.0071, +0.0018] | 0.232 |
| has_identity no | 49767 | 1140 | 0.6025 | 0.5983 | -0.0042 | [-0.0080, -0.0005] | 0.034 |
| graph view available | 29688 | 1119 | 0.6997 | 0.6954 | -0.0043 | [-0.0085, +0.0001] | 0.052 |
| graph view missing | 30914 | 995 | 0.6653 | 0.6588 | -0.0066 | [-0.0108, -0.0024] | 0.000 |
