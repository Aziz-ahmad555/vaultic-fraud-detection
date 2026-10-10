# E25: B5 trained only on labels mature 60 days before validation, over B5

Generated 2026-10-10 by `python -m vaultic.reports.gain` from development runs (**validation period only**). Gain = new − base PR-AUC, mean over seeds; paired bootstrap (1000 resamples, same rows for both runs). Groups by the uid's number of earlier transactions (point-in-time).

- base run: `20261008-211052-888539` (EXP-009)
- new run: `20261010-195707-522990` (EXP-125-L60)

| group | rows | frauds | base PR-AUC | new PR-AUC | gain | 95% CI | p |
|---|---|---|---|---|---|---|---|
| all | 60602 | 2114 | 0.6831 | 0.5873 | -0.0959 | [-0.1064, -0.0860] | 0.000 |
| >= 5 past | 20766 | 743 | 0.8568 | 0.7260 | -0.1308 | [-0.1525, -0.1105] | 0.000 |
| 1-4 past | 21071 | 783 | 0.5996 | 0.5213 | -0.0784 | [-0.0942, -0.0637] | 0.000 |
| cold start | 18765 | 588 | 0.5464 | 0.5050 | -0.0414 | [-0.0557, -0.0276] | 0.000 |
| has_identity yes | 10835 | 974 | 0.7706 | 0.7117 | -0.0589 | [-0.0717, -0.0473] | 0.000 |
| has_identity no | 49767 | 1140 | 0.6025 | 0.4835 | -0.1190 | [-0.1354, -0.1032] | 0.000 |
| graph view available | 29688 | 1119 | 0.6997 | 0.5935 | -0.1062 | [-0.1219, -0.0916] | 0.000 |
| graph view missing | 30914 | 995 | 0.6653 | 0.5799 | -0.0854 | [-0.0992, -0.0718] | 0.000 |
