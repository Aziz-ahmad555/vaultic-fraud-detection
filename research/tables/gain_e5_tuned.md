# E5 (B5 + graph features, tuned with B5's budget) over B5

Generated 2026-10-10 by `python -m vaultic.reports.gain` from development runs (**validation period only**). Gain = new − base PR-AUC, mean over seeds; paired bootstrap (1000 resamples, same rows for both runs). Groups by the uid's number of earlier transactions (point-in-time).

- base run: `20261008-211052-888539` (EXP-009)
- new run: `20261010-164859-235447` (EXP-105-tuned)

| group | rows | frauds | base PR-AUC | new PR-AUC | gain | 95% CI | p |
|---|---|---|---|---|---|---|---|
| all | 60602 | 2114 | 0.6831 | 0.6605 | -0.0226 | [-0.0277, -0.0177] | 0.000 |
| >= 5 past | 20766 | 743 | 0.8568 | 0.8473 | -0.0096 | [-0.0153, -0.0043] | 0.002 |
| 1-4 past | 21071 | 783 | 0.5996 | 0.5760 | -0.0236 | [-0.0326, -0.0145] | 0.000 |
| cold start | 18765 | 588 | 0.5464 | 0.5257 | -0.0207 | [-0.0311, -0.0105] | 0.000 |
| has_identity yes | 10835 | 974 | 0.7706 | 0.7496 | -0.0210 | [-0.0284, -0.0137] | 0.000 |
| has_identity no | 49767 | 1140 | 0.6025 | 0.5887 | -0.0139 | [-0.0207, -0.0074] | 0.000 |
| graph view available | 29688 | 1119 | 0.6997 | 0.6772 | -0.0225 | [-0.0296, -0.0157] | 0.000 |
| graph view missing | 30914 | 995 | 0.6653 | 0.6434 | -0.0220 | [-0.0284, -0.0149] | 0.000 |
