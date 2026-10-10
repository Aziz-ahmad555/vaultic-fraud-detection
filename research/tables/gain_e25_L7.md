# E25: B5 trained only on labels mature 7 days before validation, over B5

Generated 2026-10-10 by `python -m vaultic.reports.gain` from development runs (**validation period only**). Gain = new − base PR-AUC, mean over seeds; paired bootstrap (1000 resamples, same rows for both runs). Groups by the uid's number of earlier transactions (point-in-time).

- base run: `20261008-211052-888539` (EXP-009)
- new run: `20261010-194606-655896` (EXP-125-L7)

| group | rows | frauds | base PR-AUC | new PR-AUC | gain | 95% CI | p |
|---|---|---|---|---|---|---|---|
| all | 60602 | 2114 | 0.6831 | 0.6831 | +0.0000 | [+0.0000, +0.0000] | 1.000 |
| >= 5 past | 20766 | 743 | 0.8568 | 0.8568 | +0.0000 | [+0.0000, +0.0000] | 1.000 |
| 1-4 past | 21071 | 783 | 0.5996 | 0.5996 | +0.0000 | [+0.0000, +0.0000] | 1.000 |
| cold start | 18765 | 588 | 0.5464 | 0.5464 | +0.0000 | [+0.0000, +0.0000] | 1.000 |
| has_identity yes | 10835 | 974 | 0.7706 | 0.7706 | +0.0000 | [+0.0000, +0.0000] | 1.000 |
| has_identity no | 49767 | 1140 | 0.6025 | 0.6025 | +0.0000 | [+0.0000, +0.0000] | 1.000 |
| graph view available | 29688 | 1119 | 0.6997 | 0.6997 | +0.0000 | [+0.0000, +0.0000] | 1.000 |
| graph view missing | 30914 | 995 | 0.6653 | 0.6653 | +0.0000 | [+0.0000, +0.0000] | 1.000 |
