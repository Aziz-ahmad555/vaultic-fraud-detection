# E25: B5 trained only on labels mature 30 days before validation, over B5

Generated 2026-10-10 by `python -m vaultic.reports.gain` from development runs (**validation period only**). Gain = new − base PR-AUC, mean over seeds; paired bootstrap (1000 resamples, same rows for both runs). Groups by the uid's number of earlier transactions (point-in-time).

- base run: `20261008-211052-888539` (EXP-009)
- new run: `20261010-193050-481642` (EXP-125-L30)

| group | rows | frauds | base PR-AUC | new PR-AUC | gain | 95% CI | p |
|---|---|---|---|---|---|---|---|
| all | 60602 | 2114 | 0.6831 | 0.6295 | -0.0537 | [-0.0615, -0.0461] | 0.000 |
| >= 5 past | 20766 | 743 | 0.8568 | 0.7916 | -0.0652 | [-0.0779, -0.0534] | 0.000 |
| 1-4 past | 21071 | 783 | 0.5996 | 0.5502 | -0.0494 | [-0.0612, -0.0377] | 0.000 |
| cold start | 18765 | 588 | 0.5464 | 0.5155 | -0.0308 | [-0.0436, -0.0200] | 0.000 |
| has_identity yes | 10835 | 974 | 0.7706 | 0.7308 | -0.0398 | [-0.0512, -0.0301] | 0.000 |
| has_identity no | 49767 | 1140 | 0.6025 | 0.5366 | -0.0660 | [-0.0777, -0.0551] | 0.000 |
| graph view available | 29688 | 1119 | 0.6997 | 0.6377 | -0.0620 | [-0.0755, -0.0511] | 0.000 |
| graph view missing | 30914 | 995 | 0.6653 | 0.6206 | -0.0447 | [-0.0535, -0.0359] | 0.000 |
