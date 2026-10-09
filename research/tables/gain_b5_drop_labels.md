# Ablation: B5 without the label-derived group (uid_fraud_known, uid_fraud_rate_known)

Generated 2026-10-09 by `python -m vaultic.reports.gain` from development runs (**validation period only**). Gain = new − base PR-AUC, mean over seeds; paired bootstrap (1000 resamples, same rows for both runs). Groups by the uid's number of earlier transactions (point-in-time).

- base run: `20261008-211052-888539` (EXP-009)
- new run: `20261009-101431-746940` (EXP-009-drop-labels)

| group | rows | frauds | base PR-AUC | new PR-AUC | gain | 95% CI | p |
|---|---|---|---|---|---|---|---|
| all | 60602 | 2114 | 0.6831 | 0.5863 | -0.0968 | [-0.1085, -0.0864] | 0.000 |
| >= 5 past | 20766 | 743 | 0.8568 | 0.6944 | -0.1624 | [-0.1870, -0.1404] | 0.000 |
| 1-4 past | 21071 | 783 | 0.5996 | 0.5416 | -0.0580 | [-0.0739, -0.0440] | 0.000 |
| cold start | 18765 | 588 | 0.5464 | 0.5432 | -0.0032 | [-0.0066, +0.0003] | 0.060 |
