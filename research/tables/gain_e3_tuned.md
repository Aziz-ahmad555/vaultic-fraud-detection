# E3 (B5 + behavioral features, tuned with B5's budget) over B5

Generated 2026-10-10 by `python -m vaultic.reports.gain` from development runs (**validation period only**). Gain = new − base PR-AUC, mean over seeds; paired bootstrap (1000 resamples, same rows for both runs). Groups by the uid's number of earlier transactions (point-in-time).

- base run: `20261008-211052-888539` (EXP-009)
- new run: `20261010-151421-193228` (EXP-103-tuned)

| group | rows | frauds | base PR-AUC | new PR-AUC | gain | 95% CI | p |
|---|---|---|---|---|---|---|---|
| all | 60602 | 2114 | 0.6831 | 0.6762 | -0.0069 | [-0.0100, -0.0040] | 0.000 |
| >= 5 past | 20766 | 743 | 0.8568 | 0.8518 | -0.0050 | [-0.0084, -0.0017] | 0.002 |
| 1-4 past | 21071 | 783 | 0.5996 | 0.5926 | -0.0071 | [-0.0131, -0.0013] | 0.016 |
| cold start | 18765 | 588 | 0.5464 | 0.5435 | -0.0029 | [-0.0089, +0.0029] | 0.326 |
| has_identity yes | 10835 | 974 | 0.7706 | 0.7646 | -0.0061 | [-0.0108, -0.0014] | 0.004 |
| has_identity no | 49767 | 1140 | 0.6025 | 0.5954 | -0.0071 | [-0.0113, -0.0026] | 0.002 |
| graph view available | 29688 | 1119 | 0.6997 | 0.6934 | -0.0063 | [-0.0102, -0.0023] | 0.002 |
| graph view missing | 30914 | 995 | 0.6653 | 0.6568 | -0.0085 | [-0.0131, -0.0038] | 0.002 |
