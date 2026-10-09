# MVAF vs F1-F7, B5: development comparison

Generated 2026-10-09 by `python -m vaultic.fusion.dev_compare`. **Validation period only; no test rows.** Fixed plan: views trained on days 1-120; fusion fitted on gate rows (days 128-143 minus the inner tune days; 34495 rows), evaluated on the calibrate_fused slice (days 147-150; 10276 rows, 300 frauds), which the per-view calibrators (days 144-146) never saw. Default settings, no hyperparameter search, one seed; temporal view = XGBoost on the sequence inputs (D88). Bootstrap 1,000 resamples.

## Views on the evaluation rows (calibrate_fused, days 147-150)

| view | available | rows | PR-AUC (available rows) |
|---|---|---|---|
| tabular | 100.0% | 10276 | 0.4686 |
| behavioral | 68.9% | 7079 | 0.4396 |
| temporal | 68.9% | 7079 | 0.1850 |
| graph | 47.2% | 4853 | 0.1290 |
| anomaly | 100.0% | 10276 | 0.0846 |

## Fused scores on the evaluation rows (calibrate_fused, days 147-150)

| method | PR-AUC | 95% CI | MVAF − method | 95% CI | p |
|---|---|---|---|---|---|
| MVAF | 0.5868 | [0.5341, 0.6400] |  |  |  |
| F1 | 0.5483 | [0.4930, 0.6064] | +0.0385 | [+0.0159, +0.0631] | 0.000 |
| F2 | 0.5866 | [0.5332, 0.6385] | +0.0002 | [-0.0155, +0.0160] | 0.940 |
| F3 | 0.6001 | [0.5471, 0.6512] | -0.0133 | [-0.0286, +0.0029] | 0.112 |
| F4 | 0.5947 | [0.5424, 0.6477] | -0.0079 | [-0.0274, +0.0128] | 0.464 |
| F5 | 0.5958 | [0.5439, 0.6491] | -0.0090 | [-0.0177, +0.0005] | 0.064 |
| F6 | 0.5944 | [0.5404, 0.6464] | -0.0076 | [-0.0187, +0.0043] | 0.248 |
| F7 | 0.5947 | [0.5420, 0.6464] | -0.0079 | [-0.0148, +0.0004] | 0.058 |
| B5 | 0.6285 | [0.5772, 0.6777] | -0.0417 | [-0.0569, -0.0260] | 0.000 |

## Subgroups (evaluation rows)

PR-AUC per method; paired MVAF − B5 (and − F0) with 95% CI. History = the uid has an earlier transaction; graph view available = non-hub relational evidence (D52).

| subgroup | rows | frauds | MVAF | B5 | F1 | F2 | F3 | F4 | F5 | F6 | F7 | MVAF - B5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold start | 3197 | 85 | 0.5745 | 0.5975 | 0.5529 | 0.5857 | 0.5995 | 0.5800 | 0.5677 | 0.5770 | 0.5816 | -0.0230 [-0.0501, +0.0016] |
| with history | 7079 | 215 | 0.5944 | 0.6441 | 0.5478 | 0.5935 | 0.6038 | 0.6045 | 0.6057 | 0.6021 | 0.6028 | -0.0497 [-0.0733, -0.0296] |
| has_identity yes | 1836 | 141 | 0.7124 | 0.7268 | 0.6529 | 0.7165 | 0.7366 | 0.7198 | 0.7110 | 0.7018 | 0.7256 | -0.0144 [-0.0292, +0.0003] |
| has_identity no | 8440 | 159 | 0.4760 | 0.5343 | 0.4762 | 0.4699 | 0.4743 | 0.4796 | 0.4946 | 0.4991 | 0.4848 | -0.0583 [-0.0834, -0.0351] |
| graph view available | 4853 | 149 | 0.5657 | 0.6102 | 0.5243 | 0.5815 | 0.5957 | 0.5865 | 0.5650 | 0.5820 | 0.5769 | -0.0445 [-0.0675, -0.0230] |
| graph view missing | 5423 | 151 | 0.6135 | 0.6497 | 0.5685 | 0.5955 | 0.6075 | 0.6044 | 0.6307 | 0.6075 | 0.6156 | -0.0361 [-0.0602, -0.0151] |

Runtime 301 s.
