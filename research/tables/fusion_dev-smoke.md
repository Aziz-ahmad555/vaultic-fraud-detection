# MVAF vs F1-F7, B5: development comparison — SMOKE (not a result)

Generated 2026-10-09 by `python -m vaultic.fusion.dev_compare`. **Validation period only; no test rows.** Fixed plan: views trained on days 1-120; fusion fitted on gate rows (days 128-143 minus the inner tune days; 3388 rows), evaluated on the calibrate_fused slice (days 147-150; 1073 rows, 31 frauds), which the per-view calibrators (days 144-146) never saw. Default settings, no hyperparameter search, one seed; temporal view = XGBoost on the sequence inputs (D88). Bootstrap 1,000 resamples.

## Views on the evaluation rows (calibrate_fused, days 147-150)

| view | available | rows | PR-AUC (available rows) |
|---|---|---|---|
| tabular | 100.0% | 1073 | 0.3845 |
| behavioral | 71.1% | 763 | 0.1074 |
| temporal | 71.1% | 763 | 0.1822 |
| graph | 48.8% | 524 | 0.0996 |
| anomaly | 100.0% | 1073 | 0.0995 |

## Fused scores on the evaluation rows (calibrate_fused, days 147-150)

| method | PR-AUC | 95% CI | MVAF − method | 95% CI | p |
|---|---|---|---|---|---|
| MVAF | 0.3903 | [0.2293, 0.5761] |  |  |  |
| F1 | 0.3445 | [0.2039, 0.5521] | +0.0458 | [-0.0599, +0.1112] | 0.508 |
| F2 | 0.4042 | [0.2421, 0.5825] | -0.0140 | [-0.0634, +0.0362] | 0.560 |
| F3 | 0.3693 | [0.2134, 0.5494] | +0.0209 | [-0.0348, +0.0763] | 0.420 |
| F4 | 0.3762 | [0.2242, 0.5414] | +0.0141 | [-0.0862, +0.1254] | 0.820 |
| F5 | 0.3826 | [0.2258, 0.5750] | +0.0077 | [-0.0521, +0.0611] | 0.844 |
| F6 | 0.3626 | [0.2085, 0.5576] | +0.0276 | [-0.0541, +0.0976] | 0.604 |
| F7 | 0.3808 | [0.2186, 0.5602] | +0.0094 | [-0.0291, +0.0453] | 0.716 |
| B5 | 0.5199 | [0.3329, 0.6925] | -0.1297 | [-0.2236, -0.0296] | 0.010 |

## Subgroups (evaluation rows)

PR-AUC per method; paired MVAF − B5 (and − F0) with 95% CI. History = the uid has an earlier transaction; graph view available = non-hub relational evidence (D52).

| subgroup | rows | frauds | MVAF | B5 | F1 | F2 | F3 | F4 | F5 | F6 | F7 | MVAF - B5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold start | 310 | 12 | 0.5536 | 0.6246 | 0.4944 | 0.5267 | 0.5094 | 0.5514 | 0.5098 | 0.5605 | 0.5482 | -0.0710 [-0.2112, +0.0850] |
| with history | 763 | 19 | 0.2947 | 0.4877 | 0.2847 | 0.3031 | 0.2910 | 0.3276 | 0.3493 | 0.2372 | 0.2798 | -0.1930 [-0.3049, -0.0609] |
| has_identity yes | 194 | 17 | 0.6626 | 0.8025 | 0.5759 | 0.6586 | 0.6674 | 0.6302 | 0.6379 | 0.6449 | 0.6519 | -0.1400 [-0.3116, +0.0205] |
| has_identity no | 879 | 14 | 0.0716 | 0.1541 | 0.0899 | 0.0817 | 0.0596 | 0.0651 | 0.0867 | 0.0764 | 0.0782 | -0.0825 [-0.2528, +0.0091] |
| graph view available | 524 | 12 | 0.5259 | 0.5181 | 0.3795 | 0.5569 | 0.4748 | 0.5315 | 0.4642 | 0.4853 | 0.5163 | +0.0079 [-0.1358, +0.1541] |
| graph view missing | 549 | 19 | 0.3012 | 0.5774 | 0.3481 | 0.3224 | 0.3232 | 0.3132 | 0.3743 | 0.2923 | 0.2925 | -0.2762 [-0.3778, -0.1049] |

Runtime 70 s.
