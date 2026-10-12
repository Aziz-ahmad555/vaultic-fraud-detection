# Emerging fraud: every held-out group, and the Phase 6 exit gate

Generated 2026-10-10 by `python -m vaultic.reports.emerging report`. **Validation period only** (development). Each group's training frauds removed; B5 frozen hyperparameters; PR-AUC on the group's validation rows (mean over 5 seeds). Anomaly view = mean of the rank-normalised Isolation Forest and autoencoder scores (label-free, fitted on legit rows only, unaffected by the hold-out); 'beats random' = the lower end of its PR-AUC's 95% bootstrap CI lies above the group's fraud rate (D94).

| held-out group | validation frauds | base rate | anomaly-alone PR-AUC [95% CI] | beats random | B5 (saw the group's fraud) | B5 without group | B5 + anomaly without group |
|---|---|---|---|---|---|---|---|
| ProductCD = C | 768 | 0.1244 | 0.3448 [0.3108, 0.3826] | yes | 0.7748 | 0.2473 | 0.3018 |
| ProductCD = H | 81 | 0.0493 | 0.0847 [0.0664, 0.1149] | yes | 0.7320 | 0.6295 | 0.6317 |
| ProductCD = R | 93 | 0.0384 | 0.0737 [0.0555, 0.0979] | yes | 0.7953 | 0.5648 | 0.5539 |
| ProductCD = S | 60 | 0.0581 | 0.0609 [0.0450, 0.0836] | no | 0.7656 | 0.6039 | 0.5996 |
| ProductCD = W | 1112 | 0.0225 | 0.0501 [0.0451, 0.0565] | yes | 0.6010 | 0.0402 | 0.0547 |
| card4 = american express | 20 | 0.0363 | 0.0793 [0.0424, 0.1500] | yes | 0.7709 | 0.8120 | 0.8318 |
| card4 = discover | 106 | 0.1530 | 0.1946 [0.1538, 0.2511] | yes | 0.8104 | 0.7354 | 0.7403 |
| card4 = mastercard | 641 | 0.0329 | 0.1324 [0.1146, 0.1549] | yes | 0.6735 | 0.5906 | 0.5911 |
| card4 = visa | 1347 | 0.0338 | 0.1269 [0.1136, 0.1437] | yes | 0.6782 | 0.3045 | 0.2925 |

**Phase 6 exit gate:** the anomaly view beats random for 8 of 9 groups (needed: at least half). **PASSED** (Validation, development; not a test-period result.)
