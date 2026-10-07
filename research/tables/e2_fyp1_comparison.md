# E2: FYP-1's original evaluation vs. the temporal split

Generated 2026-10-07 by `python -m vaultic.reports.e2_fyp1`.

- **FYP-1 original:** FYP-1's saved XGBoost on its saved random 80/20 test split (118,108 rows), FYP-1's 0.5 threshold. FYP-1 never stored these numbers; they are reproduced from its own artifacts.
- **B6 temporal:** the same method ported to the time-based split (global XGBoost + per-user Isolation Forest + 5x-max rule, fitted on days 1-120), validation period, days 128-150 (development run; final pending), run `20261007-202216-735428`.

| metric | FYP-1 original (random split) | B6 temporal |
|---|---|---|
| PR-AUC | 0.5418 | 0.3675 |
| ROC-AUC | 0.9121 | 0.8548 |
| Recall@1%FPR | 0.4549 | 0.3198 |
| Precision@0.5 | 0.1958 | — |
| Recall@0.5 | 0.7619 | — |
| F1@0.5 | 0.3115 | — |
| Accuracy@0.5 | 0.8822 | — |

Accuracy is shown only because FYP-1 reported it; with 3.5% fraud it says little. The harness does not compute thresholded metrics at 0.5, so those cells are empty for B6.

Why the original numbers are optimistic: (1) the 80/20 split was random, so the model trained on transactions from after the ones it was tested on; (2) missing values were filled with medians computed on the whole dataset, test rows included, before splitting.
