# Tuning B1 (feature set `raw_lr`)

Generated 2026-10-09 by `python -m vaultic.eval.tune`. Grid over the inverse regularisation strength C, one fit per value (lbfgs is deterministic), scored on the validation period only. Model as redefined in D61: median imputation, clipping to the training 0.1%/99.9% quantiles, standardisation, lbfgs with up to 5000 iterations.

| C | val PR-AUC | seconds | iterations | converged |
|---|---|---|---|---|
| 0.001 | 0.3298 | 89 | 106 | yes |
| 0.01 | 0.3413 | 134 | 271 | yes |
| 0.1 | 0.3556 | 252 | 654 | yes |
| 1 | 0.3678 | 391 | 1422 | yes |
| 10 | 0.3703 | 383 | 1667 | yes |
| 100 | 0.3704 | 17829 | 1659 | yes |
| 1000 | 0.3706 | 527 | 1691 | yes |

Highest val PR-AUC: C = 1000 (0.3706). Selection rule (D62): the smallest C within 0.001 of the highest.

Chosen: **C = 10** (0.3703). Config: `experiments/configs/EXP-012.yaml`.

Grid rows: `research/tuning_B1_grid.json`.
