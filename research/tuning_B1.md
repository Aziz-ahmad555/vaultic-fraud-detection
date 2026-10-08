# Tuning B1 (feature set `raw_lr`)

Generated 2026-10-09 by `python -m vaultic.eval.tune`. Grid over the inverse regularisation strength C, one fit per value (lbfgs is deterministic), scored on the validation period only. Model as redefined in D61: median imputation, clipping to the training 0.1%/99.9% quantiles, standardisation, lbfgs with up to 5000 iterations.

| C | val PR-AUC | seconds | iterations | converged |
|---|---|---|---|---|
| 0.001 | 0.3298 | 59 | 106 | yes |
| 0.01 | 0.3413 | 99 | 271 | yes |
| 0.1 | 0.3556 | 244 | 654 | yes |
| 1 | 0.3678 | 557 | 1422 | yes |
| 10 | 0.3703 | 565 | 1667 | yes |

Chosen: **C = 10**. Config: `experiments/configs/EXP-012.yaml`.
