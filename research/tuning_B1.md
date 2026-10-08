# Tuning B1 (feature set `raw_lr`)

Generated 2026-10-08 by `python -m vaultic.eval.tune`. Grid over the inverse regularisation strength C, one fit per value (lbfgs is deterministic), scored on the validation period only.

| C | val PR-AUC | seconds |
|---|---|---|
| 0.001 | 0.3078 | 105 |
| 0.01 | 0.3163 | 163 |
| 0.1 | 0.3275 | 338 |
| 1 | 0.3396 | 504 |
| 10 | 0.3425 | 512 |

Chosen: **C = 10**. Config: `experiments/configs/EXP-012.yaml`.
