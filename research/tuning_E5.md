# Tuning E5 (xgboost, feature set `b5_graph`)

Generated 2026-10-10 by `python -m vaultic.eval.tune`. Validation period only; validation PR-AUC is checked every 25 rounds for median pruning (after 10 complete trials, never before 200 rounds) and early stopping (max 2000 trees, stop after 100 rounds without improvement; learning rate 0.02–0.3). Same sampler seed and budget per arm. Device: cpu (GPU and CPU results can differ slightly).

| weighting | trials | pruned | best val PR-AUC | trees | total minutes |
|---|---|---|---|---|---|
| none | 25 | 10 | 0.6633 | 925 | 29 |
| scale_pos_weight | 25 | 1 | 0.6568 | 425 | 43 |

Winner: **none**. Tuned config: `experiments/configs/EXP-105-tuned.yaml`.
These validation scores are optimistic (the same period chose the hyperparameters and the stopping point); the harness run of the tuned config gives the comparable numbers.

Best parameters per arm:

- none: `{'max_depth': 10, 'learning_rate': 0.033594, 'min_child_weight': 5.55111, 'subsample': 0.729452, 'colsample_bytree': 0.513044, 'reg_lambda': 0.925046, 'reg_alpha': 0.01059, 'gamma': 0.744237}`
- scale_pos_weight: `{'max_depth': 10, 'learning_rate': 0.04532, 'min_child_weight': 22.976642, 'subsample': 0.735052, 'colsample_bytree': 0.892178, 'reg_lambda': 0.001375, 'reg_alpha': 0.002355, 'gamma': 1.337856}`

## Warnings during tuning

None (apart from 0 deprecation/future warnings).
