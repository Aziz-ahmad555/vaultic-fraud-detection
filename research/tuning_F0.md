# Tuning F0 (xgboost, feature set `all_views`)

Generated 2026-10-09 by `python -m vaultic.eval.tune`. Validation period only; validation PR-AUC is checked every 25 rounds for median pruning (after 10 complete trials, never before 200 rounds) and early stopping (max 2000 trees, stop after 100 rounds without improvement; learning rate 0.02–0.3). Same sampler seed and budget per arm. Device: cpu (GPU and CPU results can differ slightly).

| weighting | trials | pruned | best val PR-AUC | trees | total minutes |
|---|---|---|---|---|---|
| none | 25 | 6 | 0.6601 | 675 | 56 |
| scale_pos_weight | 25 | 9 | 0.6575 | 1050 | 58 |

Winner: **none**. Tuned config: `experiments/configs/EXP-F0.yaml`.
These validation scores are optimistic (the same period chose the hyperparameters and the stopping point); the harness run of the tuned config gives the comparable numbers.

Best parameters per arm:

- none: `{'max_depth': 10, 'learning_rate': 0.054408, 'min_child_weight': 24.498456, 'subsample': 0.768185, 'colsample_bytree': 0.613737, 'reg_lambda': 2.63071, 'reg_alpha': 0.001695, 'gamma': 0.377703}`
- scale_pos_weight: `{'max_depth': 10, 'learning_rate': 0.04532, 'min_child_weight': 22.976642, 'subsample': 0.735052, 'colsample_bytree': 0.892178, 'reg_lambda': 0.001375, 'reg_alpha': 0.002355, 'gamma': 1.337856}`

## Warnings during tuning

None (apart from 0 deprecation/future warnings).
