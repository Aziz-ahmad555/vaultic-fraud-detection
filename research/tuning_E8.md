# Tuning E8 (xgboost, feature set `b5_anomaly`)

Generated 2026-10-10 by `python -m vaultic.eval.tune`. Validation period only; validation PR-AUC is checked every 25 rounds for median pruning (after 10 complete trials, never before 200 rounds) and early stopping (max 2000 trees, stop after 100 rounds without improvement; learning rate 0.02–0.3). Same sampler seed and budget per arm. Device: cpu (GPU and CPU results can differ slightly).

| weighting | trials | pruned | best val PR-AUC | trees | total minutes |
|---|---|---|---|---|---|
| none | 25 | 7 | 0.6797 | 1675 | 62 |
| scale_pos_weight | 25 | 5 | 0.6746 | 1625 | 64 |

Winner: **none**. Tuned config: `experiments/configs/EXP-108-tuned.yaml`.
These validation scores are optimistic (the same period chose the hyperparameters and the stopping point); the harness run of the tuned config gives the comparable numbers.

Best parameters per arm:

- none: `{'max_depth': 10, 'learning_rate': 0.029863, 'min_child_weight': 6.038668, 'subsample': 0.68351, 'colsample_bytree': 0.632142, 'reg_lambda': 0.002599, 'reg_alpha': 0.058796, 'gamma': 1.438543}`
- scale_pos_weight: `{'max_depth': 10, 'learning_rate': 0.04532, 'min_child_weight': 22.976642, 'subsample': 0.735052, 'colsample_bytree': 0.892178, 'reg_lambda': 0.001375, 'reg_alpha': 0.002355, 'gamma': 1.337856}`

## Warnings during tuning

None (apart from 0 deprecation/future warnings).
