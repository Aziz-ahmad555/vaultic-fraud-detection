# Tuning E3 (xgboost, feature set `b5_behavioral`)

Generated 2026-10-10 by `python -m vaultic.eval.tune`. Validation period only; validation PR-AUC is checked every 25 rounds for median pruning (after 10 complete trials, never before 200 rounds) and early stopping (max 2000 trees, stop after 100 rounds without improvement; learning rate 0.02–0.3). Same sampler seed and budget per arm. Device: cpu (GPU and CPU results can differ slightly).

| weighting | trials | pruned | best val PR-AUC | trees | total minutes |
|---|---|---|---|---|---|
| none | 25 | 9 | 0.6765 | 1225 | 61 |
| scale_pos_weight | 25 | 7 | 0.6664 | 1975 | 60 |

Winner: **none**. Tuned config: `experiments/configs/EXP-103-tuned.yaml`.
These validation scores are optimistic (the same period chose the hyperparameters and the stopping point); the harness run of the tuned config gives the comparable numbers.

Best parameters per arm:

- none: `{'max_depth': 9, 'learning_rate': 0.031976, 'min_child_weight': 1.393714, 'subsample': 0.710327, 'colsample_bytree': 0.553675, 'reg_lambda': 0.001634, 'reg_alpha': 0.568162, 'gamma': 1.107429}`
- scale_pos_weight: `{'max_depth': 10, 'learning_rate': 0.044802, 'min_child_weight': 6.799865, 'subsample': 0.684573, 'colsample_bytree': 0.675807, 'reg_lambda': 4.889473, 'reg_alpha': 0.00417, 'gamma': 1.218523}`

## Warnings during tuning

None (apart from 0 deprecation/future warnings).
