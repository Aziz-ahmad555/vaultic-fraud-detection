# Tuning V-anomaly (xgboost, feature set `anomaly_only`)

Generated 2026-10-09 by `python -m vaultic.eval.tune`. Validation period only; validation PR-AUC is checked every 25 rounds for median pruning (after 10 complete trials, never before 200 rounds) and early stopping (max 2000 trees, stop after 100 rounds without improvement; learning rate 0.02–0.3). Same sampler seed and budget per arm. Device: cpu (GPU and CPU results can differ slightly).

| weighting | trials | pruned | best val PR-AUC | trees | total minutes |
|---|---|---|---|---|---|
| none | 25 | 0 | 0.1540 | 25 | 1 |
| scale_pos_weight | 25 | 0 | 0.1543 | 25 | 1 |

Winner: **scale_pos_weight**. Tuned config: `experiments/configs/EXP-V-anomaly.yaml`.
These validation scores are optimistic (the same period chose the hyperparameters and the stopping point); the harness run of the tuned config gives the comparable numbers.

Best parameters per arm:

- none: `{'max_depth': 7, 'learning_rate': 0.050361, 'min_child_weight': 9.781021, 'subsample': 0.643783, 'colsample_bytree': 0.989978, 'reg_lambda': 0.386872, 'reg_alpha': 0.001648, 'gamma': 1.19266}`
- scale_pos_weight: `{'max_depth': 7, 'learning_rate': 0.066849, 'min_child_weight': 2.671018, 'subsample': 0.853833, 'colsample_bytree': 0.714861, 'reg_lambda': 5.024608, 'reg_alpha': 0.452363, 'gamma': 1.528665}`

## Warnings during tuning

None (apart from 0 deprecation/future warnings).

Scope (D95): view `anomaly`; trials scored on validation days [128, 143] only; 280203 training rows, 41950 scoring rows (1559 frauds).
