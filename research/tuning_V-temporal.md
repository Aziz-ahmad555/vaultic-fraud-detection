# Tuning V-temporal (xgboost, feature set `sequence_only`)

Generated 2026-10-09 by `python -m vaultic.eval.tune`. Validation period only; validation PR-AUC is checked every 25 rounds for median pruning (after 10 complete trials, never before 200 rounds) and early stopping (max 2000 trees, stop after 100 rounds without improvement; learning rate 0.02–0.3). Same sampler seed and budget per arm. Device: cpu (GPU and CPU results can differ slightly).

| weighting | trials | pruned | best val PR-AUC | trees | total minutes |
|---|---|---|---|---|---|
| none | 25 | 2 | 0.3800 | 250 | 11 |
| scale_pos_weight | 25 | 0 | 0.3739 | 75 | 9 |

Winner: **none**. Tuned config: `experiments/configs/EXP-V-temporal.yaml`.
These validation scores are optimistic (the same period chose the hyperparameters and the stopping point); the harness run of the tuned config gives the comparable numbers.

Best parameters per arm:

- none: `{'max_depth': 10, 'learning_rate': 0.028092, 'min_child_weight': 3.341336, 'subsample': 0.718504, 'colsample_bytree': 0.836437, 'reg_lambda': 0.008459, 'reg_alpha': 2.780697, 'gamma': 0.844798}`
- scale_pos_weight: `{'max_depth': 10, 'learning_rate': 0.068513, 'min_child_weight': 22.164839, 'subsample': 0.611897, 'colsample_bytree': 0.894124, 'reg_lambda': 0.342859, 'reg_alpha': 0.001364, 'gamma': 0.449598}`

## Warnings during tuning

None (apart from 0 deprecation/future warnings).

Scope (D95): view `temporal`; trials scored on validation days [128, 143] only; 249548 training rows, 28975 scoring rows (1137 frauds).
