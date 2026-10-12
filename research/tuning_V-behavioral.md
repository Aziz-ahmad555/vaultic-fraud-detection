# Tuning V-behavioral (xgboost, feature set `behavioral_view`)

Generated 2026-10-09 by `python -m vaultic.eval.tune`. Validation period only; validation PR-AUC is checked every 25 rounds for median pruning (after 10 complete trials, never before 200 rounds) and early stopping (max 2000 trees, stop after 100 rounds without improvement; learning rate 0.02–0.3). Same sampler seed and budget per arm. Device: cpu (GPU and CPU results can differ slightly).

| weighting | trials | pruned | best val PR-AUC | trees | total minutes |
|---|---|---|---|---|---|
| none | 25 | 9 | 0.5435 | 525 | 5 |
| scale_pos_weight | 25 | 9 | 0.5388 | 550 | 4 |

Winner: **none**. Tuned config: `experiments/configs/EXP-V-behavioral.yaml`.
These validation scores are optimistic (the same period chose the hyperparameters and the stopping point); the harness run of the tuned config gives the comparable numbers.

Best parameters per arm:

- none: `{'max_depth': 6, 'learning_rate': 0.060539, 'min_child_weight': 21.681621, 'subsample': 0.551359, 'colsample_bytree': 0.920452, 'reg_lambda': 6.695045, 'reg_alpha': 0.010246, 'gamma': 0.095787}`
- scale_pos_weight: `{'max_depth': 5, 'learning_rate': 0.053553, 'min_child_weight': 9.305661, 'subsample': 0.719301, 'colsample_bytree': 0.991862, 'reg_lambda': 0.00256, 'reg_alpha': 0.006847, 'gamma': 0.806548}`

## Warnings during tuning

None (apart from 0 deprecation/future warnings).

Scope (D95): view `behavioral`; trials scored on validation days [128, 143] only; 249548 training rows, 28975 scoring rows (1137 frauds).
