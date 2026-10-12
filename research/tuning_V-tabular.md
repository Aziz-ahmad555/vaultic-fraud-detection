# Tuning V-tabular (xgboost, feature set `tabular_view`)

Generated 2026-10-09 by `python -m vaultic.eval.tune`. Validation period only; validation PR-AUC is checked every 25 rounds for median pruning (after 10 complete trials, never before 200 rounds) and early stopping (max 2000 trees, stop after 100 rounds without improvement; learning rate 0.02–0.3). Same sampler seed and budget per arm. Device: cpu (GPU and CPU results can differ slightly).

| weighting | trials | pruned | best val PR-AUC | trees | total minutes |
|---|---|---|---|---|---|
| none | 25 | 9 | 0.6306 | 1600 | 44 |
| scale_pos_weight | 25 | 6 | 0.6245 | 1925 | 67 |

Winner: **none**. Tuned config: `experiments/configs/EXP-V-tabular.yaml`.
These validation scores are optimistic (the same period chose the hyperparameters and the stopping point); the harness run of the tuned config gives the comparable numbers.

Best parameters per arm:

- none: `{'max_depth': 10, 'learning_rate': 0.028623, 'min_child_weight': 3.615679, 'subsample': 0.745604, 'colsample_bytree': 0.609057, 'reg_lambda': 0.080541, 'reg_alpha': 0.227007, 'gamma': 0.813862}`
- scale_pos_weight: `{'max_depth': 9, 'learning_rate': 0.047739, 'min_child_weight': 17.033833, 'subsample': 0.767149, 'colsample_bytree': 0.604277, 'reg_lambda': 5.859129, 'reg_alpha': 0.01134, 'gamma': 0.206738}`

## Warnings during tuning

None (apart from 0 deprecation/future warnings).

Scope (D95): view `tabular`; trials scored on validation days [128, 143] only; 414542 training rows, 41950 scoring rows (1559 frauds).
