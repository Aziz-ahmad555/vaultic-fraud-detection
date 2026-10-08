# Tuning B5 (xgboost, feature set `b5`)

Generated 2026-10-08 by `python -m vaultic.eval.tune`. Validation period only; validation PR-AUC is checked every 25 rounds for median pruning (after 10 complete trials, never before 200 rounds) and early stopping (max 2000 trees, stop after 100 rounds without improvement; learning rate 0.02–0.3). Same sampler seed and budget per arm. Device: cpu (GPU and CPU results can differ slightly).

| weighting | trials | pruned | best val PR-AUC | trees | total minutes |
|---|---|---|---|---|---|
| none | 25 | 2 | 0.6832 | 1350 | 67 |
| scale_pos_weight | 25 | 8 | 0.6644 | 1100 | 46 |

Winner: **none**. Tuned config: `experiments/configs/EXP-009.yaml`.
These validation scores are optimistic (the same period chose the hyperparameters and the stopping point); the harness run of the tuned config gives the comparable numbers.

Best parameters per arm:

- none: `{'max_depth': 10, 'learning_rate': 0.034936, 'min_child_weight': 3.208114, 'subsample': 0.758866, 'colsample_bytree': 0.896783, 'reg_lambda': 0.001099, 'reg_alpha': 0.006097, 'gamma': 1.622601}`
- scale_pos_weight: `{'max_depth': 9, 'learning_rate': 0.025989, 'min_child_weight': 12.001076, 'subsample': 0.735048, 'colsample_bytree': 0.777823, 'reg_lambda': 0.014987, 'reg_alpha': 0.001737, 'gamma': 0.60046}`
