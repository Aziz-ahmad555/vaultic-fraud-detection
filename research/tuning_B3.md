# Tuning B3 (xgboost, feature set `raw`)

Generated 2026-10-08 by `python -m vaultic.eval.tune`. Validation period only; validation PR-AUC is checked every 25 rounds for median pruning (after 10 complete trials, never before 200 rounds) and early stopping (max 2000 trees, stop after 100 rounds without improvement; learning rate 0.02–0.3). Same sampler seed and budget per arm. Device: cpu (GPU and CPU results can differ slightly).

| weighting | trials | pruned | best val PR-AUC | trees | total minutes |
|---|---|---|---|---|---|
| none | 25 | 7 | 0.5776 | 1950 | 190 |
| scale_pos_weight | 25 | 9 | 0.5607 | 1975 | 166 |

Winner: **none**. Tuned config: `experiments/configs/EXP-011.yaml`.
These validation scores are optimistic (the same period chose the hyperparameters and the stopping point); the harness run of the tuned config gives the comparable numbers.

Best parameters per arm:

- none: `{'max_depth': 10, 'learning_rate': 0.023946, 'min_child_weight': 5.317548, 'subsample': 0.657513, 'colsample_bytree': 0.889227, 'reg_lambda': 0.001111, 'reg_alpha': 0.623602, 'gamma': 0.203405}`
- scale_pos_weight: `{'max_depth': 10, 'learning_rate': 0.056493, 'min_child_weight': 22.136915, 'subsample': 0.764447, 'colsample_bytree': 0.697631, 'reg_lambda': 5.03949, 'reg_alpha': 0.001924, 'gamma': 0.435646}`
