# Tuning B4 (lightgbm, feature set `raw`)

Generated 2026-10-08 by `python -m vaultic.eval.tune`. Validation period only; validation PR-AUC is checked every 25 rounds for median pruning (after 10 complete trials, never before 200 rounds) and early stopping (max 2000 trees, stop after 100 rounds without improvement; learning rate 0.02–0.3). Same sampler seed and budget per arm. Device: cpu (GPU and CPU results can differ slightly).

| weighting | trials | pruned | best val PR-AUC | trees | total minutes |
|---|---|---|---|---|---|
| none | 25 | 5 | 0.6000 | 1375 | 400 |
| scale_pos_weight | 25 | 3 | 0.5976 | 2000 | 208 |

Winner: **none**. Tuned config: `experiments/configs/EXP-013.yaml`.
These validation scores are optimistic (the same period chose the hyperparameters and the stopping point); the harness run of the tuned config gives the comparable numbers.

Best parameters per arm:

- none: `{'num_leaves': 137, 'learning_rate': 0.082508, 'min_child_samples': 23, 'subsample': 0.861072, 'colsample_bytree': 0.955982, 'reg_lambda': 0.001566, 'reg_alpha': 0.117654, 'min_split_gain': 0.040741}`
- scale_pos_weight: `{'num_leaves': 143, 'learning_rate': 0.072897, 'min_child_samples': 88, 'subsample': 0.806372, 'colsample_bytree': 0.742021, 'reg_lambda': 0.333093, 'reg_alpha': 0.031547, 'min_split_gain': 0.071595}`
