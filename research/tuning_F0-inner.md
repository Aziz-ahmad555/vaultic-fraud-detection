# Tuning F0-inner (xgboost, feature set `all_views`)

Generated 2026-10-09 by `python -m vaultic.eval.tune`. Validation period only; validation PR-AUC is checked every 25 rounds for median pruning (after 10 complete trials, never before 200 rounds) and early stopping (max 2000 trees, stop after 100 rounds without improvement; learning rate 0.02–0.3). Same sampler seed and budget per arm. Device: cpu (GPU and CPU results can differ slightly).

| weighting | trials | pruned | best val PR-AUC | trees | total minutes |
|---|---|---|---|---|---|
| none | 25 | 5 | 0.6918 | 800 | 64 |
| scale_pos_weight | 25 | 10 | 0.6813 | 1225 | 58 |

Winner: **none**. Tuned config: `experiments/configs/EXP-F0-inner.yaml`.
These validation scores are optimistic (the same period chose the hyperparameters and the stopping point); the harness run of the tuned config gives the comparable numbers.

Best parameters per arm:

- none: `{'max_depth': 9, 'learning_rate': 0.043586, 'min_child_weight': 1.845632, 'subsample': 0.881195, 'colsample_bytree': 0.944899, 'reg_lambda': 2.403758, 'reg_alpha': 0.005267, 'gamma': 1.568397}`
- scale_pos_weight: `{'max_depth': 10, 'learning_rate': 0.026927, 'min_child_weight': 22.83301, 'subsample': 0.64959, 'colsample_bytree': 0.664825, 'reg_lambda': 0.011689, 'reg_alpha': 0.429587, 'gamma': 0.572815}`

## Warnings during tuning

None (apart from 0 deprecation/future warnings).

Scope (D95): view `all rows`; trials scored on validation days [128, 143] only; 414542 training rows, 41950 scoring rows (1559 frauds).
