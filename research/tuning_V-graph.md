# Tuning V-graph (xgboost, feature set `graph_only`)

Generated 2026-10-09 by `python -m vaultic.eval.tune`. Validation period only; validation PR-AUC is checked every 25 rounds for median pruning (after 10 complete trials, never before 200 rounds) and early stopping (max 2000 trees, stop after 100 rounds without improvement; learning rate 0.02–0.3). Same sampler seed and budget per arm. Device: cpu (GPU and CPU results can differ slightly).

| weighting | trials | pruned | best val PR-AUC | trees | total minutes |
|---|---|---|---|---|---|
| none | 25 | 0 | 0.2275 | 50 | 2 |
| scale_pos_weight | 25 | 6 | 0.2278 | 300 | 3 |

Winner: **scale_pos_weight**. Tuned config: `experiments/configs/EXP-V-graph.yaml`.
These validation scores are optimistic (the same period chose the hyperparameters and the stopping point); the harness run of the tuned config gives the comparable numbers.

Best parameters per arm:

- none: `{'max_depth': 5, 'learning_rate': 0.05267, 'min_child_weight': 2.378455, 'subsample': 0.723924, 'colsample_bytree': 0.626963, 'reg_lambda': 0.01761, 'reg_alpha': 0.454247, 'gamma': 0.89956}`
- scale_pos_weight: `{'max_depth': 9, 'learning_rate': 0.027672, 'min_child_weight': 34.670378, 'subsample': 0.617183, 'colsample_bytree': 0.657618, 'reg_lambda': 4.924339, 'reg_alpha': 0.002547, 'gamma': 0.656676}`

## Warnings during tuning

None (apart from 0 deprecation/future warnings).

Scope (D95): view `graph`; trials scored on validation days [128, 143] only; 260413 training rows, 20639 scoring rows (858 frauds).
