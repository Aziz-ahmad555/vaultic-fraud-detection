# Development summary, 2026-10-09 (validation only)

## F0 − B5 on the evaluation rows (days 147–150, 10,276 rows, 300 frauds)

Paired bootstrap, 1,000 resamples, 5-seed mean scores. B5 = EXP-009 (Phase 2 run); F0 = EXP-F0
(tuned on the whole validation period, D90). Only the validation split of each run's
predictions is read.

| subgroup | F0 − B5 | 95% CI | p |
|---|---|---|---|
| all | −0.0228 | [−0.0362, −0.0109] | <0.001 |
| cold start | −0.0234 | [−0.0498, +0.0019] | 0.074 |
| with history | −0.0237 | [−0.0396, −0.0081] | 0.002 |
| has_identity yes | −0.0293 | [−0.0513, −0.0083] | 0.004 |
| has_identity no | −0.0165 | [−0.0301, −0.0029] | 0.012 |
| graph view available | −0.0252 | [−0.0484, −0.0048] | 0.018 |
| graph view missing | −0.0172 | [−0.0310, −0.0041] | 0.012 |

## Views before and after tuning (D95), validation days 128–143, rows where the view exists

"Before" = the untuned view of the first development table (fixed fold, B5's frozen
hyperparameters; anomaly = logistic link; temporal = XGBoost depth 6). "After" = the best trial
of the view's Optuna tuning (2 × 25 trials). That's the maximum over 50 trials scored on these
same rows, so it's optimistic by selection.

| view | rows | frauds | before | after (best trial) |
|---|---|---|---|---|
| tabular | 41,950 | 1,559 | 0.6197 | 0.6306 |
| behavioral | 28,975 | 1,137 | 0.5346 | 0.5435 |
| temporal | 28,975 | 1,137 | 0.3401 | 0.3800 |
| graph | 20,639 | 858 | 0.1842 | 0.2278 |
| anomaly | 41,950 | 1,559 | 0.1429 | 0.1543 |
