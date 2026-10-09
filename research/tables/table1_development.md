# Table 1: baselines

Generated 2026-10-09 by `python -m vaultic.reports.table1 --mode development` from harness runs. Metrics on the **validation period (development)**: mean ± std over seeds; PR-AUC also shows the 95% bootstrap CI (1,000 resamples). Recall@1%FPR, Recall@5%FPR and Precision@500 are recomputed from each run's saved predictions with the D49 metric fixes. Do not edit by hand.

| Baseline | Model | Features | Tuning | Experiment | PR-AUC | ROC-AUC | Recall@1%FPR | Recall@5%FPR | Precision@500 | Brier | ECE | Seeds | Training s/seed | Inference ms/1k | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | Logistic regression, clipped to training quantiles, lbfgs converged (D61) | raw numeric + one-hot ProductCD | grid over C (7 values), validation, D62 flat-curve rule | EXP-012 | not run | not run | not run | not run | not run | not run | not run |  |  |  |  |
| B1 (original) | Logistic regression, unclipped, lbfgs not converged (1000 iterations) | raw numeric + one-hot ProductCD | grid over C (5 values), validation; superseded by the D61 re-run | EXP-012 | 0.3425 ± 0.0000 [0.3207, 0.3641] | 0.8259 ± 0.0000 | 0.2805 ± 0.0000 | 0.4711 ± 0.0000 | 0.7100 ± 0.0000 | 0.0278 ± 0.0000 | 0.0060 ± 0.0000 | 5 | 512 | 25.1 | 20261008-185520-644386 |
| B2 | Random forest | raw | untuned reference (defaults) | EXP-002 | not run | not run | not run | not run | not run | not run | not run |  |  |  |  |
| B3 | XGBoost | raw, no engineering | Optuna, 50 trials (2 weighting arms x 25), median pruning | EXP-011 | not run | not run | not run | not run | not run | not run | not run |  |  |  |  |
| B4 | LightGBM | raw | Optuna, 50 trials (2 weighting arms x 25), median pruning | EXP-013 | not run | not run | not run | not run | not run | not run | not run |  |  |  |  |
| B5 | XGBoost | raw + V-reduction + point-in-time uid features | Optuna, 50 trials (2 weighting arms x 25), median pruning | EXP-009 | not run | not run | not run | not run | not run | not run | not run |  |  |  |  |
| B6 | FYP-1 XGBoost + per-user Isolation Forest | FYP-1 columns + behavioural | none: FYP-1 settings as built | EXP-010 | 0.3675 ± 0.0000 [0.3447, 0.3896] | 0.8548 ± 0.0000 | 0.3198 ± 0.0000 | 0.5019 ± 0.0000 | 0.8120 ± 0.0000 | 0.1006 ± 0.0000 | 0.2310 ± 0.0000 | 5 | 322 | 76.3 | 20261007-202216-735428 |
