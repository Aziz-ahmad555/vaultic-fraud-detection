# Phase 2 results

Assembled 2026-10-09T09:14:06 by `python -m vaultic.reports.phase2_summary summary`. All frozen configs are unchanged since the freeze record, which logs 5 amendments (see research/frozen_configs.md).

## Phase 2 exit gate

From `research/phase2_exit_gate.md`:

Generated 2026-10-09 by `python -m vaultic.reports.phase2_gate` from --final harness runs (test period).

Gate: B5 beats B3 with non-overlapping 95% CIs on test PR-AUC.

| baseline | experiment | run | test PR-AUC |
|---|---|---|---|
| B3 | EXP-011 | `20261008-195820-697730` | 0.5570 ± 0.0016 (95% CI 0.5403–0.5739) |
| B5 | EXP-009 | `20261008-211052-888539` | 0.6494 ± 0.0017 (95% CI 0.6331–0.6659) |

Paired bootstrap B5 − B3 (same test resamples): +0.0924 (95% CI +0.0828 to +0.1027, p = 0.000).

**Result: PASSED.**

## Table 1 (test period, final runs)

From `research/tables/table1_final.md`:

Generated 2026-10-09 by `python -m vaultic.reports.table1 --mode final` from harness runs. Metrics on the **test period (FINAL runs)**: mean ± std over seeds; PR-AUC also shows the 95% bootstrap CI (1,000 resamples). Recall@1%FPR, Recall@5%FPR and Precision@500 are recomputed from each run's saved predictions with the D49 metric fixes. Do not edit by hand.

| Baseline | Model | Features | Tuning | Experiment | PR-AUC | ROC-AUC | Recall@1%FPR | Recall@5%FPR | Precision@500 | Brier | ECE | F1 (val threshold) | Cost ($) | Seeds | Training s/seed | Inference ms/1k | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | Logistic regression, clipped to training quantiles, lbfgs converged (D61) | raw numeric + one-hot ProductCD | grid over C (7 values), validation, D62 flat-curve rule | EXP-012 | 0.3936 ± 0.0000 [0.3766, 0.4103] | 0.8338 ± 0.0000 | 0.3171 ± 0.0000 | 0.4992 ± 0.0000 | 0.8960 ± 0.0000 | 0.0263 ± 0.0000 | 0.0042 ± 0.0000 | 0.3892 ± 0.0000 | 348666.0820 ± 0.0000 | 5 | 708 | 24.8 | 20261009-091004-464760 |
| B1 (original) | Logistic regression, unclipped, lbfgs not converged (1000 iterations) | raw numeric + one-hot ProductCD | grid over C (5 values), validation; superseded by the D61 re-run | EXP-012 | 0.1827 ± 0.0000 [0.1731, 0.1926] | 0.8188 ± 0.0000 | 0.0000 ± 0.0000 | 0.4382 ± 0.0000 | 0.0166 ± 0.0000 | 0.0388 ± 0.0000 | 0.0167 ± 0.0000 | 0.3236 ± 0.0000 | 381126.4010 ± 0.0000 | 5 | 512 | 25.1 | 20261008-185520-644386 |
| B2 | Random forest | raw | untuned reference (defaults) | EXP-002 | 0.4579 ± 0.0034 [0.4412, 0.4751] | 0.8837 ± 0.0010 | 0.3508 ± 0.0114 | 0.5486 ± 0.0035 | 0.9584 ± 0.0050 | 0.0250 ± 0.0002 | 0.0102 ± 0.0003 | 0.3891 ± 0.0009 | 317433.0424 ± 2207.7601 | 5 | 110 | 17.9 | 20261008-191428-988293 |
| B3 | XGBoost | raw, no engineering | Optuna, 50 trials (2 weighting arms x 25), median pruning | EXP-011 | 0.5570 ± 0.0016 [0.5403, 0.5739] | 0.9034 ± 0.0009 | 0.4789 ± 0.0032 | 0.6599 ± 0.0046 | 0.9416 ± 0.0034 | 0.0216 ± 0.0001 | 0.0106 ± 0.0001 | 0.5320 ± 0.0078 | 251834.9994 ± 3443.7903 | 5 | 391 | 40.6 | 20261008-195820-697730 |
| B4 | LightGBM | raw | Optuna, 50 trials (2 weighting arms x 25), median pruning | EXP-013 | 0.5611 ± 0.0019 [0.5443, 0.5775] | 0.9011 ± 0.0024 | 0.4808 ± 0.0049 | 0.6678 ± 0.0032 | 0.9428 ± 0.0092 | 0.0225 ± 0.0001 | 0.0189 ± 0.0001 | 0.5438 ± 0.0052 | 255279.1554 ± 2451.1867 | 5 | 210 | 107.2 | 20261008-204121-543487 |
| B5 | XGBoost | raw + V-reduction + point-in-time uid features | Optuna, 50 trials (2 weighting arms x 25), median pruning | EXP-009 | 0.6494 ± 0.0017 [0.6331, 0.6659] | 0.9378 ± 0.0007 | 0.5696 ± 0.0019 | 0.7283 ± 0.0047 | 0.9644 ± 0.0041 | 0.0184 ± 0.0001 | 0.0082 ± 0.0001 | 0.6198 ± 0.0037 | 222178.1542 ± 4353.0398 | 5 | 178 | 30.3 | 20261008-211052-888539 |
| B6 | FYP-1 XGBoost + per-user Isolation Forest | FYP-1 columns + behavioural | none: FYP-1 settings as built | EXP-010 | 0.4216 ± 0.0000 [0.4028, 0.4397] | 0.8525 ± 0.0000 | 0.3747 ± 0.0000 | 0.5319 ± 0.0000 | 0.8620 ± 0.0000 | 0.1022 ± 0.0000 | 0.2334 ± 0.0000 | 0.4473 ± 0.0000 | 373320.4480 ± 9.4868 | 5 | 111 | 27.9 | 20261008-212952-637972 |

## E2: FYP-1 original vs temporal

From `research/tables/e2_fyp1_comparison.md`:

Generated 2026-10-08 by `python -m vaultic.reports.e2_fyp1`.

- **FYP-1 original:** FYP-1's saved XGBoost on its saved random 80/20 test split (118,108 rows), FYP-1's 0.5 threshold. FYP-1 never stored these numbers; they are reproduced from its own artifacts.
- **B6 temporal:** the same method ported to the time-based split (global XGBoost + per-user Isolation Forest + 5x-max rule, fitted on days 1-120), test period, days 151-182 (--final), run `20261008-212952-637972`.

| metric | FYP-1 original (random split) | B6 temporal |
|---|---|---|
| PR-AUC | 0.5418 | 0.4216 |
| ROC-AUC | 0.9121 | 0.8525 |
| Recall@1%FPR | 0.4549 | 0.3747 |
| Precision@0.5 | 0.1958 | — |
| Recall@0.5 | 0.7619 | — |
| F1@0.5 | 0.3115 | — |
| Accuracy@0.5 | 0.8822 | — |

Accuracy is shown only because FYP-1 reported it; with 3.5% fraud it says little. The harness does not compute thresholded metrics at 0.5, so those cells are empty for B6.

Why the original numbers are optimistic: (1) the 80/20 split was random, so the model trained on transactions from after the ones it was tested on; (2) missing values were filled with medians computed on the whole dataset, test rows included, before splitting.

## Tuning B5

From `research/tuning_B5.md`:

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

## Tuning B3

From `research/tuning_B3.md`:

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

## Tuning B4

From `research/tuning_B4.md`:

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

## Tuning B1

From `research/tuning_B1.md`:

Generated 2026-10-09 by `python -m vaultic.eval.tune`. Grid over the inverse regularisation strength C, one fit per value (lbfgs is deterministic), scored on the validation period only. Model as redefined in D61: median imputation, clipping to the training 0.1%/99.9% quantiles, standardisation, lbfgs with up to 5000 iterations.

| C | val PR-AUC | seconds | iterations | converged |
|---|---|---|---|---|
| 0.001 | 0.3298 | 89 | 106 | yes |
| 0.01 | 0.3413 | 134 | 271 | yes |
| 0.1 | 0.3556 | 252 | 654 | yes |
| 1 | 0.3678 | 391 | 1422 | yes |
| 10 | 0.3703 | 383 | 1667 | yes |
| 100 | 0.3704 | 17829 | 1659 | yes |
| 1000 | 0.3706 | 527 | 1691 | yes |

Highest val PR-AUC: C = 1000 (0.3706). Selection rule (D62): the smallest C within 0.001 of the highest.

Chosen: **C = 10** (0.3703). Config: `experiments/configs/EXP-012.yaml`.

Grid rows: `research/tuning_B1_grid.json`.

## B5 SHAP top 20

From `research/tables/b5_shap_top20.md`:

Generated 2026-10-08 by `python -m vaultic.reports.shap_summary experiments/configs/EXP-009.yaml`: EXP-009, seed 0, 5,000 random validation rows. Masked IEEE-CIS features (C, D, M, V, id_) have no published meaning.

| rank | feature | mean abs SHAP |
|---|---|---|
| 1 | `uid_fraud_rate_known` | 0.9021 |
| 2 | `C13` | 0.3234 |
| 3 | `freq_uid` | 0.2428 |
| 4 | `TransactionAmt` | 0.2398 |
| 5 | `C1` | 0.2251 |
| 6 | `C5` | 0.1949 |
| 7 | `C14` | 0.1868 |
| 8 | `card6` | 0.1638 |
| 9 | `freq_card1` | 0.1613 |
| 10 | `C11` | 0.1276 |
| 11 | `D1` | 0.1237 |
| 12 | `card2` | 0.1171 |
| 13 | `D15` | 0.1156 |
| 14 | `card1` | 0.1151 |
| 15 | `dist1` | 0.1106 |
| 16 | `V70` | 0.1053 |
| 17 | `D2` | 0.1002 |
| 18 | `freq_P_emaildomain` | 0.0939 |
| 19 | `M5` | 0.0918 |
| 20 | `M6` | 0.0904 |

Figure: `research/figures/b5_shap_summary.png`.

## Measurement conditions for the training-time column (added by hand, not generated)

"Training s/seed" in Table 1 is the wall time of `model.fit` in each --final run
(`run_info.json`, `per_seed_timing`). These timings are only comparable if nothing else
competed for the CPU. While the six --final runs ran (2026-10-08 17:02–21:30), a read-only
logger sampled all processes once a minute.
- **What it recorded:** every process outside the chain that used more than 3 CPU-seconds in
  the previous minute (about 5% of one core).
- **Machine:** 8 logical cores.
- **No test suites, builds or installs ran during the final runs.**

**Flag rule:** a run's training time is flagged for re-measurement on an idle machine if other
processes averaged at least 0.25 core, or peaked above 1 core.

| Run | Baseline | Model threads | Minutes sampled / run length | Other load, mean (cores) | Other load, peak (cores) | Other processes seen | Training time |
|---|---|---|---|---|---|---|---|
| EXP-012 | B1 | 1 (lbfgs) | 57 / 113 | 0.12 | 0.83 | Claude desktop app (53 min), Explorer, Chrome, Windows service | OK |
| EXP-002 | B2 | 8 | 19 / 19 | 0.29 | 0.53 | Claude desktop app (14 min) | **Re-measure** |
| EXP-011 | B3 | 8 | 44 / 44 | 0.22 | 0.32 | Claude desktop app (3 min) | OK (just under the rule) |
| EXP-013 | B4 | 8 | 42 / 43 | 0.08 | 0.10 | Claude desktop app (2 min) | OK |
| EXP-009 | B5 | 8 | 29 / 29 | 0.06 | 0.06 | Claude desktop app (1 min) | OK |
| EXP-010 | B6 | 8 | 19 / 19 | 0.00 | 0.00 | none | OK |
| EXP-012 re-run (D61) | B1 | 1 (lbfgs) | not logged | — | — | this Claude session only (a 4-minute `tasklist` heartbeat); no tests, builds or installs | OK, not logger-verified |

Notes:
- **EXP-012 (B1) coverage gap.** The logger missed one stretch, 17:54–18:48. That falls after
  B1's five fits ended (about 17:46: 5 × ~510 s after the start at 17:02), during the
  bootstrap phase. B1's training time is fully covered.
- **py-spy during B1's bootstrap.** At 18:52–18:55 a separate read-only profiler (py-spy, in a
  scratch venv) was installed and dumped B1's stack once. This was also after the fits, and it
  does not touch the training time.
- **Below-threshold load is not counted.** Processes under 3 CPU-s/min were not logged, so the
  "other load" figures are lower bounds.
- **Tuning times are not idle-machine numbers.** During tuning (B5, B3 and B4, 2026-10-07 22:32
  to 2026-10-08 16:30), CPU-heavy work ran alongside: test suites, venv and npm installs,
  front-end builds and a colour-palette search. The "total minutes" columns in the tuning
  sections are therefore not idle-machine numbers. Table 1 does not use them.

- **B1 re-run (2026-10-09 08:06–09:10).** After the PR #1 review B1 was redefined and re-run once with --final (D61). No CPU logger ran, but the machine was otherwise idle: the only other activity was this Claude session's text editing and a heartbeat that ran `tasklist` every 4 minutes. The original B1 run stays in Table 1 as "B1 (original)" with its own timing.

**To re-measure later on an idle machine:** EXP-002 (B2) training time. Model results do not
change; only the timing columns would.
