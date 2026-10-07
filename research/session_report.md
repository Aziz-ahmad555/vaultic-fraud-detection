# Session reports

## Session 2b — 2026-10-07 night (branch `phase3-prep`, worktree `.worktrees/phase3-prep`)

Light work only while the Phase 2 chain runs; done in a separate worktree so the chain's code
and configs are untouched (D25). To merge into `phase0-1` after the chain finishes.

| Item | Status |
|---|---|
| Kaggle portability | ✅ `VAULTIC_DATA_DIR` / `VAULTIC_RAW_DIR`; `--device cuda` for XGBoost and LightGBM (recorded in every run's saved config, metrics and run info); `vaultic.eval.transfer` export/import of runs; `vaultic.data.verify` checks Kaggle's CSVs against the DVC md5s; `docs/KAGGLE.md`. Not yet tried on Kaggle itself |
| Phase 3 behavioral features | ✅ All roadmap groups in `src/vaultic/features/behavioral.py`; 10-row hand-computed tests and synthetic leakage tests pass (D26). Not run on the full data yet |
| Push preparation | ✅ All 29 commits scanned: no data, database or `.env` files, no real secret values. One model file (FYP-1's 1 MB `xgboost_baseline.json`) is in early history and tag `v1-fyp1`: **keep or rewrite is your call** (`docs/PUSH.md`, with exact push commands) |

**Chain pace:** B5 tuning trials take 18–51 minutes each on this laptop (3 of 50 done after
2 hours). Even with pruning, B5, B3 and B4 tuning plus the final runs look like 2–3 days of CPU,
not one night. Options: let it run; move the XGBoost tuning (B3, B5) to Kaggle's GPU with the new
`--device cuda` path; or reduce the budget. Your decision.

## Session 2 — 2026-10-07 evening (branch `phase0-1`)

Nothing has been pushed. Long-running Phase 2 work continues unattended in
`tools/run_phase2.ps1` (log: `experiments/tuning/phase2_chain.log`).

### Your decisions, applied

| Decision | What was done |
|---|---|
| D3, D4 accepted | Marked accepted in `research/decisions.md` |
| D13: DVC on shared Google Drive | `dvc-gdrive` installed and pinned; **your steps are in `docs/DVC_SETUP.md`** (folder, OAuth client, `dvc remote add`, `dvc commit`, `dvc push`) |
| D8: choose uid on validation PR-AUC | **`uid` wins** (D18): 0.6480 vs 0.6244 for `uid2`; paired +0.0236 (95% CI +0.0150 to +0.0330, p < 0.001). `splits.yaml` now uses `uid` |
| Harness validation-only by default | Done: test rows are never predicted without `--final`; `--final` runs are logged **FINAL**; a test proves development output is unchanged when every test label and feature is corrupted. Earlier test-metric runs are marked EXPLORATORY (not for papers) |

### Phase 2 status

| Item | Status |
|---|---|
| 1. V-column reduction | ✅ 339 → 108 V columns (train period only); validation PR-AUC 0.6529 vs 0.6480 without it, paired +0.0049 (95% CI +0.0009 to +0.0095) (D19) |
| 2. Optuna tuning of B5 | ⏳ Running: 2 × 25 trials (no weighting vs `scale_pos_weight`), median pruning (D20, D23). First trial: validation PR-AUC 0.6548 |
| 2b. Equal-budget tuning of B3 | ⏳ Queued after B5 (needed so the exit gate compares equally tuned models) |
| 3. B6 (FYP-1 port, E2) | ✅ Validation PR-AUC **0.3675** (95% CI 0.3447–0.3896), ROC-AUC 0.855, recall at 1% FPR 0.320 (EXP-010, 5 seeds; spread < 0.0001 because only the per-user forests use the seed). For comparison the untuned B5 reaches about 0.65 on the same validation period. Only 470 customers have FYP-1's minimum of 30 training transactions (1.9% of validation rows), so B6 is almost entirely FYP-1's global XGBoost |
| 4. Table 1 and B5 SHAP plot | ✅ Generators and tests; outputs are produced by the chain once tuning is done (Table 1 on validation) |
| 5. Phase 2 exit gate | ⏳ Gate check written; runs at the end of the chain with `--final` runs of B3 and B5 only |

### E2: FYP-1's original numbers vs. the temporal split

`research/tables/e2_fyp1_comparison.md` (regenerated with B6's test-period result by the overnight chain). Reproduced from FYP-1's own saved model and saved random test split, since FYP-1 never stored its metrics: **PR-AUC 0.5418, ROC-AUC 0.912** originally, against B6's temporal **0.3675 / 0.855** (validation; final pending).

**Why the original evaluation was optimistic:** it split transactions at random (so the model trained on transactions from after the ones it was tested on), and it filled missing values with medians computed on the whole dataset, test rows included.

### Problems found and fixed

- **Seeds had no effect** on XGBoost/LightGBM without row/column subsampling (std exactly 0 over 5 seeds). Tuning now searches `subsample` and `colsample_bytree`, so seeds matter.
- **LightGBM (B4) crashed** whenever scikit-learn was imported first (Windows native-runtime clash). Fixed by loading LightGBM first; regression test (D22).
- **MLflow 3 refused the folder store**; switched to a local SQLite store (D16, earlier).
- **B6 rule rounding**: XGBoost's float32 output turned the 0.95 floor into 0.9499999; fixed.

### Your answers, applied (D24)

- **Tuning:** B3, B4, B5 get Optuna with 50 trials each (2 weighting arms × 25) and the same pruning rule (LightGBM added to the tuner); B1 a grid over C (0.001–10) on validation; B2 stays an untuned reference; B6 stays as FYP-1 built it. Table 1 states each budget in a "Tuning" column and shows training time per seed.
- **Final runs:** after all tuning, the chain hashes every Table 1 config into `research/frozen_configs.md`, then makes exactly one `--final` run per baseline B1–B6. The harness refuses a second `--final` run of an experiment unless `--rerun-reason` is given, and logs that reason in `research/decisions.md`.
- **Overnight:** `tools/run_phase2.ps1` → `research/phase2_results.md` (tuning results, Table 1 final, exit gate, E2, SHAP), with a check that no frozen config changed.

### Still needs you

- **DVC:** the steps in `docs/DVC_SETUP.md` (the Google sign-in is yours to do).

## Session 1 — 2026-10-07 afternoon (autonomous, branch `phase0-1`)

Items 1-9 are done; item 10 (Phase 2A baselines) is running. Nothing has been pushed. Research tests: 70 passing; FYP-1 tests: 18 passing.

### Done (all committed on `phase0-1` unless noted)

| Item | Status |
|---|---|
| 1. Restructure (`services/`, `.gitignore`, `.env.example`) | ✅ commit `91a0235` on `main` (see D3) |
| 2. FYP-1 runs from `legacy/fyp1/` | ✅ 18/18 tests pass, app serves its login page |
| 3. Installs | ✅ All groups installed after retrying one package at a time (B2): pandas, pyarrow, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna, shap, mlflow, dvc, pytest, black, ruff, pre-commit. FYP-1's pinned versions unchanged |
| 4. Phase 0C | ✅ Pinned `requirements.txt` (202 packages); pre-commit (ruff, black, data-file blocker); DVC tracking `data/raw` (no remote yet, D13); MLflow logging in the harness (SQLite store `experiments/mlflow/mlflow.db`; MLflow 3 refuses the plain-folder store, caught by a check and now tested); Postgres via `docker compose` on port 55432 (B1 fixed, D14); CI workflow written but **not yet run on GitHub** (needs a push) |
| 5. Phase 1A | ✅ `merged.parquet` + `research/data_report.md` |
| 6. Phase 1B | ✅ `research/uid_report.md`; provisional `uid2` (D8) |
| 7. Phase 1C | ✅ `experiments/configs/splits.yaml`, label-delay helpers |
| 8. Phase 1D | ✅ 20 point-in-time features; leakage tests pass, exact equality, real 50k sample + synthetic |
| 9. Phase 1E | ✅ Harness, metrics and bootstrap built; 67/67 tests pass. **Exit gate passed:** two runs of `EXP-000` produced byte-identical `metrics.json` and `predictions.parquet` (about 250 s each) |
| 10. Phase 2A baselines | ⏳ Configs `EXP-001`–`EXP-005` committed (untuned, D17; B5 is partial). Runs started in the background in the order B1, B4, B3, B5, B2; each finished run appends its line to `research/experiment_log.md`. Expected total: over an hour, mostly the random forest |

### Key numbers (from generated reports)

- 590,540 transactions; fraud **3.499%**; days **1–182**; identity data on 24.42% of rows (43% in days 1–30, about 18% afterwards).
- Splits:
  - train 1–120: 414,542 rows
  - gap: 22,969 rows
  - validation 128–150: 60,602 rows
  - test 151–182: 92,427 rows
  - fraud is about 3.5% in each
- uid comparison (days 1–150):

  | Variant | Pure repeat IDs | Rows in mixed IDs | Rows in repeat IDs |
  |---|---|---|---|
  | `uid2` | 97.75% | 2.60% | 68.5% |
  | `uid` | 96.70% | 5.48% | 77.8% |
  | `uid_card` | 85.7% | 74.8% | — |

- 46.4% of transactions are cold start under `uid2`.
- EXP-000 (XGBoost, raw + base features, seed 0): test PR-AUC 0.6453 (95% CI 0.6286–0.6620). **EXPLORATORY, not for papers:** it was computed on the test period before the validation-only rule existed. The same applies to the EXP-001, EXP-003 and EXP-005 results in the experiment log.

### Phase 1 exit gate

| Gate | Result |
|---|---|
| Same config + same seed gives identical metrics twice | ✅ `EXP-000` runs `20261007-170822-685400` and `20261007-171231-833707`: byte-identical outputs |
| Both leakage tests pass | ✅ Locally, on the real 50k sample and on synthetic data. CI isn't set up yet, so they haven't run in CI |
| uid choice documented with numbers | ✅ `research/uid_report.md`, D8 (provisional) |

### Needs your confirmation

These are in `research/decisions.md`:

- **D3:** the restructure commit is on `main`.
- **D4:** installs are constrained to the FYP-1 pins.
- **D8:** provisional `uid2` (runner-up `uid`).
- **D13:** choose a DVC remote; the raw files are only hashed for now.

### Blockers

These are in `research/blockers.md`:

- **B1 (resolved):** port 5432 belongs to a native PostgreSQL on this PC; compose now uses 55432.
- **B2 (resolved):** pip needed one-package-at-a-time installs on the unstable network.
