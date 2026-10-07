# Session reports

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
| 3. B6 (FYP-1 port, E2) | ✅ Code and tests; validation run in progress. Only 470 customers have FYP-1's minimum of 30 training transactions (1.9% of validation rows), so B6 is almost entirely FYP-1's global XGBoost |
| 4. Table 1 and B5 SHAP plot | ✅ Generators and tests; outputs are produced by the chain once tuning is done (Table 1 on validation) |
| 5. Phase 2 exit gate | ⏳ Gate check written; runs at the end of the chain with `--final` runs of B3 and B5 only |

### Problems found and fixed

- **Seeds had no effect** on XGBoost/LightGBM without row/column subsampling (std exactly 0 over 5 seeds). Tuning now searches `subsample` and `colsample_bytree`, so seeds matter.
- **LightGBM (B4) crashed** whenever scikit-learn was imported first (Windows native-runtime clash). Fixed by loading LightGBM first; regression test (D22).
- **MLflow 3 refused the folder store**; switched to a local SQLite store (D16, earlier).
- **B6 rule rounding**: XGBoost's float32 output turned the 0.95 floor into 0.9499999; fixed.

### Needs your input

- **Final Table 1:** you allowed `--final` for the exit gate only, so the chain runs `--final` for B3 and B5 only and builds Table 1 on validation. Say if Table 1 should also get final runs of B1, B2, B4 and B6.
- **Equal tuning for B1, B2, B4:** they are untuned (D17); B3 and B5 are tuned. Rule 9 asks for equal budgets in compared methods; tuning all of them costs several more hours each on this laptop.
- **DVC:** the steps in `docs/DVC_SETUP.md` (Google sign-in is yours to do).

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
