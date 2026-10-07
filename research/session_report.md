# Session report — 2026-10-07 (autonomous session, branch `phase0-1`)

The session stopped early at the usage limit. Nothing has been pushed.

## Done (all committed on `phase0-1` unless noted)

| Item | Status |
|---|---|
| 1. Restructure (`services/`, `.gitignore`, `.env.example`) | ✅ commit `91a0235` on `main` (see D3) |
| 2. FYP-1 runs from `legacy/fyp1/` | ✅ 18/18 tests pass, app serves its login page |
| 3. Installs | ⚠️ Partly done, see B2. Installed: pandas, pyarrow, numpy, scikit-learn, xgboost (already present), ruff, black, pre-commit, lightgbm, optuna. Still installing or not installed: shap, mlflow, dvc, catboost |
| 4. Phase 0C | ⚠️ Partly done. `docker-compose.yml` written but Postgres won't start (B1). Not done: DVC, pre-commit config, pinned `requirements.txt`, CI workflow |
| 5. Phase 1A | ✅ `merged.parquet` + `research/data_report.md` |
| 6. Phase 1B | ✅ `research/uid_report.md`; provisional `uid2` (D8) |
| 7. Phase 1C | ✅ `experiments/configs/splits.yaml`, label-delay helpers |
| 8. Phase 1D | ✅ 20 point-in-time features; leakage tests pass, exact equality, real 50k sample + synthetic |
| 9. Phase 1E | ✅ Harness, metrics and bootstrap built; 67/67 tests pass. **Exit gate passed:** two runs of `EXP-000` produced byte-identical `metrics.json` and `predictions.parquet` (about 250 s each) |
| 10. Phase 2A baselines | ❌ Not started |

## Key numbers (from generated reports)

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
- EXP-000 (XGBoost, raw + base features, seed 0): test PR-AUC **0.6453** (95% CI 0.6286–0.6620). This is one harness run, not a tuned baseline.

## Phase 1 exit gate

| Gate | Result |
|---|---|
| Same config + same seed gives identical metrics twice | ✅ `EXP-000` runs `20261007-170822-685400` and `20261007-171231-833707`: byte-identical outputs |
| Both leakage tests pass | ✅ Locally, on the real 50k sample and on synthetic data. CI isn't set up yet, so they haven't run in CI |
| uid choice documented with numbers | ✅ `research/uid_report.md`, D8 (provisional) |

## Needs your confirmation

These are in `research/decisions.md`:

- **D3:** the restructure commit is on `main`.
- **D4:** installs are constrained to the FYP-1 pins.
- **D8:** provisional `uid2` (runner-up `uid`).

## Blockers

These are in `research/blockers.md`:

- **B1:** Postgres port 5432 is blocked by Windows.
- **B2:** slow or unstable network for pip.
