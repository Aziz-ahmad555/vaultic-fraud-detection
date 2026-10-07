# Progress

Roadmap phase tracker (see `docs/ROADMAP.pdf`). Details per session in `research/session_report.md`,
decisions in `research/decisions.md`, every experiment in `research/experiment_log.md`.

## Phase 0: setup

- [x] FYP-1 preserved (`legacy/fyp1/`, tag `v1-fyp1`), alert detail page fixed, database rebuilt
- [x] Repository layout, pinned `requirements.txt`, pre-commit (ruff, black, data-file blocker)
- [x] DVC tracks `data/raw` (hashes only; Google Drive remote: see `docs/DVC_SETUP.md`)
- [x] MLflow (local SQLite store), Docker Compose Postgres (port 55432)
- [x] CI workflow written (not yet run on GitHub)
- [ ] Private GitHub repo shared with Hamza and Roshan; CONTRIBUTING.md
- [ ] Literature review, gap summaries, supervisor approval

## Phase 1: data foundation

- [x] 1A merge, order, dtypes, data report
- [x] 1B uid variants + report (choice: see D8)
- [x] 1C time-based splits, label-delay helpers
- [x] 1D point-in-time features, leakage tests (synthetic + real 50k sample)
- [x] 1E evaluation harness; development runs are validation-only, `--final` for test
- [x] Exit gate: same config + seed gives identical outputs (EXP-000)

## Phase 2: baselines

- [x] Harness: development runs are validation-only; `--final` for test (logged FINAL)
- [x] D8 → D18: `uid` beats `uid2` on validation PR-AUC (+0.0236, paired CI +0.0150 to +0.0330)
- [x] V-column reduction for B5: 339 → 108 V columns; +0.0049 validation PR-AUC (D19)
- [ ] Optuna tuning of B5 (2 × 25 trials, validation only), weighting comparison: **running**
- [ ] Optuna tuning of B3 with the same budget (needed for a fair exit gate): queued after B5
- [x] B6: FYP-1 port (code, tests, config EXP-010); validation run in progress
- [x] LightGBM crash (B4) found and fixed: import-order clash with scikit-learn (D22)
- [x] Table 1 and SHAP generators (code + tests); outputs need the tuned runs
- [ ] B1, B2, B4 development runs under the validation-only harness
- [ ] Exit gate: B5 beats B3 with non-overlapping CIs (`--final` runs)
