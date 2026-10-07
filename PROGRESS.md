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

- [ ] D8: uid vs uid2 on validation PR-AUC
- [ ] V-column reduction for B5
- [ ] Optuna tuning of B5 (50 trials, validation only), weighting comparison
- [ ] B6: FYP-1 models under the temporal split (E2)
- [ ] Table 1 and B5 SHAP summary plot generated automatically
- [ ] Exit gate: B5 beats B3 with non-overlapping CIs
