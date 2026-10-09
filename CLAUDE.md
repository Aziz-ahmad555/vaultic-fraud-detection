# Vaultic — project guide for Claude Code

Vaultic is a research framework for financial fraud detection, growing out of the FYP-1 Flask app.
Thesis: "Vaultic: An Adaptive Multi-View Framework for Explainable, Uncertainty-Aware and Drift-Aware Financial Fraud Detection".
Main method: Missing-View-Aware Adaptive Fusion (MVAF) over five evidence views:
tabular, behavioral, temporal (sequence), relational (graph/GNN), anomaly.

The full plan is in `docs/ROADMAP.pdf` (Vaultic Research Roadmap, 17 phases). Read the relevant phase section before starting any task.

## Scope (solo project, decision D31)

Aziz works on this alone. Scope, in priority order:

- **MUST:** Phases 1–3; Phase 4 (graph features + XGBoost first; GNN only if time); Phase 5 (GRU only); Phase 6 (IsolationForest + autoencoder); Phase 7 (MVAF + F1–F6 + a SimMLM-style F7 baseline with MoFe ranking loss); Phase 8; Phase 9 (research side); Phase 13. **Paper 2 is the main paper.**
- **SHOULD:** Paper 1 (graph leakage on the IEEE-CIS heterogeneous entity graph, separating edge leakage vs label leakage and label delay; Elliptic only as a replication of arXiv 2604.19514); Phase 10 as a thesis chapter; a simplified platform (FastAPI + Postgres + replay script, no Kafka); a 4-page dashboard.
- **COULD:** GraphSAGE/TGN, LLM copilot, Paper 3.
- **DROPPED:** Phases 16–17, Papers 4–5, TabPFN, the large load test, the multi-person usability study.

Do MUST work before SHOULD, and SHOULD before COULD. Don't start DROPPED items.

## Non-negotiable research rules

1. **Raw data is read-only.** Never modify anything in `data/raw/`. Derived data goes to `data/interim/` or `data/features/` and is versioned with DVC.
2. **Sort by `TransactionDT` before building any feature.**
3. **Point-in-time features only.** A feature for a transaction at time t uses only transactions strictly before t (`closed='left'` rolling windows or explicit loops). Encoders (frequency, target) are fitted on the training period only.
4. **Label delay.** A past label may be used at time t only if `its_time + L <= t` (default L = 30 days). Applies to every label-derived feature (neighbor fraud rate, customer fraud history, graph labels).
5. **Splits are by time, never random.** Default: train days 0–120, 7-day gap, val 128–150, test 151–182 (adjust to the real day range; the split lives in one definition file). Kaggle test files are never used for evaluation.
6. **Thresholds and hyperparameters are chosen on validation only.** The test period is touched only for final runs. Never tune on test.
7. **Every reported number comes from the evaluation harness** (`python -m vaultic.eval.run experiments/configs/EXP-XXX.yaml`), never from a hand-run notebook.
8. **Metrics:** PR-AUC is primary. Never report accuracy as a headline. 5 seeds, 1,000-sample bootstrap 95% CI, paired bootstrap for any "A beats B", Holm correction for multiple comparisons.
9. **Equal tuning budget** for every compared method (same Optuna trial count, same validation period).
10. **No SMOTE** on time-ordered data unless it is a separate, justified experiment.
11. **Missing views stay missing.** If a view cannot produce a score (no identity, no history, no graph edges), it is masked — never filled with a fake score.
12. **Leakage tests must pass** (`tests/test_leakage.py`) before any feature change is merged.
13. If a hypothesis fails, report it honestly. Never change the method to fit the test set.

## Wording rules (code comments, reports, generated tables)

- Data is "anonymized e-commerce transactions released by Vesta for the IEEE-CIS competition", never "real bank data".
- Masked features are described honestly ("unusual value in masked counter C13"), never given invented meanings.
- No "99% accuracy", "PCI-DSS compliant", "real-time" without latency numbers, "first ever", or "detects all fraud".

## Repository layout

```
data/raw/ interim/ features/        # raw is read-only, DVC-tracked
src/vaultic/
  data/       # loading, merging, uid reconstruction, splits
  features/   # tabular, behavioral, velocity, graph features
  views/      # tabular, temporal, graph, anomaly models
  fusion/     # MVAF gate + fusion baselines F1–F6
  trust/      # calibration, conformal, disagreement, routing
  explain/    # SHAP, counterfactuals, faithfulness, reason codes
  drift/      # PSI, KS, ADWIN, label-delay simulator
  learning/   # feedback, active learning, champion-challenger
  eval/       # harness, metrics, bootstrap, significance tests
experiments/configs/   # one YAML per experiment: EXP-001.yaml ...
experiments/runs/      # git-ignored; logged to MLflow
services/ api/ stream/ online_features/ db/   # not "platform/": that name shadows Python's stdlib module
dashboard/
legacy/fyp1/           # FYP-1 Flask app (tag v1-fyp1), kept runnable; run it from that folder
tests/                 # unit, leakage, parity tests
research/ literature/ tables/ figures/ experiment_log.md
papers/ paper1/ paper2/ paper3/ thesis/
```

Each run saves `config.yaml`, `metrics.json` (mean, std, 95% CI per metric), `predictions.parquet` (id, time, label, per-view scores, final score) and plots, all logged to MLflow with git commit hash and DVC data hash. Add a one-line entry to `research/experiment_log.md` per experiment.

## Conventions

- Python 3.11, venv, pinned `requirements.txt`. Format with black, lint with ruff (pre-commit). Tests with pytest.
- Use float32 / category dtypes; IEEE-CIS is ~590k rows × ~430 columns, so watch memory.
- Graph work: PyTorch Geometric `HeteroData` + `NeighborLoader`; sample, don't load full graphs.
- Windows machine: run Kafka, Redis, PostgreSQL via Docker Desktop + WSL2. Use `pathlib`, not hard-coded backslash paths.
- Branch per feature. Before merging to `main`, open a PR and (decision D32):
  1. Run the full test suite incl. leakage tests; all must pass.
  2. Fill the PR checklist (`.github/pull_request_template.md`): leakage rules respected,
     no test-period use outside `--final`, configs/decisions logged, no data/secrets committed,
     results reproducible from configs.
  3. Independent review: start a NEW Claude Code session (not the one that wrote the code)
     and ask it to review the PR diff against CLAUDE.md and the roadmap. Fix or answer every
     finding before merging.
  Supervisor reviews milestone results (M2, M6, M13 etc.), not individual PRs.
- Unit-test every feature function on a small hand-made example (e.g. 10 rows) with known correct answers.

## How to work in this repo

- Before a multi-file task, show a short plan and wait for approval.
- After finishing, run `pytest` (including leakage tests) and report results; don't claim success without running them.
- Stop at each phase's exit gate and report whether it passed, with numbers.
- Don't install heavy dependencies or change pinned versions without saying so.
