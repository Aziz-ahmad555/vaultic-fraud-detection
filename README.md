# Vaultic

**Vaultic: An Adaptive Multi-View Framework for Explainable, Uncertainty-Aware and Drift-Aware
Financial Fraud Detection.**

Vaultic fuses five kinds of fraud evidence (tabular, behavioral, temporal, relational and
anomaly) with Missing-View-Aware Adaptive Fusion (MVAF), evaluated on anonymized e-commerce
transactions released by Vesta for the IEEE-CIS competition, with a strictly time-based split
and no temporal leakage. The full plan is in [`docs/ROADMAP.pdf`](docs/ROADMAP.pdf); working
rules for the repository are in [`CLAUDE.md`](CLAUDE.md).

## Scope and milestones

A solo project; scope (MUST / SHOULD / COULD / DROPPED) is set in [`CLAUDE.md`](CLAUDE.md) and
decision D31 in `research/decisions.md`. Milestones follow the roadmap's M0–M16, re-scoped:

| Milestone | Scope | Status |
|---|---|---|
| FYP-1 app preserved (`legacy/fyp1/`, tag `v1-fyp1`) | done | ✅ Runs, 18 tests passing |
| M0 Repo, CI, DVC, MLflow, literature review, supervisor approval (Phase 0) | MUST | ⏳ Repo, pinned environment, pre-commit, DVC, MLflow, Docker Postgres done; CI written but not yet run on GitHub; literature review and supervisor approval not started |
| M1 Leakage tests, uid chosen, reproducible harness (Phase 1) | MUST | ✅ Exit gate passed |
| M2 Baseline table B1–B6 with CIs (Phase 2) | MUST | ⏳ Tuning and final runs in progress |
| M3 Behavioral view measured (Phase 3) | MUST | ⏳ Feature code and tests done; not yet run on the full data |
| M4 Graph view: graph features + XGBoost (Phase 4); GNN only if time | MUST | ⏳ Heterogeneous entity graph, point-in-time graph features (4.2) and leakage settings A/B/C implemented and unit-tested; not yet run on the full data |
| M5 Temporal (GRU) and anomaly (IsolationForest + autoencoder) views (Phases 5–6) | MUST | ⏳ Anomaly view (global IF, autoencoder, per-uid IF) and the sequence builder implemented and unit-tested; GRU not started (no PyTorch yet); not yet run on the full data |
| M6 MVAF vs F1–F7 (Phase 7) | MUST | ⏳ MVAF and F1–F6 implemented and unit-tested; F7 (SimMLM-style, MoFe ranking loss) implemented and unit-tested; not yet run on real views |
| M7 Calibration, conformal coverage, routing (Phase 8) | MUST | ⏳ Calibration, split/Mondrian/adaptive conformal and routing R1–R4 implemented and unit-tested; not yet run on real scores |
| M8 Explanation faithfulness, research side (Phase 9) | MUST | ⏳ Faithfulness metrics implemented and unit-tested; SHAP runs not started |
| M12 All tables regenerate from one command (Phase 13) | MUST | ⏳ `make paper-results` skeleton (Table 1, E2); statistics tooling done |
| M13 Paper 2 submitted, thesis approved, defense rehearsed (Phase 14) | MUST | ⏳ Paper 2 LaTeX skeleton |
| Paper 1: graph leakage (edge vs label leakage, label delay); Elliptic as replication of arXiv 2604.19514 | SHOULD | Not started |
| M9 Drift under delayed labels as a thesis chapter (Phase 10) | SHOULD | Not started |
| M10 Simplified platform: FastAPI + Postgres + replay script, no Kafka (Phase 11) | SHOULD | Not started |
| M11 4-page dashboard (Phase 12) | SHOULD | Not started |
| M14 GraphSAGE / TGN comparison | COULD | Not started |
| M15 LLM investigation copilot | COULD | Not started |
| Paper 3: drift under delayed labels | COULD | Not started |
| M16 Phases 16–17 (adversarial, federated); Papers 4–5; TabPFN; large load test; multi-person usability study | DROPPED | — |

## Layout

```
data/raw/ interim/ features/   IEEE-CIS data (never committed; raw is read-only)
src/vaultic/                   research code: data, features, views, fusion, trust,
                               explain, drift, learning, eval
experiments/configs/           one YAML per experiment (EXP-001.yaml ...)
services/                      api, stream, online_features, db (Phase 11)
dashboard/                     React app (Phase 12)
tests/                         research tests (unit, leakage, parity)
research/                      literature, tables, figures, experiment_log.md
papers/                        paper1, paper2, paper3, thesis
docs/ROADMAP.pdf               Vaultic Research Roadmap
legacy/fyp1/                   FYP-1 Flask app, kept runnable
```

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r legacy/fyp1/requirements.txt
copy .env.example .env   # then fill in the values
```

Put `train_transaction.csv` and `train_identity.csv` from the Kaggle IEEE-CIS Fraud Detection
competition in `data/raw/`. Only the training files are used; the Kaggle test files have no
public labels.

Run the research tests from the repository root with `python -m pytest`.

Before committing: `pre-commit install` once. The hooks (ruff, black, and a check that blocks
data, databases, models and `.env`) use the tools in `.venv`, so commit with it activated.
`requirements.txt` pins the full environment; `legacy/fyp1/requirements.txt` is FYP-1's own list. For FYP-1, see
[`legacy/fyp1/README.md`](legacy/fyp1/README.md).
