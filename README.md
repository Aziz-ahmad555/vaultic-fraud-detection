# Vaultic

**Vaultic: An Adaptive Multi-View Framework for Explainable, Uncertainty-Aware and Drift-Aware
Financial Fraud Detection.**

Vaultic fuses five kinds of fraud evidence (tabular, behavioral, temporal, relational and
anomaly) with Missing-View-Aware Adaptive Fusion (MVAF), evaluated on anonymized e-commerce
transactions released by Vesta for the IEEE-CIS competition, with a strictly time-based split
and no temporal leakage. The full plan is in [`docs/ROADMAP.pdf`](docs/ROADMAP.pdf); working
rules for the repository are in [`CLAUDE.md`](CLAUDE.md).

## Status

| Part | Status |
|---|---|
| FYP-1 app (`legacy/fyp1/`, tag `v1-fyp1`) | ✅ Runs, 18 tests passing |
| Phase 0: repo layout | ✅ Done |
| Phase 0: pinned requirements, pre-commit, CI, Docker Compose, MLflow, DVC | ⏳ In progress |
| Phase 0: literature review, supervisor approval | ⏳ Not started |
| Phases 1–17 (`src/vaultic/`) | ⏳ Not started: packages are empty placeholders |

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

Run the research tests from the repository root with `python -m pytest`. For FYP-1, see
[`legacy/fyp1/README.md`](legacy/fyp1/README.md).
