# Gap audit (2026-10-22, three months left)

Read-only audit of `phase0-1` at `fd32766` against `docs/ROADMAP.pdf` (43 pages) and
`research/STATUS.md`. No code, config or result was changed. Every status below names its
evidence (file path, decision row or experiment-log row). "Scope" is D31 (CLAUDE.md):
MUST / SHOULD / COULD / DROPPED.

## 1. Roadmap phases and items

| Phase / item | Scope | Status | Evidence |
|---|---|---|---|
| **0 Setup** | MUST | PARTIAL | |
| tag `v1-fyp1`, FYP-1 kept runnable | | DONE | `git tag` -> `v1-fyp1`; `legacy/fyp1/` (CI runs its tests, `.github/workflows/ci.yml`) |
| venv, pinned requirements, pre-commit, pytest, CI | | DONE | `requirements.txt`, `.github/workflows/ci.yml` (lint, data-file block, pytest incl. leakage, FYP-1 tests) |
| Docker Desktop + `docker compose up` test PostgreSQL | | DONE (PostgreSQL only) | `docker-compose.yml`: one `postgres:16-alpine` service; comment says Kafka, Redis, API, scorer, MLflow, dashboard "are added in Phase 11" |
| MLflow (local file store) | | DONE | `src/vaultic/eval/run.py`, `src/vaultic/eval/transfer.py` log runs to MLflow; `experiments/mlflow` (git-ignored) |
| DVC, raw CSVs tracked | | DONE | `.dvc/config`, `data/raw/train_transaction.csv.dvc`, `data/raw/train_identity.csv.dvc`, `docs/DVC_SETUP.md` |
| CONTRIBUTING.md | | NOT STARTED | no such file in `git ls-files` (the branch / PR rule lives in CLAUDE.md, D32) |
| Literature spreadsheet (30-40 papers), gap summaries, supervisor approval | | NOT FOUND in repo | no `literature/` folder tracked; `papers/paper2/references.bib` exists |
| **1 Data foundation** | MUST | DONE | `research/uid_report.md`, `research/data_report.md`, `experiments/configs/splits.yaml`, `tests/test_leakage.py`, EXP-000 (identical metrics twice) |
| **2 Baselines** | MUST | DONE for B1-B6 | `research/tables/table1_final.md` (B1-B6 FINAL test runs) |
| B7 TabPFN | DROPPED | — | D31 |
| B8 FT-Transformer, B9 CatBoost (E19) | not in D31 MUST list | NOT STARTED | no config or run in `experiments/configs`, `research/experiment_log.md` |
| **3 Behavioral view** | MUST | DONE | `src/vaultic/features/behavioral.py`, `research/tables/gain_b5_behavioral.md` |
| **4 Graph view** | MUST (features + XGBoost) | PARTIAL | |
| graph features + XGBoost (setting C) | MUST | DONE | `src/vaultic/features/graph.py`, EXP-103/105 rows in `research/experiment_log.md`, `research/tables/gain_b5_graph.md` |
| GNN / GraphSAGE (E7) | COULD | NOT STARTED | `experiments/configs/EXP-107.yaml` `status: planned`, `scope: COULD` |
| Leakage study A/B/C (E6, Paper 1, H3) | SHOULD | NOT STARTED (planned config only) | `experiments/configs/EXP-106.yaml` `status: planned`; no EXP-106 row in `research/experiment_log.md`; only a post-hoc graph-drift diagnostic exists (`research/e5_graph_drift.md`, D120) |
| Fraud-ring module | — | NOT STARTED | backend returns `"rings": []` (`services/api/demo_server.py`, `/api/network`) |
| TGN (E18), Elliptic replication | COULD / SHOULD (Elliptic only as replication) | NOT STARTED | no code or run; mentioned only in `research/decisions.md` |
| **5 Temporal view** | MUST (GRU only) | DONE | `research/phase5_temporal_report.md`, `research/tables/temporal_dev*.md`; four-architecture comparison not done (out of scope, D31) |
| **6 Anomaly view** | MUST | DONE | `research/phase6_anomaly_report.md`, `research/tables/emerging_fraud.md`, EXP-009/108 hold-out rows |
| **7 MVAF fusion** | MUST | DONE | `research/tables/final_e10.md`, `research/tables/final_e11.md` (EXP-200-final) |
| **8 Uncertainty, decisions, routing** | MUST | DONE; exit gate NOT PASSED | `research/tables/final_e12.md` ("Phase 8 exit gate ... NOT PASSED", MVAF 95% Mondrian coverage 0.9779) |
| **9 Explainability** | MUST | DONE; exit gate PASSED | `research/tables/stage2/e15_report.md` ("Phase 9 exit gate ... PASSED"); `src/vaultic/explain/` (reason codes, narrative) |
| **10 Drift and continuous learning** | SHOULD (thesis chapter) | PARTIAL | |
| drift signals / detectors / injection / label-delay code | | DONE (code, not run as E13) | `src/vaultic/drift/{signals,detectors,inject,label_delay,policies}.py` |
| E13 detectors x delays x injected drifts (H4) | | NOT STARTED (planned config only) | `experiments/configs/EXP-113.yaml` `status: planned`; no EXP-113 row in the log |
| 10.3 retraining policies + champion-challenger gate | | DONE | EXP-310-final, `research/tables/feedback/exp310_report.md`, D210-D218 |
| 10.4 analyst feedback labels | | PARTIAL | `POST /api/feedback/{tid}` keeps labels in memory for the demo session only (docstring of `services/api/demo_server.py`); audited (D208) |
| 10.4 active learning | | NOT STARTED | `experiments/configs/EXP-114.yaml` `status: planned` |
| **11 Real-time platform** | SHOULD (simplified: FastAPI + Postgres + replay, no Kafka) | PARTIAL (small) | |
| Kafka | not in the simplified platform | NOT STARTED | 0 references in `src`, `services`, `requirements.txt` |
| Redis | | NOT STARTED | 0 references |
| PostgreSQL tables (transactions, scores, ..., audit_log) | | NOT STARTED | compose service exists, but no application code uses it (0 `psycopg` references); the audit log is SQLite "PostgreSQL later" (`src/vaultic/audit/store.py`, D208) |
| FastAPI | | PARTIAL | `services/api/demo_server.py` serves the final-run test period read-only; `POST /api/score` answers 501; Test Lab `POST /api/score-live` (D182); no `/explain`, `/models`, `/health`, `/metrics` endpoints |
| Replay producer + scorer consumer | | NOT STARTED | — (the Test Lab scores one transaction on request; the Attack Lab replays days in memory) |
| JWT login with roles | | NOT STARTED | analyst / role are self-declared headers, "attribution, not proof ... until JWT" (`src/vaultic/audit/http.py`, D209); the `Flask-JWT-Extended` / `PyJWT` pins in `requirements.txt` belong to the FYP-1 Flask app |
| Online-feature parity test | | DONE (different form) | Test Lab parity gate: 2,000 real test rows, features 3.0e-8, scores 3.3e-16 / 0.0 (D183, D189); the roadmap asks 10,000 replayed rows |
| `docker compose up` full stack | | NOT STARTED | `docker-compose.yml` has PostgreSQL only |
| Locust load test / max TPS | DROPPED (large load test) | — | D31; Test Lab latency measured instead (D182: p50 472 ms, p95 577 ms features + scoring) |
| **12 Dashboard** | SHOULD (4-page) | DONE beyond scope (11 pages), many on mock or partial data, see section 2 | `dashboard/src/App.tsx` routes |
| live updates via WebSocket / SSE | | NOT STARTED | 0 references in `dashboard/src`, `services` |
| usability study | DROPPED | — | D31 |
| **13 Experiment suite** | MUST | PARTIAL | `make paper-results` exists (`Makefile`, `src/vaultic/reports/paper_results.py`); exit gate not re-checked in this audit |
| E1-E5, E8-E12, E15, E23, E24 | | DONE | `table1_final.md`, `final_e10-12.md`, `stage2/*`, `e24_compute.md` |
| E6, E7, E13, E18, E19 | | NOT STARTED | planned configs only (see above) |
| E14 | | PARTIAL | retraining done (EXP-310), active learning not |
| E16 latency / throughput | | PARTIAL | Test Lab latency only (D182) |
| E17, E21, E22 | DROPPED | — | D31 |
| E20 copilot grounding | COULD | NOT STARTED | — |
| **14 Papers, thesis, defense** | MUST (Paper 2) | PARTIAL (early) | `papers/paper2/` LaTeX skeleton, 124 lines in total across 10 section files (results 5 lines); `papers/paper1`, `paper3`, `thesis` hold only `.gitkeep`; `docs/thesis_notes/` (6 notes) |
| **15 LLM copilot** | COULD | NOT STARTED | no Ollama / copilot code |
| **16 Adversarial, 17 Federated** | DROPPED | — | D31 (the Attack Lab is a product-layer scenario tool, D190-D207, not Phase 16) |

## 2. Dashboard: real vs mock in http mode

http mode = `VITE_DATA_SOURCE=http` (`dashboard/src/api/index.ts`). The demo backend
(`services/api/demo_server.py`) serves the demo bundle written by `vaultic.reports.demo_export`
from the EXP-200-final run: per test transaction the calibrated view scores, MVAF gate weights,
E12 decision, conformal set, disagreement and view-level reasons, labelled "IEEE-CIS test
period, final run". The bundle is a local data file on drive E (`data/interim/demo_final_bundle.parquet`),
**not committed**; it is derived from the final run's predictions. Only the Model lab numbers and the
cost sensitivity come straight from a committed file (`research/tables/final_results.json`, via
`demo_export.lab_runs`).

| Page | http mode shows | Evidence | Needed for fully real data |
|---|---|---|---|
| Overview | REAL final-run test period (day summary from the bundle and frozen E12 decisions); what-if cost mode is labelled illustrative | `api.daySummary` -> `/api/day-summary`; `pages/Overview.tsx` labels | — (what-if is illustrative by design, D122/D128) |
| Live monitor | REAL recorded test-period stream (replayed by the as-of slider, not a live feed) | `api.stream` -> `/api/stream` | a replay producer / live feed (Phase 11) if "live" is required |
| Analyst queue | REAL: R1-R4 routing on the bundle with the frozen K and lambdas | `/api/queues` (`demo_server.py`, `meta["routing_lambdas"]`) | — |
| Cases | MIXED: decision, view scores, gate weights and view-level reasons are REAL; reasons are view-level only ("not feature-level SHAP", `demo_export.py` `reasons_note`); the what-if / counterfactual uses the illustrative mock model (`api.score` -> `illustrative.score`, `api/http/httpApi.ts`) | `pages/Case.tsx`, `httpApi.ts` line "score: ... illustrative.score" | per-transaction SHAP reasons and DiCE counterfactuals in the bundle (they exist for E15 alerts only, `research/tables/stage2/e15_*`) or a live explain endpoint (the Test Lab already computes top-5 feature reasons, D182) |
| Fraud network | REAL entities of today's holds / blocks and their links; **rings always empty** | `/api/network` returns `"rings": []` | a ring detector (roadmap 4.5, not started) |
| Drift monitor | PARTIAL: score / input PSI against the calibration days are REAL; **alarms, injected drifts and retraining history are empty** ("no drift detectors, injected drifts or retraining ran on the final run", comment in `/api/drift`) | `demo_server.py` `/api/drift` | E13 detector outputs (not run) and the EXP-310 retraining events (`research/tables/feedback/exp310_final.json`, run folder events) |
| Model lab | PARTIAL: runs and PR / calibration curves are REAL E10 final-run metrics (`final_results.json` + stored predictions); cost sensitivity REAL (E12 sweep); **ablation table empty** (`"ablation": []`) | `/api/lab`; `pages/Lab.tsx` | wire the stage-2 ladder (`research/tables/stage2/ladder_metrics.json`, `ladder_steps.json`) into `ablation` |
| Scenario simulator | MOCK in both modes: "Illustrative model, not the evaluated system" | `pages/Simulator.tsx` line 40; `httpApi.ts` `SCORE_SOURCE` | route it to the Test Lab's custom scoring (`POST /api/score-live`, frozen system) |
| Reports and settings | REAL: daily report from the day summary, frozen thresholds (read-only, D129), audit log viewer and chain check (D208, D209); what-if illustrative | `pages/Settings.tsx`, `/api/thresholds`, `/api/audit*` | PDF export uses the browser print; roles are self-declared (no JWT) |
| Test Lab | REAL frozen stage-2 system scoring live (parity-gated) when the backend runs with `VAULTIC_TESTLAB=1` | `services/api/testlab.py`, D182-D189 | — |
| Attack Lab | REAL frozen model and rules on SYNTHETIC attack rows: "synthetic scenario - not the evaluation" | `services/api/attack.py`, `docs/scenarios/README.md`, D190-D207 | — (by design not evaluation data) |

In mock mode (the default) every page except Test Lab and Attack Lab shows seeded synthetic
data (`dashboard/src/api/mock`, README "Every number on these screens is synthetic mock data").

## 3. Stage-1 hypotheses H1-H5

The hypotheses are defined in the roadmap's RQ table (`docs/ROADMAP.pdf` p. 4). The stage-1
pre-registration D101 names only **H2** and **H5** explicitly; H1 was pre-registered later for the
ablation ladder in stage 2 (D121, `research/preregistration_stage2.md` section 2).

| H | Roadmap wording (p. 4) | Pre-registered in | Verdict recorded | Source |
|---|---|---|---|---|
| H1 | each added view raises PR-AUC over the strong tabular baseline, non-overlapping 95% CIs for at least two views | D121 (stage 2), as "per step: difference > 0, Holm" | Per step: E3 +behavioral supported, E4 +temporal supported, E5 +graph supported, E8 +anomaly **not supported** (+0.000027, Holm p 0.978). The roadmap's "non-overlapping CIs for at least two views" criterion was **never evaluated as such** | `research/tables/stage2/ladder_meta.json` (`h1`), `ladder_steps.json` |
| H2 | MVAF beats LR and LightGBM stacking overall and by a larger margin on missing-view rows | D101 | **NOT SUPPORTED** (beats F3 overall yes, F4 no; larger margin on missing-view rows: F3 no, F4 no) | `research/tables/final_e10.md` line "H2 (D101)", `research/tables/final_results.json` key `h2` |
| H3 | static full graph reports significantly higher PR-AUC than a strictly temporal graph | not pre-registered | **Never recorded**: E6 was never run (`experiments/configs/EXP-106.yaml` `status: planned`) | — |
| H4 | with a 30-day delay, PSI / disagreement detectors flag injected drift earlier than ADWIN | not pre-registered | **Never recorded**: E13 was never run (`experiments/configs/EXP-113.yaml` `status: planned`) | — |
| H5 | at equal K, routing by expected loss + uncertainty catches more fraud value than top-K risk | D101 ("R3/R4 catch more fraud value than R1 at equal K") | **No verdict word was ever written.** The tests are recorded: for MVAF every R3 - R1 and R4 - R1 difference at K = 50, 100, 200, 500 is positive with Holm p < 0.001 (e.g. K = 50: R3 - R1 +100,164 [+75,151, +127,393]); F3 and B5 likewise. By D101's rule these numbers meet H5, but the verdict itself is not recorded anywhere | `research/tables/final_e12.md` (H5 lists), `research/tables/final_results.json` `e12.<method>.routing` |

Also without a recorded verdict word: D101's E11 hypothesis ("MVAF degrades less than stacking")
has mixed tests (e.g. tabular removed: MVAF drops 0.2429 less than F3; behavioral removed: 0.0232
more than F3) in `research/tables/final_e11.md`, but no "supported / not supported" line.

## 4. Why the Test Lab took ~30 minutes to start from the main checkout

**Cause.** The startup reuses a cached parity result only when its key matches
(`src/vaultic/live/parity.py`, `cache_key`). The key is a SHA-256 over the **raw bytes** of 10 source
files (`CODE`), of `research/frozen_stage2.md`, of the stored predictions
(`REPO_ROOT / E10_RUN / predictions.parquet`) and of `research/tables/final_results.json`, plus the
system-cache path, size and mtime. A key miss runs the full parity check on 2,000 transactions:
1,577 s and 2,431 s (D183), 1,885 s (D189).

Checked on this machine (read-only):
- The main checkout's key is `994495e3…`, the test-lab worktree's `2a6f8d52…`, although git shows
  **no content change** in any hashed file between the two branches (`git diff test-lab d5df3c0`
  over the key files is empty).
- The difference is line endings: `src/vaultic/features/pipeline.py` in the main checkout is
  `w/lf` (written 2026-10-07, before the checkout used CRLF), in the worktrees `w/crlf`
  (`git ls-files --eol`); with CRs stripped the files are identical. Different bytes, same
  content -> new key -> a full parity run. The cache entry `parity_994495e301bdf42f.json` was
  written on 2026-10-21 at 04:05, i.e. on that first main-checkout start.
- A worktree without the `experiments/runs` junction (this audit worktree) gets a third key
  (`71fdea56…`): the predictions file is hashed as `missing`, and parity would then fail (D189).
- Independently of the cache, every start rebuilds the history store, about 4 minutes (D182), and
  loads the 168 MB system cache.

**Proposed fix** (product code in `src/vaultic/live/` and `services/api/` only; no frozen research
code, config or result changes; to be logged as its own decision first):
1. Make the parity key checkout-independent: hash the content with line endings normalised (or the
   git blob id of each file) instead of raw bytes; keep everything else in the key.
2. Refuse to start with a clear message when the stored predictions or the system cache are
   missing, instead of keying on `missing` and running a parity that must fail.
3. Persist the built history store on drive E (e.g. `E:/dev-cache/testlab/store_<key>.*`, keyed by
   the stage-2 data hashes and the store code) and load it on a warm start instead of rebuilding.
4. Add `python -m vaultic.live.warm` (or reuse the start) to pre-build the store and parity once
   after a merge, so the first interactive start is warm.
Expected effect: a warm start = load the system cache + load the store + read the cached parity.
The target of under 1-2 minutes must be **measured** after the change (the load times have not
been measured separately); the full parity still runs (about 30 minutes) whenever hashed code or
data really change, by design (D183).

## 5. IBM synthetic credit-card data (TabFormer) as an external check

Not downloaded in this audit. The description below is from the dataset authors' public
description (IBM TabFormer repository; Altman, "Synthesizing credit card transactions", 2021) and
must be **verified on download**: synthetic transactions of about 2,000 synthetic US users over
several decades (about 24 million rows), columns User, Card, Year, Month, Day, Time, Amount, Use
Chip, Merchant Name (anonymised id), Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?;
fraud rate well below 1%.

| View | Fits? | Why |
|---|---|---|
| Tabular | YES, with a new feature set | B5's features are IEEE-CIS specific (V / C / D columns, `card1`-`card6`, `addr1`; `src/vaultic/data/uid.py`, `features/pipeline.py`); a TabFormer tabular config (amount, MCC, chip use, merchant, time, errors) is needed |
| Behavioral | YES | explicit User / Card ids replace uid reconstruction; amount deviation, velocity, recency, novelty (new merchant / MCC / city / state) port directly (`features/behavioral.py` logic, new column mapping) |
| Temporal | YES | per-card sequences are natural (TabFormer itself is a sequence benchmark) |
| Graph | PARTIAL | only user-card-merchant(-location) links; no email, device, address-of-card or identity entities, so shared-device / shared-email features do not exist; merchants are high-degree hubs |
| Anomaly | YES | label-free models need only the tabular inputs |
| Missing views (MVAF's premise) | WEAK | no identity table, no device data: views are almost always available, so H2's missing-view question has little to test |

Other fit issues: the data are **synthetic** (wording: "synthetic data, not real transactions");
dates span decades, so the split must be chosen (e.g. the last N months as test); the row count
needs a pre-registered user-level subsample to fit this laptop; label delay L applies unchanged;
the cost model works (amounts exist).

**What a limited pre-registered external check would need** (a decision row and a short
pre-registration before any download-based result, as D101 / D121 / D210):
1. One question only, e.g. "does C3 / the B5-vs-fusion finding (or C2, the label-delay
   overstatement) replicate on a second dataset?" with the primary comparison, metric (PR-AUC,
   paired bootstrap, Holm) and decision rule fixed in advance.
2. A frozen data recipe: download hash, the user-level subsample (seeded), the time split
   (train / gap / validation / test) and L.
3. Ported feature builders with the leakage tests run on the new data (point-in-time, label
   curtain), and the same tuning budget for every model.
4. One final run on the TabFormer test period behind the existing guards, results reported as
   found, labelled synthetic.
5. A dataset-loader abstraction is the main engineering cost: today the loaders and uid code are
   IEEE-CIS specific.

## Summary of the largest gaps (for planning, no priority set here)

- Paper 1's core experiment (E6 / H3) has not been run; H4 (E13) neither.
- The platform is a read-only demo backend: no Kafka, Redis, PostgreSQL tables, JWT, replay
  scorer or full `docker compose` stack.
- Papers: Paper 2 is a skeleton; Paper 1, Paper 3 and the thesis have no text yet.
- Dashboard pages on partial data: Cases (view-level reasons only, illustrative what-if), Fraud
  network (no rings), Drift monitor (no alarms / retrains), Model lab (no ablation), Scenario
  simulator (illustrative model).
- H5 and E11 have test results without a written verdict; H1's roadmap criterion was not
  evaluated as written.
- Test Lab cold starts: parity key depends on line endings; history store not cached.
