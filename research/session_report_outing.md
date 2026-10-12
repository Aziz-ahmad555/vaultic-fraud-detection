# Session report: Aziz's outing list (2026-10-11)

Worked through the five items in order, without --final runs, without logging any new
pre-registration, one heavy job at a time, pushing after each item, data and caches on drive E.

## Results

### 1. F3 sensitivity (POST-HOC / EXPLORATORY, D119) — done, pushed `58665f4`

F3-log = F3 with ctx_hist_n_past -> log1p(min(count, 366)), 366 being its maximum on the gate rows;
D95-tuned C kept (not re-tuned), same seeds, gate rows and other inputs; scored on the cached final
test inputs. research/f3_log_sensitivity.md.

| method (test period) | PR-AUC [95% CI] | recall@1%FPR | cost ($) |
|---|---|---|---|
| F3-log (post-hoc) | 0.5908 [0.5731, 0.6080] | 0.520 | 255,651 |
| F3 (pre-registered) | 0.4229 | 0.280 | 278,070 |
| MVAF | 0.6219 | 0.547 | 236,084 |

Bounding the one input recovers most of F3's collapse (+0.168), but F3-log stays below MVAF
(-0.031, CI -0.035 to -0.026), F4 (-0.030) and B5 (-0.062). Without the collapse MVAF's margin over
the logistic stacker is about +0.03, not +0.20. No pre-registered number changed.

### 2. E5 graph-drift diagnostic (POST-HOC, validation only, D120) — done, pushed `3231d84`, `68d024e`

research/e5_graph_drift.md. The setting-C entity counts are cumulative since the data start and grow
with calendar time (Spearman with day 0.29-0.84); e.g. email counts' median doubles and 48% of
validation rows exceed the training maximum. Time-normalising recovers only about a third of E5's
penalty (validation PR-AUC, seeds 0-2): B5 0.6858, E5 0.6651, counts per elapsed day 0.6698,
counts dropped 0.6715; log1p = as built (tree invariance; caps also cannot change trees). The
remaining ~0.014 gap holds with and without graph evidence, so the count drift is a real design
issue of point-in-time graph features for Paper 1, but it does not explain E5. No frozen view changed.

### 3. make paper-results — done, pushed `3177ea8`

20 steps, each reading only saved runs / saved JSON: Table 1 (incl. B1 original), E2, the
label-feature ablation, the frozen and retuned E3/E5/E8 gains, E25 (L7/30/60), emerging fraud (+ the
two ProductCD-R gains), D95, E10-E12 (+ reliability figures), the F3 diagnosis and sensitivity and the
E5 drift report. New reports/render_saved.py re-renders the JSON-backed reports with their own
writers and keeps hand-written sections; d95_compare's writer was split out for this.
**End-to-end test:** all 20 steps run (56 min) and regenerate the committed files exactly — the only
difference is the "Generated" date line; E10-E12 and the figures are byte-identical. One fix: E2 needs
the git-ignored FYP-1 model and CSVs; they are not in this worktree, so E2 now honours
VAULTIC_FYP1_DIR (pointed at the main checkout's legacy/fyp1). Run:
`make paper-results` (or `python -m vaultic.reports.paper_results`).

### 4. Stage-2 pre-registration — DRAFT only, pushed `5c22bc6`, not logged

research/drafts/preregistration_stage2.md: E15 (deletion vs random-k and permutation importance with
AOPC, insertion, stability under ±1/±5% amount, seed consistency, counterfactual validity on 500
alerts, exit gate as the roadmap), the ablation ladder under F4 stacking (rungs 0-5, F4 re-tuned per
rung with E10's budget, Holm across steps), E23 subgroup gaps (ProductCD, DeviceType, email domain,
cold start; FPR at the cost threshold as the fairness-relevant error) and E24 compute cost. Section 7
lists 8 decisions for you.

### 5. Demo backend (dashboard branch) — done, pushed `9fbbcea`, `30a9b92` (dashboard) and the export (phase3-run)

- services/api/demo_server.py: FastAPI app over a demo bundle of the final run, one endpoint per
  method of the dashboard's typed Api, with the mock's as-of and label-delay rules; threshold
  changes, feedback and audit kept in memory for the session; POST /api/score answers 501 (the
  final-run models score recorded transactions only), so the simulator keeps the illustrative model.
- dashboard/src/api/http/httpApi.ts: real-data implementation of the Api adapter; api/index.ts picks
  it with VITE_DATA_SOURCE=http (the mock stays the default); the top bar labels the data
  "IEEE-CIS test period, final run".
- Tests: 4 endpoint tests against a real uvicorn server (pass); 4 adapter tests with a fake fetch;
  the dashboard's 17 vitest tests and tsc pass.
- reports/demo_export.py (phase3-run) writes the bundle from the cached final inputs and stops unless
  the refitted MVAF reproduces the stored final-run scores and the rebuilt decisions reproduce E12's
  default cost. **Run on the real data:** MVAF reproduced exactly (max |diff| 0.0) and the E12
  default cost exactly ($99,263.19); bundle of 92,427 test rows written to
  data/interim/demo_final_bundle.parquet (drive E, not committed).
- **Smoke test of the backend on the real bundle:** every endpoint answers (alerts 0.3-0.5 s, others
  < 0.15 s); the label curtain works (a day's queue outcome appears 31 days later: day 165, 33 frauds
  caught by R1 at K = 50). It found one real issue, now fixed: the frozen decision thresholds have
  t_block = inf (on validation the cost chose never to block) and t_dis finite, which JSON cannot
  carry; the contract now has `t_block: number | null` (null = never block), handled by the server,
  the mock and the Settings form.

## Needs your decision

1. **Stage-2 draft, section 7** (8 questions: ladder form, rung 4 vs E10's F4, raw vs bounded history
   count, E15 primary model, counterfactual budget, extra E23 attributes, carbon estimate, E6 separate).
2. **Paper 2 reporting of the primary result:** the pre-registered MVAF - F3 (+0.199) must be stated
   with the F3 diagnosis (D118) and the F3-log sensitivity (D119: margin about +0.03), and with MVAF
   level with F4-F7 and behind B5 / F0. Wording is yours.
3. **Paper 1:** whether to propose time-normalised or windowed entity counts as a recommendation
   (D120 shows they help a little on validation but do not close E5's gap).
4. **Demo:** the E12 decision thresholds never block (t_block = inf on validation), so the demo
   shows Step-up / Hold but no Block on the test period; view-level reasons only (feature-level SHAP for 92k rows would be a heavy job), the
   simulator stays illustrative — acceptable for the demo?
5. Merging: nothing was merged; phase3-run and dashboard are pushed for your review (D32 review by a
   new session before any merge).

## Blockers

None. Notes: the endpoint tests use a synthetic bundle; the real bundle was smoke-tested by hand
(above); how to run it is in dashboard/README.md. httpx is not installed, so the endpoint tests start a real
uvicorn server instead of FastAPI's TestClient (no dependency added).
