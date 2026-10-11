# DRAFT v2 — Stage-2 pre-registration: E15, the ablation ladder, E23, E24

**Status: DRAFT v2 for Aziz's final review. Not yet pre-registered or logged in
research/decisions.md; nothing in it runs on the test period until Aziz approves it and it is
logged as a decision, then frozen and hashed like D101 / D108 / D112–D114.** v1 written
2026-10-11 during the outing; v2 the same day with Aziz's answers to the eight open questions
(section 7 records them). E6 (graph leakage settings, Paper 1) gets its own pre-registration.

Sources: roadmap Phase 9.3 (faithfulness and stability), Phase 13 (E15, E23, E24 and the main
ablation table), D38 (immutable / derived / actionable features), D39 (EXP-1NN configs; stacking
by F4), D89 (view definitions), D95 / D101 / D102 (pipeline and final-run protocol), D117–D120
(post-hoc findings behind some safeguards).

---

## 0. Protocol shared by every experiment here (as D101)

- **Splits and roles unchanged** (splits.yaml): train 1–120, validation 128–150 (calibrate_views
  144–146, calibrate_fused 147–150), test 151–182. Every choice (hyperparameters, thresholds,
  sample sizes, perturbation sizes, k ranges, counterfactual settings) is made on validation or
  fixed in this document before the test period is touched.
- **Frozen inputs:** each experiment gets a frozen config (status frozen-final, refused by the
  harness), hashed with the data files, data paths and library versions into a freeze record;
  the final runs use the EXP-200-final guards (clean tree on the frozen commit, origin check, log
  guard, O_EXCL lock, FINAL-STARTED pushed before anything is scored; D108–D115).
- **Statistics:** 5 seeds (0–4) where a model is trained; 1,000-resample bootstrap 95% CIs on the
  test period; paired bootstrap on the same resamples for every "A beats B"; two-sided p; Holm
  within each table; effect sizes (difference, relative difference, paired Cohen's d over seeds
  where available). Negative results are reported as such.
- **Models explained or compared are the frozen final-run models** (EXP-200-final, Table 1);
  nothing is re-tuned except where stated (ladder rungs 1–4).
- **One final run per experiment**; a re-run needs a logged reason (FINAL-RERUN).

---

## 1. E15 — explanation faithfulness, stability, seed consistency, counterfactual validity (RQ6)

**Explained system: MVAF** (primary), with **B5 alongside** (secondary).

**How MVAF is explained.** Two levels, both from the frozen final-run models:
- *view level:* the gate's contribution of each available view, c_v = w_v × logit(p_v) (gate
  weight × calibrated view logit; explain/view_contrib.py);
- *feature level:* TreeExplainer SHAP (log-odds) on each XGBoost view model of the fixed fold
  (the models that score the test period, D77), weighted by that view's gate weight:
  a_f = Σ_v w_v × s_{v,f}. This passes over the per-view calibrators, which are monotone, so it
  is an approximation; the deletion and insertion tests below check it end to end.
B5 is explained by TreeExplainer SHAP on its 5 seeds (averaged).

**Alert samples** (seed 0, drawn once, before any explanation is computed):
- MVAF: test rows whose frozen E12 decision is step-up, hold or block — **1,000 alerts** for
  deletion / insertion / stability / seeds, **500 alerts** for counterfactuals;
- B5: test rows above B5's frozen cost threshold (Table 1) — same sizes.
If fewer alerts exist, all are used.

**1.1 Deletion test (primary).** For each alert, replace its top-k features (k = 1…10, by |a_f|
for MVAF, by |SHAP| for B5) with their **training-period medians** (days 1–120) and recompute
the **full** score: for MVAF the affected view models, their calibrators and the gate (5-seed
mean); for B5 the model (5-seed mean). Summary per alert: AOPC = mean over k of
(risk(original) − risk(top-k replaced)). Comparators on the same alerts and k:
- **random-k:** k features drawn at random from the explained model's inputs, 20 draws per alert
  and k (fixed seed), averaged;
- **permutation importance:** the global top-k by permutation importance of the explained system
  (computed on validation, 5 repeats), replaced the same way.

Hypotheses (paired bootstrap over alerts):
- **H6a (MVAF):** AOPC(attribution) − AOPC(random) > 0;
- **H6b (MVAF):** AOPC(attribution) − AOPC(permutation) > 0.
Holm across H6a and H6b. The same two tests for B5 form a second family (Holm within it),
reported alongside.

**1.2 View-level deletion (MVAF).** Mask the top-1 and top-2 contributing views (by |c_v|) vs
one random available view; drop in MVAF risk; paired comparison, reported with CIs (not in the
H6 family).

**1.3 Insertion test.** Start from all features at training medians and add features back in
attribution order (k = 1…10); summary = mean over k of risk(top-k inserted) − risk(all medians).
Same comparators; reported with CIs (secondary).

**1.4 Stability.** Perturb TransactionAmt by ±1% and ±5% (recomputing only the inputs derived
from the amount — the list is fixed in the implementation-details entry before any run) and
compute the Jaccard overlap of the top-3 features with the unperturbed explanation, plus the
share of alerts whose top-1 feature changes. Mean with CI; descriptive.

**1.5 Seed consistency.** MVAF: top-5 features under each of the 5 gate seeds (the view models
are one fixed-fold model each, so seeds act through the gate weights); B5: top-5 under each of
its 5 seeds. Mean pairwise Jaccard over the 10 seed pairs, with CI over alerts. Descriptive.

**1.6 Counterfactual validity.** On the 500-alert samples, counterfactuals with
explain/counterfactual.py restricted to **actionable** features (D38), scored end to end (MVAF:
views, calibrators, gate; B5: the model); search settings fixed on validation alerts before the
test period. Metrics: **validity** (risk falls below the decision boundary that triggered the
alert), proximity (normalised L1), sparsity (features changed), share of changed features that
are actionable (100% by construction; reported as a check).
**Exit criterion (roadmap 9.3): validity > 90% for MVAF, with its 95% CI.**

**Phase 9 exit gate:** H6a supported for MVAF (CI excludes 0) and MVAF counterfactual validity
> 90%.

---

## 2. Ablation ladder (E3, E4, E5, E8) — cumulative views under F4 stacking (RQ1)

With the D89 view definitions (tabular view = B5's features without the label-derived ones;
behavioral view = behavioral features + the label-derived ones; temporal, graph, anomaly as in
D95), each rung adds one view and is fused by **F4** (LightGBM stacking, the E10 baseline):

| rung | inputs (calibrated views, plus the context as in E10) | ID | run |
|---|---|---|---|
| 1 | tabular | — | new |
| 2 | tabular + behavioral | E3 | new |
| 3 | + temporal | E4 | new |
| 4 | + graph | E5 | new |
| 5 | + anomaly (all five views) | E8 | **E10's F4, reused** (no re-run) |
| ref | MVAF (all five views) | E10 | reused |
| ref | B5 (strong tabular, feature level) | E1 / Table 1 | reused |

- **Rungs 1–4** use the D95 pipeline unchanged (cross-fitted views, calibrate_views, gate rows
  fit + tune) and the same 5 seeds; F4 is **tuned per rung on the inner split with E10's F4
  budget** (50 Optuna trials, D95). Rung 5 is E10's F4 (tuned with that budget on all five views);
  it and the two references are taken from the EXP-200-final run, so no test-period number is
  recomputed for them.
- **Comparisons:** each rung minus the previous one (rungs 2–5 vs 1–4: four steps), test PR-AUC
  (primary), recall at 1% FPR and total cost (threshold chosen on calibrate_fused); paired
  bootstrap on the same rows and resamples as E10 (the rung-5 scores are the stored E10 scores);
  Holm across the four steps per metric. MVAF and B5 are shown as reference rows, with MVAF −
  rung 5 and B5 − rung 5 reported (these equal E10's comparisons; not a new family). Subgroups as
  E10 (cold start / history, has_identity, graph available, number of views), descriptive.
- **Hypothesis H1 (RQ1, per step):** adding the view improves test PR-AUC (difference > 0, Holm
  p < 0.05). Reported for every step; no step is dropped after seeing results.
- **History count (D117–D119):** F4 receives the context with the **raw** ctx_hist_n_past, as in
  E10, so the ladder stays comparable. A **labelled sensitivity analysis** repeats rungs 1–5 with
  the bounded count log1p(min(count, gate-row max)) (rung 5 then refitted with E10's F4 settings,
  not re-tuned); reported separately, not in the H1 family. (Trees are close to invariant to this
  monotone transform, so little change is expected.)
- **Out of scope:** feature-level E3 / E5 / E8 rows (B5 + features; development results stay as
  reported in D99 / D100), E6 (separate pre-registration), E7 (COULD, D31).

---

## 3. E23 — subgroup performance gaps (fairness check)

No new training: computed from the stored final-run predictions (Table 1, EXP-200-final) and the
raw columns of the test period.

- **Methods:** B5, F0, F4, MVAF.
- **Groups (roadmap list only):** ProductCD (W, C, R, H, S); DeviceType (desktop, mobile,
  missing); purchaser email domain (the 5 most frequent in the **training** period, other,
  missing); cold start vs history.
- **Metrics per group:** PR-AUC with bootstrap CI; recall at the method's global 1%-FPR threshold
  (as in E10); **false-positive rate at the frozen cost threshold** (legitimate customers wrongly
  flagged — the fairness-relevant error); prevalence and row counts.
- **Gap statistics:** max − min and max / min across the groups of each attribute, with bootstrap
  CIs; MVAF − B5 per group, paired, Holm across the groups of each attribute.
- Groups with < 30 frauds are reported but flagged as unreliable.
- Statement (roadmap): IEEE-CIS has no demographic data, so this is a performance-gap check, not
  a fairness audit.

---

## 4. E24 — compute cost per model (reporting)

- **Hardware:** CPU model, cores, RAM, OS (recorded automatically).
- **Training time per model:** wall-clock for one seed's fit on days 1–120 with the frozen
  settings, timed **on validation-only refits** (no test data), 3 repetitions, median; tuning time
  from the tuning logs (trials x mean trial time). Models: B1–B6, the five views, F1–F7, MVAF, F0,
  and the ladder's F4 rungs. Reported as **compute hours**.
- **Inference time:** per 1,000 rows on validation, median of 5 repetitions.
- **Carbon: a labelled estimate only** — energy = measured CPU time x the CPU's TDP (no PUE,
  laptop), times a cited grid carbon intensity for Pakistan (source and year stated); marked as an
  estimate, not a measurement. No new dependency (codecarbon).
- Reported as a table; no hypothesis test.

---

## 5. Run order and compute budget (estimate)

1. E24 timing (validation only; ~1–2 h) — any time, no freeze needed.
2. E23 (no training; minutes) — after freeze.
3. Ladder: F4 tuning for rungs 1–4 on validation (~1 h), then one final run (~3 h, view table
   rebuilt as in EXP-200-final) for rungs 1–4 and the bounded-count sensitivity.
4. E15 (~3–5 h: SHAP for five view models and B5 on 1,000 alerts, end-to-end rescoring for
   deletion / insertion with 20 random draws, counterfactual search on 500 alerts each).

One heavy job at a time; nothing alongside a final run.

---

## 6. Fixed by this document once approved

Alert definitions, sample sizes (1,000 / 500) and seed; k = 1…10; training medians as the
reference; 20 random draws; permutation importance on validation; ±1% / ±5% perturbations;
top-3 / top-5 Jaccard; the actionable-feature restriction; the ladder rungs, F4's tuning budget
and the reuse of E10's F4 and references; the raw history count in the primary ladder and the
bounded count as sensitivity; the E23 groups (email domains from the training period) and the
30-fraud flag; the E24 recipe; the hypotheses and Holm families above.

---

## 7. Aziz's answers to the v1 questions (2026-10-11)

1. Ladder: cumulative over the new view definitions (tabular → +behavioral → +temporal → +graph →
   +anomaly) via F4, with MVAF and B5 as references.
2. Top rung: reuse E10's F4.
3. History count: raw; bounded only as a labelled sensitivity analysis.
4. E15: explain MVAF (gate weights + SHAP on the tree view models), with B5 alongside.
5. Counterfactuals: 500 alerts.
6. E23: no attributes beyond the roadmap's.
7. E24: compute hours and hardware; carbon only as a labelled estimate.
8. E6: its own separate pre-registration.

## 8. Remaining implementation details (to be fixed in a D102-style entry before any run)

The exact list of amount-derived inputs recomputed in 1.4; the counterfactual search settings
(chosen on validation alerts); the grid carbon-intensity source.
