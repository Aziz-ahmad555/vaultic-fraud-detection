# DRAFT — Stage-2 pre-registration: E15, the ablation-ladder final runs, E23, E24

**Status: DRAFT for Aziz's review. Not pre-registered, not logged in research/decisions.md, and
nothing in it may be run on the test period until Aziz approves it and it is logged as a
decision (then frozen and hashed like D101 / D108 / D112–D114).** Written 2026-10-11 by Claude
while Aziz was out (outing list item 4). Section 7 lists the choices that need his decision.

Sources: roadmap Phase 9.3 (faithfulness and stability), Phase 13 (experiment table E15, E23,
E24 and the main ablation table), D38 (immutable / derived / actionable features), D39 (EXP-1NN
configs; ablation stacking — Aziz has now chosen **F4 stacking**), D95 / D101 / D102 (pipeline
and final-run protocol), D117–D120 (post-hoc findings that motivate some safeguards below).

---

## 0. Protocol shared by every experiment here (as D101)

- **Splits and roles unchanged** (splits.yaml): train 1–120, validation 128–150 (calibrate_views
  144–146, calibrate_fused 147–150), test 151–182. Every choice (hyperparameters, thresholds,
  sample sizes, perturbation sizes, k ranges, counterfactual settings) is made on validation or
  fixed in this document before the test period is touched.
- **Frozen inputs:** each experiment gets a frozen config (status frozen-final, refused by the
  harness), hashed with the data files, data paths and library versions into a freeze record;
  the final runs use the same guards as EXP-200-final (clean tree on the frozen commit, origin
  check, log guard, O_EXCL lock, FINAL-STARTED pushed before anything is scored, D108–D115).
- **Statistics:** 5 seeds (0–4) where a model is trained; 1,000-resample bootstrap 95% CIs on the
  test period; paired bootstrap on the same resamples for every "A beats B"; two-sided p; Holm
  within each table; effect sizes (difference, relative difference, paired Cohen's d over seeds
  where available). Negative results are reported as such.
- **One final run per experiment**; a re-run needs a logged reason (FINAL-RERUN).

---

## 1. E15 — explanation faithfulness, stability, seed consistency, counterfactual validity (RQ6)

**Models explained.** (a) B5 (EXP-009, the strongest single model; TreeExplainer SHAP on its 5
seeds); (b) MVAF at the view level (the gate's weights x calibrated view logits, explain/
view_contrib.py), as a secondary analysis. The explained models are the frozen final-run models;
nothing is refitted for E15 except the seed-wise B5 models already defined by EXP-009.

**Alert sample.** The test rows B5 flags at its frozen cost threshold (chosen on validation,
Table 1), sampled at random with seed 0: **1,000 alerts** for deletion / insertion / stability /
seeds, **500 alerts** for counterfactuals (roadmap exit gate). If fewer alerts exist, all are used.

**1.1 Deletion test (primary).** For each alert, replace its top-k SHAP features (k = 1…10) by
their **training-period medians** (days 1–120) and record the drop in B5's risk. Summary per
alert: AOPC = mean over k of (risk(original) − risk(top-k replaced)).
Comparators on the same alerts and the same k:
- **random-k:** k features drawn at random from the model's features, 20 draws per alert and k,
  averaged (fixed seed);
- **permutation importance:** the global top-k by permutation importance (computed on validation,
  B5, 5 repeats), replaced the same way.
**Primary hypothesis H6a:** AOPC(SHAP) − AOPC(random) > 0 (paired bootstrap over alerts, Holm with
H6b below). **H6b:** AOPC(SHAP) − AOPC(permutation) > 0.

**1.2 Insertion test.** Start from all features at training medians, add features back in SHAP
order (k = 1…10); summary = mean over k of risk(top-k inserted) − risk(all medians). Same
comparators, reported with CIs (secondary, not in the Holm family).

**1.3 Stability.** For each alert, perturb TransactionAmt by ±1% and ±5% (recomputing only the
features derived from the amount, point-in-time as built) and compute the Jaccard overlap of the
top-3 SHAP features with the unperturbed explanation. Report mean Jaccard with CI per
perturbation; descriptive (no threshold), plus the share of alerts whose top-1 feature changes.

**1.4 Seed consistency.** Top-5 SHAP features of each alert under each of B5's 5 seeds; mean
pairwise Jaccard over the 10 seed pairs, with CI over alerts. Descriptive.

**1.5 Counterfactual validity.** On the 500-alert sample, counterfactuals with the counterfactual
module (explain/counterfactual.py) restricted to **actionable** features (D38); settings (search
budget, distance) fixed on validation alerts before the test period. Metrics: **validity** (risk
falls below the threshold), proximity (normalised L1 to the original), sparsity (features
changed), and share whose changed features are all actionable (by construction 100%; reported as
a check). **Exit-gate criterion (roadmap 9.3): validity > 90%, with its 95% CI reported.**

**1.6 MVAF view level (secondary).** Delete (mask) the top-1 / top-2 contributing views vs a
random available view; drop in MVAF risk; paired comparison, reported, not in the family.

**Phase 9 exit gate:** H6a supported (SHAP deletion beats random with CI excluding 0) and
counterfactual validity > 90%.

---

## 2. Final runs of the ablation ladder (E3, E4, E5, E8) under F4 stacking (RQ1)

The roadmap's main ablation table adds behavioral features at feature level and the temporal,
graph and anomaly views by stacking. With F4 (LightGBM stacking, the E10 baseline) as the
stacker (Aziz's D39 choice):

| rung | model | inputs | ID |
|---|---|---|---|
| 0 | B5 | strong tabular (feature level) | E1 (Table 1, already final) |
| 1 | + behavioral | B5 + behavioral features, retuned with B5's budget (EXP-103-tuned) | E3 |
| 2 | + temporal | F4 over the calibrated views {tabular, behavioral, temporal} | E4 |
| 3 | + graph | F4 over {tabular, behavioral, temporal, graph} | E5 (stacking form) |
| 4 | + anomaly | F4 over all five views | E8 (stacking form) |
| 5 | MVAF (full) | all five views, MVAF | E10 (already final) |

- **Stacking rungs (2–4)** use the D95 pipeline unchanged (cross-fitted views, calibrate_views,
  gate rows fit + tune); F4 is **tuned per rung** on the inner split with the same budget as
  E10's F4 (50 Optuna trials, D95). Rung 4 uses the same inputs as E10's F4 but is tuned and run
  within this table, so the ladder is self-contained (its E10 counterpart is shown for reference).
- **Feature-level rows (secondary):** E5 = B5 + graph features (EXP-105-tuned) and E8 = B5 +
  anomaly scores (EXP-108-tuned) as harness --final runs, reported beside the ladder, because
  D99/D100 developed them in that form.
- **Comparisons:** each rung minus the previous one (rungs 1–4 vs 0–3) and rung 5 minus rung 4,
  test PR-AUC (primary), recall at 1% FPR and total cost; paired bootstrap; Holm across the five
  steps per metric. Subgroups as in E10 (cold start / history, has_identity, graph available,
  number of views).
- **Hypothesis H1 (RQ1, per step):** adding the view improves test PR-AUC (difference > 0, Holm
  p < 0.05). Reported per step; no step is dropped after seeing results.
- **Safeguard from D117–D120 (fixed now, not after seeing results):** F4 receives the context
  columns like every fusion method; the unbounded history count is passed **as in E10** (raw), so
  the ladder stays comparable with E10. The feature-level E5 row keeps its graph features as
  built (the E5 drift diagnostic, D120, is exploratory and changes no frozen view). Both points
  are stated as threats to validity.
- **Out of scope here:** E6 (graph leakage settings A/B/C × label delay, Paper 1) gets its own
  pre-registration; E7 (GNN) is COULD (D31).

---

## 3. E23 — subgroup performance gaps (fairness check)

No new training: computed from the stored final-run predictions (Table 1, EXP-200-final) and
the raw columns of the test period.

- **Methods:** B5, F0, F4, MVAF.
- **Groups:** ProductCD (W, C, R, H, S); DeviceType (desktop, mobile, missing); purchaser email
  domain (the 5 most frequent in the **training** period, other, missing); cold start vs history.
- **Metrics per group:** PR-AUC with bootstrap CI; recall at the method's global 1%-FPR threshold
  (chosen on all test rows' legitimate scores, as in E10) and **false-positive rate at the frozen
  cost threshold** (legitimate customers wrongly flagged — the fairness-relevant error);
  prevalence and row counts.
- **Gap statistics:** max − min and max / min across the groups of each attribute, with
  bootstrap CIs; MVAF − B5 per group, paired, Holm across the groups of each attribute.
- Groups with < 30 frauds are reported but flagged as unreliable.
- Statement (roadmap): IEEE-CIS has no demographic data, so this is a performance-gap check, not
  a fairness audit.

---

## 4. E24 — compute and carbon cost per model (reporting)

- **Hardware:** CPU model, cores, RAM, OS (recorded automatically).
- **Training time per model:** wall-clock for one seed's fit on days 1–120 with the frozen
  settings, timed **on validation-only refits** (no test data needed), 3 repetitions, median;
  plus tuning time from the tuning logs (Optuna trials x mean trial time). Models: B1–B6, the
  five views, F1–F7, MVAF, F0.
- **Inference time:** per 1,000 rows on validation, median of 5 repetitions.
- **Energy and carbon (estimate):** energy = measured CPU time x the CPU's TDP x 1.0 (no PUE;
  laptop); carbon = energy x the grid carbon intensity of Pakistan from a cited public source
  (value and year stated). No new dependency (codecarbon) without approval.
- Reported as a table; no hypothesis test.

---

## 5. Run order and compute budget (estimate)

1. E24 timing (validation only; ~1–2 h) — can run any time, no freeze needed.
2. E23 (no training; minutes) — after freeze.
3. Ablation ladder: rungs 2–4 F4 tuning on validation (~1 h), then one final run (~3 h, view
   table rebuilt as in EXP-200-final); E5 / E8 feature-level --final (~30 min each).
4. E15 (~2–4 h: SHAP on 1,000 alerts x 5 seeds, 20 random draws, counterfactual search).

One heavy job at a time; nothing alongside a final run.

---

## 6. What is fixed by this document once approved

The alert sample sizes and seed, k = 1…10, training medians as the reference, 20 random draws,
permutation importance on validation, ±1% / ±5% perturbations, top-3 / top-5 Jaccard, the
counterfactual feature restriction, the ladder rungs and F4's tuning budget, the subgroup
definitions (email domains from the training period), the 30-fraud flag, the E24 measurement
recipe, the hypotheses and Holm families above.

---

## 7. Decisions needed from Aziz before this is pre-registered

1. **Ladder form:** stacking rungs over views (as drafted) with E3 at feature level for rung 1 —
   or E3 also as a stacking rung F4{tabular, behavioral}? (Draft: E3 feature level, as the roadmap
   table says "behavioral: feature level".)
2. **Rung 4 vs E10's F4:** re-tune and re-run within the ladder (draft), or reuse E10's F4
   result as rung 4 (no extra run, but tuned once for all five views only)?
3. **Bounded history count in the ladder:** keep the raw count as in E10 (draft, comparability)
   or use the bounded form explored post-hoc in D119 (would make the ladder differ from E10)?
4. **E15 primary model:** B5 (draft) or MVAF's tabular view? And is the MVAF view-level analysis
   wanted as secondary?
5. **Counterfactual budget:** 500 alerts (roadmap) — confirm; and whether actionability should
   follow D38 exactly.
6. **E23 attributes:** add card4 / card6 (card network / type)? Draft keeps the roadmap's four.
7. **E24 carbon:** accept a TDP-based estimate and a cited Pakistan grid factor, or skip carbon
   and report time only?
8. **E6 (Paper 1):** separate pre-registration (draft assumption), or add here?
