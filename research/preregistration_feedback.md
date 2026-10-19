# Pre-registration: feedback loop with champion–challenger retraining (Phase C (b), D210)

**Status: PRE-REGISTERED as D210 (2026-10-19), approved by Aziz with amendment 1 (D211).** Written and committed
before any simulation code is written or run. Nothing here touches the test period except a single
`--final` run, started only with Aziz's go-ahead. Any change after approval is logged as an
amendment (D2xx) **before** the run it affects; a change made after seeing test results is reported
as post hoc and never replaces the primary analysis.

Sources: roadmap Phase 10.3 (retraining policies P0–P3, champion–challenger gate) and 10.4
(feedback); D44 (Phase 10 drift code: signals, detectors, label-delay simulator, policies), D45
(cost model), D58 (fit / threshold / gate split), D101 (protocol), D179 (refits reproduce the
final run). The code in `src/vaultic/drift/` (D44) is the starting point; this document fixes the
settings D44 left "needs confirmation" (gate margin, retry rule, training window).

---

## 0. Protocol (shared with D101)

- **Splits unchanged** (`experiments/configs/splits.yaml`): train 1–120, gap 121–127, validation
  128–150, test 151–182. Day = `TransactionDT // 86400`.
- **Statistics:** 5 seeds (0–4); 1,000-resample bootstrap 95% CIs over the evaluation rows;
  paired bootstrap on the same resamples for every "A beats B"; two-sided p; Holm within each
  table; effect sizes (difference and relative difference). Negative results are reported as such.
- **Frozen inputs:** a frozen config (`EXP-310-final`, status frozen-final) hashed with the data
  files, library versions and this document into a freeze record (as `research/frozen_stage2.md`);
  the final run uses the existing final-run guards (clean tree on the frozen commit, FINAL-STARTED
  logged and pushed before anything is scored, one run; a re-run needs a logged reason).
- **Product layer untouched:** no frozen research result, config or freeze record changes; this
  is a new experiment with its own IDs (EXP-310-dev, EXP-310-final).

## 1. Question and hypotheses

**Question.** When labels arrive late, does retraining the deployed model on newly matured labels,
behind a champion–challenger gate, improve fraud ranking over the test period compared with never
retraining, and how does the answer depend on the label delay?

- **H-FB1 (primary).** At the system's label delay **L = 30 days**, at least one gated retraining
  policy (P1, P2 or P3) has a higher primary metric than P0 on the test days, by a paired
  bootstrap (Holm over the three comparisons, α = 0.05).
- **H-FB2.** At L = 30, no gated policy has a higher total cost than P0 on the test days (paired
  bootstrap of the cost difference; a cost increase whose 95% CI lies entirely above 0 rejects it).
- **H-FB3 (delay).** The PR-AUC gain of P1 over P0 does not increase with the delay:
  gain(L=0) ≥ gain(L=7) ≥ gain(L=30) ≥ gain(L=60) (point estimates; reported with CIs, a
  descriptive check, not a significance test).
- **H-FB4 (gate).** Gated P1 has a total cost on the test days no higher than ungated P1 (every
  challenger accepted) at L = 30.

**Predicted structural results, stated before any run** (from the arithmetic in section 4): at
L = 60 no challenger can contain a post-training label within the simulation span, so P1–P3
should equal P0 (no promotion); at L = 30 at most four gate decisions fall in the span (days 156,
163, 170, 177). These are reported as results, not removed.

## 2. Primary metric and cost

- **Primary:** PR-AUC of the scores actually served by the policy on **test days 151–182**,
  computed after the fact with every test label. (All test labels have matured by the time the
  evaluation runs; the simulation itself only ever sees labels matured at the simulated time,
  section 6.)
- Estimate: mean over the 5 seeds of the per-seed PR-AUC; 95% CI by bootstrap over the test rows
  (the same resampled rows for all seeds, mean over seeds per resample); "A beats B" by paired
  bootstrap on the same resamples.
- **Cost (co-primary for H-FB2, H-FB4):** the D44 / D45 cost of the served decisions on the test
  days: missed fraud amount + C_FP × false positives + C_rev × reviews, with C_FP = $10,
  C_rev = $5, a review whenever the served score reaches the serving model's cost threshold
  (`eval.stats.cost_breakdown`, as in `drift/policies.py`).
- **Secondary:** PR-AUC per test week (151–157, 158–164, 165–171, 172–182); number of retrain
  attempts, promotions and rejections (with reasons); detector alarm days (P2, P3); the warm-up
  days 128–150 reported descriptively (not in any hypothesis).

## 3. Models

- **Champion at deployment:** the frozen B5 XGBoost (EXP-009 settings and its five seed models,
  trained on days 1–120; its cost threshold chosen on calibrate_fused 147–150). Seed s of the
  simulation uses seed model s.
- **Challenger:** the **same B5 XGBoost configuration** (EXP-009 hyperparameters, the `b5`
  feature set, seed s), retrained on matured labels. "On frozen views" (Aziz): the five view
  models, the MVAF gate and the E12 layer are not retrained or used; the challenger's inputs come
  from the frozen point-in-time feature pipeline. MVAF as the deployed model is out of scope here
  (it would need its own amendment).
- **Parity before anything is compared:** at L = 30 the champion's scores must reproduce the
  stored EXP-009 validation predictions (dev run) and the stored EXP-200-final `score_B5` test
  predictions (final run) within 1e-6, or the run stops.

## 4. Simulation span, cadence, design vs evaluation

| Run | Champion | Simulated days | Used for |
|---|---|---|---|
| **EXP-310-dev** (design) | B5 configuration fitted on days 1–60 (5 seeds) | **61–120** (training period only) | checking the code end to end and that the gate is not degenerate; no number from it enters any hypothesis |
| **EXP-310-final** (evaluation) | the frozen B5 (days 1–120) | **121–182**: warm-up 121–150, evaluation **151–182** | the hypotheses (test days only), once, with Aziz's go-ahead |

- The dev run never reads a validation or test row (a guard refuses any row after day 120).
  Every setting below is fixed now; if the dev run shows a setting is broken, the change is
  logged as an amendment before EXP-310-final, and stated in the report.
- **Cadence (addresses the 32-day test period).** Retrain attempts every **7 days** from
  deployment: in EXP-310-final on days **128, 135, 142, 149, 156, 163, 170, 177** (8 attempts, 4
  of them inside the test period); in EXP-310-dev on days 68, 75, …, 117. A monthly cadence would
  give at most one attempt inside the test period. Monthly P1 (every 30 days) is reported as a
  secondary analysis only.
- **Why the warm-up starts at 121:** the champion is trained on 1–120, so deployment is day 121;
  the warm-up lets matured labels accumulate and the first gates happen before the test days.
  Scores served on 121–150 are not part of any hypothesis.

## 5. Policies, delays, gate

**Policies** (D44 code, settings fixed here):
- **P0** never retrain.
- **P1** attempt a retrain at every 7-day slot (above).
- **P2** label-free: PSI of the served scores over the trailing 7 days vs the champion's reference
  scores (its threshold slice) > 0.2, checked daily. An alarm makes the **next 7-day slot** an
  attempt; a pending alarm stays pending (one attempt per slot) until a challenger is accepted, at
  most 6 attempts per alarm (D44 retry rule, confirmed here).
- **P3** label-based: ADWIN (δ = 0.002, D44) on the 0/1 error stream of the serving model's
  decisions in label-arrival order (time + L), checked daily; same slot, pending and retry rule as
  P2.
- **Deliberate change from D44:** D44's P2 / P3 attempted on the alarm day itself. Here every
  policy attempts only on the shared 7-day slots, so P1, P2 and P3 differ only in *whether* they
  retrain at a slot, never in *when*; challengers are shared (section 8) and the alarm-to-attempt
  wait is at most 6 days.
- **Ungated P1** (secondary, H-FB4): P1 with every challenger accepted.

**Label delays:** L ∈ {0, 7, 30, 60} days, each a separate simulation of every policy × 5 seeds.

**Training data of a challenger at attempt day t:** every row with `TransactionDT + L·86400 ≤
t·86400` (labels matured by the start of day t), all history from day 1 (D44 default; a 60-day
window is NOT run here). Split by time (D58): fit = all but the most recent 14 matured days;
threshold slice = the next 7 days (cost threshold by `choose_cost_threshold`, and P2's reference
scores); gate window = the most recent 7 matured days.

**Gate (both conditions, on the gate window, after removing every row the current champion or
the challenger trained on):**
1. **PR-AUC:** challenger − champion gain ≥ **0.005** (point estimate) **and** the 95% paired
   bootstrap CI of the gain (1,000 resamples) entirely above 0;
2. **Cost:** the challenger's cost on the window (at its own threshold) ≤ the champion's cost (at
   its threshold).
3. **Evidence floor:** fewer than **50 fraud rows** in the window after the removal → rejected as
   "insufficient evidence" (logged).
A rejected challenger is logged with its reason; the champion keeps serving. Champion and
challenger are compared on the same rows. The gate window length (7 days), the threshold slice
(7 days) and the margin (0.005) are fixed here and not tuned.

**Arithmetic of the span (why the structural predictions hold):** the gate window is the 7
matured days ending at t − L; after removing the frozen champion's training rows (days 1–120) it
keeps only days 121 … t − L, so a gate decision is possible only when t − L ≥ 121 (and those days
hold ≥ 50 frauds; about 90 frauds a day on validation). With the 7-day attempt slots: L = 0 and
L = 7 from day 128 (8 attempts each); L = 30 on days 156, 163, 170, 177 (4); L = 60: none (t ≥ 181
is after the last slot, 177).

## 6. Leakage prevention

- **Label curtain:** a label is used (training, threshold, gate, ADWIN) only if
  `its_time + L ≤ the simulated time` (start of the attempt day for training; the arrival time
  for ADWIN). No row with `time ≥` the attempt day enters training.
- **Serving:** the scores of day d come from the model serving at the start of day d; a model
  trained at day t serves from day t on. A transaction never influences its own score.
- **Label-derived inputs follow the arm's delay.** The three label-derived B5 columns
  (`uid_fraud_known`, `uid_fraud_rate_known`, `uid_n_labels_known`, D89) are stored with a 30-day
  delay. For L = 0, 7 and 60 they are recomputed once, before any run, with the same
  point-in-time code at delay L (for L = 60 the stored 30-day values would reveal labels the loop
  is not allowed to know yet); every other input is label-free. The champion's weights are
  unchanged; its served scores at L ≠ 30 therefore use the arm's inputs (parity is checked at
  L = 30 only).
- **Tests before any run** (hand-made data with known answers, plus real-data checks): (i) no
  training, threshold or gate row violates the curtain; (ii) flipping every label not yet matured
  at t changes no model and no served score up to t; (iii) truncating the data after day t
  changes nothing up to t; (iv) the recomputed label-derived columns at delay L use no label with
  `its_time + L > t`; (v) the dev run refuses any row after day 120; (vi) the gate's row removal
  and evidence floor on a toy example.
- **Test period:** read only by EXP-310-final behind the final-run guards.

## 7. What counts as a negative result (reported in every case)

- H-FB1 not supported: no gated policy beats P0 at L = 30 after Holm (including all challengers
  rejected, or gains whose CI includes 0 or is negative).
- H-FB2 not supported: a gated policy raises cost (CI entirely above 0).
- H-FB3 not supported: the gain is not monotone in L.
- H-FB4 not supported: ungated P1 costs less than gated P1.
- The predicted structural nulls (L = 60, at most four gates at L = 30) are reported as findings, with the
  span arithmetic, and as a limitation of a 32-day test period, not hidden or removed.
- The champion was trained on labels a real deployment at day 121 could not have had (days
  91–120 at L = 30); this favours P0 and is stated as a limitation.

## 8. Compute and outputs

- Challengers are deterministic given (attempt day, L, seed), so P1, P2, P3 and ungated P1 share
  cached fits. The evidence floor is checked before a challenger is fitted (no gate is possible →
  no fit, logged as "insufficient evidence"), so EXP-310-final needs at most (8 + 8 + 4 + 0)
  attempt days over L = 0 / 7 / 30 / 60 × 5 seeds = 100 B5 fits, plus the dev run. Ungated P1
  accepts a challenger only where one was fitted. Run detached, one job at a time, logs on E:.
- Outputs per run: `metrics.json` (per policy × L: test PR-AUC mean, std, 95% CI; cost; paired
  comparisons with Holm), `events.json` (every attempt: day, L, seed, accepted, reason, gate
  numbers), served scores (parquet, git-ignored), and a report `research/tables/feedback/` as JSON
  and markdown; one line in `research/experiment_log.md`.

## 9. Open points for Aziz (before approval)

1. Confirm the reading of "challenger = B5 XGBoost on frozen views" (section 3).
2. Confirm the gate margin 0.005 PR-AUC plus a CI above 0, and the 50-fraud evidence floor.
3. Confirm recomputing the three label-derived columns per delay (section 6) instead of dropping
   them.
4. Confirm the EXP-310-dev design span (champion on 1–60, simulate 61–120).

---

## Amendment 1 (D211, 2026-10-19, before any code): Aziz's answers and four additions

**Section 9 answered (Aziz):** (1) challenger = the B5 XGBoost retrained on matured labels; views,
MVAF and E12's fitted components untouched; (2) gate margin 0.005, CI entirely above 0 and >= 50
frauds approved; a gate that rejects every challenger is a reportable finding; (3) the three
label-derived inputs are recomputed per delay, not dropped; (4) design span approved. The shared
weekly slots for P2 / P3 are approved as an amendment to D44 (D44's same-day attempt is replaced
for this experiment).

**a) Second use of the test period.** Days 151-182 were already used once, in the stage-1 final
run (EXP-200-final) and the stage-2 final runs. EXP-310-final is a second use of the same days
under this new pre-registration. No B5 setting was chosen using test results: B5's hyperparameters
come from validation tuning (EXP-009, D-log), its seed models from days 1-120, and every setting of
this experiment is fixed in this document before EXP-310-final runs.

**b) PSI reference resets after each promotion.** P2 compares the scores served by the CURRENT
champion with that champion's own reference scores (its threshold slice). After a promotion the
reference is replaced by the new champion's threshold-slice scores and the trailing 7-day window
restarts with the new champion's served days; no P2 alarm is possible until it has served 7 days.
Otherwise the change of model would itself trigger the alarm.

**c) Cost model: E12's, not the binary review cost.** The D44 / D45 policy code used the E10 binary
cost (flag = review: legitimate flagged $15, fraud flagged $5, missed fraud its amount). That equals
E12's cost model only for its allow / block actions; E12 (trust.decision.CostModel, D43 / D45 / D46
defaults: C_FP $10, C_rev $5, step-up $0.50 + $1 friction for a legitimate customer and 90% of fraud
stopped, hold $5 and 100% stopped) also has step-up and hold. **This experiment uses E12's cost
model and decision rule:** every serving model's action is E12's decide(p, amount, thresholds) on
its expected loss p x amount (no conformal sets or disagreement for B5: the expected loss alone
decides), with thresholds chosen by E12's choose_thresholds (same grid) on the model's threshold
slice; cost = E12's total_cost of the served actions; P3's error stream uses alert = step-up or
higher (D159). B5's scores are not calibrated; the thresholds are chosen on its own scores, so the
expected-loss scale is absorbed by the threshold choice (stated). The champion's thresholds: in
EXP-310-final, chosen once on calibrate_fused (days 147-150), like the frozen system's other
thresholds (part of the champion's head start, stated as a limitation); in EXP-310-dev the champion
is fitted on days 1-56 and its thresholds and P2 reference are chosen on days 57-60 (it is still
built from days 1-60, deployed day 61, with no label from the simulated span).

**d) What varies across the 5 seeds.** Only the XGBoost random seed s (0-4), which drives B5's row
and column subsampling (subsample 0.76, colsample_bytree 0.90): the champion is the frozen seed-s
B5 model (dev: the seed-s fit on days 1-56) and every challenger of simulation s is fitted with seed
s. Identical across seeds: the data, splits, delays, slots, the recomputed inputs, the threshold
procedure, the gate's bootstrap resamples (seed 0, the same resampled gate rows for every seed and
attempt of a given day and delay) and the evaluation bootstrap resamples (seed 0). The five
simulations can therefore make different gate decisions; each policy's metric is the mean over the
five, with CIs by the shared bootstrap. Parity (section 3) compares the mean of the five seed-model
champion scores with the stored score_B5, which is that same 5-seed mean.
