# Feedback loop: champion-challenger retraining under label delay (EXP-310)

## What it is
A simulation of retraining a deployed fraud model as labels arrive late (roadmap Phase 10.3 / 10.4):
a champion (the frozen B5 model) serves from day 121; challengers are the same configuration
retrained on matured labels and replace the champion only if a gate accepts them. Policies: P0 never
retrain, P1 weekly, P2 on a PSI drift alarm, P3 on an ADWIN error alarm (plus ungated and monthly P1
as secondary). Label delays L = 0, 7, 30, 60 days. Pre-registered in D210 with amendments
(D211, D214-D216); one final run on test days 151-182; results recorded as-is (D218).

## Why we built it
Labels in fraud arrive weeks late. The question is whether scheduled or drift-triggered retraining
helps a model when its labels are delayed, and whether a gate stops harmful promotions.

## How it works (simple)
At each weekly slot a challenger is fitted on labels already known at that time (its_time + L < t).
The gate accepts it only if its PR-AUC gain is at least 0.005 with a bootstrap CI above 0, its cost is
not higher, and the gate window (rows neither model trained on) holds at least 50 frauds. Every
policy is run for 5 seeds at every delay and compared with P0 on the test days.

## Key results (with sources)
From `research/tables/feedback/exp310_final.json` (report: `research/tables/feedback/exp310_report.md`):
- **H-FB1 (PR-AUC gain over P0 at L = 30, Holm): NOT SUPPORTED.** P1 +0.00049 [-0.00027, +0.00123],
  Holm p 0.69; P2 = P0; P3 +0.00016, Holm p 1.0.
- H-FB2: supported in the sense of **no cost increase** vs P0 (P1 -598, CI -1,120 to +5); not
  evidence of a cost reduction.
- H-FB3 (gain does not grow with delay, descriptive): supported; P1 gain +0.0491 / +0.0220 /
  +0.0005 / 0 at L = 0 / 7 / 30 / 60. Partly true by construction (L = 60 is 0, L = 30's CI includes 0).
- H-FB4 (gated minus ungated P1 cost, point estimate): supported, -415, with a CI from -1,282 to +531
  that includes 0.
- Parity before the simulation: the five champion models reproduced the stored scores on all
  92,427 test rows (`research/tables/feedback/exp310_report.md`).

## Limitations
From `research/tables/feedback/exp310_report.md`:
- The 32-day test period limits gate decisions: none possible at L = 60, at most 4 per seed at
  L = 30.
- The champion has a head start (trained on labels a day-121 deployment could not have had;
  thresholds from days 147-150).
- Days 151-182 are a second use of the test period under a new pre-registration (D211 a).
- P2 never fired (PSI never above 0.2), so P2 = P0 at every delay.

## Screenshots to take
1. A chart of the gain by delay from `research/tables/feedback/exp310_gain_by_delay.json`
   (final vs design run, with CIs).
2. The hypothesis table in `research/tables/feedback/exp310_report.md`.
3. The "Promotions and rejections" table in the same report (insufficient evidence at L = 30 / 60).
