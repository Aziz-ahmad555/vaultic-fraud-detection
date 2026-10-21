# Stage 1: Paper 2 main comparison (E10-E12)

## What it is
The main experiment of Paper 2: Missing-View-Aware Adaptive Fusion (MVAF) over five evidence views
(tabular, behavioral, temporal, graph, anomaly) compared with fusion baselines F0-F7 and the single
strongest model B5 (E10), its robustness when a view is removed (E11), and the trust layer:
calibration, conformal sets and cost-sensitive decisions (E12). Pre-registered in D101 (details D102),
run once on the test period (days 151-182) as EXP-200-final (a logged re-run, D115).

## Why we built it
Paper 2's question is whether a fusion that knows which views are missing beats simpler fusion, and
whether the system's scores can be trusted for decisions (calibrated, with coverage guarantees and
costs). The pre-registration fixed every claim before the test period was read.

## How it works (simple)
Each view's model scores a transaction; views that cannot produce a score are masked, never filled.
MVAF learns a gate that weights the available views per transaction. Thresholds and calibration are
chosen on the last validation days only. The test period is scored once, with 5 seeds and a
1,000-sample paired bootstrap for every "A vs B" claim, Holm-corrected.

## Key results (with sources)
Data: anonymized e-commerce transactions released by Vesta for the IEEE-CIS competition; 92,427
test rows, 3,213 frauds (`research/tables/final_e10.md`).

- Test PR-AUC: **MVAF 0.6219 [0.6037, 0.6392] vs B5 0.6532 [0.6368, 0.6697]**; MVAF - B5 -0.0313
  [-0.0374, -0.0255], Holm p < 0.001. The single strongest model beats MVAF on PR-AUC
  (`research/tables/final_e10.md`).
- Primary pre-registered comparison: MVAF - F3 (logistic stacking) +0.1990 [+0.1880, +0.2096],
  Holm p < 0.001 (`research/tables/final_e10.md`). MVAF - F4 (LightGBM stacking) +0.0013
  [-0.0035, +0.0057], p 0.678: no difference (same file).
- **H2 (MVAF's margin is larger on rows with missing views): NOT SUPPORTED**: beats F3 overall yes,
  F4 no; larger margin on missing-view rows: F3 no, F4 no (`research/tables/final_e10.md`,
  `research/tables/final_results.json` key `h2`).
- Cost at the E10 threshold: MVAF $236,084 vs B5 $268,023; MVAF - B5 -$31,939 [-44,161, -19,062]
  (`research/tables/final_e10.md`).
- E11 (remove the tabular view): MVAF drops 0.0928 [+0.0826, +0.1049] PR-AUC vs B5 0.4378
  [+0.4206, +0.4540] (`research/tables/final_e11.md`).
- E12 calibration (MVAF, isotonic): ECE 0.1471 -> 0.0050, Brier 0.0840 -> 0.0195
  (`research/tables/final_e12.md`).
- E12 conformal (MVAF, Mondrian): 0.90 target -> coverage 0.9063; 0.95 target -> 0.9779, outside the
  2-point band, so the **Phase 8 gate did NOT pass** (`research/tables/final_e12.md`,
  `research/tables/final_results.json` key `phase8_gate`).
- H5 (review policy by fraud value): R3 - R1 at K = 50 reviews/day +100,164 [+75,151, +127,393] of
  fraud value caught, Holm p < 0.001 (`research/tables/final_e12.md`).

## Limitations
- MVAF does not beat B5 or F0 on PR-AUC (MVAF 0.622 vs B5 0.653); this is reported as found.
- E12's cost model is a stated limitation of Paper 2 (D123).
- H2 not supported; Phase 8 gate not passed (95% conformal coverage over-covers).
- One dataset (IEEE-CIS); masked features cannot be interpreted.
- The final run is a logged re-run (D115).

## Screenshots to take
1. `research/figures/final_e12_reliability_MVAF.png` (reliability diagram before / after calibration).
2. The E10 headline table at the top of `research/tables/final_e10.md` (rendered).
3. The E11 drop table for the tabular view in `research/tables/final_e11.md`.
(The dashboard's Model lab page shows mock numbers; do not use it for results.)
