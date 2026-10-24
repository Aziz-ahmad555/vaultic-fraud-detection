# Sprint 1: demo-critical fixes (D220)

## What it is
A set of product-layer fixes after the gap audit, so the demo shows real committed results or says
plainly where it does not. No frozen research code, config, freeze record or result changed; all
four freeze records were re-checked at the end (`research/frozen_{configs,feedback,final,stage2}.md`:
46 hashed files, none changed).

## Why we built it
The gap audit found that several dashboard pages in http mode were empty or illustrative without
saying so clearly, that the Test Lab could take about 30 minutes to start, and that some
pre-registered verdicts were never written down. A defense demo must not imply results that do not
exist.

## How it works (simple)
- **Test Lab fast start.** The parity cache key now hashes each file with line endings normalised,
  so the same code in a checkout with different line endings no longer forces a full parity run. The
  history store is saved on drive E (`VAULTIC_CACHE_DIR/testlab/store_<key>.joblib`) and reloaded.
  Missing inputs stop the start with a message naming them. `python -m vaultic.live.warm` warms both
  caches once.
- **Model lab.** The ablation table is filled from the stage-2 ladder files.
- **Drift monitor.** Alarms and retraining history come from EXP-310, labelled "EXP-310 simulation on
  test period", through a new file generated from the run folder (with the source run id and the
  SHA-256 of each source events file).
- **Cases.** The page says feature-level SHAP reasons are not stored per transaction and can fetch the
  frozen system's live top-5 feature reasons from the Test Lab on request; the what-if is labelled
  "Illustrative model, not the frozen system".
- **Scenario simulator** is labelled "Illustrative mock" and points to the Test Lab's custom mode;
  wiring it to the Attack Lab was not feasible (the Attack Lab runs whole replay days; the sliders do
  not map to it).
- **Fraud network** says "Ring detection: not yet built" in http mode.
- **Verdicts** that were never written are recorded in `research/stage1_verdicts.md`.

## Key results / numbers (with sources)
- Test Lab start, measured on the laptop (i5-8265U, 15.9 GB) with the logs on drive E:
  - **cold** (no store cache, new parity key): **1,672.8 s** total = 261.6 s system + store build
    (218 s) and save, then 1,411.2 s parity (passed) (`E:/dev-cache/testlab/warm_cold.log`, not
    committed);
  - **warm**: **41.8 s** total, store loaded in 32 s, parity read from cache in 0.1 s
    (`E:/dev-cache/testlab/warm_warm.log`);
  - backend start with the Test Lab (`VAULTIC_TESTLAB=1`), warm: ready after 68 s wall-clock, Test Lab
    part 53.3 s (status `timings`).
  - After the review fixes (D221: every module in the store key, the store key in the parity key)
    the store rebuilt once: cold 1,502.2 s (store build 264 s, full parity 1,177.1 s, passed), then
    **warm 23.5 s** (store loaded in 13 s, likely helped by the OS file cache right after the build;
    parity cached) (`E:/dev-cache/testlab/warm_cold2.log`, `warm_warm2.log`).
  - The cached store is 2.7 GB on drive E. A full parity run still happens whenever the hashed live or
    feature code, the data or the system cache really change (D183).
- Ablation ladder shown in the Model lab: rung 1 tabular 0.548 PR-AUC; steps E3 +0.0666, E4 +0.0036,
  E5 +0.0025, E8 +0.000027 (Holm p 0.978, not supported) (`research/tables/stage2/ladder_metrics.json`,
  `research/tables/stage2/ladder_steps.json`).
- EXP-310 events: 149 / 130 / 113 / 95 events at L = 0 / 7 / 30 / 60 from run `20261020-174032-191246`
  (`research/tables/feedback/exp310_events.json`; `check_events` confirms it matches the run folder).
- H5 is supported under D101's rule for MVAF, F3 and B5, but the uncertainty weight chosen on
  calibrate_fused was 0 at every K, so R3 reduces to R2 (and MVAF's R4 to R3 at K = 50 and 500):
  the gain comes from ranking by expected loss (D221; `research/stage1_verdicts.md`, from
  `research/tables/final_results.json` `e12.<method>.routing.lambdas`).

## Limitations
- The warm start depends on the caches on drive E; the first start after a code or data change is
  still slow by design (full parity).
- Live feature reasons need the Test Lab backend running; otherwise only view-level reasons exist.
- The Drift monitor shows one seed (0) and two policies (P1, P3) of the EXP-310 simulation; it is not
  the Phase 10 detector study (E13 was never run).
- The simulator remains an illustrative model.
- E11 has per-view verdicts only: D101 fixed no overall rule.

## Screenshots to take
1. Drift monitor at day 165, L = 30: the "EXP-310 simulation on test period" note, the ADWIN alarm and
   the retraining table with "Rejected: insufficient evidence" rows.
2. Model lab: the ablation table with the "Step vs previous rung" column and the source line.
3. Cases: the "Compute feature reasons live" result labelled "Live-computed by the frozen system in the
   Test Lab (matches the frozen run ✓)" next to the what-if label.
