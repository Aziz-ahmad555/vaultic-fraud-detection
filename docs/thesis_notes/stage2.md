# Stage 2: explanations, ablation ladder, subgroup gaps, compute (E15, E3-E8, E23, E24)

## What it is
The second set of pre-registered experiments on the frozen final-run models (D121, details
D158-D167): E15 explanation faithfulness, stability and counterfactual validity; the ablation
ladder adding one view at a time (E3 behavioral, E4 temporal, E5 graph, E8 anomaly); E23 subgroup
performance gaps; E24 compute cost per model.

## Why we built it
Paper 2 needs to show which views earn their place (RQ1), whether explanations are faithful
(RQ6), whether performance differs across customer groups (fairness check), and what the system
costs to run.

## How it works (simple)
- E15: remove each alert's top-k attributed features (set them to training medians) and measure the
  risk drop (AOPC), against random and permutation orders; counterfactuals with DiCE.
- Ladder: cumulative views under F4 stacking; each step compared with the previous rung (paired
  bootstrap, Holm).
- E23: per subgroup (ProductCD, card type, device, email domain) PR-AUC and recall gaps.
- E24: fit and inference time per model on validation-only refits.

## Key results (with sources)
- E15, MVAF primary ordering: H6a (beats random) **supported**, +0.0323 [+0.0168, +0.0476];
  H6b (beats permutation importance) **not supported**, -0.1510 [-0.1656, -0.1362]. B5: H6a and H6b
  supported (`research/tables/stage2/e15_report.md`).
- E15 counterfactual validity: MVAF 0.976 [0.963, 0.987]; B5 0.259 [0.221, 0.296]. Different decision
  boundaries, not directly comparable (same file). Phase 9 exit gate PASSED (same file).
- Ladder (H1, PR-AUC per step): E3 +behavioral +0.0666 [+0.0590, +0.0756]; E4 +temporal +0.0036;
  E5 +graph +0.0025; E8 +anomaly +0.000027, Holm p 0.978, **not supported**
  (`research/tables/stage2/ladder_steps.json`, `research/tables/stage2/ladder_meta.json`).
- E23, B5 by ProductCD: PR-AUC max - min 0.341 (highest R, lowest W)
  (`research/tables/stage2/e23_gaps.json`). MVAF - B5 PR-AUC by product: W -0.0379
  [-0.0478, -0.0283], C -0.0216 [-0.0311, -0.0121] (`research/tables/stage2/e23_mvaf_minus_b5.json`).
- E24 (validation refits, i5-8265U laptop CPU): B5 fit 137.8 s, inference 34.40 ms / 1,000 rows
  (`research/tables/e24_compute.md`).

## Limitations
- E15's registered absolute-attribution order suits MVAF alerts poorly (D166); a secondary ordering
  was added after a validation check and never decides H6a / H6b (D167).
- The anomaly view adds nothing measurable on top of four views (E8 not supported).
- E23 is a descriptive fairness check on product / card / device / email groups, not protected
  attributes (IEEE-CIS has none).
- E15 needs `.venv-dice`, which exists only in the stage2 worktree (D219).

## Screenshots to take
1. The deletion-test table in `research/tables/stage2/e15_report.md`.
2. The counterfactual validity table in the same report.
3. A small chart of the ladder steps (from `research/tables/stage2/ladder_steps.json`).
