# Project status (phase0-1, 2026-10-21)

Short summary of where Vaultic stands after PR #10. Details live in `research/decisions.md`,
`research/experiment_log.md` and the per-track handoff / report files named below.

## Done (merged into phase0-1)

| Track | What | Record |
|---|---|---|
| Stage 1: Paper 2 E10-E12 | MVAF vs fusion baselines F0-F7, calibration and conformal (E11), cost-sensitive decisions (E12); one final run on test days 151-182 (logged re-run) | D101-D115, EXP-200-final, `research/tables/final_results.json` |
| Stage 2 | E15 explanations, ablation ladder (E3-E5, E8), E23 subgroup gaps, E24; frozen system cache | D121-D181, EXP-223/210/215-final |
| Test Lab (Phase A) | live scoring of one transaction or a CSV batch by the frozen stage-2 system; parity gate on 2,000 real test rows | D182-D189 |
| Attack Lab (Phase B) | scenario framework: five synthetic attack generators (v3 primary), rules fixed on validation, 5-seed suite and findings | D190-D207 |
| Audit log (Phase C a) | persistent SQLite audit log with named analyst and role, hash chain and anchor (tamper-evident, not tamper-proof) | D208-D209 |
| Feedback loop (Phase C b, roadmap Phase 10.3 / 10.4) | EXP-310 champion-challenger under label delay; design run and one final run, results recorded as-is: H-FB1 not supported, H-FB2 supported (no cost increase), H-FB3 supported (descriptive), H-FB4 supported (point estimate) | D210-D218, `research/tables/feedback/exp310_report.md` |

## Optional (COULD, D31)

- LLM copilot.
- GraphSAGE / TGN (the graph view uses graph features + XGBoost).

## Known gaps

- **`.venv-dice` exists only in the `stage2` worktree** (`.worktrees/stage2/.venv-dice`, checked:
  imports dice_ml and xgboost 2.0.3; `.venv-torch` is there too). The main checkout and the other
  worktrees have neither, so E15 (EXP-215-final) and the dice / torch tests can only be re-run from
  the stage2 worktree, or after rebuilding the venvs from `requirements-dice.txt` /
  `requirements-torch.txt` (`tools/setup_dice_venv.ps1`). D216, the feedback handoff and the PR #10
  text said `.venv-dice` was "missing on this machine"; corrected in D219 (D216's decision is
  unchanged: EXP-310 never uses dice).
- **Test counts.** Locally (with `VAULTIC_DATA_DIR` = the main checkout's data) the full suite skips 5:
  torch x1, dice_ml x2, a Windows temp-folder link test, `.venv-torch`. CI (Linux, no data, no venvs,
  no run folders) skips 30: those 4 (the link test runs on Linux) + 19 real-data leakage cases + 8
  tests that need the data files and the frozen system + 2 PowerShell-only tests + 1 that needs
  `experiments/runs`.
- Worktrees share `experiments/runs` through junctions to the main checkout's folder; remove a
  worktree's junctions (`rmdir`, not a recursive delete) before removing the worktree.

## Next decision number

D220.
