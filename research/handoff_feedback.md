# Handoff: Phase C (b) feedback loop (EXP-310), 2026-10-20

## Branch state
- Branch `feedback` (worktree `.worktrees/feedback`, from `phase0-1` after PR #9), pushed.
- `experiments/runs` in this worktree is a junction to the main checkout's `experiments/runs`
  (not committed).

## Commits (in order)
| Commit | What |
|---|---|
| `4f551af` | D210: pre-registration `research/preregistration_feedback.md` (before any code) |
| `16045c3` | D211: amendment 1 (Aziz's answers to section 9; second use of days 151-182; PSI reset after promotion; E12 cost model; only the XGBoost seed varies), committed alone before any code |
| `6f06709` | Simulation code (`src/vaultic/learning/feedback.py`, `feedback_inputs.py`, `feedback_run.py`), `experiments/configs/EXP-310-dev.yaml`, `tests/test_feedback.py` (the six pre-registered leakage tests + engine tests, 17 passed) |
| this commit | this handoff note |

## What EXP-310-dev is doing
The design run (training period only, days 1-120; no number from it enters any hypothesis).
For each label delay L = 0 / 7 / 30 / 60 and seed 0-4: the champion (B5 configuration, seed s,
fitted on days 1-56; E12 thresholds and P2 reference on days 57-60) serves from day 61 to 120;
policies P0, P1, P2, P3, ungated P1 and 4-weekly P1 run in one pass; challengers are fitted at
the weekly slots (68, 75, ..., 117) when the gate's evidence floor (>= 50 frauds) allows. About
115 B5 fits in total, roughly 4.5 hours.

- Started 2026-10-20 ~00:41 (detached: `E:\dev-cache\feedback\run_dev.cmd`), keep-awake on.
  At the time of this note: L = 0 finished for all seeds; L = 7 next.
- Already checked in the run: recomputing the label-derived inputs at L = 30 reproduces the
  stored columns exactly (max difference 0.0, same missing pattern).
- An earlier attempt (`experiments/runs/EXP-310-dev/20261019-141149`, log
  `E:\dev-cache\feedback\exp310_dev_interrupted_20261019.log`) was killed by a PC shutdown after
  L = 0; kept as the record, not reused (the simulation is deterministic, so the re-run
  reproduces L = 0).

## Where the log and results go
- Log: `E:\dev-cache\feedback\exp310_dev.log` (ends with `exit <code>`).
- Run folder: `experiments/runs/EXP-310-dev/20261020-004133/` (git-ignored): `config.yaml`,
  `run.log`, `metrics.json` and `events.json` (written after each delay), `served.parquet` and
  `report.md` (at the end).
- Cached recomputed inputs: `E:\dev-cache\feedback\label_derived_dev_L{0,7,60}.parquet`.

## What will be reported when it finishes
Per delay: each policy's PR-AUC on days 61-120 (seed mean, 95% CI) and its paired difference
with P0; E12 cost and the cost difference with P0; attempts, promotions and rejection reasons
(insufficient evidence, margin / CI, cost); P2 / P3 alarm counts; weekly PR-AUC. Plus whether the
gate is degenerate (accepts everything or nothing), whether the pre-registered structural
predictions hold in the design span, and any setting that looks broken (which would need a
logged amendment before EXP-310-final). Labelled as the design run, not the evaluation.

## Not done, needs Aziz
- **EXP-310-final needs Aziz's go-ahead.** Its mode is not implemented yet: it needs the
  final-run guards (frozen config, freeze record, FINAL-STARTED before scoring, one run) and the
  parity check (the 5-seed champion mean vs the stored `score_B5`, within 1e-6) as its first
  step. Next decision number: D212.
