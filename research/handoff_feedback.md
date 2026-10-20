# Handoff: Phase C (b) feedback loop (EXP-310), updated 2026-10-20 (after the design run)

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
- **EXP-310-final needs Aziz's go-ahead.** The final mode is now built (see the update below) but
  NOT run; next: Aziz's independent review of the final-mode code, the freeze, then the run.
  Next decision number: D215.

## Update after the design run (2026-10-20)
- EXP-310-dev completed (run `20261020-004133`, exit 0, 4.8 h). P1 vs P0 PR-AUC: +0.064 (L=0),
  +0.021 (L=7), +0.0016 (L=30), 0 (L=60, all attempts "insufficient evidence"). Gate not
  degenerate. P2 never fired at any delay.
- P2 PSI diagnostic (design days only, `python -m vaultic.learning.psi_diagnostic`): code correct,
  daily PSI max 0.018 / 0.018 / 0.024 / 0.191 (L = 0/7/30/60), never > 0.2: P2 = P0, threshold
  kept (D212).
- D212: predictions before the final mode (small effect at L=30, H-FB1 may fail, H-FB4 may fail,
  L=60 structural null, P0 not comparable across delays, dev frequency-encoder limitation).
- D213: final-mode tooling. Built in `src/vaultic/learning/feedback_final.py` with
  `experiments/configs/EXP-310-final.yaml`; tests in `tests/test_feedback_final.py` (9 passed).
  Order: guards and FINAL-STARTED pushed -> data loaded (`load("final", final_started=True)`) ->
  parity (5-seed B5 mean vs stored score_B5, 1e-6, abort on failure) -> delays one by one with a
  DONE marker; `--resume <run folder>` continues in the same folder and skips completed delays.
- **Not run.** Before a run: Aziz's independent review of the final-mode code, then
  `python -m vaultic.learning.feedback_final --freeze` (commit the freeze record + FREEZE row),
  then the run with Aziz's go-ahead.
- Full test suite: 620 passed, 5 skipped, 1 flaky (`test_demo_server.py::test_whatif_is_stateless...`,
  ConnectionResetError WinError 10054; the file passes 32/32 on re-run; this branch does not touch
  the demo server).

## Update after the independent review of the final mode (2026-10-20)
- D214 (amendment 2, `0c68b19`, committed alone before the code): resume runs the same guards as a
  start plus the recorded-commit and origin FINAL-STARTED checks, and pushes FINAL-RESUMED before
  any test row; parity failure aborts (status "aborted", FINAL-ABORTED pushed, no resume); the
  stage-2 system cache is a frozen data file and each seed's booster hash is checked at parity;
  strict label curtain (its_time + L < t); monthly P1 every 30 days (P1_monthly; the design run
  used 28 days); cached recomputed inputs verified by input key + file hash; nits (bootstrap seed
  from config, warm-up 121-150, load('final') checks the FINAL-STARTED row, champion
  threshold-slice overlap reported in each gate, H-FB3 descriptive, H-FB4 point estimate, D210
  status, CLAUDE.md resume wording).
- Not frozen, not run. Next: Aziz's short second review of the resume / guard code, then
  `python -m vaultic.learning.feedback_final --freeze`, then the run with his go-ahead.

## Status after the second review (2026-10-20)
- D214 fixes done and pushed (`d585919`); full suite 631 passed, 5 skipped.
- Second review: FIX FIRST, no blockers in the normal CLI path.
- Next step: log D215 (amendment, committed alone; includes a one-line correction to D214 point 8:
  the 5 skips are torch x1, dice_ml x2, a Windows temp-folder link test and .venv-torch, not
  real-data tests), then the second-review fixes (per-process guard token + origin row check in
  the loader; resume requires data hashes equal to the start's and treats a re-freeze as a
  change; refuse resume on a FINAL-ABORTED row at HEAD or origin; per-run RUNNING lock;
  freeze() writes the D113 FREEZE row; a temp-repo + bare-origin test; nits). Do NOT freeze yet.

## Status after the second-review fixes (2026-10-20)
- D215 (amendment 3, `ebc521a`, committed alone before the code; corrects D214 point 8: the 5
  skips are torch x1, dice_ml x2, a Windows temp-folder link test and .venv-torch).
- Fixes: load('final') needs the per-process token from start() / resume() and the run's latest
  FINAL-STARTED / FINAL-RESUMED row on origin; resume needs the start's data hashes, refuses after
  a re-freeze, a FINAL-ABORTED row (HEAD or origin) or while `<run>/RUNNING.lock` exists (held for
  the whole run; delete by hand after a crash once no run is active); `--freeze` writes the D113
  FREEZE row; temp-repo + bare-origin tests; nits.
- Not frozen, not run. Next: Aziz's review, then `--freeze` (commit the record + FREEZE row), then
  the run with his go-ahead. Next decision number: D216.

## Frozen, not run (2026-10-20)
- `7fcf447`: docstring wording (RUNNING.lock is left only by a hard kill or shutdown).
- D216 (`26be2dd`, alone) + code `845f51b`: EXP-310-final records and checks library versions of
  `.venv` only (`.venv-dice` out of scope); stage-2 code unchanged. Full suite 642 passed, 5 skipped.
- **Frozen** `93f9730`: `research/frozen_feedback.md` (frozen at `845f51b`) + its D113 FREEZE row.
  Recorded with VAULTIC_DATA_DIR = `E:/old work/fyp 11/fraud-detection/data`, VAULTIC_RAW_DIR
  unset: the run must use exactly these, the `.venv` interpreter, PYTHONPATH = this worktree's
  `src`, from this worktree (the E10 predictions path is recorded through its runs junction).
- Dry start-guard check (guards + read-only re-run checks; no start, no lock, no row, no data
  loaded): passed at HEAD `93f9730`, 10 data files, 5 champion hashes, no earlier final run
  folders, 0 FINAL rows at HEAD / origin.
- Cosmetic: the record's title lists D210-D215 (fixed string in code); D216 is in the library
  section heading.
- **Not run.** Next: Aziz's go-ahead, then
  `python -m vaultic.learning.feedback_final` (detached, keep-awake, log on E:). Next decision D217.
- Note: `.venv-dice` is missing on this machine, so E15 (stage-2 explanation final run) cannot
  currently be re-run; to be rebuilt later from `requirements-dice.txt`
  (`tools/setup_dice_venv.ps1`).
