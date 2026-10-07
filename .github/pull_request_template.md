## What this PR does

<!-- One or two sentences. Link the roadmap phase / milestone and any decision (Dxx). -->

## Merge checklist (CLAUDE.md, decision D32)

**1. Tests**
- [ ] Full test suite passes (`python -m pytest`), including the leakage tests in
      `tests/test_leakage.py` (run where the real 50k sample exists, so they are not skipped)

**2. Research rules**
- [ ] Leakage rules respected: point-in-time features only, label delay applied to every
      label-derived feature, encoders fitted on the training period only
- [ ] No test-period use outside `--final` runs; every choice (thresholds, hyperparameters,
      variants) made on validation
- [ ] New configs are in `experiments/configs/`; every non-obvious choice is logged in
      `research/decisions.md`; experiments are listed in `research/experiment_log.md`
- [ ] No data, databases, model binaries or secrets committed (pre-commit `no-data-files` hook
      passed; `.env` untouched)
- [ ] Results are reproducible from the configs (harness runs, not notebooks; seeds fixed)

**3. Independent review**
- [ ] A NEW Claude Code session (not the one that wrote the code) reviewed this PR's diff
      against CLAUDE.md and the roadmap
- [ ] Every finding is fixed or answered below

### Review findings and answers

<!-- Paste each finding from the reviewing session and how it was resolved. -->
