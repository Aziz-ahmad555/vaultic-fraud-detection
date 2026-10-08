# Phase 2 exit gate

Generated 2026-10-08 by `python -m vaultic.reports.phase2_gate` from --final harness runs (test period).

Gate: B5 beats B3 with non-overlapping 95% CIs on test PR-AUC.

| baseline | experiment | run | test PR-AUC |
|---|---|---|---|
| B3 | EXP-011 | `20261008-195820-697730` | 0.5570 ± 0.0016 (95% CI 0.5403–0.5739) |
| B5 | EXP-009 | `20261008-211052-888539` | 0.6494 ± 0.0017 (95% CI 0.6331–0.6659) |

Paired bootstrap B5 − B3 (same test resamples): +0.0924 (95% CI +0.0828 to +0.1027, p = 0.000).

**Result: PASSED.**
