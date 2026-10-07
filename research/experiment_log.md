# Experiment log

One line per experiment, filled in from harness output only (`python -m vaultic.eval.run experiments/configs/EXP-XXX.yaml`).

| ID | Date | Question | Result (from metrics.json) | Conclusion |
|---|---|---|---|---|
| EXP-000 | 2026-10-07 | Phase 1 exit gate: does the same config + seed give identical metrics twice? | PR-AUC 0.6453 ± 0.0000 (95% CI 0.6286–0.6620), 1 seeds, run `experiments/runs/EXP-000/20261007-170822-685400` | — |
| EXP-000 | 2026-10-07 | Phase 1 exit gate: does the same config + seed give identical metrics twice? | PR-AUC 0.6453 ± 0.0000 (95% CI 0.6286–0.6620), 1 seeds, run `experiments/runs/EXP-000/20261007-171231-833707` | Identical to the previous run: harness is reproducible (exit gate passed) |
