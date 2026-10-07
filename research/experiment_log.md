# Experiment log

One line per experiment, filled in from harness output only (`python -m vaultic.eval.run experiments/configs/EXP-XXX.yaml`).

Development runs report validation metrics only. Test-period metrics come only from `--final` runs, marked **FINAL**. Rows marked EXPLORATORY were run before this rule and must not be used in papers.

| ID | Date | Question | Result (from metrics.json) | Conclusion |
|---|---|---|---|---|
| EXP-000 | 2026-10-07 | Phase 1 exit gate: does the same config + seed give identical metrics twice? | EXPLORATORY test-period result from before the validation-only rule (not for papers): PR-AUC 0.6453 ± 0.0000 (95% CI 0.6286–0.6620), 1 seeds, run `experiments/runs/EXP-000/20261007-170822-685400` | — |
| EXP-000 | 2026-10-07 | Phase 1 exit gate: does the same config + seed give identical metrics twice? | EXPLORATORY test-period result from before the validation-only rule (not for papers): PR-AUC 0.6453 ± 0.0000 (95% CI 0.6286–0.6620), 1 seeds, run `experiments/runs/EXP-000/20261007-171231-833707` | Identical to the previous run: harness is reproducible (exit gate passed) |
| EXP-001 | 2026-10-07 | B1 logistic regression: raw numeric + one-hot ProductCD, scaled (untuned) | EXPLORATORY test-period result from before the validation-only rule (not for papers): PR-AUC 0.1799 ± 0.0000 (95% CI 0.1702–0.1898), 5 seeds, run `experiments/runs/EXP-001/20261007-180548-232460` | — |
| EXP-003 | 2026-10-07 | B3 XGBoost on raw columns, no feature engineering (untuned) | EXPLORATORY test-period result from before the validation-only rule (not for papers): PR-AUC 0.5287 ± 0.0000 (95% CI 0.5114–0.5462), 5 seeds, run `experiments/runs/EXP-003/20261007-182603-613676` | — |
| EXP-005 | 2026-10-07 | B5 PARTIAL: XGBoost on raw + point-in-time base features; no V-reduction, untuned | EXPLORATORY test-period result from before the validation-only rule (not for papers): PR-AUC 0.6453 ± 0.0000 (95% CI 0.6286–0.6620), 5 seeds, run `experiments/runs/EXP-005/20261007-184554-349262` | — |
| EXP-006 | 2026-10-07 | D8: B5-partial with uid_variant uid (seed 0, same settings as EXP-007/006); validation only | val PR-AUC 0.6480 ± 0.0000 (95% CI 0.6288–0.6663) (development run), 1 seeds, uid `uid`, run `experiments/runs/EXP-006/20261007-185950-066314` | — |
| EXP-007 | 2026-10-07 | D8: B5-partial with uid_variant uid2 (seed 0, same settings as EXP-007/006); validation only | val PR-AUC 0.6244 ± 0.0000 (95% CI 0.6046–0.6428) (development run), 1 seeds, uid `uid2`, run `experiments/runs/EXP-007/20261007-190134-481504` | — |
| EXP-008 | 2026-10-07 | B5 with V-column reduction (108 of 339 V kept), untuned, uid; compare with EXP-006 on validation | val PR-AUC 0.6529 ± 0.0000 (95% CI 0.6335–0.6713) (development run), 1 seeds, uid `uid`, run `experiments/runs/EXP-008/20261007-191452-827489` | — |
| EXP-010 | 2026-10-07 | B6 / E2: FYP-1 models (global XGBoost + per-user Isolation Forest + extreme-amount rule) under the temporal split | val PR-AUC 0.3675 ± 0.0000 (95% CI 0.3447–0.3896) (development run), 5 seeds, uid `uid`, run `experiments/runs/EXP-010/20261007-202216-735428` | — |
