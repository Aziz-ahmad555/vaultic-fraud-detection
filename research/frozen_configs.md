# Frozen configs for the Phase 2 final runs

Frozen 2026-10-08T16:57:37 at git commit `efa0ec8b467580b1d0abbc0e157f40011a16e919`, before any --final run. Final runs are made once per baseline and never repeated after seeing results (a forced re-run is logged in research/decisions.md as FINAL-RERUN).

| file | sha256 |
|---|---|
| `experiments/configs/EXP-012.yaml` | `c6659e73d06bfa8c343d65da1e64540b779d75db2852e4914514e2b3935e09ee` |
| `experiments/configs/EXP-002.yaml` | `08feaa7479d10cbbdb7f677716add60a65161d184b36de29de7539495342b416` |
| `experiments/configs/EXP-011.yaml` | `4fdc55c85369a96617ca2a574033e2107646897178e56744466b96394fc80c54` |
| `experiments/configs/EXP-013.yaml` | `3a21ccdb6559678b76b4d8d30be2f22d981a3edb39cdeacb5e987fd7f056dbe7` |
| `experiments/configs/EXP-009.yaml` | `fe949b9cb9fc2074c8d649e7bfdab9d3c13c8042ebb9b40107ef6b538515e29a` |
| `experiments/configs/EXP-010.yaml` | `70baa14c8337903f4015cc91f08e24cba1e345b3931f6e437d33436d40870ee6` |
| `experiments/configs/splits.yaml` | `96666efd2da333cffab20c1f4ce8a567481d8fb5da5724a2e9757bd40e5c6955` |
| `experiments/configs/v_columns.yaml` | `34909dc424a2db19cf8684368213847dc2edeff159deed7e07e7ba8c62c04e7b` |
| `experiments/configs/table1.yaml` | `0412b274078d258e8201b733129ce642aed434aa435732736138e50d70e3c723` |

## Amendments
- 2026-10-09 `experiments/configs/EXP-012.yaml`: `e9355d054186…` → `0518b38085f2…`. B1 redefined after the PR #1 review (M6, D61): clipped to training quantiles, lbfgs converged; FINAL-RERUN of EXP-012. table1.yaml pins the original B1 run.
- 2026-10-09 `experiments/configs/splits.yaml`: `0ad9bcd34aeb…` → `7e580ad5b747…`. Review M1 (D53) added `fixed.calibrate: [144, 150]`, a sub-range of the existing validation period used by the Phase 7 fusion work; the train, validation and test ranges are unchanged, so no Phase 2 run is affected.
- 2026-10-09 `experiments/configs/table1.yaml`: `96bee2580c64…` → `2b2cc587e9e6…`. B1 redefined after the PR #1 review (M6, D61): clipped to training quantiles, lbfgs converged; FINAL-RERUN of EXP-012. table1.yaml pins the original B1 run.
- 2026-10-09 `experiments/configs/EXP-012.yaml`: `0518b38085f2…` → `c6659e73d06b…`. B1 grid extended on validation to C = 1000 (D61) and C chosen by the D62 flat-curve rule: still C = 10; only the config's question text changed.
- 2026-10-09 `experiments/configs/table1.yaml`: `2b2cc587e9e6…` → `0412b274078d…`. Table 1 B1 row describes the redefined model and the 7-value grid with the D62 rule (D61).
- 2026-10-09 `experiments/configs/splits.yaml`: `7e580ad5b747…` → `58fb5496c75e…`. Review N1 (D76): the calibrate tail is split into calibrate_views 144-146 and calibrate_fused 147-150; the train, validation and test ranges are unchanged, so no Phase 2 run is affected.
- 2026-10-09 `experiments/configs/splits.yaml`: `58fb5496c75e…` → `96666efd2da3…`. D95: splits.yaml gains gate_folds (cross-fitted gate folds over the training period); the train, validation, test and calibrate ranges are unchanged, so no Phase 2 run is affected.
