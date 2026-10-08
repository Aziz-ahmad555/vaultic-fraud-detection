# Frozen configs for the Phase 2 final runs

Frozen 2026-10-08T16:57:37 at git commit `efa0ec8b467580b1d0abbc0e157f40011a16e919`, before any --final run. Final runs are made once per baseline and never repeated after seeing results (a forced re-run is logged in research/decisions.md as FINAL-RERUN).

| file | sha256 |
|---|---|
| `experiments/configs/EXP-012.yaml` | `0518b38085f2ae5f9ab2e22a4faa8aa7f9159f9d42cb634eb337497faa657fdc` |
| `experiments/configs/EXP-002.yaml` | `08feaa7479d10cbbdb7f677716add60a65161d184b36de29de7539495342b416` |
| `experiments/configs/EXP-011.yaml` | `4fdc55c85369a96617ca2a574033e2107646897178e56744466b96394fc80c54` |
| `experiments/configs/EXP-013.yaml` | `3a21ccdb6559678b76b4d8d30be2f22d981a3edb39cdeacb5e987fd7f056dbe7` |
| `experiments/configs/EXP-009.yaml` | `fe949b9cb9fc2074c8d649e7bfdab9d3c13c8042ebb9b40107ef6b538515e29a` |
| `experiments/configs/EXP-010.yaml` | `70baa14c8337903f4015cc91f08e24cba1e345b3931f6e437d33436d40870ee6` |
| `experiments/configs/splits.yaml` | `7e580ad5b747d44eaf161f937675cbdd8a38ab200db918fc43c30818c142802f` |
| `experiments/configs/v_columns.yaml` | `34909dc424a2db19cf8684368213847dc2edeff159deed7e07e7ba8c62c04e7b` |
| `experiments/configs/table1.yaml` | `2b2cc587e9e6fad2175c66ae703dda3884a4405424a88fe929cf144cf6f06d34` |

## Amendments
- 2026-10-09 `experiments/configs/EXP-012.yaml`: `e9355d054186…` → `0518b38085f2…`. B1 redefined after the PR #1 review (M6, D61): clipped to training quantiles, lbfgs converged; FINAL-RERUN of EXP-012. table1.yaml pins the original B1 run.
- 2026-10-09 `experiments/configs/splits.yaml`: `0ad9bcd34aeb…` → `7e580ad5b747…`. Review M1 (D53) added `fixed.calibrate: [144, 150]`, a sub-range of the existing validation period used by the Phase 7 fusion work; the train, validation and test ranges are unchanged, so no Phase 2 run is affected.
- 2026-10-09 `experiments/configs/table1.yaml`: `96bee2580c64…` → `2b2cc587e9e6…`. B1 redefined after the PR #1 review (M6, D61): clipped to training quantiles, lbfgs converged; FINAL-RERUN of EXP-012. table1.yaml pins the original B1 run.
