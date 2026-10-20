# Frozen config for the feedback-loop final run (EXP-310-final; D210, D211, D213, D214, D215)

Frozen 2026-10-20T16:59:31 at git commit `845f51b7dc04c02379c1349e31d53d986ad37799`, before any --final run. Final runs are made once per baseline and never repeated after seeing results (a forced re-run is logged in research/decisions.md as FINAL-RERUN).

| file | sha256 |
|---|---|
| `experiments/configs/EXP-310-final.yaml` | `3aa26349ba627718c8e253c0e5792fa1895d607d0ae307105e9b57d570933ec4` |
| `experiments/configs/splits.yaml` | `96666efd2da333cffab20c1f4ce8a567481d8fb5da5724a2e9757bd40e5c6955` |
| `experiments/configs/v_columns.yaml` | `34909dc424a2db19cf8684368213847dc2edeff159deed7e07e7ba8c62c04e7b` |
| `experiments/configs/EXP-009.yaml` | `fe949b9cb9fc2074c8d649e7bfdab9d3c13c8042ebb9b40107ef6b538515e29a` |
| `research/preregistration_feedback.md` | `6e585211fd423b1214d4c46502f0fb5afc58e0a5e9ab06f3bf3b7ddb2609aca4` |
| `research/tables/final_results.json` | `e1885c02605087cf5ff161b335444ec7b9638e2606f24c7e2e1577bf718fba8c` |
| `src/vaultic/learning/feedback.py` | `3543eff7e350d1e94282d012d7d397e50560eea61751d23202b7cfa08d6dc9e3` |
| `src/vaultic/learning/feedback_inputs.py` | `d904b4b2ee12a74ad0809a7218e6a15a5fb53404e13e60d0dee65e24f4d6a4d7` |
| `src/vaultic/learning/feedback_run.py` | `03345898abecd7e3edd7efd90478fb1faac7cf8d7345515e646aae72271fb4be` |
| `src/vaultic/learning/feedback_final.py` | `4bf70ca4e91f3da12ab65253c08ea84c57e20d352dc2a594d3dcede7d4e67f83` |

## Data files (D112)

Environment at freeze: VAULTIC_DATA_DIR = `E:/old work/fyp 11/fraud-detection/data`; VAULTIC_RAW_DIR = `unset`.

| file | bytes | sha256 |
|---|---|---|
| `E:/old work/fyp 11/fraud-detection/data/interim/merged.parquet` | 84177147 | `0be2b559a3840e7df589ac0a8ac7cad1c0b1f28331a5eac0fa94c59c79035fee` |
| `E:/old work/fyp 11/fraud-detection/data/interim/uids.parquet` | 9805651 | `ebba50931df3b801ae96d4894eb04745aac3ee39027849ec2acbee7ee80d3b0e` |
| `E:/old work/fyp 11/fraud-detection/data/features/anomaly_uid.parquet` | 9638403 | `d453c67317eb1fa5e5d53f56c096397cd49b98c79055f9208188d03ba6ea8c71` |
| `E:/old work/fyp 11/fraud-detection/data/features/base_features_uid.parquet` | 17117352 | `7eb2ff73088704c5302db18812682bd5662bc50d2a39c0783f029de409fcfad5` |
| `E:/old work/fyp 11/fraud-detection/data/features/base_features_uid2.parquet` | 14928891 | `5058eaea910d6253b3276c2b5cf07d1105b7724b0059b3d8fe47df043f3aa0b0` |
| `E:/old work/fyp 11/fraud-detection/data/features/behavioral_uid.parquet` | 25681372 | `ddd3e870714db35ed88e2ee735164aca0d8da0d6faef6720c6fe8405420f2082` |
| `E:/old work/fyp 11/fraud-detection/data/features/graph_uid.parquet` | 25914794 | `c58d7c1d85050fd8a7c5857b84a386ef10c6899f4bd9ff8513285ff34311f0f2` |
| `E:/old work/fyp 11/fraud-detection/data/features/sequence_uid.parquet` | 41240190 | `2109ecc344edb5534951f855baa8d0c5492e06dabc5bb8ab83e475bc1e53626a` |
| `E:/old work/fyp 11/fraud-detection/.worktrees/feedback/experiments/runs/EXP-200-final/20261011-002400-202994/predictions.parquet` | 8967778 | `e536614d134fc7131c035f399c4fc1c9abf75fe3a892164d0bcfabc08d9a5612` |
| `E:/dev-cache/stage2/system_dev.joblib` | 168209182 | `9d65b476042f3d62c584f2332ac99d13421ecda4fa5419e4480a62a788e43a59` |

## Library versions (D114; .venv only, D216)

| library | version |
|---|---|
| library `.venv:python` | `3.11.9` |
| library `.venv:numpy` | `1.26.4` |
| library `.venv:pandas` | `2.2.3` |
| library `.venv:scikit-learn` | `1.3.2` |
| library `.venv:scipy` | `1.17.1` |
| library `.venv:xgboost` | `2.0.3` |
| library `.venv:pyarrow` | `25.0.1` |
| library `.venv:optuna` | `5.0.0` |
| library `.venv:lightgbm` | `4.7.0` |
| library `.venv:pyyaml` | `6.0.3` |
| library `.venv:matplotlib` | `3.11.2` |
| library `.venv:torch` | `not installed` |
| library `.venv:dice-ml` | `not installed` |

## Champion B5 seed models (SHA-256 of each XGBoost booster, JSON)

- seed 0: `8675bd9f1fac372cabbdaf8867b514f66e5c88451ed0e446c00a0bbeb23e5129`
- seed 1: `564b8f4bb2f2544e0f52bd8fa5e5fc50b1daf386416c4328951220cd52eb30df`
- seed 2: `3dad400ae3f401affa75af617109b014ee40308afa92412e07e3f27f418d2332`
- seed 3: `2b2947b79ea8b38bae414aad145c6cfcf19c5d62102326a87b09357767168d5f`
- seed 4: `1905951537df2e17bcf104fc527ea14debccf1af4183575960365daa70e8070d`
