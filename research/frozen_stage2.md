# Frozen configs for the stage-2 final runs (E23, ablation ladder, E15; D121)

Frozen 2026-10-14T01:22:38 at git commit `291207c4a9f6c4404f2e1acd33d2aa3e76e58e99`, before any --final run. Final runs are made once per baseline and never repeated after seeing results (a forced re-run is logged in research/decisions.md as FINAL-RERUN).

| file | sha256 |
|---|---|
| `experiments/configs/EXP-223-final.yaml` | `e9eb03478a1eed999542f2a9d347dc649a28e5a7ab6d23533a4ecec848808b22` |
| `experiments/configs/EXP-210-final.yaml` | `b96881af0023ca1ec4d451e2ecc30de32ae0e5371050d3dde5a15890c864d45c` |
| `experiments/configs/EXP-215-final.yaml` | `deb80e742957dd266fe09c59410c9b5501a09ece243a92ab759788117deae50e` |
| `experiments/configs/splits.yaml` | `96666efd2da333cffab20c1f4ce8a567481d8fb5da5724a2e9757bd40e5c6955` |
| `experiments/configs/v_columns.yaml` | `34909dc424a2db19cf8684368213847dc2edeff159deed7e07e7ba8c62c04e7b` |
| `experiments/configs/EXP-009.yaml` | `fe949b9cb9fc2074c8d649e7bfdab9d3c13c8042ebb9b40107ef6b538515e29a` |
| `experiments/configs/EXP-200-final.yaml` | `280127bff8e70ecab050662442c1e074c9bb81f8cd6c17a249455cc50e4a6ccd` |
| `experiments/configs/EXP-V-tabular.yaml` | `83debaa11bf4b7c0d1de917518cce74c388001a2ef7d28ef6674d7b37675d11b` |
| `experiments/configs/EXP-V-behavioral.yaml` | `7e7ec9a1a6ea90569231db28fe7cb03d42b79c5f84de339cbdeae03a4ecd073f` |
| `experiments/configs/EXP-V-temporal.yaml` | `a8d0205e2a2db3077b4c903c3a0757c82dbe9d4646713f3cc8fab7ce87bea013` |
| `experiments/configs/EXP-V-graph.yaml` | `009252ecee28f22ab0ec21197737b8a9c3866e818b260b015df4f3c4ce8880d3` |
| `experiments/configs/EXP-V-anomaly.yaml` | `9d856db2eca5bb80e4bcad787a14823e73fab4d80c5991f259963513d5a7058b` |
| `research/tables/fusion_d95.json` | `c8880e70ed502b112064f21ff2b4d4f1bbf5ee86a47766aa1da7bcb45318880e` |
| `research/tables/final_results.json` | `e1885c02605087cf5ff161b335444ec7b9638e2606f24c7e2e1577bf718fba8c` |
| `research/tables/stage2_ladder_tuned.json` | `b367ef11db1581c829dd4f38d234902ffec6e353cad90bd93cf4ab3a8552f06e` |
| `research/tables/stage2_cf_setting.json` | `6dfafc53b8e747dd310fbbff03e3813d18d6921e2c7923a5f853692ea2daf480` |

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
| `E:/old work/fyp 11/fraud-detection/data/interim/view_table_d95.parquet` | 33521161 | `99c73606053607080d75196ea25df8c081a5ad08ccdaf7ad1f95cdab54240f6f` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/diag.json` | 858 | `d2b69bf5bcf4232160c3d2c4b4136c07fbc1eaab815ec4fba88bae3880b7dd5d` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/encoder.json` | 93 | `2c9da02307743322ab56a603964bc4f1bf7e2177983e5ab44c06eb914b15b6c1` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/inputs.parquet` | 10368334 | `96224939832f212e230cd19d5ff717f3283c253b3f531941b817c3bb43374823` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/p_temporal.history.json` | 1962 | `339d9a51b5b3125503a42cc96005d99095697f3eb42bb1ac0122b65e457a260d` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/p_temporal.parquet` | 701958 | `351c565d83b7cb33756fde8dc9194338a8ef2f3dcb1909b2fe26f8a35d896399` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/p_temporal_xgb.parquet` | 703643 | `6b0e90ca87e91452358499aa329b3f542c3e0757c7e941ae14b50377427d3678` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/plan.json` | 273 | `ed357ed2c9106f86c50e563516125a50ea554f6eb15aab51cbc9d76228ee43c1` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/settings.json` | 314 | `263c987556bc946cdff3e9244b9739427535318e1a4955ba0cf258788e00074a` |
| `E:/old work/fyp 11/fraud-detection/.worktrees/stage2/experiments/runs/EXP-200-final/20261011-002400-202994/predictions.parquet` | 8967778 | `e536614d134fc7131c035f399c4fc1c9abf75fe3a892164d0bcfabc08d9a5612` |
| `E:/old work/fyp 11/fraud-detection/.worktrees/stage2/experiments/runs/EXP-200-final/20261011-002400-202994/metrics.json` | 5896 | `780634d9a40ce0aa00359ade68bfdd4b30b5fec0e04d42855946a5c36c2247f5` |
| `E:/old work/fyp 11/fraud-detection/.worktrees/stage2/experiments/runs/EXP-009/20261008-211052-888539/predictions.parquet` | 9217943 | `9512f4918d07a1d7b2ed11a3736929e067c83c6489458265ce647761346ee224` |

## Library versions per environment (D114)

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
| library `.venv-dice:python` | `3.11.9` |
| library `.venv-dice:numpy` | `1.26.4` |
| library `.venv-dice:pandas` | `2.2.3` |
| library `.venv-dice:scikit-learn` | `1.3.2` |
| library `.venv-dice:scipy` | `1.17.1` |
| library `.venv-dice:xgboost` | `2.0.3` |
| library `.venv-dice:pyarrow` | `25.0.1` |
| library `.venv-dice:optuna` | `not installed` |
| library `.venv-dice:lightgbm` | `4.7.0` |
| library `.venv-dice:pyyaml` | `6.0.3` |
| library `.venv-dice:matplotlib` | `not installed` |
| library `.venv-dice:torch` | `not installed` |
| library `.venv-dice:dice-ml` | `0.12` |
