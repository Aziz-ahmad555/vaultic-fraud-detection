# Frozen configs for the Paper 2 final runs (E10-E12, D101)

Frozen 2026-10-10T22:49:59 at git commit `3fb214bf1bb3208ca1480d990d49437aa32792dd`, before any --final run. Final runs are made once per baseline and never repeated after seeing results (a forced re-run is logged in research/decisions.md as FINAL-RERUN).

| file | sha256 |
|---|---|
| `experiments/configs/EXP-200-final.yaml` | `280127bff8e70ecab050662442c1e074c9bb81f8cd6c17a249455cc50e4a6ccd` |
| `experiments/configs/splits.yaml` | `96666efd2da333cffab20c1f4ce8a567481d8fb5da5724a2e9757bd40e5c6955` |
| `experiments/configs/v_columns.yaml` | `34909dc424a2db19cf8684368213847dc2edeff159deed7e07e7ba8c62c04e7b` |
| `experiments/configs/EXP-009.yaml` | `fe949b9cb9fc2074c8d649e7bfdab9d3c13c8042ebb9b40107ef6b538515e29a` |
| `experiments/configs/EXP-F0-inner.yaml` | `5ea1f6ef288ba22b4c15d9cae9303ca3cfd208ec37ece4acd461d98613e9435c` |
| `research/tables/fusion_d95.json` | `c8880e70ed502b112064f21ff2b4d4f1bbf5ee86a47766aa1da7bcb45318880e` |
| `experiments/configs/EXP-V-tabular.yaml` | `83debaa11bf4b7c0d1de917518cce74c388001a2ef7d28ef6674d7b37675d11b` |
| `experiments/configs/EXP-V-behavioral.yaml` | `7e7ec9a1a6ea90569231db28fe7cb03d42b79c5f84de339cbdeae03a4ecd073f` |
| `experiments/configs/EXP-V-temporal.yaml` | `a8d0205e2a2db3077b4c903c3a0757c82dbe9d4646713f3cc8fab7ce87bea013` |
| `experiments/configs/EXP-V-graph.yaml` | `009252ecee28f22ab0ec21197737b8a9c3866e818b260b015df4f3c4ce8880d3` |
| `experiments/configs/EXP-V-anomaly.yaml` | `9d856db2eca5bb80e4bcad787a14823e73fab4d80c5991f259963513d5a7058b` |

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
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/diag.json` | 858 | `d2b69bf5bcf4232160c3d2c4b4136c07fbc1eaab815ec4fba88bae3880b7dd5d` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/encoder.json` | 93 | `2c9da02307743322ab56a603964bc4f1bf7e2177983e5ab44c06eb914b15b6c1` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/inputs.parquet` | 10368334 | `96224939832f212e230cd19d5ff717f3283c253b3f531941b817c3bb43374823` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/p_temporal.history.json` | 1962 | `339d9a51b5b3125503a42cc96005d99095697f3eb42bb1ac0122b65e457a260d` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/p_temporal.parquet` | 701958 | `351c565d83b7cb33756fde8dc9194338a8ef2f3dcb1909b2fe26f8a35d896399` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/p_temporal_xgb.parquet` | 703643 | `6b0e90ca87e91452358499aa329b3f542c3e0757c7e941ae14b50377427d3678` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/plan.json` | 273 | `ed357ed2c9106f86c50e563516125a50ea554f6eb15aab51cbc9d76228ee43c1` |
| `E:/old work/fyp 11/fraud-detection/data/interim/temporal_dev_v4/settings.json` | 314 | `263c987556bc946cdff3e9244b9739427535318e1a4955ba0cf258788e00074a` |
| `E:/old work/fyp 11/fraud-detection/.worktrees/phase7-prep/experiments/runs/EXP-009/20261008-211052-888539/predictions.parquet` | 9217943 | `9512f4918d07a1d7b2ed11a3736929e067c83c6489458265ce647761346ee224` |
| `E:/old work/fyp 11/fraud-detection/.worktrees/phase7-prep/experiments/runs/EXP-F0-inner/20261009-213740-324011/predictions.parquet` | 3702830 | `e25b6157b2be7f5e4b925cddf260d46d98ca65065524af415121fcaf4e2faa0a` |

## Library versions (D114)

| library | version |
|---|---|
| library `python` | `3.11.9` |
| library `numpy` | `1.26.4` |
| library `pandas` | `2.2.3` |
| library `scikit-learn` | `1.3.2` |
| library `scipy` | `1.17.1` |
| library `xgboost` | `2.0.3` |
| library `pyarrow` | `25.0.1` |
| library `optuna` | `5.0.0` |
| library `lightgbm` | `4.7.0` |
| library `pyyaml` | `6.0.3` |
| library `matplotlib` | `3.11.2` |
| library `torch` | `not installed` |
