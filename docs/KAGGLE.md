# Running Vaultic experiments on Kaggle Notebooks

Kaggle gives a free GPU (about 30 GPU-hours a week, sessions up to 12 hours) and already hosts
the IEEE-CIS data. Use it for the heavy jobs (tuning, GPU training); keep the record of results
(`experiments/runs/`, `research/`) on the laptop and copy Kaggle runs back with
`vaultic.eval.transfer`.

Not tested end to end yet: the steps below follow the code paths that the unit tests cover, but
the first real Kaggle session should be treated as a trial.

## One-time setup

1. Kaggle account with a **verified phone number** (needed for GPU and internet access).
2. Accept the competition rules: <https://www.kaggle.com/competitions/ieee-fraud-detection>.
3. GitHub: create a **fine-grained personal access token** with *read-only* access to the
   private Vaultic repo (Contents: read). In Kaggle: notebook → *Add-ons → Secrets* → add
   `GITHUB_TOKEN`. Never paste the token into a cell or commit it.

## Notebook settings

- *Accelerator*: **GPU T4 x2** (or P100). *Internet*: **on**.
- *Add input*: the competition **ieee-fraud-detection**. Its files appear read-only at
  `/kaggle/input/ieee-fraud-detection/`.

## Cell 1: get the code

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone -q -b phase0-1 https://x-access-token:{token}@github.com/<your-user>/vaultic.git /kaggle/working/vaultic
del token
%cd /kaggle/working/vaultic
!git log --oneline -1
```

## Cell 2: match the laptop's library versions

Kaggle's preinstalled versions differ from ours. Pin the ones that affect results (this takes a
few minutes; restart the session if pip asks you to):

```python
!pip install -q xgboost==2.0.3 lightgbm==4.7.0 scikit-learn==1.3.2 pandas==2.2.3 numpy==1.26.4 \
    pyarrow==25.0.1 optuna==5.0.0 shap==0.49.1 mlflow==3.17.0 pyyaml
```

The pip XGBoost wheel includes CUDA support. **The pip LightGBM wheel does not**: for LightGBM on
the GPU you would have to build it (`pip install lightgbm==4.7.0 --no-binary lightgbm
--config-settings=cmake.define.USE_GPU=ON`, OpenCL), which is slow and not tested here. Run
LightGBM with `--device cpu` on Kaggle unless that build works.

## Cell 3: point Vaultic at Kaggle's folders

```python
import os
os.environ["VAULTIC_RAW_DIR"] = "/kaggle/input/ieee-fraud-detection"   # read-only CSVs
os.environ["VAULTIC_DATA_DIR"] = "/kaggle/working/data"                # interim + features
os.environ["PYTHONPATH"] = "/kaggle/working/vaultic/src"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
```

## Cell 4: check the data and build the inputs

```python
!python -m vaultic.data.verify            # md5 of Kaggle's CSVs vs the repo's DVC pointers
!python -m vaultic.data.load              # -> /kaggle/working/data/interim/merged.parquet
!python -m vaultic.data.uid               # -> uids.parquet (also rewrites research/uid_report.md)
!python -m vaultic.features.pipeline --uid-variant uid
```

Stop if `verify` prints anything other than `ok` for both files: the data would not be the data
the laptop results were computed on. `experiments/configs/v_columns.yaml` (the V-column
reduction) comes with the repo, so it is identical by construction.

## Cell 5: run experiments on the GPU

```python
!python -m vaultic.eval.run experiments/configs/EXP-009.yaml --device cuda
!python -m vaultic.eval.tune --name B5 --features b5 --model xgboost --device cuda --config-id EXP-009
```

`--device` can also be set for the whole session with `os.environ["VAULTIC_DEVICE"] = "cuda"`.
Models that only run on CPU (logistic regression, random forest, B6) ignore it, and the
harness records `cpu` for them.

## Cell 6: bring the results home

```python
!python -m vaultic.eval.transfer export /kaggle/working/vaultic_runs.zip --experiments EXP-009
```

Download `vaultic_runs.zip` from the notebook's *Output* panel, then on the laptop (repo root,
`.venv` active):

```powershell
python -m vaultic.eval.transfer import path\to\vaultic_runs.zip
```

This copies the run folders into `experiments/runs/` (never overwriting an existing run) and
logs them into the local MLflow store. Add their lines to `research/experiment_log.md` by hand
from `metrics.json`, marking them as Kaggle runs.

## Rules that still apply on Kaggle

- **Device is part of the result.** GPU and CPU training can differ slightly (different
  floating-point summation order). Every run records the device in its saved `config.yaml`,
  in `metrics.json` and, with the GPU name, in `run_info.json`. Do not mix CPU and GPU runs in
  one paired comparison without saying so; make all final runs for one table on one device.
- **One `--final` run per experiment.** The Kaggle clone has no `experiments/runs/` history,
  so the harness there cannot see laptop finals. Make `--final` runs on the laptop, or first
  copy the laptop's runs to Kaggle (`transfer export` here, upload as a dataset, `transfer
  import` there).
- **Validation only for every choice**: tuning on Kaggle uses the same validation-only tuner.
- Don't download or commit `/kaggle/working/data`; it is derived and rebuilt in Cell 4.
