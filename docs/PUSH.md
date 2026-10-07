# Pushing the repository to a private GitHub repo

## History scan (2026-10-07, all 29 commits on main, phase0-1, phase3-prep and tag v1-fyp1)

| Check | Result |
|---|---|
| CSV, Parquet, database (.db/.sqlite), joblib/pickle/torch files ever committed | **None** |
| `.env` ever committed | **No** (only `.env.example`, whose every version holds placeholders) |
| Real values from the local `.env` (FLASK_SECRET, FERNET_KEY, JWT_SECRET_KEY) in any commit | **None.** The only matches were `FLASK_ENV=development` and `SMTP_HOST=smtp.gmail.com`, which are not secrets and are in `.env.example` |
| Token / key patterns (GitHub, AWS, private keys, Fernet tokens) in any commit | **None**; only placeholder lines such as `SMTP_PASS=your-app-password` in READMEs |
| Model binaries | **One:** `models_store/xgboost_baseline.json` (FYP-1's trained XGBoost, 1.0 MB) is in the first three commits and in tag `v1-fyp1`. It has been untracked since the restructure (`91a0235`) |
| Largest files | `docs/ROADMAP.pdf` 1.5 MB, the model above 1.0 MB; everything else under 25 KB |

How it was checked: every path ever added (`git log --all --name-only`), every blob size
(`git rev-list --objects --all`), and `git grep` of every commit for key patterns and for each
real `.env` value (values compared without printing them, with a positive and a negative
control to prove the search works).

### Your decision: the model file in history

- **Keep it (recommended):** it contains no data or secrets, and `v1-fyp1` is meant to be the
  exact FYP-1 submission, which needs that model to run.
- **Remove it:** rewrite history with `git filter-repo` before the first push. This changes the
  hash of every commit and of the tag, so the commit hashes already recorded in
  `research/decisions.md`, `research/frozen_configs.md` and every run's `run_info.json` would
  no longer exist. If you choose this, do it only after the Phase 2 chain has finished:

  ```powershell
  pip install git-filter-repo
  git filter-repo --invert-paths --path models_store/xgboost_baseline.json
  ```

## Create the private repo and push

The GitHub CLI is not installed here, so the steps use the website (or install the CLI with
`winget install GitHub.cli`, then `gh auth login`).

1. On <https://github.com/new>: name `fraud-detection` (or `vaultic`), **Private**, and **no**
   README, .gitignore or licence (the repo must be empty).
2. In the repo root (`E:\old work\fyp 11\fraud-detection`):

   ```powershell
   git remote add origin https://github.com/<your-user>/fraud-detection.git
   git push -u origin main
   git push -u origin phase0-1
   git push origin v1-fyp1
   ```

   The first push opens a browser window for GitHub sign-in (Git Credential Manager).
   `phase3-prep` is already merged into `phase0-1`; `phase7-prep` goes up after it is merged too.
3. Check on GitHub that `data/raw/` shows only `.gitkeep` and the two `.dvc` files, and that
   the *Actions* tab runs the CI workflow on the pushed branches.
4. Protect `main`: *Settings → Branches → Add rule* → require a pull request before merging,
   with **0 required approvals**: GitHub does not let you approve your own pull request, so a
   required approval would block every merge in a solo project (D31).

Nothing in `.gitignore`d folders is pushed (`data/`, `experiments/runs/`, `experiments/mlflow/`,
`experiments/tuning/`, `.venv/`, `.env`, `legacy/fyp1/instance/`, `legacy/fyp1/data/`).
