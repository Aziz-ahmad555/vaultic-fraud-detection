# DVC remote on a Google Drive folder

The raw IEEE-CIS CSVs are registered with DVC (`data/raw/*.dvc` hold their md5 and size) but are
**not yet in any remote**: they were added with `--no-commit`, so nothing was copied or moved.
These steps put them on a Google Drive folder, so the data can be restored on another machine
with `dvc pull` (solo project, D31).

Already done in the repo: `dvc init`, the `.dvc` pointer files, and the Google Drive plugin
(`dvc-gdrive`, pinned in `requirements.txt`).

Google blocks the OAuth app that ships with DVC, so you need your own OAuth client (free,
about 10 minutes, done once). Secrets go in `.dvc/config.local`, which is git-ignored, so
they are never committed.

## 1. Create the folder

1. In Google Drive, create a folder, e.g. `vaultic-dvc`.
2. Open the folder and copy its ID from the address bar:
   `https://drive.google.com/drive/folders/<FOLDER_ID>`.

## 2. Create an OAuth client (once)

1. Go to <https://console.cloud.google.com/>, create a project (e.g. `vaultic-dvc`).
2. **APIs & Services → Library**: enable **Google Drive API**.
3. **APIs & Services → OAuth consent screen**: user type *External*, app name `vaultic-dvc`,
   your email as support/developer contact. Leave it in **Testing** and add your own Gmail
   address as a **test user**.
4. **APIs & Services → Credentials → Create credentials → OAuth client ID**, type
   **Desktop app**. Copy the **client ID** and **client secret** and keep them private
   (not in git, not in chat logs that get committed).

## 3. Connect the repo (run in the repo root with `.venv` activated)

```powershell
dvc remote add -d gdrive gdrive://<FOLDER_ID>
dvc remote modify --local gdrive gdrive_client_id "<CLIENT_ID>"
dvc remote modify --local gdrive gdrive_client_secret "<CLIENT_SECRET>"
```

The first command changes `.dvc/config` (commit it); the `--local` ones write to the
git-ignored `.dvc/config.local`.

## 4. Upload the data

```powershell
dvc commit data/raw/train_transaction.csv.dvc data/raw/train_identity.csv.dvc
dvc push
```

- `dvc commit` copies the two CSVs (about 710 MB) into `.dvc/cache` and re-links the
  workspace files from it. The content is unchanged (DVC verifies the md5), but it does
  rewrite the files in `data/raw/`, which is why this step was left for you to run.
  The files are read-only; if DVC reports a permission error, run
  `attrib -R data\raw\*.csv` first and `attrib +R data\raw\*.csv` afterwards.
- `dvc push` opens a browser the first time: sign in with the Google account that owns the
  folder and allow access. The token is cached locally.

## 5. Another machine

To restore the data elsewhere (e.g. a new laptop), after cloning the repo and creating `.venv`:

```powershell
dvc remote modify --local gdrive gdrive_client_id "<CLIENT_ID>"
dvc remote modify --local gdrive gdrive_client_secret "<CLIENT_SECRET>"
dvc pull
```

Sign in with the same Google account (the test user from step 2.3).

## Later: CI

GitHub Actions cannot use the browser sign-in. If CI needs the real data, create a Google
service account, share the folder with its email, store its JSON key as a GitHub secret, and
set `gdrive_use_service_account true`. Until then CI runs the leakage tests on synthetic data.
