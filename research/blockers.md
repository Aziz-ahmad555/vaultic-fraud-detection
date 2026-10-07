# Blockers

Problems that could not be fixed in about three attempts. Each entry: what failed, what was tried, what is needed.

| # | Date | Item | Problem | Tried | Needed |
|---|---|---|---|---|---|
| B1 | 2026-10-07 | Phase 0C Docker | `docker compose up` fails: Windows forbids binding 127.0.0.1:5432 (port reserved/excluded or used by another service) | 1 attempt | Check `netsh interface ipv4 show excludedportrange protocol=tcp`; or map to another host port (e.g. 55432) |
| B2 | 2026-10-07 | Installs | Group xgboost/lightgbm/catboost hung 70 min on the flaky network; retried one package at a time: ruff, black, pre-commit, lightgbm, optuna OK; shap, mlflow, dvc, catboost still installing at session end | 2 | Check the install log; then `pip freeze > requirements.txt` |
