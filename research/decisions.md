# Decisions log

Every non-obvious choice made while building Vaultic: what was chosen, the alternatives, and why.
Entries marked **NEEDS CONFIRMATION** are provisional until Aziz confirms them.

| # | Date | Area | Decision | Alternatives | Why | Status |
|---|---|---|---|---|---|---|
| D1 | 2026-10-07 | Repo | FYP-1 kept runnable in `legacy/fyp1/`; research code in `src/vaultic/` | Separate repo; leave FYP-1 at root | One history, as the roadmap asks; FYP-1 still needed for baseline B6 and Phase 11 | Approved |
| D2 | 2026-10-07 | Repo | `platform/` renamed `services/` | Keep `platform/` but never import it | `platform` shadows Python's stdlib module used by pandas/numpy/pytest | Approved |
| D3 | 2026-10-07 | Repo | The restructure commit (`91a0235`) is on `main`; all later work is on branch `phase0-1` | Move the restructure onto the branch too | It was finished and verified before the branch instruction; rewriting `main` history is riskier than leaving it | NEEDS CONFIRMATION |
| D4 | 2026-10-07 | Installs | Installs run with `-c legacy/fyp1/requirements.txt` (constraints) | Unconstrained installs | Stops new packages from silently upgrading pandas/numpy/scikit-learn/xgboost, which would break FYP-1's saved models | NEEDS CONFIRMATION |
| D5 | 2026-10-07 | Data | Exact dtypes kept for `TransactionID`, `TransactionDT` (int) and `TransactionAmt` (float64); other numeric columns float32; strings as category | Downcast everything to float32 | uid reconstruction and amount-based features need exact keys and exact cents; float32 elsewhere halves memory on a 16 GB machine | Default |
| D6 | 2026-10-07 | Data | A `month` is a 30-day block counted from the first day in the data (day 1); the last block (days 181-182) is partial | Calendar months (impossible: no real dates); 6 equal blocks of ~30.3 days | Simple, integer days, and matches the roadmap's monthly analysis; partial block stated in the report | Default |
