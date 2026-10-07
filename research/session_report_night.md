# Night session report — 2026-10-08

Worked only in the `.worktrees/phase7-prep` worktree (branch `phase7-prep`). Synthetic or small hand-made data only; no full-data training. The Phase 2 chain and its `.venv` were not touched, nothing was merged, and nothing was pushed.

## Chain status (read-only check)

B5 tuning is finished (EXP-009 written at 00:25). B3 tuning was running at 02:53 (`none` arm, trial 10 of 25, best validation PR-AUC so far 0.5751). B4 and B1, then the final runs, are still to come.

## What was done

| # | Item | Commit | Decision |
|---|---|---|---|
| 1 | Shared category codes fitted on the training period (fixes the D36 caveat) | e0d040a | D37 |
| 2 | Phase 9 explanation layer: reason codes, view contributions, case narrative, DiCE wrapper | a072a86 | D38 |
| 3 | Phase 13: planned configs E3–E15 and the new E25, plus harness options; E25 added to README as MUST | d3273f1 | D39 |
| 4 | Phase 5 GRU temporal view in a separate CPU PyTorch venv | e049d81 | D40 |

### 1. Category codes (`src/vaultic/features/categories.py`)

- `CategoryEncoder` is fitted on training-period rows only and can be saved as JSON.
- Codes: 0 = missing, 1 = unknown (a value never seen in training), 2 onwards = training categories in sorted order.
- The sequence builder now requires it, and the anomaly view uses it for DataFrame inputs.
- It is deliberately **not** applied to the graph or to the baselines; see "Needs your confirmation".

### 2. Explanation layer (`src/vaultic/explain/`)

- **`reason_codes.py`** maps each feature to a plain-language template.
  - A test builds the base, behavioral, graph and anomaly features on synthetic data and fails if any of them has no template.
  - Masked raw features only get honest wording: "Unusual value in masked counter C13" when they raise risk, and "Value in masked counter C13 lowers the risk score" when they lower it. Missing values get an explicit "no value" sentence.
  - Points = SHAP log-odds × 20 / ln 2 (the credit-scoring "points to double the odds" convention), so +20 points means the fraud odds doubled.
- **`view_contrib.py`**: each view's contribution is its gate weight × its logit. Together with the gate's bias this exactly rebuilds MVAF's fused logit (tested). Missing views get no share at all, rather than 0%. Output reads like "Graph evidence 45% (raises risk), …".
- **`narrative.py`**: a fixed template (no LLM) that combines the decision, top reasons, view shares, graph evidence and counterfactual. Any part with no evidence is stated as missing.
- **`counterfactual.py`**: a DiCE wrapper with three feature classes.
  - Immutable (never varied): card, addr, dist, C/D, all history, frequency, entity and graph features, time.
  - Derived (amount z-score and ratio, novelty flags, anomaly scores): locked unless a `derive` callback recomputes them.
  - Actionable (the default set DiCE may vary): amount, product, email domains, device.
  - The lock is enforced twice: DiCE is told what it may vary, then every returned candidate is checked again. The 9.3 quality metrics (validity, proximity, features changed, actionable) are included.

### 3. Experiment configs (`experiments/configs/EXP-103` … `EXP-115`, `EXP-125`)

- Numbering: EXP-1NN = roadmap experiment ENN, so these never clash with the chain's EXP-0xx configs.
- Every config has `status: planned` and a `needs:` list, and the harness refuses to run them.
- New harness options, both tested on synthetic data:
  - `extends:` inherits another config from the same folder (status and needs are not inherited).
  - `train_label_maturity_days` keeps a training row only if its time + L ≤ the start of validation. For L = 30 that means TransactionDT ≤ day 98.
- **E25** extends EXP-009 (the frozen B5) with `train_label_maturity_days: 30`, keeping the same hyperparameters. It can't run in the worktree until the chain finishes and EXP-009 is merged.

### 4. GRU temporal view (`src/vaultic/views/temporal.py`)

- **Environment:** CPU PyTorch 2.14.1 in `.venv-torch` (git-ignored), recreated by `tools/setup_torch_venv.ps1`. The shared libraries are pinned to the same versions as the main venv; the full package list is in `requirements-torch.txt`.
- **Model:**
  - A masked GRU reads the history; a padded step provably leaves the state unchanged.
  - The scored transaction's own features are added: `Sequences.current`, a new field in the sequence builder.
  - ProductCD goes through an embedding.
  - Training uses weighted BCE (balanced) and stops early on validation PR-AUC, keeping the best epoch.
  - Uids with no history get NaN (masked view).
- **Synthetic check:** PR-AUC 1.0 at 12.9% prevalence (3,600 rows, 3,895 parameters, best epoch 13 of 20). The pattern is deliberately easy, so this shows the model uses history correctly; it says nothing about real-data performance.
- **Not done:** LSTM, TCN and Transformer (roadmap Phase 5). D31 scope is GRU only.

## Test results

- **Main `.venv`:** 251 passed, 10 skipped (full suite, 3 min 14 s). The skips are the torch and DiCE tests plus the earlier data-dependent skips.
- **`.venv-torch`:** 36 passed: temporal, explain (including the real DiCE lock test), sequences and categories.
- ruff and black are clean, and the pre-commit hooks passed on every commit.
- **Bug found by a test and fixed before committing:** garbage values in padded steps could reach the ProductCD embedding. Padded steps are now forced to code 0, and out-of-range codes raise a clear error.

## Needs your confirmation

1. **D37:**
   - The category encoder is **not** applied to graph node keys. Nodes are entity identities; mapping unseen cards or devices to "unknown" would merge them all into one fake hub.
   - **Baselines B1–B5 keep their current encoding** in `features/sets.py`, so the frozen Phase 2 configs and their `--final` runs stay reproducible.
2. **D38:**
   - The immutable / derived / actionable feature lists.
   - PDO = 20 for reason-code points.
   - Masked C/D columns count as "history" (locked).
3. **D39:**
   - The ablation rows E4, E5 and E8 add views by **F3 logistic-regression stacking**, as the roadmap's ablation table says "stacking". Should it be F4 (LightGBM) instead?
   - The graph view = XGBoost on the graph features alone.
   - E3 is retuned with the same 50-trial budget; E25 is **not** retuned, to isolate the effect of the label set.
   - E6 adds a label-delay sweep (L = 0/7/30/60) for Paper 1.
4. **D40:** dice-ml pulled **xgboost 3.2.0 and lightgbm 4.7.0** into `.venv-torch` (the main venv has xgboost 2.0.3). Baselines must only ever run from the main `.venv`. If you prefer, dice-ml can go in a third venv.

## Blockers

None. `research/blockers.md` is unchanged.

## Next (after the chain finishes)

1. Commit the chain results.
2. Merge `phase7-prep` into `phase0-1` following the D32 checklist (PR + review by a new session).
3. Remove `status: planned` from EXP-125 once you approve, and run its development run.
