# Phase 5: temporal view: GRU attempts and diagnosis (D74, D85–D88)

Development work, **validation period only**.

**Setup:**
- Fixed plan: views trained on days 1–120, predicting validation days 128–150.
- Early stopping on the latest 20% of the training rows (D45), one seed.
- Inputs: the uid's last 20 transactions. Step channels: log amount, log gap, ProductCD, C1, C13, D1, dist1. Plus the scored transaction's own step.

**Evaluation:**
- Scored on the 41,837 validation rows with history (1,526 frauds).
- The temporal view is masked for cold-start uids (rule 11).
- B5 on the same rows: 0.7354.

## Result

| attempt | change | validation PR-AUC (rows with history) | 1–4 past | 5–19 past | 20+ past | best epoch (early-stop PR-AUC) |
|---|---|---|---|---|---|---|
| v1 | original training: balanced class weight, lr 1e-3, plain standardisation | 0.122 | 0.129 | 0.125 | 0.088 | 0 (0.133) |
| v2 | pos_weight sqrt(neg/pos), lr decay 0.9 per epoch, gradient clipping 1.0 | 0.124 | 0.131 | 0.129 | 0.091 | 0 (0.130) |
| v3 | v2 + **robust inputs**: signed log1p, then standardise and clip to ±5 SD | 0.263 | 0.322 | 0.258 | 0.141 | 32 (0.356) |
| v4 | v3 with a slower schedule: lr decay 0.97, patience 8, up to 80 epochs | 0.267 | 0.333 | 0.254 | 0.123 | 15 (0.377) |
| **XGBoost, same inputs** | last 20 steps flattened (padding as NaN) + number of real steps + current step; depth 6, lr 0.05, early stopping | **0.311** | | | | iteration 641 (0.475) |

Bar set before v3 ran (D86): 90% of XGBoost, i.e. 0.28. The best GRU (v4) reaches 86%.

## Diagnosis

1. **The inputs carry signal** (step 1, D74). XGBoost on exactly the GRU's inputs reaches 0.311,
   so the GRU's 0.12 was a training or input problem, not missing information.
2. **The GRU can learn** (step 3). It overfits a 256-row batch: train PR-AUC 0.92 in v2 and 0.95 in v3/v4.
   - v1 and v2 peak after their first epoch and then decline, so the optimiser settings (v2) were not the problem.
3. **The input scale was the main fault** (D85).
   - The C and D counters are heavy-tailed. Under plain standardisation one extreme value squashes every other value to almost one point, and a neural network then sees almost no variation.
   - XGBoost only uses the order of values, so it is unaffected.
   - The signed log1p + clipping fix (v3) more than doubles validation PR-AUC (0.12 → 0.26).
4. **A slower learning-rate schedule (v4) helps the early-stopping rows** (0.356 → 0.377) but barely moves validation (0.263 → 0.267).
   - The remaining gap to XGBoost (0.267 vs 0.311) is not closed by training longer.
5. **The GRU gets weaker as history grows** (0.33 for 1–4 past transactions, 0.12 for 20+), where B5
   is strongest. It doesn't capture the long-history signal that B5 gets from its delay-respecting fraud-history features (D65).
   - This was not investigated further (the one allowed attempt was used).

## Decision (D86 → D88)

The GRU is **not** kept as the temporal view. MVAF's temporal view is **XGBoost on the sequence
inputs**. `vaultic.views.temporal_diag` writes its predictions in the external-view format
(`p_temporal_xgb.parquet`, NaN without history).

The GRU code, `robust_inputs` and every attempt's predictions stay in the repository for the thesis
chapter. Reported as a negative finding: a small GRU over the last 20 transactions does not beat a
tree model on the same flattened inputs here.

**Sources:**
- `research/tables/temporal_dev.md` (v1), `temporal_dev_v2.md`, `temporal_dev_v3.md`, `temporal_dev_v4.md`
- training curves in `data/interim/temporal_dev*/p_temporal.history.json`
- XGBoost baseline in `data/interim/temporal_dev/diag.json`

Each table also compares the attempt with B5 alone and with an unfitted rank-mean fusion. The fusion is below B5 in every
version: a weak view averaged in with equal weight costs accuracy. Learning when to trust a view is
the job of a gate such as MVAF.
