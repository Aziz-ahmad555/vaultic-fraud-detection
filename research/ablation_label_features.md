# Ablation: B5 without its label-derived group (item 2, D65)

Development run, **validation period only**, tuned B5 hyperparameters frozen (EXP-009). Config
`experiments/configs/EXP-009-drop-labels.yaml`, run
`experiments/runs/EXP-009-drop-labels/20261009-101431-746940` (5 seeds, no flagged warnings).

**Dropped:** `uid_fraud_known` and `uid_fraud_rate_known`, the uid's known fraud count and known
fraud rate. Both respect the label delay, L = 30 days.

**Kept:** `uid_n_labels_known`, the number of the uid's labels already known. This is label timing,
not label values.

## Result

| model | validation PR-AUC (95% CI) |
|---|---|
| B5 (EXP-009) | 0.6831 (0.6659–0.7001) |
| B5 without uid_fraud_rate_known only (EXP-009-drop-ufrk, earlier) | 0.6768 (0.6592–0.6939) |
| **B5 without the label-derived group** | **0.5863 (0.5657–0.6051)** |
| B4 LightGBM, raw (EXP-013) | 0.5904 (0.5701–0.6096) |
| B3 XGBoost, raw (EXP-011) | 0.5767 (0.5549–0.5960) |

The B3, B4 and B5 values are the validation metrics of their Phase 2 runs; only their validation
predictions were read.

### Paired bootstrap, drop-labels vs B5, by customer history

From `research/tables/gain_b5_drop_labels.md`: 1,000 resamples, same rows for both runs.

| group | rows | frauds | B5 | drop-labels | difference | 95% CI | p |
|---|---|---|---|---|---|---|---|
| all | 60,602 | 2,114 | 0.6831 | 0.5863 | −0.0968 | [−0.1085, −0.0864] | <0.001 |
| ≥ 5 past transactions | 20,766 | 743 | 0.8568 | 0.6944 | −0.1624 | [−0.1870, −0.1404] | <0.001 |
| 1–4 past | 21,071 | 783 | 0.5996 | 0.5416 | −0.0580 | [−0.0739, −0.0440] | <0.001 |
| cold start | 18,765 | 588 | 0.5464 | 0.5432 | −0.0032 | [−0.0066, +0.0003] | 0.060 |

### Paired bootstrap, drop-labels vs the raw GBDT baselines (validation)

- vs B3: +0.0096 (95% CI +0.0032 to +0.0160, p = 0.004)
- vs B4: −0.0041 (95% CI −0.0107 to +0.0023, p = 0.24)

## Reading

- **Almost all of B5's margin over the raw GBDTs comes from the two label-derived features.**
  B5's validation margin over B3 is +0.106. Without the group, the margin is +0.010, and B5 is
  level with B4.
  - The V-reduction and the label-free point-in-time uid features (history, velocity,
    frequency encodings) add little on their own.
- **The loss grows with customer history** and is about zero for cold-start uids, who have no
  known labels. This is what a genuine, delay-respecting fraud-history signal looks like.
  - The leakage tests cover these features: label delay, truncation and label flips
    (`tests/test_leakage.py`).
- **Dropping only the rate costs little (−0.006)** because the known fraud count carries the same
  information. That's why the earlier single-feature ablation looked harmless.
- **Thesis framing:** B5 beating B3 on test (+0.092, Phase 2 gate) should be described as
  "point-in-time customer fraud history (label-delayed)", not as general feature engineering.

## SHAP top 10 of the ablated model

Validation sample, TreeExplainer. Table: `research/tables/b5_drop_labels_shap_top20.md`; figure:
`research/figures/b5_drop_labels_shap_summary.png`.

| rank | feature | mean \|SHAP\| |
|---|---|---|
| 1 | C13 | 0.3508 |
| 2 | C5 | 0.3078 |
| 3 | freq_uid | 0.2489 |
| 4 | TransactionAmt | 0.2404 |
| 5 | C1 | 0.2239 |
| 6 | C14 | 0.1937 |
| 7 | freq_card1 | 0.1667 |
| 8 | card6 | 0.1650 |
| 9 | card1 | 0.1327 |
| 10 | D2 | 0.1282 |

The C and D columns are masked counters and timedeltas with no published meaning. With the label
group gone, `uid_n_labels_known` drops out of the top 10. It ranked first when only the rate was
dropped, where it worked together with `uid_fraud_known`.

## Every label-derived feature in B5

Found by perturbation, not by name: `python -m vaultic.reports.label_features` rebuilds the base
features with every label flipped and with a longer label delay
(`research/tables/b5_label_features.md`).

| feature | derived from |
|---|---|
| `uid_fraud_known` | label values (known frauds of the uid) |
| `uid_fraud_rate_known` | label values |
| `uid_n_labels_known` | label timing (how many labels are known, not their values) |

No other base feature depends on labels. The raw IEEE-CIS columns come from Vesta; whether any of
them encode past labels can't be checked from this data.
