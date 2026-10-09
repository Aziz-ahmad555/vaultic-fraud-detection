# Phase 6: anomaly view and the emerging-fraud experiment (D68, D69, D93)

Development runs, **validation period only**, with B5's frozen tuned hyperparameters.

**Anomaly scores** (`views/anomaly_build.py`):
- Global Isolation Forest, autoencoder, and per-customer Isolation Forest.
- Fitted only on legit rows of earlier days (forward-chained 30-day blocks), so every score is out-of-sample.
- Inputs: the B5 features without the label-derived ones.
- Coverage: 77% of rows. The training period's first 30 days have no earlier data and stay masked.
- The per-customer IF covers only 0.6% of validation rows: it needs ≥ 20 of the customer's transactions inside the fitting window.

## Anomaly scores alone (label-free detectors)

| rows | frauds | base rate | Isolation Forest PR-AUC | autoencoder PR-AUC |
|---|---|---|---|---|
| all validation | 2,114 | 3.5% | 0.105 | 0.103 |
| ProductCD R (validation) | 93 | 3.8% | 0.066 | 0.067 |

On all rows the scores rank fraud about 3× better than chance. On ProductCD R they're only 1.7× better.

## B5 + anomaly scores

From `research/tables/gain_b5_anomaly.md`:

| group | B5 | B5 + anomaly | gain | 95% CI |
|---|---|---|---|---|
| all | 0.6831 | 0.6771 | −0.0060 | [−0.0087, −0.0034] |
| ≥ 5 past | 0.8568 | 0.8524 | −0.0044 | [−0.0073, −0.0018] |
| 1–4 past | 0.5996 | 0.5888 | −0.0108 | [−0.0161, −0.0057] |
| cold start | 0.5464 | 0.5471 | +0.0007 | [−0.0058, +0.0070] |

## Emerging fraud (D69)

ProductCD R's 1,063 training-period frauds were removed from training. Both models below were then scored on the 2,424 ProductCD R validation rows (93 frauds).

| model | PR-AUC on R rows | vs the row above |
|---|---|---|
| full B5 (saw R frauds), reference | 0.7953 | |
| B5 without R frauds | 0.5648 | −0.2305 [−0.3065, −0.1629] (`gain_holdout_R_b5.md`) |
| B5 + anomaly without R frauds | 0.5539 | −0.0109 [−0.0349, +0.0109], p = 0.34 (`gain_holdout_R_anomaly.md`) |

## Reading

1. **The hold-out creates a real "unseen fraud" gap.** B5 loses 0.23 PR-AUC on a product whose
   fraud it never saw.
2. **The anomaly scores do not close it.**
   - B5 + anomaly is no better on those rows (−0.011, not significant).
   - The scores alone barely separate R fraud (0.066 against a 3.8% base rate).
   - This hypothesis does not hold here: on this data, global label-free novelty does not pick out R fraud.
3. **As extra features, the anomaly scores slightly hurt B5** (−0.006), as behavioral (−0.006) and
   graph (−0.019) did with frozen hyperparameters. None of these feature-level additions was
   retuned; E3/E5/E8 retune with B5's budget.
4. **Small numbers.** 93 R frauds in validation: the CIs are wide.

Under the view redesign (D89), the anomaly view enters MVAF as its own view. Whether a gate can
use it where B5 is weak is MVAF's question, not this one's.
