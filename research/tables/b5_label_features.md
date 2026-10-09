# Label-derived features in B5

Generated 2026-10-09 by `python -m vaultic.reports.label_features` on the 50k sample: base features rebuilt with all labels flipped (values) and with a longer label delay (timing). All of them respect the label delay L (its_time + L <= t).

| feature | derived from |
|---|---|
| `uid_fraud_known` | label values (known frauds) |
| `uid_fraud_rate_known` | label values (known frauds) |
| `uid_n_labels_known` | label timing (number of known labels, not their values) |

No other base feature changes. The raw IEEE-CIS columns (C, D, M, V, id_ and the transaction fields) are supplied by Vesta, not built here; whether any encode past labels cannot be checked from this data.
