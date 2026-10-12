# B5_DROP_UFRK SHAP top 20 (mean |SHAP|)

Generated 2026-10-08 by `python -m vaultic.reports.shap_summary experiments/configs/EXP-009-drop-ufrk.yaml`: EXP-009-drop-ufrk, seed 0, 5,000 random validation rows. Masked IEEE-CIS features (C, D, M, V, id_) have no published meaning.

| rank | feature | mean abs SHAP |
|---|---|---|
| 1 | `uid_n_labels_known` | 0.5979 |
| 2 | `C13` | 0.3007 |
| 3 | `C5` | 0.2672 |
| 4 | `uid_fraud_known` | 0.2656 |
| 5 | `TransactionAmt` | 0.2395 |
| 6 | `freq_uid` | 0.2318 |
| 7 | `C1` | 0.2221 |
| 8 | `C14` | 0.1912 |
| 9 | `card6` | 0.1650 |
| 10 | `freq_card1` | 0.1485 |
| 11 | `D1` | 0.1357 |
| 12 | `C11` | 0.1229 |
| 13 | `card1` | 0.1194 |
| 14 | `D2` | 0.1193 |
| 15 | `card2` | 0.1108 |
| 16 | `D3` | 0.1080 |
| 17 | `dist1` | 0.1053 |
| 18 | `D15` | 0.1025 |
| 19 | `M5` | 0.0982 |
| 20 | `M6` | 0.0929 |
