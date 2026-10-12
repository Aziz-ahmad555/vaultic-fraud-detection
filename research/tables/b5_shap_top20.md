# B5 SHAP top 20 (mean |SHAP|)

Generated 2026-10-08 by `python -m vaultic.reports.shap_summary experiments/configs/EXP-009.yaml`: EXP-009, seed 0, 5,000 random validation rows. Masked IEEE-CIS features (C, D, M, V, id_) have no published meaning.

| rank | feature | mean abs SHAP |
|---|---|---|
| 1 | `uid_fraud_rate_known` | 0.9021 |
| 2 | `C13` | 0.3234 |
| 3 | `freq_uid` | 0.2428 |
| 4 | `TransactionAmt` | 0.2398 |
| 5 | `C1` | 0.2251 |
| 6 | `C5` | 0.1949 |
| 7 | `C14` | 0.1868 |
| 8 | `card6` | 0.1638 |
| 9 | `freq_card1` | 0.1613 |
| 10 | `C11` | 0.1276 |
| 11 | `D1` | 0.1237 |
| 12 | `card2` | 0.1171 |
| 13 | `D15` | 0.1156 |
| 14 | `card1` | 0.1151 |
| 15 | `dist1` | 0.1106 |
| 16 | `V70` | 0.1053 |
| 17 | `D2` | 0.1002 |
| 18 | `freq_P_emaildomain` | 0.0939 |
| 19 | `M5` | 0.0918 |
| 20 | `M6` | 0.0904 |
