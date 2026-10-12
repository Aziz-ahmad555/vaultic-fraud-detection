# B5_BEHAVIORAL SHAP top 20 (mean |SHAP|)

Generated 2026-10-09 by `python -m vaultic.reports.shap_summary experiments/configs/EXP-103-frozen.yaml`: EXP-103-frozen, seed 0, 5,000 random validation rows. Masked IEEE-CIS features (C, D, M, V, id_) have no published meaning.

| rank | feature | mean abs SHAP |
|---|---|---|
| 1 | `uid_fraud_rate_known` | 0.9615 |
| 2 | `C13` | 0.3119 |
| 3 | `freq_uid` | 0.2673 |
| 4 | `C1` | 0.2279 |
| 5 | `TransactionAmt` | 0.2255 |
| 6 | `C5` | 0.2124 |
| 7 | `C14` | 0.1802 |
| 8 | `freq_card1` | 0.1690 |
| 9 | `card6` | 0.1650 |
| 10 | `hist_days_since_first` | 0.1389 |
| 11 | `C11` | 0.1141 |
| 12 | `card2` | 0.1081 |
| 13 | `dist1` | 0.1044 |
| 14 | `card1` | 0.1040 |
| 15 | `D1` | 0.1040 |
| 16 | `D15` | 0.1006 |
| 17 | `M5` | 0.0971 |
| 18 | `D2` | 0.0960 |
| 19 | `M4` | 0.0912 |
| 20 | `M6` | 0.0889 |
