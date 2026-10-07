# Customer ID (uid) reconstruction report

Generated 2026-10-07 by `python -m vaultic.data.uid`. Do not edit by hand.

IEEE-CIS has no customer ID, so one is reconstructed. Every label-based number below uses **days 1–150 only** (train + validation); the test period (days 151–182) is not used for this choice.

## Variants

| variant | definition |
|---|---|
| `card1` | card1 (coarse reference, not a candidate) |
| `uid_card` | card1 + card2 + card3 + card5 |
| `uid` | card1 + addr1 + (day − D1) |
| `uid2` | card1 + addr1 + (day − D1) + P_emaildomain |

Missing components are kept as an explicit `NA` token, so every row gets an ID. `complete key %` is the share of rows where every component is present.

## Comparison (days 1–150, 498,113 rows)

| variant | ids | rows per id (mean) | rows per id (median) | rows per id (p99) | rows in repeat ids % | repeat ids that are pure % | rows in mixed ids % | fraud rows in all-fraud repeat ids % | fraud rate of singleton ids % | complete key % |
|---|---|---|---|---|---|---|---|---|---|---|
| `card1` | 12,904 | 38.60 | 4.00 | 695.88 | 99.33 | 85.48 | 75.43 | 0.99 | 3.90 | 100.00 |
| `uid_card` | 13,932 | 35.75 | 4.00 | 604.76 | 99.22 | 85.70 | 74.82 | 1.07 | 3.69 | 98.16 |
| `uid` | 190,501 | 2.61 | 1.00 | 19.00 | 77.82 | 96.70 | 5.48 | 51.75 | 2.34 | 88.54 |
| `uid2` | 239,271 | 2.08 | 1.00 | 14.00 | 68.49 | 97.75 | 2.60 | 67.61 | 2.49 | 73.39 |

How to read it:

- **repeat ids that are pure %**: among IDs with 2+ transactions, the share whose transactions are all fraud or all legitimate. Higher means an ID looks more like one person.
- **rows in mixed ids %**: rows sitting in IDs that mix fraud and legitimate transactions (lower is better).
- **fraud rows in all-fraud repeat ids %**: of fraud rows in repeat IDs, the share in IDs that are entirely fraud.
- **rows in repeat ids %**: how much history the ID provides; splitting everyone into singletons would give perfect purity and no history.

## History carry-over (no labels)

Share of rows on days 121–150 whose ID already appeared on days 1–120, i.e. how often the validation period can use history built in training:

| variant | carry-over % | IDs over all 182 days |
|---|---|---|
| `card1` | 98.89 | 13,553 |
| `uid_card` | 98.68 | 14,845 |
| `uid` | 46.79 | 217,850 |
| `uid2` | 42.42 | 273,919 |

## Choice

Purity rule (first, provisional pick, D8): among `uid_card`, `uid` and `uid2`, the highest **repeat ids that are pure %** with **rows in repeat ids %** ≥ 50 gives `uid2`.

**In use: `uid`** (`experiments/configs/splits.yaml`). The final choice was made on the downstream metric, B5 validation PR-AUC (research/decisions.md D18). The uid reconstruction is a stated limitation in every paper.
