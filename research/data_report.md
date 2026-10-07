# IEEE-CIS data report

Generated 2026-10-07 by `python -m vaultic.data.report` from `merged.parquet`. Do not edit by hand.

Data: anonymized e-commerce transactions released by Vesta for the IEEE-CIS competition (training files only; the Kaggle test files have no public labels and are not used).

## Overview

| Fact | Value |
|---|---|
| Transactions | 590,540 |
| Columns after merge | 436 (incl. added `has_identity`, `day`) |
| Fraud | 20,663 (3.499%) |
| Rows with identity data | 144,233 (24.42%) |
| TransactionDT range (s) | 86,400 – 15,811,131 |
| Day range (TransactionDT // 86400) | 1 – 182 (182 distinct days) |
| TransactionID unique | True |
| Sorted by TransactionDT | True |
| TransactionAmt min / median / p99 / max | 0.251 / 68.77 / 1,104.00 / 31,937.39 |
| Transactions per day min / median / max | 2,048 / 3,049 / 6,852 |

TransactionDT is seconds from an unknown reference point, so there are no calendar dates or local hours. `day` counts days from that reference.

## Per 30-day block

Blocks are 30-day windows counted from the first day in the data; the last block can be partial.

| block | first day | last day | days | rows | fraud rate % | has_identity % |
|---|---|---|---|---|---|---|
| 0 | 1 | 30 | 30 | 134,339 | 2.53 | 43.22 |
| 1 | 31 | 60 | 30 | 89,399 | 4.00 | 18.22 |
| 2 | 61 | 90 | 30 | 92,189 | 4.04 | 18.81 |
| 3 | 91 | 120 | 30 | 98,615 | 3.95 | 18.67 |
| 4 | 121 | 150 | 30 | 83,571 | 3.41 | 17.67 |
| 5 | 151 | 180 | 30 | 86,934 | 3.42 | 21.13 |
| 6 | 181 | 182 | 2 | 5,493 | 4.39 | 18.15 |

## Missing values by column group

| group | columns | mean missing % | min % | max % |
|---|---|---|---|---|
| card1-6 | 6 | 0.51 | 0.00 | 1.51 |
| addr1-2 | 2 | 11.13 | 11.13 | 11.13 |
| dist1-2 | 2 | 76.64 | 59.65 | 93.63 |
| email domains | 2 | 46.37 | 15.99 | 76.75 |
| C1-C14 (counts) | 14 | 0.00 | 0.00 | 0.00 |
| D1-D15 (time deltas) | 15 | 58.15 | 0.21 | 93.41 |
| M1-M9 (matches) | 9 | 49.92 | 28.68 | 59.35 |
| V1-V339 (Vesta features) | 339 | 43.04 | 0.00 | 86.12 |
| id_01-id_38 (identity) | 38 | 84.82 | 75.58 | 99.20 |
| DeviceType/DeviceInfo | 2 | 78.03 | 76.16 | 79.91 |

## ProductCD

| ProductCD | rows | fraud rate % |
|---|---|---|
| C | 68,519 | 11.69 |
| H | 33,024 | 4.77 |
| R | 37,699 | 3.78 |
| S | 11,628 | 5.90 |
| W | 439,670 | 2.04 |
