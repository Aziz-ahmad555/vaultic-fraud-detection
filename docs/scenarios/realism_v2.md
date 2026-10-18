# Generator realism check (v2) - synthetic scenario - not the evaluation

Masked-field missingness of synthetic rows (seeds 0-4 x low / medium / high) vs real rows of the same customer type (new: young accounts, masked time-delta D1 0-13; existing: customer with >= 3 earlier rows). Cells: share of rows with the whole group missing / mean share of the group's fields missing. 'Unseen profile': share of synthetic rows whose exact missingness pattern over all M, V and id fields occurs in no real row of the type.

Scope: this check shows only that the synthetic rows' missingness patterns are realistic (they occur in real rows of the customer type); it does not show that their values, or the joint distribution of values, are realistic.

| rows | n | M | V | id | C | D2-D15 | unseen profile |
|---|---|---|---|---|---|---|---|
| real new | 324,650 | 0.23 / 0.56 | 0.00 / 0.40 | 0.62 / 0.77 | 0.00 / 0.00 | 0.09 / 0.68 | - |
| real existing | 223,483 | 0.07 / 0.46 | 0.00 / 0.44 | 0.80 / 0.88 | 0.00 / 0.00 | 0.01 / 0.55 | - |
| card_testing (existing) | 210 | 0.00 / 0.21 | 0.00 / 0.50 | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.59 | 0.00 |
| account_takeover (existing) | 30 | 0.20 / 0.91 | 0.00 / 0.27 | 0.00 / 0.43 | 0.00 / 0.00 | 0.10 / 0.45 | 0.00 |
| mule_ring (new) | 175 | 0.40 / 0.93 | 0.00 / 0.26 | 0.00 / 0.38 | 0.00 / 0.00 | 0.18 / 0.61 | 0.03 |
| bust_out (new) | 265 | 0.01 / 0.21 | 0.00 / 0.50 | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.71 | 0.00 |
| mimicry (existing) | 50 | 0.00 / 0.14 | 0.00 / 0.49 | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.57 | 0.00 |
