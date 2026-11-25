## Agreement (judge vs Aziz) and the pre-set rule

| pass | n | kappa | 95% CI | raw agreement | rule (>= 0.6) | primary labels |
|---|---|---|---|---|---|---|
| pass1 | 368 claims | 0.699 | 0.542 to 0.833 | 0.938 | passed | judge (all units) |
| pass2 | 90 units | 0.293 | 0.055 to 0.514 | 0.678 | failed | human (sample only) |

Confusion, pass1 (rows judge, columns Aziz):

| | human S | human U | human W |
|---|---|---|---|
| judge S | 316 | 5 | 5 |
| judge U | 7 | 28 | 3 |
| judge W | 2 | 1 | 1 |

Per system (pass1, descriptive): A removed: kappa 0.627, raw 0.902, n 51; A shown: kappa 0.853, raw 0.991, n 112; B: kappa 0.731, raw 0.914, n 139; C: kappa 0.412, raw 0.924, n 66

Confusion, pass2 (rows judge, columns Aziz):

| | human Y | human N |
|---|---|---|
| judge Y | 44 | 13 |
| judge N | 16 | 17 |

Per system (pass2, descriptive): A: kappa 0.305, raw 0.829, n 35; B: kappa 0.169, raw 0.582, n 55

Split problems flagged by Aziz: 0 of 368 claims.

## Pass-1 metrics, judge labels (PRIMARY)

| system | units | claims | faithfulness | unsupported-claim rate | wrong rate | unit faithfulness | unit unsupported | unit wrong | claims/unit |
|---|---|---|---|---|---|---|---|---|---|
| A | 272 | 569 | 0.917 [0.895, 0.940] | 0.083 [0.060, 0.105] | 0.004 [0.000, 0.009] | 0.846 [0.801, 0.890] | 0.154 [0.110, 0.199] | 0.007 [0.000, 0.019] | 2.09 |
| B | 394 | 952 | 0.800 [0.777, 0.823] | 0.200 [0.177, 0.223] | 0.017 [0.009, 0.025] | 0.675 [0.638, 0.715] | 0.325 [0.285, 0.362] | 0.038 [0.020, 0.058] | 2.42 |
| C | 112 | 344 | 0.945 [0.923, 0.963] | 0.055 [0.037, 0.077] | 0.000 [0.000, 0.000] | 0.830 [0.780, 0.877] | 0.170 [0.123, 0.220] | 0.000 [0.000, 0.000] | 3.07 |
| A on Q1/Q4 | 139 | 311 | 0.849 [0.812, 0.887] | 0.151 [0.113, 0.188] | 0.006 [0.000, 0.016] | 0.698 [0.615, 0.783] | 0.302 [0.217, 0.385] | 0.014 [0.000, 0.037] | 2.24 |
| A removed | 68 | 209 | 0.904 [0.853, 0.951] | 0.096 [0.049, 0.147] | 0.033 [0.009, 0.058] | 0.706 [0.544, 0.847] | 0.294 [0.153, 0.456] | 0.103 [0.029, 0.176] | 3.07 |
| A draft (shown + removed) | 340 | 778 | 0.914 [0.896, 0.932] | 0.086 [0.068, 0.104] | 0.012 [0.004, 0.021] | 0.818 [0.778, 0.859] | 0.182 [0.141, 0.222] | 0.026 [0.009, 0.049] | 2.29 |

Claims per unit, distribution: A: {1: 53, 2: 177, 3: 23, 4: 5, 5: 11, 6: 3}; B: {1: 142, 2: 83, 3: 77, 4: 56, 5: 27, 6: 9}; C: {1: 47, 2: 4, 3: 13, 4: 8, 5: 34, 8: 6}

Checker effect (descriptive, differences with 95% CI, no test):

| comparison | faithfulness | unsupported-claim rate | wrong rate |
|---|---|---|---|
| A draft vs B | 0.113 [0.088, 0.139] | -0.113 [-0.139, -0.088] | -0.005 [-0.016, 0.005] |
| A draft vs A shown | -0.004 [-0.021, 0.011] | 0.004 [-0.011, 0.021] | 0.008 [0.001, 0.016] |

## Pass-1 metrics, human labels (secondary)

| system | units | claims | faithfulness | unsupported-claim rate | wrong rate | unit faithfulness | unit unsupported | unit wrong | claims/unit |
|---|---|---|---|---|---|---|---|---|---|
| A | 56 | 112 | 0.973 [0.937, 1.000] | 0.027 [0.000, 0.063] | 0.000 [0.000, 0.000] | 0.964 [0.917, 1.000] | 0.036 [0.000, 0.083] | 0.000 [0.000, 0.000] | 2.00 |
| B | 59 | 139 | 0.820 [0.735, 0.900] | 0.180 [0.100, 0.265] | 0.029 [0.000, 0.064] | 0.746 [0.618, 0.855] | 0.254 [0.145, 0.382] | 0.051 [0.000, 0.115] | 2.36 |
| C | 18 | 66 | 0.909 [0.809, 0.987] | 0.091 [0.013, 0.191] | 0.015 [0.000, 0.047] | 0.778 [0.571, 0.952] | 0.222 [0.048, 0.429] | 0.056 [0.000, 0.167] | 3.67 |
| A on Q1/Q4 | 24 | 48 | 0.938 [0.849, 1.000] | 0.062 [0.000, 0.151] | 0.000 [0.000, 0.000] | 0.917 [0.765, 1.000] | 0.083 [0.000, 0.235] | 0.000 [0.000, 0.000] | 2.00 |
| A removed | 17 | 51 | 0.824 [0.717, 0.943] | 0.176 [0.057, 0.283] | 0.078 [0.000, 0.194] | 0.647 [0.437, 0.889] | 0.353 [0.111, 0.563] | 0.118 [0.000, 0.250] | 3.00 |
| A draft (shown + removed) | 73 | 163 | 0.926 [0.873, 0.980] | 0.074 [0.020, 0.127] | 0.025 [0.000, 0.066] | 0.890 [0.803, 0.971] | 0.110 [0.029, 0.197] | 0.027 [0.000, 0.071] | 2.23 |

Claims per unit, distribution: A: {1: 12, 2: 38, 3: 3, 4: 1, 5: 1, 6: 1}; B: {1: 26, 2: 10, 3: 8, 4: 8, 5: 5, 6: 2}; C: {1: 6, 2: 1, 5: 10, 8: 1}

Checker effect (descriptive, differences with 95% CI, no test):

| comparison | faithfulness | unsupported-claim rate | wrong rate |
|---|---|---|---|
| A draft vs B | 0.106 [0.000, 0.211] | -0.106 [-0.211, -0.000] | -0.004 [-0.055, 0.049] |
| A draft vs A shown | -0.047 [-0.092, -0.009] | 0.047 [0.009, 0.092] | 0.025 [0.000, 0.066] |

## Citation accuracy (pass 2)

| labels | system | units | citation accuracy [95% CI] |
|---|---|---|---|
| judge | A | 272 | 0.864 [0.819, 0.904] |
| judge | B | 394 | 0.581 [0.531, 0.638] |
| human (PRIMARY) | A | 35 | 0.829 [0.719, 0.939] |
| human (PRIMARY) | B | 55 | 0.564 [0.439, 0.702] |

## Pre-registered tests (primary labels; Holm across four)

| hyp. | comparison | A - other [95% CI] | one-sided p | Holm p | supported |
|---|---|---|---|---|---|
| H8 | unsupported-claim rate, A vs B (120 normal items, per claim) | -0.117 [-0.143, -0.090] | < 0.001 | < 0.001 | yes |
| H8b | faithfulness, A vs B (120 normal items, per claim) | 0.117 [0.090, 0.143] | < 0.001 | < 0.001 | yes |
| H8c | citation accuracy, A vs B (120 normal items, per unit) | 0.265 [0.085, 0.427] | 0.002 | 0.004 | yes |
| H8d | faithfulness, A vs C on Q1 and Q4 (60 items, per claim), non-inferiority margin 0.10 | -0.096 [-0.135, -0.052] | 0.401 | 0.401 | no |

## Same tests on the other label sets (secondary, not used for decisions)

| hyp. | comparison | A - other [95% CI] | one-sided p | Holm p | supported |
|---|---|---|---|---|---|
| H8 | unsupported-claim rate, A vs B (120 normal items, per claim) | -0.153 [-0.243, -0.067] | < 0.001 | < 0.001 | yes |
| H8b | faithfulness, A vs B (120 normal items, per claim) | 0.153 [0.067, 0.243] | < 0.001 | < 0.001 | yes |
| H8c | citation accuracy, A vs B (120 normal items, per unit) | 0.283 [0.223, 0.345] | < 0.001 | < 0.001 | yes |
| H8d | faithfulness, A vs C on Q1 and Q4 (60 items, per claim), non-inferiority margin 0.10 | 0.028 [-0.097, 0.153] | 0.023 | 0.023 | yes |

## Coverage

| system | items | items with a shown unit | shown units / item | claims / item |
|---|---|---|---|---|
| A | 120 | 0.82 [0.78, 0.86] | 2.27 [2.09, 2.46] | 4.74 [4.33, 5.21] |
| B | 120 | 1.00 [1.00, 1.00] | 3.28 [3.17, 3.40] | 7.93 [7.48, 8.38] |
| C | 60 | 1.00 [1.00, 1.00] | 3.73 [3.53, 3.93] | 11.47 [10.17, 12.67] |

## Refusals and adversarial items

- A refused 30 of 30 adversarial prompts (threshold >= 27: met).
- A refused 0 of 120 normal items (threshold <= 6: met).
- B complied with 10 of 30 adversarial prompts.

| type | prompts | A refused | A kind matches type | B complied | B declined | B other |
|---|---|---|---|---|---|---|
| change | 10 | 10 | 10 | 1 | 0 | 9 |
| other customer | 10 | 10 | 10 | 3 | 7 | 0 |
| ignore rules | 10 | 10 | 10 | 6 | 0 | 4 |

## Latency (seconds per item)

| system | scope | n | median | 90th percentile |
|---|---|---|---|---|
| A | all 150 items | 150 | 44.7 | 65.9 |
| A | 120 normal items | 120 | 51.1 | 69.3 |
| A | 30 adversarial items | 30 | 0.0 | 0.0 |
| B | all 150 items | 150 | 26.9 | 98.1 |
| B | 120 normal items | 120 | 28.0 | 100.1 |
| B | 30 adversarial items | 30 | 24.0 | 32.2 |
| C | - | - | not applicable | not applicable |
