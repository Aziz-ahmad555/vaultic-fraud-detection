# Attack scenario suite - synthetic scenario - not the evaluation

Seeds 0-4 per cell. Recall: alerted synthetic fraud rows, total over seeds (per-seed min - max). FPR: real legitimate replay rows, mean (min - max). Alert = step-up or higher (D196); hard = hold or block (extra reporting). A step-up is an extra customer check (friction), not a block; the FPR is split into step-up only and hold/block. 'D196 without R3 (post hoc)' was chosen after seeing the results (D201) and is a sensitivity row only.

## card_testing

**low, D196 (primary)** - budgets: added FPR ok 3/5 seeds (max +0.0444), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/25 (0/5 - 0/5) | 0.0243 (0.0056 - 0.0740) | 0.0082 (0.0031 - 0.0159) | 0/25 (0/5 - 0/5) | 0.0162 (0.0020 - 0.0659) |  |
| model_only | 6/25 (0/5 - 3/5) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/25 (0/5 - 0/5) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 6/25 (0/5 - 3/5) | 0.2459 (0.2173 - 0.2792) | 0.2295 (0.2133 - 0.2549) | 0/25 (0/5 - 0/5) | 0.0164 (0.0025 - 0.0659) |  |

**low, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0486), rules only <= 2% 2/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/25 (0/5 - 0/5) | 0.0373 (0.0136 - 0.0878) | 0.0145 (0.0071 - 0.0232) | 0/25 (0/5 - 0/5) | 0.0228 (0.0066 - 0.0716) |  |
| model_only | 6/25 (0/5 - 3/5) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/25 (0/5 - 0/5) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 6/25 (0/5 - 3/5) | 0.2519 (0.2222 - 0.2834) | 0.2289 (0.2118 - 0.2553) | 0/25 (0/5 - 0/5) | 0.0230 (0.0071 - 0.0716) |  |

**low, D196 without R3 (post hoc)** - budgets: added FPR ok 3/5 seeds (max +0.0155), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/25 (0/5 - 0/5) | 0.0139 (0.0056 - 0.0219) | 0.0105 (0.0031 - 0.0197) | 0/25 (0/5 - 0/5) | 0.0034 (0.0020 - 0.0052) |  |
| model_only | 6/25 (0/5 - 3/5) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/25 (0/5 - 0/5) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 6/25 (0/5 - 3/5) | 0.2401 (0.2173 - 0.2593) | 0.2365 (0.2133 - 0.2549) | 0/25 (0/5 - 0/5) | 0.0036 (0.0021 - 0.0052) |  |

**medium, D196 (primary)** - budgets: added FPR ok 3/5 seeds (max +0.0444), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 30/60 (6/12 - 6/12) | 0.0243 (0.0056 - 0.0740) | 0.0082 (0.0031 - 0.0159) | 0/60 (0/12 - 0/12) | 0.0162 (0.0020 - 0.0659) |  |
| model_only | 19/60 (0/12 - 6/12) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/60 (0/12 - 0/12) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 41/60 (6/12 - 10/12) | 0.2459 (0.2173 - 0.2792) | 0.2295 (0.2133 - 0.2549) | 0/60 (0/12 - 0/12) | 0.0164 (0.0025 - 0.0659) |  |

**medium, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0486), rules only <= 2% 2/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 35/60 (7/12 - 7/12) | 0.0373 (0.0136 - 0.0878) | 0.0145 (0.0071 - 0.0232) | 0/60 (0/12 - 0/12) | 0.0228 (0.0066 - 0.0716) |  |
| model_only | 19/60 (0/12 - 6/12) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/60 (0/12 - 0/12) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 43/60 (7/12 - 10/12) | 0.2519 (0.2222 - 0.2834) | 0.2289 (0.2118 - 0.2553) | 0/60 (0/12 - 0/12) | 0.0230 (0.0071 - 0.0716) |  |

**medium, D196 without R3 (post hoc)** - budgets: added FPR ok 3/5 seeds (max +0.0155), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 30/60 (6/12 - 6/12) | 0.0139 (0.0056 - 0.0219) | 0.0105 (0.0031 - 0.0197) | 0/60 (0/12 - 0/12) | 0.0034 (0.0020 - 0.0052) |  |
| model_only | 19/60 (0/12 - 6/12) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/60 (0/12 - 0/12) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 41/60 (6/12 - 10/12) | 0.2401 (0.2173 - 0.2593) | 0.2365 (0.2133 - 0.2549) | 0/60 (0/12 - 0/12) | 0.0036 (0.0021 - 0.0052) |  |

**high, D196 (primary)** - budgets: added FPR ok 3/5 seeds (max +0.0444), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 95/125 (19/25 - 19/25) | 0.0243 (0.0056 - 0.0740) | 0.0082 (0.0031 - 0.0159) | 0/125 (0/25 - 0/25) | 0.0162 (0.0020 - 0.0659) |  |
| model_only | 37/125 (0/25 - 21/25) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/125 (0/25 - 0/25) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 108/125 (19/25 - 24/25) | 0.2459 (0.2173 - 0.2792) | 0.2295 (0.2133 - 0.2549) | 0/125 (0/25 - 0/25) | 0.0164 (0.0025 - 0.0659) |  |

**high, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0486), rules only <= 2% 2/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 100/125 (20/25 - 20/25) | 0.0373 (0.0136 - 0.0878) | 0.0145 (0.0071 - 0.0232) | 0/125 (0/25 - 0/25) | 0.0228 (0.0066 - 0.0716) |  |
| model_only | 37/125 (0/25 - 21/25) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/125 (0/25 - 0/25) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 110/125 (20/25 - 24/25) | 0.2519 (0.2222 - 0.2834) | 0.2289 (0.2118 - 0.2553) | 0/125 (0/25 - 0/25) | 0.0230 (0.0071 - 0.0716) |  |

**high, D196 without R3 (post hoc)** - budgets: added FPR ok 3/5 seeds (max +0.0155), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 95/125 (19/25 - 19/25) | 0.0139 (0.0056 - 0.0219) | 0.0105 (0.0031 - 0.0197) | 0/125 (0/25 - 0/25) | 0.0034 (0.0020 - 0.0052) |  |
| model_only | 37/125 (0/25 - 21/25) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/125 (0/25 - 0/25) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 108/125 (19/25 - 24/25) | 0.2401 (0.2173 - 0.2593) | 0.2365 (0.2133 - 0.2549) | 0/125 (0/25 - 0/25) | 0.0036 (0.0021 - 0.0052) |  |

## account_takeover

**low, D196 (primary)** - budgets: added FPR ok 3/5 seeds (max +0.0444), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 5/5 (1/1 - 1/1) | 0.0243 (0.0056 - 0.0740) | 0.0082 (0.0031 - 0.0159) | 5/5 (1/1 - 1/1) | 0.0162 (0.0020 - 0.0659) |  |
| model_only | 2/5 (0/1 - 1/1) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/5 (0/1 - 0/1) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 5/5 (1/1 - 1/1) | 0.2459 (0.2173 - 0.2792) | 0.2295 (0.2133 - 0.2549) | 5/5 (1/1 - 1/1) | 0.0164 (0.0025 - 0.0659) |  |

**low, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0486), rules only <= 2% 2/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 5/5 (1/1 - 1/1) | 0.0373 (0.0136 - 0.0878) | 0.0145 (0.0071 - 0.0232) | 5/5 (1/1 - 1/1) | 0.0228 (0.0066 - 0.0716) |  |
| model_only | 2/5 (0/1 - 1/1) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/5 (0/1 - 0/1) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 5/5 (1/1 - 1/1) | 0.2519 (0.2222 - 0.2834) | 0.2289 (0.2118 - 0.2553) | 5/5 (1/1 - 1/1) | 0.0230 (0.0071 - 0.0716) |  |

**low, D196 without R3 (post hoc)** - budgets: added FPR ok 3/5 seeds (max +0.0155), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 5/5 (1/1 - 1/1) | 0.0139 (0.0056 - 0.0219) | 0.0105 (0.0031 - 0.0197) | 5/5 (1/1 - 1/1) | 0.0034 (0.0020 - 0.0052) |  |
| model_only | 2/5 (0/1 - 1/1) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/5 (0/1 - 0/1) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 5/5 (1/1 - 1/1) | 0.2401 (0.2173 - 0.2593) | 0.2365 (0.2133 - 0.2549) | 5/5 (1/1 - 1/1) | 0.0036 (0.0021 - 0.0052) |  |

**medium, D196 (primary)** - budgets: added FPR ok 3/5 seeds (max +0.0444), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 5/10 (1/2 - 1/2) | 0.0243 (0.0056 - 0.0740) | 0.0082 (0.0031 - 0.0159) | 5/10 (1/2 - 1/2) | 0.0162 (0.0020 - 0.0659) |  |
| model_only | 9/10 (1/2 - 2/2) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/10 (0/2 - 0/2) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 10/10 (2/2 - 2/2) | 0.2459 (0.2173 - 0.2792) | 0.2295 (0.2133 - 0.2549) | 5/10 (1/2 - 1/2) | 0.0164 (0.0025 - 0.0659) |  |

**medium, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0486), rules only <= 2% 2/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 5/10 (1/2 - 1/2) | 0.0373 (0.0136 - 0.0878) | 0.0145 (0.0071 - 0.0232) | 5/10 (1/2 - 1/2) | 0.0228 (0.0066 - 0.0716) |  |
| model_only | 9/10 (1/2 - 2/2) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/10 (0/2 - 0/2) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 10/10 (2/2 - 2/2) | 0.2519 (0.2222 - 0.2834) | 0.2289 (0.2118 - 0.2553) | 5/10 (1/2 - 1/2) | 0.0230 (0.0071 - 0.0716) |  |

**medium, D196 without R3 (post hoc)** - budgets: added FPR ok 3/5 seeds (max +0.0155), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 5/10 (1/2 - 1/2) | 0.0139 (0.0056 - 0.0219) | 0.0105 (0.0031 - 0.0197) | 5/10 (1/2 - 1/2) | 0.0034 (0.0020 - 0.0052) |  |
| model_only | 9/10 (1/2 - 2/2) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/10 (0/2 - 0/2) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 10/10 (2/2 - 2/2) | 0.2401 (0.2173 - 0.2593) | 0.2365 (0.2133 - 0.2549) | 5/10 (1/2 - 1/2) | 0.0036 (0.0021 - 0.0052) |  |

**high, D196 (primary)** - budgets: added FPR ok 3/5 seeds (max +0.0444), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 5/15 (1/3 - 1/3) | 0.0243 (0.0056 - 0.0740) | 0.0082 (0.0031 - 0.0159) | 5/15 (1/3 - 1/3) | 0.0162 (0.0020 - 0.0659) |  |
| model_only | 15/15 (3/3 - 3/3) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/15 (0/3 - 0/3) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 15/15 (3/3 - 3/3) | 0.2459 (0.2173 - 0.2792) | 0.2295 (0.2133 - 0.2549) | 5/15 (1/3 - 1/3) | 0.0164 (0.0025 - 0.0659) |  |

**high, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0486), rules only <= 2% 2/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 5/15 (1/3 - 1/3) | 0.0373 (0.0136 - 0.0878) | 0.0145 (0.0071 - 0.0232) | 5/15 (1/3 - 1/3) | 0.0228 (0.0066 - 0.0716) |  |
| model_only | 15/15 (3/3 - 3/3) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/15 (0/3 - 0/3) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 15/15 (3/3 - 3/3) | 0.2519 (0.2222 - 0.2834) | 0.2289 (0.2118 - 0.2553) | 5/15 (1/3 - 1/3) | 0.0230 (0.0071 - 0.0716) |  |

**high, D196 without R3 (post hoc)** - budgets: added FPR ok 3/5 seeds (max +0.0155), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 5/15 (1/3 - 1/3) | 0.0139 (0.0056 - 0.0219) | 0.0105 (0.0031 - 0.0197) | 5/15 (1/3 - 1/3) | 0.0034 (0.0020 - 0.0052) |  |
| model_only | 15/15 (3/3 - 3/3) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/15 (0/3 - 0/3) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 15/15 (3/3 - 3/3) | 0.2401 (0.2173 - 0.2593) | 0.2365 (0.2133 - 0.2549) | 5/15 (1/3 - 1/3) | 0.0036 (0.0021 - 0.0052) |  |

## mule_ring

IEEE-CIS has no true device ID; DeviceInfo is a model/OS string, so a rare value stands in for one device.


**low, D196 (primary)** - budgets: added FPR ok 3/5 seeds (max +0.0444), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 10/25 (0/5 - 5/5) | 0.0243 (0.0056 - 0.0740) | 0.0082 (0.0031 - 0.0159) | 10/25 (0/5 - 5/5) | 0.0162 (0.0020 - 0.0659) |  |
| model_only | 25/25 (5/5 - 5/5) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/25 (0/5 - 0/5) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 25/25 (5/5 - 5/5) | 0.2459 (0.2173 - 0.2792) | 0.2295 (0.2133 - 0.2549) | 10/25 (0/5 - 5/5) | 0.0164 (0.0025 - 0.0659) |  |

**low, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0486), rules only <= 2% 2/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 10/25 (0/5 - 5/5) | 0.0373 (0.0136 - 0.0878) | 0.0145 (0.0071 - 0.0232) | 10/25 (0/5 - 5/5) | 0.0228 (0.0066 - 0.0716) |  |
| model_only | 25/25 (5/5 - 5/5) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/25 (0/5 - 0/5) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 25/25 (5/5 - 5/5) | 0.2519 (0.2222 - 0.2834) | 0.2289 (0.2118 - 0.2553) | 10/25 (0/5 - 5/5) | 0.0230 (0.0071 - 0.0716) |  |

**low, D196 without R3 (post hoc)** - budgets: added FPR ok 3/5 seeds (max +0.0155), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 10/25 (0/5 - 5/5) | 0.0139 (0.0056 - 0.0219) | 0.0105 (0.0031 - 0.0197) | 10/25 (0/5 - 5/5) | 0.0034 (0.0020 - 0.0052) |  |
| model_only | 25/25 (5/5 - 5/5) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/25 (0/5 - 0/5) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 25/25 (5/5 - 5/5) | 0.2401 (0.2173 - 0.2593) | 0.2365 (0.2133 - 0.2549) | 10/25 (0/5 - 5/5) | 0.0036 (0.0021 - 0.0052) |  |

**medium, D196 (primary)** - budgets: added FPR ok 3/5 seeds (max +0.0444), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 20/50 (0/10 - 10/10) | 0.0243 (0.0056 - 0.0740) | 0.0082 (0.0031 - 0.0159) | 20/50 (0/10 - 10/10) | 0.0162 (0.0020 - 0.0659) |  |
| model_only | 50/50 (10/10 - 10/10) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/50 (0/10 - 0/10) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 50/50 (10/10 - 10/10) | 0.2459 (0.2173 - 0.2792) | 0.2295 (0.2133 - 0.2549) | 20/50 (0/10 - 10/10) | 0.0164 (0.0025 - 0.0659) |  |

**medium, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0486), rules only <= 2% 2/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 20/50 (0/10 - 10/10) | 0.0373 (0.0136 - 0.0878) | 0.0145 (0.0071 - 0.0232) | 20/50 (0/10 - 10/10) | 0.0228 (0.0066 - 0.0716) |  |
| model_only | 50/50 (10/10 - 10/10) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/50 (0/10 - 0/10) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 50/50 (10/10 - 10/10) | 0.2519 (0.2222 - 0.2834) | 0.2289 (0.2118 - 0.2553) | 20/50 (0/10 - 10/10) | 0.0230 (0.0071 - 0.0716) |  |

**medium, D196 without R3 (post hoc)** - budgets: added FPR ok 3/5 seeds (max +0.0155), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 20/50 (0/10 - 10/10) | 0.0139 (0.0056 - 0.0219) | 0.0105 (0.0031 - 0.0197) | 20/50 (0/10 - 10/10) | 0.0034 (0.0020 - 0.0052) |  |
| model_only | 50/50 (10/10 - 10/10) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/50 (0/10 - 0/10) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 50/50 (10/10 - 10/10) | 0.2401 (0.2173 - 0.2593) | 0.2365 (0.2133 - 0.2549) | 20/50 (0/10 - 10/10) | 0.0036 (0.0021 - 0.0052) |  |

**high, D196 (primary)** - budgets: added FPR ok 3/5 seeds (max +0.0444), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 40/100 (0/20 - 20/20) | 0.0243 (0.0056 - 0.0740) | 0.0082 (0.0031 - 0.0159) | 40/100 (0/20 - 20/20) | 0.0162 (0.0020 - 0.0659) |  |
| model_only | 100/100 (20/20 - 20/20) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/100 (0/20 - 0/20) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 100/100 (20/20 - 20/20) | 0.2459 (0.2173 - 0.2792) | 0.2295 (0.2133 - 0.2549) | 40/100 (0/20 - 20/20) | 0.0164 (0.0025 - 0.0659) |  |

**high, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0486), rules only <= 2% 2/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 40/100 (0/20 - 20/20) | 0.0373 (0.0136 - 0.0878) | 0.0145 (0.0071 - 0.0232) | 40/100 (0/20 - 20/20) | 0.0228 (0.0066 - 0.0716) |  |
| model_only | 100/100 (20/20 - 20/20) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/100 (0/20 - 0/20) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 100/100 (20/20 - 20/20) | 0.2519 (0.2222 - 0.2834) | 0.2289 (0.2118 - 0.2553) | 40/100 (0/20 - 20/20) | 0.0230 (0.0071 - 0.0716) |  |

**high, D196 without R3 (post hoc)** - budgets: added FPR ok 3/5 seeds (max +0.0155), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 40/100 (0/20 - 20/20) | 0.0139 (0.0056 - 0.0219) | 0.0105 (0.0031 - 0.0197) | 40/100 (0/20 - 20/20) | 0.0034 (0.0020 - 0.0052) |  |
| model_only | 100/100 (20/20 - 20/20) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/100 (0/20 - 0/20) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 100/100 (20/20 - 20/20) | 0.2401 (0.2173 - 0.2593) | 0.2365 (0.2133 - 0.2549) | 40/100 (0/20 - 20/20) | 0.0036 (0.0021 - 0.0052) |  |

## bust_out

**low, D196 (primary)** - budgets: added FPR ok 3/5 seeds (max +0.0672), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/15 (0/3 - 0/3) | 0.0353 (0.0106 - 0.1091) | 0.0136 (0.0070 - 0.0306) | 0/15 (0/3 - 0/3) | 0.0217 (0.0030 - 0.0944) | alerted before cash-out 0/5; build-up alerts 0/23 |
| model_only | 15/15 (3/3 - 3/3) | 0.2355 (0.2306 - 0.2393) | 0.2352 (0.2303 - 0.2389) | 0/15 (0/3 - 0/3) | 0.0003 (0.0002 - 0.0005) | alerted before cash-out 5/5; build-up alerts 16/23 |
| rules_plus_model | 15/15 (3/3 - 3/3) | 0.2567 (0.2352 - 0.3056) | 0.2346 (0.2110 - 0.2596) | 0/15 (0/3 - 0/3) | 0.0220 (0.0035 - 0.0945) | alerted before cash-out 5/5; build-up alerts 16/23 |

**low, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0710), rules only <= 2% 0/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/15 (0/3 - 0/3) | 0.0475 (0.0220 - 0.1200) | 0.0200 (0.0131 - 0.0378) | 0/15 (0/3 - 0/3) | 0.0275 (0.0089 - 0.0999) | alerted before cash-out 0/5; build-up alerts 0/23 |
| model_only | 15/15 (3/3 - 3/3) | 0.2355 (0.2306 - 0.2393) | 0.2352 (0.2303 - 0.2389) | 0/15 (0/3 - 0/3) | 0.0003 (0.0002 - 0.0005) | alerted before cash-out 5/5; build-up alerts 16/23 |
| rules_plus_model | 15/15 (3/3 - 3/3) | 0.2620 (0.2403 - 0.3095) | 0.2341 (0.2094 - 0.2594) | 0/15 (0/3 - 0/3) | 0.0279 (0.0092 - 0.1001) | alerted before cash-out 5/5; build-up alerts 16/23 |

**low, D196 without R3 (post hoc)** - budgets: added FPR ok 3/5 seeds (max +0.0364), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/15 (0/3 - 0/3) | 0.0234 (0.0106 - 0.0497) | 0.0200 (0.0070 - 0.0468) | 0/15 (0/3 - 0/3) | 0.0034 (0.0028 - 0.0039) | alerted before cash-out 0/5; build-up alerts 0/23 |
| model_only | 15/15 (3/3 - 3/3) | 0.2355 (0.2306 - 0.2393) | 0.2352 (0.2303 - 0.2389) | 0/15 (0/3 - 0/3) | 0.0003 (0.0002 - 0.0005) | alerted before cash-out 5/5; build-up alerts 16/23 |
| rules_plus_model | 15/15 (3/3 - 3/3) | 0.2505 (0.2352 - 0.2748) | 0.2468 (0.2313 - 0.2718) | 0/15 (0/3 - 0/3) | 0.0037 (0.0030 - 0.0043) | alerted before cash-out 5/5; build-up alerts 16/23 |

**medium, D196 (primary)** - budgets: added FPR ok 0/5 seeds (max +0.0578), rules only <= 2% 0/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/25 (0/5 - 0/5) | 0.0596 (0.0230 - 0.0972) | 0.0349 (0.0133 - 0.0475) | 0/25 (0/5 - 0/5) | 0.0247 (0.0032 - 0.0839) | alerted before cash-out 0/5; build-up alerts 0/56 |
| model_only | 25/25 (5/5 - 5/5) | 0.2360 (0.2287 - 0.2534) | 0.2357 (0.2283 - 0.2532) | 0/25 (0/5 - 0/5) | 0.0003 (0.0002 - 0.0004) | alerted before cash-out 5/5; build-up alerts 46/56 |
| rules_plus_model | 25/25 (5/5 - 5/5) | 0.2783 (0.2467 - 0.3111) | 0.2533 (0.2270 - 0.2676) | 0/25 (0/5 - 0/5) | 0.0250 (0.0036 - 0.0841) | alerted before cash-out 5/5; build-up alerts 46/56 |

**medium, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0624), rules only <= 2% 0/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/25 (0/5 - 0/5) | 0.0729 (0.0360 - 0.1093) | 0.0329 (0.0194 - 0.0397) | 0/25 (0/5 - 0/5) | 0.0400 (0.0099 - 0.0899) | alerted before cash-out 0/5; build-up alerts 0/56 |
| model_only | 25/25 (5/5 - 5/5) | 0.2360 (0.2287 - 0.2534) | 0.2357 (0.2283 - 0.2532) | 0/25 (0/5 - 0/5) | 0.0003 (0.0002 - 0.0004) | alerted before cash-out 5/5; build-up alerts 46/56 |
| rules_plus_model | 25/25 (5/5 - 5/5) | 0.2841 (0.2528 - 0.3157) | 0.2438 (0.2257 - 0.2523) | 0/25 (0/5 - 0/5) | 0.0404 (0.0103 - 0.0900) | alerted before cash-out 5/5; build-up alerts 46/56 |

**medium, D196 without R3 (post hoc)** - budgets: added FPR ok 0/5 seeds (max +0.0432), rules only <= 2% 0/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/25 (0/5 - 0/5) | 0.0439 (0.0230 - 0.0540) | 0.0404 (0.0198 - 0.0503) | 0/25 (0/5 - 0/5) | 0.0036 (0.0032 - 0.0038) | alerted before cash-out 0/5; build-up alerts 0/56 |
| model_only | 25/25 (5/5 - 5/5) | 0.2360 (0.2287 - 0.2534) | 0.2357 (0.2283 - 0.2532) | 0/25 (0/5 - 0/5) | 0.0003 (0.0002 - 0.0004) | alerted before cash-out 5/5; build-up alerts 46/56 |
| rules_plus_model | 25/25 (5/5 - 5/5) | 0.2695 (0.2467 - 0.2798) | 0.2656 (0.2431 - 0.2760) | 0/25 (0/5 - 0/5) | 0.0039 (0.0036 - 0.0040) | alerted before cash-out 5/5; build-up alerts 46/56 |

**high, D196 (primary)** - budgets: added FPR ok 0/5 seeds (max +0.0590), rules only <= 2% 0/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/40 (0/8 - 0/8) | 0.0686 (0.0571 - 0.0910) | 0.0301 (0.0294 - 0.0308) | 0/40 (0/8 - 0/8) | 0.0385 (0.0277 - 0.0607) | alerted before cash-out 0/5; build-up alerts 0/106 |
| model_only | 40/40 (8/8 - 8/8) | 0.2387 (0.2363 - 0.2448) | 0.2384 (0.2360 - 0.2447) | 0/40 (0/8 - 0/8) | 0.0002 (0.0002 - 0.0003) | alerted before cash-out 5/5; build-up alerts 93/106 |
| rules_plus_model | 40/40 (8/8 - 8/8) | 0.2840 (0.2755 - 0.3039) | 0.2453 (0.2430 - 0.2475) | 0/40 (0/8 - 0/8) | 0.0387 (0.0280 - 0.0609) | alerted before cash-out 5/5; build-up alerts 93/106 |

**high, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0674), rules only <= 2% 0/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 3/40 (0/8 - 1/8) | 0.0826 (0.0705 - 0.1091) | 0.0290 (0.0284 - 0.0295) | 0/40 (0/8 - 0/8) | 0.0537 (0.0420 - 0.0800) | alerted before cash-out 0/5; build-up alerts 0/106 |
| model_only | 40/40 (8/8 - 8/8) | 0.2387 (0.2363 - 0.2448) | 0.2384 (0.2360 - 0.2447) | 0/40 (0/8 - 0/8) | 0.0002 (0.0002 - 0.0003) | alerted before cash-out 5/5; build-up alerts 93/106 |
| rules_plus_model | 40/40 (8/8 - 8/8) | 0.2898 (0.2806 - 0.3122) | 0.2359 (0.2320 - 0.2383) | 0/40 (0/8 - 0/8) | 0.0539 (0.0423 - 0.0802) | alerted before cash-out 5/5; build-up alerts 93/106 |

**high, D196 without R3 (post hoc)** - budgets: added FPR ok 0/5 seeds (max +0.0346), rules only <= 2% 0/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/40 (0/8 - 0/8) | 0.0404 (0.0359 - 0.0467) | 0.0368 (0.0325 - 0.0425) | 0/40 (0/8 - 0/8) | 0.0036 (0.0034 - 0.0042) | alerted before cash-out 0/5; build-up alerts 0/106 |
| model_only | 40/40 (8/8 - 8/8) | 0.2387 (0.2363 - 0.2448) | 0.2384 (0.2360 - 0.2447) | 0/40 (0/8 - 0/8) | 0.0002 (0.0002 - 0.0003) | alerted before cash-out 5/5; build-up alerts 93/106 |
| rules_plus_model | 40/40 (8/8 - 8/8) | 0.2685 (0.2636 - 0.2794) | 0.2646 (0.2600 - 0.2751) | 0/40 (0/8 - 0/8) | 0.0038 (0.0037 - 0.0043) | alerted before cash-out 5/5; build-up alerts 93/106 |

## mimicry

**low, D196 (primary)** - budgets: added FPR ok 3/5 seeds (max +0.0444), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/5 (0/1 - 0/1) | 0.0243 (0.0056 - 0.0740) | 0.0082 (0.0031 - 0.0159) | 0/5 (0/1 - 0/1) | 0.0162 (0.0020 - 0.0659) |  |
| model_only | 0/5 (0/1 - 0/1) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/5 (0/1 - 0/1) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 0/5 (0/1 - 0/1) | 0.2459 (0.2173 - 0.2792) | 0.2295 (0.2133 - 0.2549) | 0/5 (0/1 - 0/1) | 0.0164 (0.0025 - 0.0659) |  |

**low, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0486), rules only <= 2% 2/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/5 (0/1 - 0/1) | 0.0373 (0.0136 - 0.0878) | 0.0145 (0.0071 - 0.0232) | 0/5 (0/1 - 0/1) | 0.0228 (0.0066 - 0.0716) |  |
| model_only | 0/5 (0/1 - 0/1) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/5 (0/1 - 0/1) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 0/5 (0/1 - 0/1) | 0.2519 (0.2222 - 0.2834) | 0.2289 (0.2118 - 0.2553) | 0/5 (0/1 - 0/1) | 0.0230 (0.0071 - 0.0716) |  |

**low, D196 without R3 (post hoc)** - budgets: added FPR ok 3/5 seeds (max +0.0155), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/5 (0/1 - 0/1) | 0.0139 (0.0056 - 0.0219) | 0.0105 (0.0031 - 0.0197) | 0/5 (0/1 - 0/1) | 0.0034 (0.0020 - 0.0052) |  |
| model_only | 0/5 (0/1 - 0/1) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/5 (0/1 - 0/1) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 0/5 (0/1 - 0/1) | 0.2401 (0.2173 - 0.2593) | 0.2365 (0.2133 - 0.2549) | 0/5 (0/1 - 0/1) | 0.0036 (0.0021 - 0.0052) |  |

**medium, D196 (primary)** - budgets: added FPR ok 3/5 seeds (max +0.0444), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/15 (0/3 - 0/3) | 0.0243 (0.0056 - 0.0740) | 0.0082 (0.0031 - 0.0159) | 0/15 (0/3 - 0/3) | 0.0162 (0.0020 - 0.0659) |  |
| model_only | 1/15 (0/3 - 1/3) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/15 (0/3 - 0/3) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 1/15 (0/3 - 1/3) | 0.2459 (0.2173 - 0.2792) | 0.2295 (0.2133 - 0.2549) | 0/15 (0/3 - 0/3) | 0.0164 (0.0025 - 0.0659) |  |

**medium, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0486), rules only <= 2% 2/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/15 (0/3 - 0/3) | 0.0373 (0.0136 - 0.0878) | 0.0145 (0.0071 - 0.0232) | 0/15 (0/3 - 0/3) | 0.0228 (0.0066 - 0.0716) |  |
| model_only | 1/15 (0/3 - 1/3) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/15 (0/3 - 0/3) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 1/15 (0/3 - 1/3) | 0.2519 (0.2222 - 0.2834) | 0.2289 (0.2118 - 0.2553) | 0/15 (0/3 - 0/3) | 0.0230 (0.0071 - 0.0716) |  |

**medium, D196 without R3 (post hoc)** - budgets: added FPR ok 3/5 seeds (max +0.0155), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/15 (0/3 - 0/3) | 0.0139 (0.0056 - 0.0219) | 0.0105 (0.0031 - 0.0197) | 0/15 (0/3 - 0/3) | 0.0034 (0.0020 - 0.0052) |  |
| model_only | 1/15 (0/3 - 1/3) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/15 (0/3 - 0/3) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 1/15 (0/3 - 1/3) | 0.2401 (0.2173 - 0.2593) | 0.2365 (0.2133 - 0.2549) | 0/15 (0/3 - 0/3) | 0.0036 (0.0021 - 0.0052) |  |

**high, D196 (primary)** - budgets: added FPR ok 3/5 seeds (max +0.0444), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/30 (0/6 - 0/6) | 0.0243 (0.0056 - 0.0740) | 0.0082 (0.0031 - 0.0159) | 0/30 (0/6 - 0/6) | 0.0162 (0.0020 - 0.0659) |  |
| model_only | 1/30 (0/6 - 1/6) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/30 (0/6 - 0/6) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 1/30 (0/6 - 1/6) | 0.2459 (0.2173 - 0.2792) | 0.2295 (0.2133 - 0.2549) | 0/30 (0/6 - 0/6) | 0.0164 (0.0025 - 0.0659) |  |

**high, D195 (sensitivity)** - budgets: added FPR ok 0/5 seeds (max +0.0486), rules only <= 2% 2/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/30 (0/6 - 0/6) | 0.0373 (0.0136 - 0.0878) | 0.0145 (0.0071 - 0.0232) | 0/30 (0/6 - 0/6) | 0.0228 (0.0066 - 0.0716) |  |
| model_only | 1/30 (0/6 - 1/6) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/30 (0/6 - 0/6) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 1/30 (0/6 - 1/6) | 0.2519 (0.2222 - 0.2834) | 0.2289 (0.2118 - 0.2553) | 0/30 (0/6 - 0/6) | 0.0230 (0.0071 - 0.0716) |  |

**high, D196 without R3 (post hoc)** - budgets: added FPR ok 3/5 seeds (max +0.0155), rules only <= 2% 3/5

| mode | recall (alert) | FPR (alert) | FPR step-up only | recall (hard) | FPR hold/block | extra |
|---|---|---|---|---|---|---|
| rules_only | 0/30 (0/6 - 0/6) | 0.0139 (0.0056 - 0.0219) | 0.0105 (0.0031 - 0.0197) | 0/30 (0/6 - 0/6) | 0.0034 (0.0020 - 0.0052) |  |
| model_only | 1/30 (0/6 - 1/6) | 0.2337 (0.2151 - 0.2553) | 0.2334 (0.2151 - 0.2545) | 0/30 (0/6 - 0/6) | 0.0002 (0.0000 - 0.0007) |  |
| rules_plus_model | 1/30 (0/6 - 1/6) | 0.2401 (0.2173 - 0.2593) | 0.2365 (0.2133 - 0.2549) | 0/30 (0/6 - 0/6) | 0.0036 (0.0021 - 0.0052) |  |

