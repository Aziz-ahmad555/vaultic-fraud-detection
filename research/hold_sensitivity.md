# Block decisions when a Hold can fail (POST-HOC / EXPLORATORY)

**POST-HOC / EXPLORATORY (D124), motivated by D123.** Not pre-registered; E12 and every pre-registered number stand as reported. In the D43 cost model a Hold always stops the fraud, so it weakly dominates Block and Block is never chosen. Here a held fraud is stopped only with probability hold_success_rate (1.0 = D43). Generated 2026-10-11 by `python -m vaultic.trust.hold_sensitivity`.

MVAF on the test period (92,427 rows, 3,213 frauds); thresholds re-chosen on the calibration days (147-150) for each of the 135 D46 cost settings and each hold_success_rate. Inputs reproduce E12's default cost ($99,263.19). Cost change: test total cost minus the same setting with hold_success_rate = 1.0, each under its own cost model (a leakier Hold makes every held fraud dearer, so costs rise). Saving from Block: test cost with Block forbidden (thresholds otherwise the same) minus the cost with the chosen thresholds.

| hold_success_rate | settings where Block is reachable | mean share of test rows blocked | max share blocked | mean cost change vs 1.0 ($) | mean saving from Block ($) |
|---|---|---|---|---|---|
| 1.00 | 0 of 135 | 0.00% | 0.00% | +0 | 0 |
| 0.95 | 95 of 135 | 0.58% | 5.20% | +9,639 | 2,080 |
| 0.90 | 91 of 135 | 0.99% | 5.20% | +14,633 | 6,294 |
| 0.80 | 90 of 135 | 1.40% | 8.65% | +21,300 | 16,439 |
| 0.70 | 90 of 135 | 1.44% | 8.65% | +24,197 | 27,042 |

## Default costs (C_FP $10, review $5, OTP $0.50, step-up 90%, friction $1)

| hold_success_rate | Block reachable | t_block (expected loss, $) | test rows blocked | frauds among them | test cost ($) | change vs 1.0 ($) |
|---|---|---|---|---|---|---|
| 1.00 | no | never | 0 | 0 | 99,263 | +0 |
| 0.95 | yes | 546.00 | 59 | 36 | 99,567 | +304 |
| 0.90 | yes | 546.00 | 59 | 36 | 99,600 | +337 |
| 0.80 | yes | 546.00 | 59 | 36 | 99,600 | +337 |
| 0.70 | yes | 546.00 | 59 | 36 | 99,600 | +337 |

## Reading (added by hand, not generated)

POST-HOC / EXPLORATORY (D124); E12 and every pre-registered number stand as reported.

- **"Never block" is a knife-edge of the D43 model.** With a perfect Hold (1.0) Block is reachable
  in none of the 135 cost settings. As soon as a held fraud can slip through 5% of the time
  (0.95), Block becomes part of the cost-optimal policy in 95 of 135 settings, and in about 90 of
  135 for 0.9-0.7. The settings without Block are mostly those with a high false-positive cost.
- **Blocking stays rare.** On average 0.6-1.4% of test transactions are blocked (at most 8.7%),
  so Block takes over the highest-risk cases, not the bulk of the alerts.
- **At the default costs** the re-chosen Block threshold is an expected loss of $546 (where Hold
  started in E12): 59 test transactions are blocked, 36 of them fraud. The test cost rises by
  only $304-337 against the perfect-Hold model, because at these costs Holds are rare to begin
  with (step-up does most of the work).
- **What Block is worth grows with Hold's failure rate:** forbidding Block (other thresholds
  unchanged) would cost on average $2.1k more at 0.95 and $27k more at 0.7, against total costs
  that themselves rise by $10k-24k as held fraud gets dearer.
- For Paper 2 (D123): E12's "never block" follows from the assumption that a review always stops
  the fraud. Under a mildly imperfect review, the same pipeline blocks a small, high-risk slice.
  This is a sensitivity of the cost model, not a result on detection quality, and these numbers
  are exploratory (single post-hoc sweep, no CIs).
