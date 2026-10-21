# Attack Lab: synthetic attack scenarios and rules (Phase B)

## What it is
A scenario framework and dashboard page (`/attack`) that injects synthetic attacks into a replay of
real test-period traffic and compares three detectors: rules only, the frozen model only, and rules
plus model (D190-D207). Five generators: card testing, account takeover, mule ring, bust-out and
mimicry. **Every number here is a synthetic scenario, not the evaluation**
(`docs/scenarios/README.md`).

## Why we built it
To show how the system behaves on attack patterns that are rare or absent in IEEE-CIS, and how
simple rules complement the model, without touching any research result.

## How it works (simple)
A generator writes synthetic transactions (masked fields copied from real legitimate
validation-period rows, generator v3, D205) into a day of real traffic. The features are recomputed
point-in-time, the frozen model and the rules score every row, and the suite reports recall on the
synthetic fraud rows and the false-positive rate on real legitimate rows: 5 seeds x 3 intensities
(D199). Rule thresholds were fixed on validation before any scenario (D192, D195, D196).

## Key results (with sources; synthetic scenario - not the evaluation)
- Card testing, high intensity, D196 primary: rules only 95/125, model only 20/125, rules + model
  100/125 alerted synthetic rows; model-only FPR on real legitimate rows 0.2337
  (`docs/scenarios/suite_generator_v3.md`).
- Mule ring, model only (v3): 17/25, 37/50, 73/100 by intensity (`research/decisions.md`, D207).
- Bust-out: cash-outs caught in every version, but a post-hoc check shows E12 steps up 99.9% of 1,463
  real legitimate young card-not-present rows with amounts 500-2,000, so **this is not bust-out
  detection** (`research/decisions.md`, D207; `docs/scenarios/posthoc_diagnostics.md`).
- Realism v3: masked-field missingness patterns not seen in real rows of the same customer type:
  0% for four generators, 6% for mule ring; this checks missingness only, not values
  (`docs/scenarios/realism_v3.md`, D207).

## Limitations
- Synthetic scenarios are not the evaluation and never enter a research number.
- IEEE-CIS has no true device ID; a rare DeviceInfo string stands in for one device (D199).
- Generator realism is checked for missingness patterns only.
- The "D196 without R3" row was chosen after seeing results and is a sensitivity row only (D201).

## Screenshots to take
1. The Attack Lab page after a card-testing run (rules vs model vs both, with the "synthetic
   scenario - not the evaluation" label).
2. The mule-ring result showing the rare-DeviceInfo stand-in note.
3. The bust-out result with the post-hoc note that cash-out step-ups are not bust-out detection.
