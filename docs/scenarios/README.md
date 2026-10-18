# Attack scenarios (Phase B) - synthetic scenario - not the evaluation

Product-layer results only (D190-D203); nothing here is a research number.

| file | what | decision |
|---|---|---|
| suite_v1.md | 5 seeds x 3 intensities, generator v1, D196 primary and D195 sensitivity | D200 |
| suite_v2.md | the same suite re-run (generator v1) with the post-hoc 'D196 without R3' row and the step-up / hold-block FPR split | D201, D202 |
| suite_generator_v2.md | the suite with generator v2 (masked fields from real validation-period rows), the primary since D203 | D203 |
| realism_v1.md / .json | masked-field missingness of generator v1 rows vs real rows of the same customer type | D203 |
| realism_v2.md / .json | the same check for generator v2 | D203 |
| posthoc_diagnostics.md / .json | per-day rules-only FPR (per rule, with and without R3, Windows weekly count vs 778) and model alerts on real new accounts vs bust-out build-up rows (v2 and v1) | D201, D202, D203 |

'suite_v1' / 'suite_v2' name the two runs of generator v1, not the generator version.
