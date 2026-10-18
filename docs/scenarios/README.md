# Attack scenarios (Phase B) - synthetic scenario - not the evaluation

Product-layer results only (D190-D203); nothing here is a research number.

| file | what | decision |
|---|---|---|
| suite_v1.md | 5 seeds x 3 intensities, generator v1, D196 primary and D195 sensitivity | D200 |
| suite_gen_v1_run2.md | the same suite re-run (generator v1) with the post-hoc 'D196 without R3' row and the step-up / hold-block FPR split | D201, D202 |
| suite_generator_v2.md | the suite with generator v2 (masked fields from real validation-period rows); superseded: donor pool included fraud rows (review finding 1) | D203, D204 |
| suite_generator_v3.md | the suite with generator v3 (as v2, donor pools of legitimate validation rows only), the primary since D205 | D205 |
| realism_v1.md / .json | masked-field missingness of generator v1 rows vs real rows of the same customer type | D203 |
| realism_v2.md / .json | the same check for generator v2 (superseded) | D203 |
| realism_v3.md / .json | the same check for generator v3 (primary) | D205 |
| posthoc_diagnostics.md / .json | per-day rules-only FPR (per rule, with and without R3, Windows weekly count vs 778) and model alerts on real new accounts vs bust-out build-up rows (v2 and v1) | D201, D202, D203 |

'suite_v1' and 'suite_gen_v1_run2' are the two runs of generator v1 (the second was named suite_v2.md until D206).

**Stored (parity-verified) rows.** A real legitimate replay row keeps its stored score unless it shares the customer,
card key or device key with a synthetic row (D193, D197). A row that shares only an email domain or an address with
the attack keeps its stored score, so its email and address degree features miss at most the attack's few synthetic
rows: a known approximation.

**Realism checks** show only that the synthetic rows' missingness patterns occur in real rows of the customer type,
not that their values are realistic.
