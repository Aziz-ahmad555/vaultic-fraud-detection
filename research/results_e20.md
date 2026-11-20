# E20 results: copilot evaluation (EXP-320)

**Copilot evaluation on IEEE-CIS test-period cases, not a fraud-detection result.** The data are
anonymized e-commerce transactions released by Vesta for the IEEE-CIS competition.

| | |
|---|---|
| Run | `20261026-220704-227d24` (generation at `6142cf2`, 330 items: A 150, B 150, C 30; 0 failed) |
| Analysis | `python -m vaultic.eval.e20_analysis --final`, code at `a9d2f34` (sha256 `5ec7cc59…`) |
| Started | FINAL-STARTED row D225n, `01bb283` |
| Pre-registration | `research/preregistration_e20.md`, sections 6, 7 and 11 (D225, amended by D225a-D225m; post-review addendum D225p) |
| Sealed key | Opened only by that run, after its sha256 matched the value recorded at generation |
| Full numbers | `research/tables/e20/e20_results.json`; the tables below are `research/tables/e20/e20_tables.md` verbatim, except the H8c row label, corrected after review (D225p) |

## Which labels each metric uses (the pre-set rule, section 11.4)

**Pass 1 (claims, S / U / W):**
- **Agreement:** judge vs Aziz on the seeded 150-unit sample (368 claims): Cohen's kappa **0.699**, 95% CI
  0.542 to 0.833, raw agreement 0.938.
- **Rule: passed** (0.699 >= 0.6 on the point estimate). The lower end of the CI is below 0.6; the rule was
  fixed on the point estimate.
- **Labels used:** the judge's labels on all 846 units are **primary** for faithfulness, the
  unsupported-claim and wrong rates, the per-unit rates, H8, H8b, H8d and the checker effect. Aziz's sample
  is the validation; its numbers are shown as secondary.

**Pass 2 (citations, Y / N):**
- **Agreement:** judge vs Aziz on the seeded 90-unit sample: kappa **0.293**, 95% CI 0.055 to 0.514, raw
  agreement 0.678.
- **Rule: failed.** By the pre-registered fallback, **Aziz's labels on the 90-unit sample are primary** for
  citation accuracy and H8c (n = 35 A units and 55 B units; smaller n stated). The judge's labels on all 666
  units are secondary.

**Adversarial:** Aziz's labels on all 30 rows (human only, D225e).

## Plain-language summary

1. **A shows fewer unsupported claims than B (H8: supported).**
   - **Per claim**, the unsupported-claim rate (unsupported + wrong) is **0.083** [0.060, 0.105] for the
     copilot (A) and **0.200** [0.177, 0.223] for the same model on raw text (B).
   - **Difference** A - B = -0.117 [-0.143, -0.090], Holm p < 0.001.
   - **Faithfulness (H8b: supported)** mirrors it: 0.917 vs 0.800.
   - Read with coverage: A showed at least one fact on 82% of normal items (B 100%) and 4.74 claims per item (B 7.93).
   - **Wrong claims** are rare in both: 0.004 for A, 0.017 for B.
   - This is a **whole-system** comparison (tools, trimmed evidence, schema, checker and refusal rules
     against raw text); it does not say which part does the work.
2. **A's citations are more often accurate than B's (H8c: supported)**, on Aziz's 90-unit sample:
   - **A:** 0.829 [0.719, 0.939], n = 35. **B:** 0.564 [0.439, 0.702], n = 55.
   - **Difference** +0.265 [0.085, 0.427], Holm p = 0.004.
   - The judge's labels on all 666 units give similar levels: A 0.864, B 0.581.
   - However, the judge and Aziz agreed poorly on *which* units are accurate (kappa 0.293), which is why the
     human sample is primary.
3. **A is not shown to be non-inferior to the code-built template C on Q1 and Q4 (H8d: not supported).**
   - **Faithfulness:** A 0.849 [0.812, 0.887] on those questions; C 0.945 [0.923, 0.963].
   - **Difference** A - C = -0.096 [-0.135, -0.052]. The lower bound is below the -0.10 margin; Holm
     p = 0.401.
   - C's faithfulness was higher (-0.096, CI excludes 0; descriptive, not a pre-registered test), and a loss
     larger than 0.10 cannot be ruled out.
   - **H8d depends on the label set:** on the judge's labels (primary) it is not supported; on Aziz's sample
     labels (secondary, not used for the decision) it would be supported (+0.028 [-0.097, 0.153]). The primary
     result is the one above.
4. **Refusals met both pre-set thresholds (descriptive).**
   - **A** refused **30 of 30** adversarial prompts (threshold >= 27), and its refusal kind matched the prompt
     type in all 30.
   - **A refused 0 of 120** normal items (threshold <= 6).
   - **B complied with 10 of 30** adversarial prompts: 1 of 10 "change", 3 of 10 "other customer", 6 of 10
     "ignore rules".
5. **Coverage: A says less.**
   - A showed at least one fact on **82%** of normal items [78%, 86%]; B and C on 100%.
   - **Per item**, A showed 2.27 units and 4.74 claims, B 3.28 units and 7.93 claims.
   - So A's better faithfulness comes with less content and must be read together with coverage, as
     pre-registered.
6. **Checker effect (descriptive, no test).**
   - Most of A's advantage over B is already present **before** the checker: the A draft (shown + removed)
     vs B gives faithfulness +0.113 [0.088, 0.139].
   - **The checker itself did not measurably improve faithfulness per claim** on the judge's labels: draft
     vs shown -0.004 [-0.021, 0.011].
   - **Its two measured effects:**
     - a lower wrong rate: draft 0.012, shown 0.004, difference 0.008 [0.001, 0.016];
     - a content cost: it removed 68 units (209 claims), and the judge labelled 0.904 of those claims as
       supported. It left 17 normal items with no fact although facts had been drafted (addendum).
   - **On Aziz's sample the checker effect on faithfulness is larger:** draft minus shown = -0.047 [-0.092,
     -0.009], so on his labels the shown facts were 0.047 more faithful than the draft.
   - **Removed units have the lowest raw agreement** between the two label sets (0.902; kappa 0.627).
7. **Latency (normal items).** A median **51.1 s** (90th percentile 69.3 s); B median 28.0 s (90th
   percentile 100.1 s). Both were warm, on one laptop CPU. A's refusals take no model call.

## What was surprising

- **Pass-2 agreement was low (kappa 0.293)**, although the two raters' citation-accuracy *rates* are close.
  They disagree on which units are accurate, mostly B's (kappa 0.169 on B, raw agreement 0.58). The
  pre-registered fallback therefore made the 90-unit human sample primary for H8c.
- **The checker mostly removed supported claims.** By the judge's labels, the removed units were 90%
  supported per claim, so on those labels the checker gave no measurable per-claim faithfulness gain.
  - **Its two measured effects:** a lower wrong rate (0.008 [0.001, 0.016]) and a content cost (68 units, 209
    claims removed; 17 items left empty, 15 of them Q3).
  - **Possible cause:** this is consistent with the limit stated in D225d (true facts about earlier
    transactions can be removed by the conservative history rule; not checked).
- **On Q1 and Q4, C's faithfulness was higher** (-0.096, CI excludes 0; descriptive, not a pre-registered
  test).
- **Per-system agreement on pass 1 was lowest for C** (kappa 0.412, raw agreement 0.924). That is a kappa
  effect of nearly all claims being S, not of frequent disagreement.

## Limitations (stated in advance, plus what happened)

- **One human labeller**, who also built the system, labelled the validation samples; no second human
  labeller was available.
- **The judge is Claude**, a different model family from the copilot's Qwen 2.5, from the same company as
  the assistant that helped build the system.
- **The judge's claim split** defines the claims for both raters; Aziz flagged 0 split problems.
- **Pass 2 is not blind**, and C's wording is recognisable.
- **Small study:** 30 cases, one generation each, on one laptop CPU, with the Test Lab off (view-level
  reasons only).
- **B gets more evidence than A**, and A vs B is a whole-system comparison.
- **H8c rests on 90 human-labelled units** (35 A, 55 B).
- **Small hold strata:** the bundle has 22 legit and 42 fraud holds.
- **The checker does not check meaning** (negation, number words, phrasings outside its word lists). That is
  exactly what the labels measure.
- **H8d depends on the label set:** not supported on the judge's labels (primary); supported on Aziz's
  sample (secondary).
- **Known gaps in the frozen analysis code, recorded and not changed:**
  - it checks that Aziz's pass-1 units equal the drawn sample, but not his pass-2 units. The addendum checks
    pass 2: equal.
  - its FINAL-STARTED guard reads `decisions.md` from the working tree, not from the pushed commit. In this
    run the row was pushed (`01bb283`) before the key was opened.
- **Deviations from and amendments to the pre-registration after generation**, each logged alone in
  `research/decisions.md`:
  - **D225e / D225f:** labelling changed to an LLM judge plus human validation, with the judge's isolation
    and tooling; before any labelling.
  - **D225g:** two desktop-app interface calls in the pass-1 judge session. They touched no files and held
    no labels; accepted as a declared deviation.
  - **D225h / D225l:** locked labelling workbooks for Aziz's samples and the adversarial sheet, with the
    same checks as the CSVs.
  - **D225i:** two more tools denied for the pass-2 judge before it ran.
  - **D225j:** pass-2 case 8 was incomplete (2 of 23 units) after a truncated read, and was redone in a
    fresh session by the pre-set rule.
  - **D225k:** that redo ran in Auto mode instead of Ask. The deny rule was in force and access was
    unchanged.
  - **D225m:** analysis details fixed before the key was opened, including the **output path**:
    `research/tables/e20/` instead of `research/tables/copilot/` as named in section 8.
  - **D225n / D225o:** FINAL-STARTED / FINAL-COMPLETED.
  - **D225p:** this post-review addendum and the wording fixes.

  An accidental session opened in the wrong folder read and wrote nothing (recorded in the manifest).

# Tables

### Agreement (judge vs Aziz) and the pre-set rule

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

### Pass-1 metrics, judge labels (PRIMARY)

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

### Pass-1 metrics, human labels (secondary)

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

### Citation accuracy (pass 2)

| labels | system | units | citation accuracy [95% CI] |
|---|---|---|---|
| judge | A | 272 | 0.864 [0.819, 0.904] |
| judge | B | 394 | 0.581 [0.531, 0.638] |
| human (PRIMARY) | A | 35 | 0.829 [0.719, 0.939] |
| human (PRIMARY) | B | 55 | 0.564 [0.439, 0.702] |

### Pre-registered tests (primary labels; Holm across four)

| hyp. | comparison | A - other [95% CI] | one-sided p | Holm p | supported |
|---|---|---|---|---|---|
| H8 | unsupported-claim rate, A vs B (120 normal items, per claim) | -0.117 [-0.143, -0.090] | < 0.001 | < 0.001 | yes |
| H8b | faithfulness, A vs B (120 normal items, per claim) | 0.117 [0.090, 0.143] | < 0.001 | < 0.001 | yes |
| H8c | citation accuracy, A vs B (Aziz's 90-unit sample: 35 A, 55 B units; per unit) | 0.265 [0.085, 0.427] | 0.002 | 0.004 | yes |
| H8d | faithfulness, A vs C on Q1 and Q4 (60 items, per claim), non-inferiority margin 0.10 | -0.096 [-0.135, -0.052] | 0.401 | 0.401 | no |

### Same tests on the other label sets (secondary, not used for decisions)

| hyp. | comparison | A - other [95% CI] | one-sided p | Holm p | supported |
|---|---|---|---|---|---|
| H8 | unsupported-claim rate, A vs B (120 normal items, per claim) | -0.153 [-0.243, -0.067] | < 0.001 | < 0.001 | yes |
| H8b | faithfulness, A vs B (120 normal items, per claim) | 0.153 [0.067, 0.243] | < 0.001 | < 0.001 | yes |
| H8c | citation accuracy, A vs B (120 normal items, per unit) | 0.283 [0.223, 0.345] | < 0.001 | < 0.001 | yes |
| H8d | faithfulness, A vs C on Q1 and Q4 (60 items, per claim), non-inferiority margin 0.10 | 0.028 [-0.097, 0.153] | 0.023 | 0.023 | yes |

### Coverage

| system | items | items with a shown unit | shown units / item | claims / item |
|---|---|---|---|---|
| A | 120 | 0.82 [0.78, 0.86] | 2.27 [2.09, 2.46] | 4.74 [4.33, 5.21] |
| B | 120 | 1.00 [1.00, 1.00] | 3.28 [3.17, 3.40] | 7.93 [7.48, 8.38] |
| C | 60 | 1.00 [1.00, 1.00] | 3.73 [3.53, 3.93] | 11.47 [10.17, 12.67] |

### Refusals and adversarial items

- A refused 30 of 30 adversarial prompts (threshold >= 27: met).
- A refused 0 of 120 normal items (threshold <= 6: met).
- B complied with 10 of 30 adversarial prompts.

| type | prompts | A refused | A kind matches type | B complied | B declined | B other |
|---|---|---|---|---|---|---|
| change | 10 | 10 | 10 | 1 | 0 | 9 |
| other customer | 10 | 10 | 10 | 3 | 7 | 0 |
| ignore rules | 10 | 10 | 10 | 6 | 0 | 4 |

### Latency (seconds per item)

| system | scope | n | median | 90th percentile |
|---|---|---|---|---|
| A | all 150 items | 150 | 44.7 | 65.9 |
| A | 120 normal items | 120 | 51.1 | 69.3 |
| A | 30 adversarial items | 30 | 0.0 | 0.0 |
| B | all 150 items | 150 | 26.9 | 98.1 |
| B | 120 normal items | 120 | 28.0 | 100.1 |
| B | 30 adversarial items | 30 | 24.0 | 32.2 |
| C | - | - | not applicable | not applicable |

# Addendum (D225p)

Added on 2026-11-21, **after** the final analysis and its independent review.
- **Descriptive only:** not pre-registered, not tested and not used for any decision.
- **Frozen code untouched:** the frozen analysis module (`a9d2f34`, sha256 `5ec7cc59…`) is unchanged. The
  addendum script (`python -m vaultic.eval.e20_addendum`) reads the same verified inputs through that
  module's loader.
- **Outputs:** `research/tables/e20/addendum/e20_addendum.json` and `e20_addendum.md`, reproduced below.

**Reading notes:**
- **Removed units:** where the judge and Aziz differ on A's removed units, the judge said S and Aziz said W or
  U on 4 claims, against 1 claim the other way. Aziz therefore found the removed facts less supported than
  the judge did, which is why the checker effect on his sample is larger (summary, item 6).
- **Pass 2, B:** the disagreement goes both ways: 9 units judge Y / Aziz N and 14 units judge N / Aziz Y.
- **A's empty normal items:** 22 of 120. 17 were emptied by the checker (49 drafted facts removed) and 5 had
  nothing drafted. 15 of the 17 emptied items are Q3 ("Has this customer done something like this
  before?"). This is the content cost of the checker: 17 items show nothing although the model had drafted
  facts.
- **Pass-2 sample check:** the 90 units Aziz labelled are exactly the 90 drawn units. The frozen module
  checks this for pass 1 only.

### Per-system confusion (rows judge, columns Aziz)

pass1, A removed (n = 51):

| | human S | human U | human W |
|---|---|---|---|
| judge S | 41 | 1 | 3 |
| judge U | 0 | 4 | 0 |
| judge W | 1 | 0 | 1 |

pass1, A shown (n = 112):

| | human S | human U | human W |
|---|---|---|---|
| judge S | 108 | 0 | 0 |
| judge U | 1 | 3 | 0 |
| judge W | 0 | 0 | 0 |

pass1, B (n = 139):

| | human S | human U | human W |
|---|---|---|---|
| judge S | 108 | 1 | 1 |
| judge U | 5 | 19 | 3 |
| judge W | 1 | 1 | 0 |

pass1, C (n = 66):

| | human S | human U | human W |
|---|---|---|---|
| judge S | 59 | 3 | 1 |
| judge U | 1 | 2 | 0 |
| judge W | 0 | 0 | 0 |

pass2, A (n = 35):

| | human Y | human N |
|---|---|---|
| judge Y | 27 | 4 |
| judge N | 2 | 2 |

pass2, B (n = 55):

| | human Y | human N |
|---|---|---|
| judge Y | 17 | 9 |
| judge N | 14 | 15 |

### A's empty normal items, per question

| question | items | empty | emptied by the checker | nothing drafted | facts removed in emptied items |
|---|---|---|---|---|---|
| Q1 | 30 | 2 | 2 | 0 | 6 |
| Q2 | 30 | 0 | 0 | 0 | 0 |
| Q3 | 30 | 20 | 15 | 5 | 43 |
| Q4 | 30 | 0 | 0 | 0 | 0 |
| total | 120 | 22 | 17 | 5 | 49 |

Pass-2 sample check: 90 labelled units, 90 drawn; equal: True.
