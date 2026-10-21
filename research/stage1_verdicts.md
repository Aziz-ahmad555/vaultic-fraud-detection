# Stage-1 hypothesis verdicts written from the recorded results (D220)

Written 2026-10-22 from results recorded at the time of the final runs. **No new analysis**: every
number below is copied from the named file; nothing was recomputed. The hypotheses are the
roadmap's (docs/ROADMAP.pdf p. 4); the decision rules are D101's (stage 1) and D121's (stage 2).
This file only writes down verdicts that were never written; it changes no result.

| H | Verdict | Rule | Source |
|---|---|---|---|
| H1 | Per step (D121): E3 +behavioral supported, E4 +temporal supported, E5 +graph supported, E8 +anomaly **not supported**. The roadmap's own test ("non-overlapping 95% CIs for at least two views") **was never evaluated as written** | D121: per step, difference > 0, Holm across the four steps | `research/tables/stage2/ladder_meta.json` (`h1`), `research/tables/stage2/ladder_steps.json` |
| H2 | **NOT SUPPORTED** (recorded at the time) | D101: MVAF beats F3 and F4 overall AND by a larger margin on missing-view subgroups | `research/tables/final_e10.md`, `research/tables/final_results.json` key `h2` |
| H3 | **Never run** (E6, the graph-leakage settings A / B / C) | — | `experiments/configs/EXP-106.yaml` (`status: planned`) |
| H4 | **Never run** (E13, drift detectors x label delays x injected drifts) | — | `experiments/configs/EXP-113.yaml` (`status: planned`) |
| H5 | **SUPPORTED** under D101's rule, for MVAF, F3 and B5; see the note below | D101: "R3/R4 catch more fraud value than R1 at equal K" | `research/tables/final_e12.md` (H5 lists), `research/tables/final_results.json` `e12.<method>.routing.h5` |

## H5 in detail (D101 rule; day-block bootstrap, Holm within each method, D104)

Every recorded H5 test is positive with Holm p < 0.001 (`research/tables/final_e12.md`):

- MVAF, 8 tests (R3 - R1 and R4 - R1 at K = 50, 100, 200, 500), e.g. K = 50: R3 - R1 +100,164
  [+75,151, +127,393] of fraud value caught; K = 500: +28,007 [+15,800, +41,061].
- F3, 8 tests, e.g. K = 50: R3 - R1 +102,540 [+79,237, +129,651].
- B5, 4 tests (R3 - R1; B5 has no view disagreement, so no R4), e.g. K = 50: R3 - R1 +104,516
  [+77,512, +133,208].

**Note.** In the recorded routing results R3 (expected loss + uncertainty bonus) catches exactly the
same fraud value as R2 (expected loss only) at every K, for all three methods
(`research/tables/final_results.json` `e12.<method>.routing.policies`). So H5 holds as D101 states
it, but the whole gain over top-K risk comes from ranking by expected loss; the uncertainty bonus
added nothing measurable. The roadmap's wording ("expected loss plus uncertainty") should be
reported with this qualification.

## E11 hypothesis (D101: "MVAF, trained with view dropout, degrades less than stacking")

D101 fixed the comparison (MVAF's drop minus F3's or F4's drop when a view is removed for all rows;
paired bootstrap; Holm across the 5 views) but **no rule for an overall verdict**, so none is
claimed here (D220). Per view, from `research/tables/final_e11.md` (negative = MVAF drops less):

| removed view | vs F3 | vs F4 |
|---|---|---|
| tabular | MVAF degrades less: -0.2429 [-0.2593, -0.2258], Holm p < 0.001 | less: -0.2233 [-0.2374, -0.2085], Holm p < 0.001 |
| behavioral | MVAF degrades more: +0.0232 [+0.0179, +0.0287], Holm p < 0.001 | less: -0.0172 [-0.0216, -0.0132], Holm p < 0.001 |
| temporal | more: +0.0079 [+0.0047, +0.0109], Holm p < 0.001 | no difference: +0.0002 [-0.0018, +0.0022], Holm p 0.872 |
| graph | more: +0.0019 [+0.0005, +0.0032], Holm p 0.012 | no difference: -0.0013 [-0.0029, +0.0002], Holm p 0.176 |
| anomaly | no difference: -0.0013 [-0.0032, +0.0006], Holm p 0.194 | less: -0.0149 [-0.0175, -0.0123], Holm p < 0.001 |
| tabular only (not in the Holm family) | more: +0.0379 [+0.0317, +0.0445], p < 0.001 | less: -0.0299 [-0.0339, -0.0260], p < 0.001 |

Counts over the Holm family (5 views): vs F3, MVAF degrades less for 1 view, more for 3, no
difference for 1; vs F4, less for 3, no difference for 2, more for none.
