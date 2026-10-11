# F3 test-period collapse: diagnosis (POST-HOC / EXPLORATORY)

**POST-HOC / EXPLORATORY.** Not pre-registered and not part of the Paper 2 final results (D101). It was run after the EXP-200-final final run, reads the test period, and changes no reported number: research/tables/final_e10.md - final_e12.md stand as generated. Generated 2026-10-11 by `python -m vaultic.fusion.f3_diagnosis` (D117).

## Reproduction

The diagnosis rebuilds the final run's fusion inputs with the same frozen pipeline and refits F3 (D95-tuned C, seeds 0-4, fit + tune rows). Against the stored F3 test scores of run `20261011-002400-202994`: 92427 of 92427 rows matched, max |diff| 0.00e+00, correlation 1.000000; test PR-AUC 0.4229 (stored 0.4229).

F3 PR-AUC by period: gate rows (fit + tune; in-sample) 0.6551; calibrate_fused (147-150) 0.6047; test (151-182) 0.4229.

## 1. Coefficients and input shift (gate rows -> test rows)

Coefficient: F3's logistic-regression weight on the standardised input (mean over the 5 seeds). PSI / KS on the input's natural scale (calibrated view probability on rows where the view is available; context as is). z shift: change in the input's mean in the units F3 sees (gate-row standard deviations; a missing view enters as logit 0). coef x z: the resulting average change in F3's logit.

### all rows

| input | coef | PSI | KS | missing gate -> test | mean gate -> test | max gate -> test | z shift | coef x z |
|---|---|---|---|---|---|---|---|---|
| view_tabular | +2.173 | 2.348 | 0.317 | 0.000 -> 0.000 | 0.042 -> 0.039 | 1.0 -> 1.0 | -0.57 | -1.24 |
| view_behavioral | +1.018 | 3.308 | 0.389 | 0.336 -> 0.296 | 0.042 -> 0.036 | 1.0 -> 1.0 | -0.65 | -0.67 |
| view_temporal | -0.512 | 0.055 | 0.087 | 0.336 -> 0.296 | 0.042 -> 0.039 | 1.0 -> 1.0 | -0.12 | +0.06 |
| view_graph | -0.037 | 0.500 | 0.291 | 0.431 -> 0.540 | 0.046 -> 0.031 | 0.7 -> 0.6 | +0.04 | -0.00 |
| view_anomaly | +0.015 | 0.038 | 0.049 | 0.277 -> 0.000 | 0.035 -> 0.039 | 0.4 -> 0.3 | -0.56 | -0.01 |
| ctx_log_amount | +0.112 | 0.003 | 0.013 | 0.000 -> 0.000 | 4.380 -> 4.383 | 10.4 -> 8.6 | +0.00 | +0.00 |
| ctx_hist_n_past | +0.096 | 0.049 | 0.089 | 0.000 -> 0.000 | 4.803 -> 17.101 | 366.0 -> 1413.0 | +1.02 | +0.10 |
| ctx_has_identity | +0.476 | 0.000 | 0.025 | 0.000 -> 0.000 | 0.185 -> 0.209 | 1.0 -> 1.0 | +0.06 | +0.03 |
| ctx_product | +0.307 | 0.000 | 0.019 | 0.000 -> 0.000 | 5.364 -> 5.364 | 6.0 -> 6.0 | +0.00 | +0.00 |
| ctx_hour | +0.019 | 0.016 | 0.036 | 0.000 -> 0.000 | 13.844 -> 13.979 | 23.0 -> 23.0 | +0.02 | +0.00 |

### with history

| input | coef | PSI | KS | missing gate -> test | mean gate -> test | max gate -> test | z shift | coef x z |
|---|---|---|---|---|---|---|---|---|
| view_tabular | +2.173 | 1.242 | 0.339 | 0.000 -> 0.000 | 0.041 -> 0.036 | 1.0 -> 1.0 | -0.67 | -1.46 |
| view_behavioral | +1.018 | 3.308 | 0.389 | 0.000 -> 0.000 | 0.042 -> 0.036 | 1.0 -> 1.0 | -0.85 | -0.87 |
| view_temporal | -0.512 | 0.055 | 0.087 | 0.000 -> 0.000 | 0.042 -> 0.039 | 1.0 -> 1.0 | -0.07 | +0.04 |
| view_graph | -0.037 | 0.557 | 0.304 | 0.460 -> 0.565 | 0.046 -> 0.028 | 0.6 -> 0.6 | +0.02 | -0.00 |
| view_anomaly | +0.015 | 0.032 | 0.048 | 0.264 -> 0.000 | 0.039 -> 0.044 | 0.4 -> 0.3 | -0.52 | -0.01 |
| ctx_log_amount | +0.112 | 0.002 | 0.016 | 0.000 -> 0.000 | 4.333 -> 4.339 | 10.4 -> 8.5 | +0.01 | +0.00 |
| ctx_hist_n_past | +0.096 | 0.064 | 0.104 | 0.000 -> 0.000 | 7.229 -> 24.288 | 366.0 -> 1413.0 | +1.42 | +0.14 |
| ctx_has_identity | +0.476 | 0.000 | 0.020 | 0.000 -> 0.000 | 0.163 -> 0.184 | 1.0 -> 1.0 | +0.05 | +0.02 |
| ctx_product | +0.307 | 0.000 | 0.017 | 0.000 -> 0.000 | 5.399 -> 5.430 | 6.0 -> 6.0 | +0.02 | +0.01 |
| ctx_hour | +0.019 | 0.011 | 0.032 | 0.000 -> 0.000 | 14.187 -> 14.242 | 23.0 -> 23.0 | +0.01 | +0.00 |

### 5 views

| input | coef | PSI | KS | missing gate -> test | mean gate -> test | max gate -> test | z shift | coef x z |
|---|---|---|---|---|---|---|---|---|
| view_tabular | +2.173 | 1.705 | 0.339 | 0.000 -> 0.000 | 0.045 -> 0.038 | 1.0 -> 1.0 | -0.80 | -1.74 |
| view_behavioral | +1.018 | 1.515 | 0.354 | 0.000 -> 0.000 | 0.044 -> 0.042 | 1.0 -> 1.0 | -0.71 | -0.73 |
| view_temporal | -0.512 | 0.097 | 0.108 | 0.000 -> 0.000 | 0.046 -> 0.043 | 1.0 -> 1.0 | -0.07 | +0.03 |
| view_graph | -0.037 | 0.323 | 0.238 | 0.000 -> 0.000 | 0.039 -> 0.028 | 0.6 -> 0.6 | -0.28 | +0.01 |
| view_anomaly | +0.015 | 0.067 | 0.065 | 0.000 -> 0.000 | 0.042 -> 0.054 | 0.4 -> 0.3 | +0.04 | +0.00 |
| ctx_log_amount | +0.112 | 0.009 | 0.029 | 0.000 -> 0.000 | 4.280 -> 4.310 | 8.8 -> 8.2 | +0.03 | +0.00 |
| ctx_hist_n_past | +0.096 | 0.074 | 0.105 | 0.000 -> 0.000 | 6.898 -> 43.018 | 231.0 -> 1413.0 | +3.01 | +0.29 |
| ctx_has_identity | +0.476 | 0.000 | 0.030 | 0.000 -> 0.000 | 0.206 -> 0.236 | 1.0 -> 1.0 | +0.08 | +0.04 |
| ctx_product | +0.307 | 0.010 | 0.039 | 0.000 -> 0.000 | 5.224 -> 5.313 | 6.0 -> 6.0 | +0.07 | +0.02 |
| ctx_hour | +0.019 | 0.016 | 0.033 | 0.000 -> 0.000 | 14.001 -> 14.105 | 23.0 -> 23.0 | +0.01 | +0.00 |

## 2. One input at a time mapped onto its gate-row distribution

F3's test PR-AUC when the available test values of one input are mapped by rank onto that input's gate-row distribution (ranks kept, marginal shift removed, missingness unchanged). An input whose swap restores PR-AUC is the one whose shift drives the collapse.

| swapped input | all rows | cold start | with history | 1-2 views | 3-4 views | 5 views |
|---|---|---|---|---|---|---|
| none (as in the final run) | 0.4229 | 0.5896 | 0.3856 | 0.5055 | 0.5676 | 0.3587 |
| view_tabular | 0.3462 | 0.5969 | 0.3073 | 0.5146 | 0.5767 | 0.2539 |
| view_behavioral | 0.3949 | 0.5896 | 0.3514 | 0.5055 | 0.5426 | 0.3334 |
| view_temporal | 0.4221 | 0.5896 | 0.3847 | 0.5055 | 0.5671 | 0.3580 |
| view_graph | 0.4230 | 0.5898 | 0.3857 | 0.5055 | 0.5678 | 0.3587 |
| view_anomaly | 0.4229 | 0.5896 | 0.3856 | 0.5055 | 0.5677 | 0.3587 |
| ctx_log_amount | 0.4227 | 0.5895 | 0.3855 | 0.5055 | 0.5671 | 0.3586 |
| ctx_hist_n_past | 0.5947 | 0.5896 | 0.5991 | 0.5055 | 0.5677 | 0.6647 |
| ctx_has_identity | 0.4229 | 0.5896 | 0.3856 | 0.5055 | 0.5676 | 0.3587 |
| ctx_product | 0.4194 | 0.5895 | 0.3823 | 0.5058 | 0.5678 | 0.3553 |
| ctx_hour | 0.4229 | 0.5896 | 0.3857 | 0.5054 | 0.5676 | 0.3588 |
| all five views | 0.3186 | 0.5969 | 0.2749 | 0.5146 | 0.5471 | 0.2291 |
| all five context columns | 0.5937 | 0.5893 | 0.5976 | 0.5058 | 0.5673 | 0.6621 |

## 3. Over time (7-day blocks of the test period)

| days | rows | frauds | F3 | MVAF | mean ctx_hist_n_past | share above the gate-row max |
|---|---|---|---|---|---|---|
| 151-157 | 22071 | 681 | 0.6598 | 0.6718 | 6.2 | 0.001 |
| 158-164 | 20726 | 650 | 0.5525 | 0.5661 | 9.6 | 0.001 |
| 165-171 | 20332 | 717 | 0.2796 | 0.5953 | 33.4 | 0.037 |
| 172-178 | 19010 | 760 | 0.3978 | 0.6730 | 26.0 | 0.016 |
| 179-182 | 10288 | 405 | 0.5413 | 0.5961 | 6.9 | 0.001 |

## 4. Interpretation (added by hand, not generated)

Everything here is POST-HOC / EXPLORATORY and changes no reported number.

**Cause: one context input, `ctx_hist_n_past`, through its extreme tail.** F3 is a logistic
regression on standardised inputs, so the number of past transactions of the reconstructed
customer enters its logit linearly and without bound. On the gate rows this count has mean 4.8,
standard deviation 12.0, 99.9th percentile 165 and maximum 366. On the test rows its 99th
percentile is 489, its 99.9th percentile 1,321 and its maximum 1,413: a small group of customers
with very long histories appears that F3 never saw. Its coefficient is small (+0.096 per gate
standard deviation), but at a count of 1,413 it adds about +11.2 to the logit, which puts these
rows at the top of F3's ranking whatever the views say. They are mostly legitimate: the 1,509
test rows with more than 100 past transactions have a fraud rate of 1.0% (1.1% for such rows in
the gate period, where there are fewer of them and the counts are lower). Half of F3's top 1% of
test scores (50.8%) are rows above the gate-row maximum, which is what destroys
precision at the top of the ranking and so the PR-AUC.

**Evidence.**

- Mapping only this input onto its gate-row distribution (ranks kept) restores F3's test PR-AUC
  from 0.4229 to 0.5947, close to its 0.6047 on the calibrate_fused days. On rows with history it
  goes from 0.3856 to 0.5991, and on 5-view rows from 0.3587 to 0.6647. Cold-start rows (count 0)
  are unchanged. Mapping all five context columns gives the same 0.5937, so the other context
  columns play no part.
- The bulk shift is small (PSI 0.05, KS 0.09): the damage comes from the tail beyond the training
  range, not from a general drift. The mean shift (+1.0 gate standard deviations on all rows,
  +3.0 on 5-view rows) mostly reflects that tail.
- Over time, F3 collapses in days 165-171 (PR-AUC 0.28) and 172-178 (0.40), exactly the blocks
  where the mean count jumps (33.4 and 26.0, against 6-10 elsewhere) and 3.7% / 1.6% of rows lie
  above the gate-row maximum. MVAF stays at 0.60-0.67 in the same blocks. In the first and last
  blocks, where almost no row exceeds the training range, F3 is close to MVAF (0.66 vs 0.67;
  0.54 vs 0.60).
- The views are not the cause. The tabular and behavioral calibrated scores do shift (PSI 2.3
  and 3.3, KS 0.32 and 0.39), but mapping them onto their gate-row distribution lowers F3's
  PR-AUC further (0.35 and 0.39). Their test-period ranking carries the signal, and their shift
  partly reflects the different case mix of the test period. Temporal, graph and anomaly have
  near-zero effect.

**Link to the E5 graph-feature drift (D99/D100): not through the graph view.** F3's coefficient
on the graph view is about zero (-0.04) and mapping the graph view onto its gate-row distribution
leaves F3's PR-AUC unchanged (0.4230). The graph view does shift (PSI 0.50; missing on 54% of test
rows against 43% of gate rows), and the mechanism is of the same kind as the one suspected for
E5: count features that grow with time since the data starts, so later rows take values outside
the training range. Whether this also explains E5 was not examined here.

**What this means for the Paper 2 results (no number changes).** The pre-registered primary
comparison (MVAF - F3, +0.1990, Holm p < 0.001) stands as reported, but it mainly measures F3's
fragility to one unbounded context input on the test period, not an advantage of MVAF's
missing-view handling. Against baselines that did not break (F4 +0.0013, p 0.68; F5-F7) MVAF is
level or slightly behind, and it is behind B5 and F0. Both points need to be stated together
wherever the primary result is reported. The finding also shows a limitation of the baseline
design: F3 received the raw context, as every fusion method did by design, without a bounded
transform of the count. Tree models (F4, B5, F0) are insensitive to values beyond the training
range, which is consistent with their not collapsing. That MVAF's gate is robust to the same
input was not examined beyond its per-block PR-AUC above.

Not done, deliberately: any F3 variant with a bounded or log-transformed count. It would be a
new, post-hoc method, so it could only appear as a separately labelled exploratory analysis.
