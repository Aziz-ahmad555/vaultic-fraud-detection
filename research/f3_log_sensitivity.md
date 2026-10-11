# F3 with a bounded history count: sensitivity analysis (POST-HOC / EXPLORATORY)

**POST-HOC / EXPLORATORY, not a replacement result.** Not pre-registered; designed after the EXP-200-final final run and the F3 diagnosis (research/f3_collapse_diagnosis.md, D117 / D118). The pre-registered F3 and every number in research/tables/final_e10.md - final_e12.md stand as reported. P-values below are unadjusted and exploratory. Generated 2026-10-11 by `python -m vaultic.fusion.f3_sensitivity` (D119).

F3-log = F3 with ctx_hist_n_past replaced by log1p(min(count, 366)), 366 being the count's maximum on the gate rows; same D95-tuned C (not re-tuned), seeds 0-4, gate rows (fit + tune) and other inputs as the pre-registered F3; cost threshold chosen on the calibrate_fused rows. F3, MVAF, F4 and B5 are the stored final-run scores and thresholds. Test period, 92,427 rows, 3,213 frauds.

| method | PR-AUC [95% CI] | recall@1%FPR [95% CI] | cost ($) [95% CI] |
|---|---|---|---|
| F3-log (post-hoc) | 0.5908 [0.5731, 0.6080] | 0.5198 [0.5013, 0.5393] | 255,651 [233,617, 278,547] |
| F3 | 0.4229 [0.4049, 0.4413] | 0.2801 [0.2546, 0.3066] | 278,070 [255,674, 301,601] |
| MVAF | 0.6219 [0.6037, 0.6392] | 0.5468 [0.5274, 0.5666] | 236,084 [215,815, 258,102] |
| F4 | 0.6206 [0.6026, 0.6386] | 0.5490 [0.5317, 0.5673] | 234,256 [213,379, 255,716] |
| B5 | 0.6532 [0.6368, 0.6697] | 0.5727 [0.5550, 0.5909] | 268,023 [243,921, 293,297] |

Paired comparisons (F3-log minus method, same 1,000 resamples; for cost, positive = F3-log costs more; unadjusted p, exploratory):

| metric | comparison | diff | 95% CI | p |
|---|---|---|---|---|
| pr_auc | F3-log (post-hoc) - F3 | +0.1679 | [+0.1580, +0.1777] | < 0.001 |
| pr_auc | F3-log (post-hoc) - MVAF | -0.0311 | [-0.0354, -0.0264] | < 0.001 |
| pr_auc | F3-log (post-hoc) - F4 | -0.0298 | [-0.0359, -0.0239] | < 0.001 |
| pr_auc | F3-log (post-hoc) - B5 | -0.0624 | [-0.0689, -0.0557] | < 0.001 |
| recall_at_1pct_fpr | F3-log (post-hoc) - F3 | +0.2397 | [+0.2160, +0.2639] | < 0.001 |
| recall_at_1pct_fpr | F3-log (post-hoc) - MVAF | -0.0271 | [-0.0347, -0.0177] | < 0.001 |
| recall_at_1pct_fpr | F3-log (post-hoc) - F4 | -0.0293 | [-0.0365, -0.0197] | < 0.001 |
| recall_at_1pct_fpr | F3-log (post-hoc) - B5 | -0.0529 | [-0.0615, -0.0414] | < 0.001 |
| cost | F3-log (post-hoc) - F3 | -22,420 | [-30,339, -14,694] | < 0.001 |
| cost | F3-log (post-hoc) - MVAF | 19,567 | [10,619, 28,231] | < 0.001 |
| cost | F3-log (post-hoc) - F4 | 21,395 | [11,288, 31,527] | 0.002 |
| cost | F3-log (post-hoc) - B5 | -12,372 | [-25,972, 2,237] | 0.088 |

## PR-AUC by subgroup

| subgroup | rows | frauds | F3-log (post-hoc) | F3 | MVAF | F4 | B5 |
|---|---|---|---|---|---|---|---|
| all rows | 92427 | 3213 | 0.5908 | 0.4229 | 0.6219 | 0.6206 | 0.6532 |
| cold start | 27350 | 896 | 0.5863 | 0.5896 | 0.5947 | 0.5925 | 0.6230 |
| with history | 65077 | 2317 | 0.5931 | 0.3856 | 0.6348 | 0.6307 | 0.6664 |
| 1-2 views | 13131 | 365 | 0.5026 | 0.5055 | 0.5152 | 0.5149 | 0.5488 |
| 3-4 views | 51004 | 1678 | 0.5670 | 0.5676 | 0.5790 | 0.5881 | 0.6205 |
| 5 views | 28292 | 1170 | 0.6537 | 0.3587 | 0.7125 | 0.6980 | 0.7372 |

## PR-AUC by 7-day block

| days | F3-log (post-hoc) | F3 | MVAF |
|---|---|---|---|
| 151-157 | 0.6610 | 0.6598 | 0.6718 |
| 158-164 | 0.5508 | 0.5525 | 0.5661 |
| 165-171 | 0.5320 | 0.2796 | 0.5953 |
| 172-178 | 0.6471 | 0.3978 | 0.6730 |
| 179-182 | 0.5597 | 0.5413 | 0.5961 |

## Reading (added by hand, not generated)

POST-HOC / EXPLORATORY; the pre-registered results stand as reported.

- Bounding the one input recovers most of F3's loss: test PR-AUC 0.5908 against 0.4229 (+0.168),
  recall at 1% FPR 0.520 against 0.280, and $22k less cost. On rows with history (0.593 vs
  0.386) and with all five views (0.654 vs 0.359) the collapse is gone; cold-start and 1-4-view
  rows are unchanged, as expected for a transform of the history count. Over time F3-log follows
  F3 in the blocks without extreme counts and holds up in days 165-178 (0.53 / 0.65 vs 0.28 / 0.40).
- Even so, F3-log stays below MVAF (-0.031 PR-AUC, 95% CI -0.035 to -0.026, $19.6k more cost),
  below F4 (-0.030) and below B5 (-0.062). Without the collapse, MVAF's margin over the logistic
  stacker is about +0.03 rather than +0.20: the logistic stacker is a weaker fusion than the
  gate, but by a small margin, and MVAF is still level with F4 and behind B5 / F0 (E10).
- F3-log was not re-tuned (D95 C kept on purpose); a re-tuned version could do somewhat better.
  These are unadjusted, post-hoc comparisons on the test period and are not evidence for or
  against H1 / H2; they only bound how much of the pre-registered MVAF - F3 difference is due to
  the unbounded input.
