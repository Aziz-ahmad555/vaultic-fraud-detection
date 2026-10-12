# MVAF after the pre-registered changes (D95): the one evaluation on days 147-150 — SMOKE (not a result)

Generated 2026-10-09 by `python -m vaultic.fusion.d95_compare`. **Validation only; no test rows.** Plan, tuning and this single evaluation were pre-registered in D95 before any of it ran. Evaluation rows: 1073 (31 frauds), never used to fit or tune anything here. B5 (frozen Phase 2 model) was tuned on the whole validation period, so it is favoured on these days.

## Gate-training rows (cross-fitted, D95)

| fold | rows | frauds |
|---|---|---|
| crossfit_0 | 8532 | 358 |
| crossfit_1 | 8072 | 276 |
| crossfit_2 | 8852 | 315 |
| fixed | 4106 | 156 |
| all | 29562 | 1105 |

## Views on the evaluation rows

| view | available | PR-AUC (available rows) |
|---|---|---|
| tabular | 100.0% | 0.3951 |
| behavioral | 71.1% | 0.0869 |
| temporal | 71.1% | 0.1907 |
| graph | 48.8% | 0.1202 |
| anomaly | 100.0% | 0.1443 |

## Fused and single models

| method | PR-AUC (5-seed mean score) | 95% CI | per-seed mean ± std | MVAF − method | 95% CI | p | tuned on the inner split |
|---|---|---|---|---|---|---|---|
| MVAF | 0.4167 | [0.2431, 0.5903] | 0.4010 ± 0.0095 |  | |  | 3 trials, tune PR-AUC 0.5868 |
| F0 | 0.4760 | [0.3060, 0.6453] | (run's 5 seeds) | -0.0593 | [-0.1227, +0.0065] | 0.092 | F0: Optuna on 128-143 |
| B5 | 0.5199 | [0.3329, 0.6925] | (run's 5 seeds) | -0.1033 | [-0.1851, -0.0099] | 0.036 | — |
| F1 | 0.3278 | [0.1899, 0.5302] | 0.3278 ± 0.0000 | +0.0889 | [+0.0146, +0.1402] | 0.010 | — |
| F2 | 0.4051 | [0.2363, 0.5788] | 0.4051 ± 0.0003 | +0.0116 | [-0.0190, +0.0482] | 0.422 | — |
| F3 | 0.4029 | [0.2345, 0.5772] | 0.4029 ± 0.0000 | +0.0137 | [-0.0195, +0.0525] | 0.354 | 3 trials, tune PR-AUC 0.5622 |
| F4 | 0.4183 | [0.2492, 0.5933] | 0.4183 ± 0.0000 | -0.0016 | [-0.0662, +0.0606] | 0.964 | 3 trials, tune PR-AUC 0.5400 |
| F5 | 0.3831 | [0.2212, 0.5621] | 0.3886 ± 0.0041 | +0.0335 | [-0.0018, +0.0710] | 0.070 | 3 trials, tune PR-AUC 0.5917 |
| F6 | 0.4100 | [0.2352, 0.5912] | 0.4135 ± 0.0008 | +0.0066 | [-0.0328, +0.0542] | 0.762 | 3 trials, tune PR-AUC 0.5993 |
| F7 | 0.4168 | [0.2447, 0.5934] | 0.4159 ± 0.0116 | -0.0001 | [-0.0278, +0.0298] | 0.970 | 3 trials, tune PR-AUC 0.6197 |

## Subgroups

| subgroup | rows | frauds | MVAF | F0 | B5 | F1 | F2 | F3 | F4 | F5 | F6 | F7 | MVAF - B5 | MVAF - F0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold start | 310 | 12 | 0.5569 | 0.5873 | 0.6246 | 0.4719 | 0.5695 | 0.5487 | 0.5540 | 0.5635 | 0.5608 | 0.5444 | -0.0677 [-0.2681, +0.1192] | -0.0304 [-0.1970, +0.1304] |
| with history | 763 | 19 | 0.3348 | 0.4252 | 0.4877 | 0.2693 | 0.3230 | 0.3308 | 0.3296 | 0.3044 | 0.3438 | 0.3629 | -0.1529 [-0.2554, -0.0534] | -0.0904 [-0.1617, -0.0234] |
| has_identity yes | 194 | 17 | 0.6874 | 0.7670 | 0.8025 | 0.5417 | 0.6646 | 0.6642 | 0.7003 | 0.6479 | 0.6570 | 0.6997 | -0.1152 [-0.2666, +0.0225] | -0.0796 [-0.1976, +0.0410] |
| has_identity no | 879 | 14 | 0.0780 | 0.1183 | 0.1541 | 0.0756 | 0.0750 | 0.0660 | 0.0680 | 0.0736 | 0.0764 | 0.0736 | -0.0761 [-0.2543, +0.0164] | -0.0402 [-0.2009, +0.0269] |
| graph view available | 524 | 12 | 0.5039 | 0.5298 | 0.5181 | 0.3482 | 0.5179 | 0.5028 | 0.5697 | 0.4873 | 0.5022 | 0.5130 | -0.0141 [-0.1534, +0.1647] | -0.0259 [-0.1560, +0.1469] |
| graph view missing | 549 | 19 | 0.4248 | 0.4976 | 0.5774 | 0.3925 | 0.3821 | 0.4035 | 0.3626 | 0.3826 | 0.3871 | 0.4126 | -0.1526 [-0.2687, -0.0407] | -0.0728 [-0.1718, +0.0227] |

Runtime 276 s.
