# MVAF after the pre-registered changes (D95): the one evaluation on days 147-150

Generated 2026-10-10 by `python -m vaultic.fusion.d95_compare`. **Validation only; no test rows.** Plan, tuning and this single evaluation were pre-registered in D95 before any of it ran. Evaluation rows: 10276 (300 frauds), never used to fit or tune anything here. B5 (frozen Phase 2 model) was tuned on the whole validation period, so it is favoured on these days.

## Gate-training rows (cross-fitted, D95)

| fold | rows | frauds |
|---|---|---|
| crossfit_0 | 81384 | 3199 |
| crossfit_1 | 82589 | 3403 |
| crossfit_2 | 88195 | 3455 |
| fixed | 41950 | 1559 |
| all | 294118 | 11616 |

## Views on the evaluation rows

| view | available | PR-AUC (available rows) |
|---|---|---|
| tabular | 100.0% | 0.4654 |
| behavioral | 68.9% | 0.4291 |
| temporal | 68.9% | 0.2071 |
| graph | 47.2% | 0.1738 |
| anomaly | 100.0% | 0.1053 |

## Fused and single models

| method | PR-AUC (5-seed mean score) | 95% CI | per-seed mean ± std | MVAF − method | 95% CI | p | tuned on the inner split |
|---|---|---|---|---|---|---|---|
| MVAF | 0.5935 | [0.5436, 0.6447] | 0.5924 ± 0.0069 |  | |  | 50 trials, tune PR-AUC 0.7098 |
| F0 | 0.6172 | [0.5637, 0.6653] | (run's 5 seeds) | -0.0237 | [-0.0401, -0.0081] | 0.006 | F0: Optuna on 128-143 |
| B5 | 0.6285 | [0.5772, 0.6777] | (run's 5 seeds) | -0.0349 | [-0.0490, -0.0223] | 0.000 | — |
| F1 | 0.5416 | [0.4860, 0.6008] | 0.5416 ± 0.0000 | +0.0519 | [+0.0298, +0.0750] | 0.000 | — |
| F2 | 0.6042 | [0.5551, 0.6539] | 0.6041 ± 0.0008 | -0.0107 | [-0.0252, +0.0035] | 0.158 | — |
| F3 | 0.6047 | [0.5555, 0.6547] | 0.6047 ± 0.0000 | -0.0112 | [-0.0258, +0.0034] | 0.126 | 50 trials, tune PR-AUC 0.6832 |
| F4 | 0.6042 | [0.5535, 0.6548] | 0.6029 ± 0.0023 | -0.0107 | [-0.0214, -0.0001] | 0.048 | 50 trials, tune PR-AUC 0.7065 |
| F5 | 0.5969 | [0.5475, 0.6478] | 0.5950 ± 0.0045 | -0.0034 | [-0.0096, +0.0031] | 0.314 | 50 trials, tune PR-AUC 0.7104 |
| F6 | 0.5990 | [0.5494, 0.6498] | 0.5984 ± 0.0019 | -0.0055 | [-0.0106, -0.0001] | 0.040 | 50 trials, tune PR-AUC 0.7106 |
| F7 | 0.5958 | [0.5456, 0.6483] | 0.5954 ± 0.0016 | -0.0023 | [-0.0080, +0.0055] | 0.452 | 50 trials, tune PR-AUC 0.7098 |

## Subgroups

| subgroup | rows | frauds | MVAF | F0 | B5 | F1 | F2 | F3 | F4 | F5 | F6 | F7 | MVAF - B5 | MVAF - F0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold start | 3197 | 85 | 0.5837 | 0.5769 | 0.5975 | 0.5568 | 0.6054 | 0.5715 | 0.5876 | 0.5764 | 0.5818 | 0.5866 | -0.0138 [-0.0433, +0.0130] | +0.0069 [-0.0273, +0.0408] |
| with history | 7079 | 215 | 0.6059 | 0.6369 | 0.6441 | 0.5387 | 0.6094 | 0.6184 | 0.6150 | 0.6130 | 0.6120 | 0.6095 | -0.0381 [-0.0548, -0.0220] | -0.0310 [-0.0504, -0.0134] |
| has_identity yes | 1836 | 141 | 0.6944 | 0.7101 | 0.7268 | 0.6566 | 0.7178 | 0.7209 | 0.7029 | 0.7030 | 0.6982 | 0.7020 | -0.0323 [-0.0502, -0.0141] | -0.0157 [-0.0363, +0.0067] |
| has_identity no | 8440 | 159 | 0.4963 | 0.5232 | 0.5343 | 0.4589 | 0.4933 | 0.4891 | 0.5115 | 0.4982 | 0.4981 | 0.4855 | -0.0380 [-0.0595, -0.0189] | -0.0269 [-0.0521, -0.0051] |
| graph view available | 4853 | 149 | 0.5793 | 0.5987 | 0.6102 | 0.5309 | 0.6148 | 0.6060 | 0.5941 | 0.5869 | 0.5869 | 0.5864 | -0.0309 [-0.0525, -0.0103] | -0.0194 [-0.0416, +0.0039] |
| graph view missing | 5423 | 151 | 0.6126 | 0.6363 | 0.6497 | 0.5551 | 0.5980 | 0.6038 | 0.6199 | 0.6132 | 0.6142 | 0.6098 | -0.0371 [-0.0558, -0.0198] | -0.0237 [-0.0473, -0.0034] |

Runtime 12184 s.
