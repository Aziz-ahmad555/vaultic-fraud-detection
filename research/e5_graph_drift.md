# E5 graph-feature drift (POST-HOC / EXPLORATORY, validation only)

**POST-HOC / EXPLORATORY, for Paper 1.** Not pre-registered; uses the training period (days 1-120) and validation (128-150) only, never the test period. No frozen view, config or feature file changes. Generated 2026-10-11 by `python -m vaultic.reports.e5_graph_drift` (D120).

## 1. Drift of each graph feature (train -> validation)

Cumulative = counts every earlier transaction since the data start (setting C entity counts); the others are rates or 30-day-window structure. PSI with 10 training-quantile bins and KS on rows where the feature is available. Spearman: correlation with the day inside the training period (growth with calendar time).

| feature | cumulative | available train -> val | median train -> val | max train -> val | PSI | KS | val above train max | Spearman(day) |
|---|---|---|---|---|---|---|---|---|
| g_comp_fraud_rate | no | 0.64 -> 0.97 | 0.03 -> 0.04 | 1 -> 1 | 4.644 | 0.766 | 0.000 | 0.569 |
| g_comp_tx | no | 0.96 -> 0.98 | 69,700.00 -> 66,193.00 | 124,140 -> 70,839 | 4.206 | 0.485 | 0.000 | 0.045 |
| g_comp_uids | no | 0.96 -> 0.98 | 34,654.00 -> 32,896.00 | 64,039 -> 35,041 | 3.298 | 0.526 | 0.000 | -0.028 |
| g_shared_uids_email | yes | 0.86 -> 0.85 | 19,989.00 -> 43,340.00 | 80,434 -> 93,899 | 2.858 | 0.479 | 0.479 | 0.575 |
| g_deg_uid_email | yes | 0.86 -> 0.85 | 19,990.00 -> 43,340.50 | 80,435 -> 93,899 | 2.858 | 0.479 | 0.479 | 0.575 |
| g_deg_tx_email | yes | 0.86 -> 0.85 | 32,843.00 -> 88,563.50 | 169,180 -> 203,590 | 2.657 | 0.479 | 0.479 | 0.604 |
| g_fraud_rate_device | no | 0.12 -> 0.17 | 0.04 -> 0.06 | 1 -> 1 | 1.841 | 0.444 | 0.000 | 0.254 |
| g_twohop_fraud | yes | 1.00 -> 1.00 | 223.00 -> 2,249.00 | 9,164 -> 11,724 | 1.809 | 0.447 | 0.013 | 0.843 |
| g_fraud_rate_email | no | 0.58 -> 0.85 | 0.03 -> 0.04 | 0 -> 1 | 1.504 | 0.308 | 0.000 | 0.326 |
| g_shared_uids_device | yes | 0.26 -> 0.17 | 3,583.00 -> 5,204.50 | 22,468 -> 25,027 | 1.497 | 0.305 | 0.305 | 0.311 |
| g_deg_uid_device | yes | 0.26 -> 0.17 | 3,583.00 -> 5,204.50 | 22,468 -> 25,027 | 1.496 | 0.305 | 0.305 | 0.311 |
| g_deg_tx_device | yes | 0.26 -> 0.17 | 5,219.00 -> 9,964.50 | 36,703 -> 41,085 | 1.456 | 0.305 | 0.305 | 0.344 |
| g_shared_uids_address | yes | 0.89 -> 0.90 | 2,525.00 -> 5,419.00 | 11,999 -> 13,763 | 0.998 | 0.377 | 0.103 | 0.592 |
| g_deg_uid_address | yes | 0.89 -> 0.90 | 2,525.00 -> 5,419.50 | 12,000 -> 13,764 | 0.995 | 0.377 | 0.103 | 0.592 |
| g_deg_tx_address | yes | 0.89 -> 0.90 | 5,014.00 -> 13,401.50 | 33,006 -> 39,333 | 0.983 | 0.402 | 0.191 | 0.631 |
| g_fraud_rate_address | no | 0.59 -> 0.90 | 0.02 -> 0.02 | 1 -> 1 | 0.635 | 0.279 | 0.000 | 0.303 |
| g_deg_tx_card | yes | 1.00 -> 1.00 | 198.00 -> 618.00 | 9,686 -> 11,753 | 0.241 | 0.188 | 0.025 | 0.334 |
| g_deg_uid_card | yes | 1.00 -> 1.00 | 96.00 -> 223.00 | 2,377 -> 2,834 | 0.190 | 0.179 | 0.015 | 0.294 |
| g_shared_uids_card | yes | 1.00 -> 1.00 | 96.00 -> 223.00 | 2,376 -> 2,834 | 0.187 | 0.179 | 0.015 | 0.292 |
| g_comm_fraud_rate | no | 0.64 -> 0.97 | 0.02 -> 0.02 | 1 -> 1 | 0.092 | 0.098 | 0.000 | 0.069 |
| g_comm_tx | no | 0.96 -> 0.98 | 862.00 -> 472.00 | 32,424 -> 21,767 | 0.087 | 0.127 | 0.000 | -0.101 |
| g_shared_nonhub | yes | 1.00 -> 1.00 | 7.00 -> 0.00 | 212 -> 212 | 0.085 | 0.138 | 0.000 | -0.099 |
| g_fraud_rate_card | no | 0.64 -> 0.98 | 0.01 -> 0.02 | 1 -> 1 | 0.062 | 0.122 | 0.000 | 0.142 |

## 2. Does time-normalising the cumulative counts remove E5's penalty?

Validation PR-AUC. Seeds [0, 1, 2] (stored runs: the same seeds' scores). Reproduction of EXP-105-tuned seed 0 by the 'as built' refit: max |diff| 0.00e+00, correlation 1.000000. Rows: all validation rows 60,602 (2,114 frauds); graph evidence 29,688 (1,119 frauds); no graph evidence 30,914 (995 frauds).

| model | all validation rows | graph evidence | no graph evidence |
|---|---|---|---|
| B5 (stored EXP-009) | 0.6858 | 0.7018 | 0.6689 |
| E5 (stored EXP-105-tuned) | 0.6651 | 0.6823 | 0.6468 |
| B5+graph, cumulative counts as built (seeds 0) | 0.6636 | 0.6808 | 0.6450 |
| B5+graph, cumulative counts log1p (seeds 0) | 0.6636 | 0.6808 | 0.6450 |
| B5+graph, cumulative counts per elapsed day (seeds 0,1,2) | 0.6698 | 0.6915 | 0.6467 |
| B5+graph, cumulative counts dropped (seeds 0,1,2) | 0.6715 | 0.6910 | 0.6505 |

## 3. Reading (added by hand, not generated)

POST-HOC / EXPLORATORY, validation only; no frozen view or reported number changes.

- **The cumulative entity counts drift with calendar time.** Every setting-C entity count grows
  with the day inside training (Spearman 0.29-0.84) and is higher on validation: the email
  counts' median more than doubles (about 20k -> 43k distinct customers) and 48% of validation
  rows lie above the training maximum; device 31%, address 10-19%; two-hop frauds' median
  223 -> 2,249. Card counts drift less (PSI 0.19-0.24).
- **Windowed structure also shifts, for a different reason.** The 30-day component features
  have high PSI (comp size 3.3-4.2) but no values beyond the training range; their known-fraud
  rates become available on far more rows (0.64 -> 0.97), because early in training no window
  label is known yet under L = 30 (D50). That is label-availability drift, not count growth.
- **Time-normalising helps only partly.** E5 is 0.0207 below B5 on validation (seeds 0-2: 0.6651
  vs 0.6858). Counts per elapsed day give 0.6698 and dropping the cumulative counts 0.6715: about
  a third of the gap (+0.005 to +0.006) is recovered, and B5+graph stays about 0.014 below B5,
  on rows with and without graph evidence. log1p gives exactly the as-built result (trees ignore
  monotone transforms), as expected; caps would not change trees either.
- **So the E5 penalty is not mainly a count-drift artefact.** The remaining gap, also on rows
  without non-hub graph evidence, points to the other graph inputs (component / community
  structure and the label-availability shift of the graph fraud rates) or to the extra
  features' variance under B5's tuning budget. For Paper 1 this means: report the cumulative
  counts' drift as a design issue of point-in-time graph features (they need time
  normalisation or a window), but do not claim it explains E5.
- Caveats: three seeds, validation only, one hyperparameter set (EXP-105-tuned, not re-tuned per
  variant); no CIs were computed for these screening numbers.
