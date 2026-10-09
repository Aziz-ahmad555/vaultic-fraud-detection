# Temporal view (GRU), development run — SMOKE (1 in 10 uids, not a result)

Generated 2026-10-09 by `python -m vaultic.views.temporal_dev report`. **Validation period only.** GRU over the uid's last 20 transactions (amount, gap, ProductCD, C1, C13, D1, dist1), trained on days 1-120 with early stopping on the latest 20% of the training rows (D45), one seed. Compared with tuned B5 (5 seeds) on the validation rows where the temporal view exists (uid has history); fusion = mean of the two scores' ranks (nothing fitted on validation). Paired bootstrap, 1,000 resamples.

- B5 run: `20261008-211052-888539`; temporal rows scored: 4224 of 6064 validation rows (the rest have no history: masked).

| history | rows | frauds | B5 | GRU | fusion | GRU − B5 | fusion − B5 |
|---|---|---|---|---|---|---|---|
| 1-4 past | 2048 | 82 | 0.5517 | 0.1404 | 0.3267 | -0.4113 [-0.5182, -0.2817], p = 0.000 | -0.2250 [-0.3209, -0.1089], p = 0.000 |
| 5-19 past | 1794 | 62 | 0.8925 | 0.1957 | 0.6370 | -0.6968 [-0.7760, -0.5974], p = 0.000 | -0.2555 [-0.3744, -0.1483], p = 0.000 |
| 20+ past | 382 | 7 | 1.0000 | 0.0492 | 0.4426 | -0.9508 [-0.9846, -0.8561], p = 0.000 | -0.5574 [-0.9352, -0.1941], p = 0.014 |
| all with history | 4224 | 151 | 0.7280 | 0.1335 | 0.4660 | -0.5945 [-0.6543, -0.5124], p = 0.000 | -0.2620 [-0.3387, -0.1811], p = 0.000 |
