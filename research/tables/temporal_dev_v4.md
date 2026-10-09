# Temporal view (GRU), development run

Generated 2026-10-09 by `python -m vaultic.views.temporal_dev report`. **Validation period only.** GRU over the uid's last 20 transactions (amount, gap, ProductCD, C1, C13, D1, dist1), trained on days 1-120 with early stopping on the latest 20% of the training rows (D45), one seed. Compared with tuned B5 (5 seeds) on the validation rows where the temporal view exists (uid has history); fusion = mean of the two scores' ranks (nothing fitted on validation). Paired bootstrap, 1,000 resamples.

- B5 run: `20261008-211052-888539`; temporal rows scored: 41837 of 60602 validation rows (the rest have no history: masked).

| history | rows | frauds | B5 | GRU | fusion | GRU − B5 | fusion − B5 |
|---|---|---|---|---|---|---|---|
| 1-4 past | 21071 | 783 | 0.5996 | 0.3334 | 0.4750 | -0.2662 [-0.2949, -0.2385], p = 0.000 | -0.1246 [-0.1485, -0.1034], p = 0.000 |
| 5-19 past | 16811 | 564 | 0.8285 | 0.2542 | 0.4443 | -0.5743 [-0.6122, -0.5302], p = 0.000 | -0.3842 [-0.4255, -0.3467], p = 0.000 |
| 20+ past | 3955 | 179 | 0.9362 | 0.1233 | 0.2907 | -0.8129 [-0.8552, -0.7574], p = 0.000 | -0.6455 [-0.7071, -0.5792], p = 0.000 |
| all with history | 41837 | 1526 | 0.7354 | 0.2668 | 0.4369 | -0.4686 [-0.4923, -0.4428], p = 0.000 | -0.2985 [-0.3215, -0.2762], p = 0.000 |
