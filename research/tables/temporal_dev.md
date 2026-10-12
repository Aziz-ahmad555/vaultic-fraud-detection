# Temporal view (GRU), development run

Generated 2026-10-09 by `python -m vaultic.views.temporal_dev report`. **Validation period only.** GRU over the uid's last 20 transactions (amount, gap, ProductCD, C1, C13, D1, dist1), trained on days 1-120 with early stopping on the latest 20% of the training rows (D45), one seed. Compared with tuned B5 (5 seeds) on the validation rows where the temporal view exists (uid has history); fusion = mean of the two scores' ranks (nothing fitted on validation). Paired bootstrap, 1,000 resamples.

- B5 run: `20261008-211052-888539`; temporal rows scored: 41837 of 60602 validation rows (the rest have no history: masked).

| history | rows | frauds | B5 | GRU | fusion | GRU − B5 | fusion − B5 |
|---|---|---|---|---|---|---|---|
| 1-4 past | 21071 | 783 | 0.5996 | 0.1292 | 0.4060 | -0.4704 [-0.5008, -0.4391], p = 0.000 | -0.1936 [-0.2238, -0.1639], p = 0.000 |
| 5-19 past | 16811 | 564 | 0.8285 | 0.1253 | 0.3887 | -0.7032 [-0.7330, -0.6724], p = 0.000 | -0.4398 [-0.4799, -0.4005], p = 0.000 |
| 20+ past | 3955 | 179 | 0.9362 | 0.0876 | 0.2968 | -0.8487 [-0.8754, -0.8147], p = 0.000 | -0.6394 [-0.6927, -0.5759], p = 0.000 |
| all with history | 41837 | 1526 | 0.7354 | 0.1218 | 0.4020 | -0.6136 [-0.6318, -0.5940], p = 0.000 | -0.3334 [-0.3578, -0.3105], p = 0.000 |
