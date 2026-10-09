# Temporal view (GRU), development run

Generated 2026-10-09 by `python -m vaultic.views.temporal_dev report`. **Validation period only.** GRU over the uid's last 20 transactions (amount, gap, ProductCD, C1, C13, D1, dist1), trained on days 1-120 with early stopping on the latest 20% of the training rows (D45), one seed. Compared with tuned B5 (5 seeds) on the validation rows where the temporal view exists (uid has history); fusion = mean of the two scores' ranks (nothing fitted on validation). Paired bootstrap, 1,000 resamples.

- B5 run: `20261008-211052-888539`; temporal rows scored: 41837 of 60602 validation rows (the rest have no history: masked).

| history | rows | frauds | B5 | GRU | fusion | GRU − B5 | fusion − B5 |
|---|---|---|---|---|---|---|---|
| 1-4 past | 21071 | 783 | 0.5996 | 0.1308 | 0.3938 | -0.4688 [-0.4983, -0.4385], p = 0.000 | -0.2058 [-0.2353, -0.1773], p = 0.000 |
| 5-19 past | 16811 | 564 | 0.8285 | 0.1286 | 0.3993 | -0.6999 [-0.7298, -0.6699], p = 0.000 | -0.4292 [-0.4729, -0.3883], p = 0.000 |
| 20+ past | 3955 | 179 | 0.9362 | 0.0910 | 0.3291 | -0.8452 [-0.8740, -0.8086], p = 0.000 | -0.6071 [-0.6677, -0.5431], p = 0.000 |
| all with history | 41837 | 1526 | 0.7354 | 0.1240 | 0.3937 | -0.6114 [-0.6298, -0.5917], p = 0.000 | -0.3417 [-0.3664, -0.3182], p = 0.000 |
