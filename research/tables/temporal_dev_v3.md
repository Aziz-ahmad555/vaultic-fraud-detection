# Temporal view (GRU), development run

Generated 2026-10-09 by `python -m vaultic.views.temporal_dev report`. **Validation period only.** GRU over the uid's last 20 transactions (amount, gap, ProductCD, C1, C13, D1, dist1), trained on days 1-120 with early stopping on the latest 20% of the training rows (D45), one seed. Compared with tuned B5 (5 seeds) on the validation rows where the temporal view exists (uid has history); fusion = mean of the two scores' ranks (nothing fitted on validation). Paired bootstrap, 1,000 resamples.

- B5 run: `20261008-211052-888539`; temporal rows scored: 41837 of 60602 validation rows (the rest have no history: masked).

| history | rows | frauds | B5 | GRU | fusion | GRU − B5 | fusion − B5 |
|---|---|---|---|---|---|---|---|
| 1-4 past | 21071 | 783 | 0.5996 | 0.3219 | 0.4700 | -0.2778 [-0.3064, -0.2493], p = 0.000 | -0.1296 [-0.1537, -0.1083], p = 0.000 |
| 5-19 past | 16811 | 564 | 0.8285 | 0.2577 | 0.4559 | -0.5708 [-0.6102, -0.5269], p = 0.000 | -0.3726 [-0.4148, -0.3339], p = 0.000 |
| 20+ past | 3955 | 179 | 0.9362 | 0.1409 | 0.3196 | -0.7953 [-0.8425, -0.7384], p = 0.000 | -0.6167 [-0.6804, -0.5484], p = 0.000 |
| all with history | 41837 | 1526 | 0.7354 | 0.2631 | 0.4464 | -0.4723 [-0.4966, -0.4472], p = 0.000 | -0.2891 [-0.3116, -0.2672], p = 0.000 |
