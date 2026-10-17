"""Stage 2 (D121, implementation details D158-): E15 explanations, the ablation ladder, E23
subgroup gaps and E24 compute cost. Development runs are tested on synthetic data and run on
validation rows: the data loaders read every row, and the test-period rows are dropped right
after loading, never scored (D180). Test-period runs go through the frozen final-run guards
(stage2/final.py)."""
