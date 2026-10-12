"""D48: the threaded bootstrap must give IDENTICAL CIs to the original single-threaded code."""

import numpy as np
import pytest

from vaultic.eval.bootstrap import (
    _iter_resample_indices,
    _resample_indices,
    paired_bootstrap,
    paired_bootstrap_reference,
    seed_mean_ci,
    seed_mean_ci_reference,
)
from vaultic.eval.metrics import RANKING_METRICS, pr_auc


def _data(n=3000, n_seeds=3, prevalence=0.05, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    seeds = [np.clip(0.3 * y + rng.normal(0.2, 0.15, n), 0, 1) for _ in range(n_seeds)]
    other = [np.clip(0.2 * y + rng.normal(0.25, 0.15, n), 0, 1) for _ in range(n_seeds)]
    return y, seeds, other


def test_indices_are_the_same_draws_in_the_same_order():
    y, _, _ = _data(n=200, prevalence=0.02)  # rare positives: some resamples get rejected
    old = _resample_indices(len(y), 300, 7, y)
    new = list(_iter_resample_indices(len(y), 300, 7, y))
    assert len(old) == len(new) == 300
    assert all(np.array_equal(a, b) for a, b in zip(old, new, strict=True))


@pytest.mark.parametrize("name", sorted(RANKING_METRICS))
@pytest.mark.parametrize("jobs", [1, 3, 8])
def test_seed_mean_ci_identical_to_reference(name, jobs):
    y, seeds, _ = _data()
    metric = RANKING_METRICS[name]
    ref = seed_mean_ci_reference(y, seeds, metric, n_boot=200, seed=11)
    new = seed_mean_ci(y, seeds, metric, n_boot=200, seed=11, n_jobs=jobs)
    assert new == ref  # exact, not approximate


@pytest.mark.parametrize("jobs", [1, 4])
def test_paired_bootstrap_identical_to_reference(jobs):
    y, a, b = _data()
    ref = paired_bootstrap_reference(y, a, b, pr_auc, n_boot=200, seed=3)
    new = paired_bootstrap(y, a, b, pr_auc, n_boot=200, seed=3, n_jobs=jobs)
    assert new == ref


def test_identical_when_many_resamples_are_rejected():
    y, seeds, other = _data(n=120, prevalence=0.02, seed=5)  # often 0 positives in a resample
    assert seed_mean_ci(y, seeds, pr_auc, n_boot=150, seed=2, n_jobs=4) == seed_mean_ci_reference(
        y, seeds, pr_auc, n_boot=150, seed=2
    )
    assert paired_bootstrap(
        y, seeds, other, pr_auc, n_boot=150, seed=2, n_jobs=4
    ) == paired_bootstrap_reference(y, seeds, other, pr_auc, n_boot=150, seed=2)


def test_progress_line(capsys):
    y, seeds, _ = _data(n=500)
    seed_mean_ci(y, seeds, pr_auc, n_boot=40, seed=0, n_jobs=2, label="test pr_auc")
    err = capsys.readouterr().err.strip().splitlines()
    assert err and err[-1].startswith("bootstrap test pr_auc: 40/40 resamples")
    seed_mean_ci(y, seeds, pr_auc, n_boot=40, seed=0, n_jobs=2)  # no label: silent
    assert capsys.readouterr().err == ""
