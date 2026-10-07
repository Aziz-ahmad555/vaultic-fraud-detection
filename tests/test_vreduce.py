import numpy as np
import pandas as pd

from vaultic.features.vreduce import missing_pattern_groups, reduce_v, v_columns


def _df():
    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(size=n)
    mask_a = np.arange(n) % 4 == 0  # pattern A: every 4th row missing
    df = pd.DataFrame(
        {
            "V1": np.round(base, 1),  # pattern A, few distinct values
            "V2": base * 2 + 1,  # pattern A, r = 1 with V1, more distinct values
            "V3": rng.normal(size=n),  # pattern A, uncorrelated -> own cluster
            "V10": base,  # no missing values (pattern B): correlated with V1, other group
            "C1": base,  # not a V column
        }
    )
    for c in ("V1", "V2", "V3"):
        df.loc[mask_a, c] = np.nan
    return df


def test_v_columns_sorted_numerically():
    assert v_columns(["V10", "V2", "C1", "V1", "id_01"]) == ["V1", "V2", "V10"]


def test_groups_by_identical_missing_pattern():
    groups = missing_pattern_groups(_df(), ["V1", "V2", "V3", "V10"])
    assert sorted(map(sorted, groups)) == [["V1", "V2", "V3"], ["V10"]]


def test_keeps_one_per_correlated_cluster_with_most_distinct_values():
    r = reduce_v(_df())
    # V1 and V2 cluster (r = 1): V2 has more distinct values. V3 stands alone. V10 is in
    # another missing-pattern group, so it is kept even though it correlates with V1.
    assert r["keep"] == ["V2", "V3", "V10"]
    assert r["n_input"] == 4 and r["n_kept"] == 3
    assert {"keep": "V2", "members": ["V1", "V2"]} in r["clusters"]
