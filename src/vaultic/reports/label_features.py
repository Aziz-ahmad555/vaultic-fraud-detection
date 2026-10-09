"""Which B5 features are derived from labels? Found by perturbation, not by name.

Run:  python -m vaultic.reports.label_features      (uses data/interim/merged_sample.parquet)

The base features (vaultic.features.pipeline.build_features) are built three times on the same
rows: as is, with every label flipped, and with a longer label delay. A column that changes when
the labels are flipped is derived from label VALUES; a column that changes only with the delay
is derived from label TIMING (how many labels are known, not what they are). The sample spans
only ~13 days, so a 1-day delay is used to make labels mature inside it (the dependency, not the
value, is what is measured). The raw IEEE-CIS columns in B5 come from Vesta and are not built
by this code; whether any of them encode past labels cannot be checked here.

Writes research/tables/b5_label_features.md.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from vaultic.features.pipeline import build_features, fit_encoders
from vaultic.paths import RESEARCH_DIR


def _changed(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    out = []
    for c in a.columns:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        if not np.array_equal(x, y, equal_nan=True):
            out.append(c)
    return out


def label_dependent_columns(
    df: pd.DataFrame, uid: pd.Series, delay_days: int = 1, longer_delay_days: int = 3
) -> dict[str, list[str]]:
    """{'values': columns that change when labels flip, 'timing': change only with the delay}."""
    encoders = fit_encoders(df, uid)  # frequency encoders use no labels
    base = build_features(df, uid, encoders, delay_days).drop(columns="TransactionID")
    flipped = df.assign(isFraud=1 - df["isFraud"])
    flip = build_features(flipped, uid, encoders, delay_days).drop(columns="TransactionID")
    later = build_features(df, uid, encoders, longer_delay_days).drop(columns="TransactionID")
    values = _changed(base, flip)
    timing = [c for c in _changed(base, later) if c not in values]
    return {"values": values, "timing": timing}


def main() -> None:
    from vaultic.data.load import load_merged
    from vaultic.data.splits import load_splits
    from vaultic.data.uid import build_uids
    from vaultic.paths import INTERIM_DIR

    df = load_merged(INTERIM_DIR / "merged_sample.parquet")
    uid = build_uids(df)[load_splits().uid_variant]
    found = label_dependent_columns(df, uid)
    lines = [
        "# Label-derived features in B5",
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.reports.label_features` on the "
        "50k sample: base features rebuilt with all labels flipped (values) and with a longer label "
        "delay (timing). All of them respect the label delay L (its_time + L <= t).",
        "",
        "| feature | derived from |",
        "|---|---|",
        *[f"| `{c}` | label values (known frauds) |" for c in found["values"]],
        *[
            f"| `{c}` | label timing (number of known labels, not their values) |"
            for c in found["timing"]
        ],
        "",
        "No other base feature changes. The raw IEEE-CIS columns (C, D, M, V, id_ and the "
        "transaction fields) are supplied by Vesta, not built here; whether any encode past "
        "labels cannot be checked from this data.",
        "",
    ]
    out = RESEARCH_DIR / "tables" / "b5_label_features.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}: {found}")


if __name__ == "__main__":
    main()
