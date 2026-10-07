"""Phase 1.2: reconstruct a customer ID (uid). IEEE-CIS does not provide one.

uid = card1 + addr1 + (day - D1). D1 behaves like "days since the card was first used", so
day - D1 is roughly constant for one card holder. Variants are compared on purity and
coverage in research/uid_report.md; the chosen variant is recorded in research/decisions.md.

Missing components are kept as an explicit "NA" token, so every row gets an ID; the
*_complete flag says whether all components were present.

Run:  python -m vaultic.data.uid   (writes data/interim/uids.parquet and research/uid_report.md)
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from vaultic.data.load import load_merged
from vaultic.paths import INTERIM_DIR, MERGED_PATH, RESEARCH_DIR

UID_PATH = INTERIM_DIR / "uids.parquet"

# Only labels from this window may inform the choice of uid (train + validation; the test
# period starts on day 151). Kept in sync with experiments/configs/splits.yaml.
SELECTION_LAST_DAY = 150

D1N = "D1n"  # day - D1: the card holder's approximate "first seen" day

VARIANTS: dict[str, list[str]] = {
    "card1": ["card1"],  # coarse reference, not a candidate
    "uid_card": ["card1", "card2", "card3", "card5"],
    "uid": ["card1", "addr1", D1N],
    "uid2": ["card1", "addr1", D1N, "P_emaildomain"],
}
CANDIDATES = ["uid_card", "uid", "uid2"]
UID_INPUT_COLUMNS = [
    "TransactionID", "TransactionDT", "day", "isFraud",
    "card1", "card2", "card3", "card5", "addr1", "D1", "P_emaildomain",
]


def _token(s: pd.Series) -> pd.Series:
    """String form of one key component; numeric values without a trailing '.0'."""
    if pd.api.types.is_numeric_dtype(s):
        out = s.astype("Float64").round(6).astype(str).str.replace(r"\.0$", "", regex=True)
    else:
        out = s.astype(object).astype(str)
    return out.where(s.notna(), "NA")


def build_uids(df: pd.DataFrame) -> pd.DataFrame:
    """Return TransactionID plus one integer-coded column and one *_complete flag per variant."""
    parts = df.assign(**{D1N: df["day"] - df["D1"]})
    out = pd.DataFrame({"TransactionID": df["TransactionID"].to_numpy()}, index=df.index)
    for name, cols in VARIANTS.items():
        key = _token(parts[cols[0]])
        for col in cols[1:]:
            key = key + "_" + _token(parts[col])
        out[name] = pd.factorize(key, sort=False)[0].astype("int32")
        out[f"{name}_complete"] = parts[cols].notna().all(axis=1).astype("int8")
    return out


def uid_metrics(uid: pd.Series, is_fraud: pd.Series, complete: pd.Series) -> dict[str, float]:
    """Quality of one uid variant. Higher purity among repeat IDs = better separation."""
    g = pd.DataFrame({"uid": uid.to_numpy(), "y": is_fraud.to_numpy()}).groupby("uid")["y"]
    size = g.size()
    fraud_share = g.mean()
    repeat = size >= 2
    pure = (fraud_share == 0) | (fraud_share == 1)
    rows_in_repeat = size[repeat].sum()
    fraud_rows_repeat = (size * fraud_share)[repeat]
    fraud_rows_in_all_fraud = fraud_rows_repeat[fraud_share[repeat] == 1].sum()
    singletons = size == 1
    return {
        "ids": int(len(size)),
        "rows per id (mean)": float(size.mean()),
        "rows per id (median)": float(size.median()),
        "rows per id (p99)": float(size.quantile(0.99)),
        "rows in repeat ids %": float(rows_in_repeat / size.sum() * 100),
        "repeat ids that are pure %": float(pure[repeat].mean() * 100),
        "rows in mixed ids %": float(size[repeat & ~pure].sum() / size.sum() * 100),
        "fraud rows in all-fraud repeat ids %": float(
            fraud_rows_in_all_fraud / max(fraud_rows_repeat.sum(), 1) * 100
        ),
        "fraud rate of singleton ids %": float(fraud_share[singletons].mean() * 100),
        "complete key %": float(complete.mean() * 100),
    }


def compare_variants(df: pd.DataFrame, uids: pd.DataFrame) -> pd.DataFrame:
    rows = {
        name: uid_metrics(uids[name], df["isFraud"], uids[f"{name}_complete"])
        for name in VARIANTS
    }
    return pd.DataFrame(rows).T


def history_carryover(df: pd.DataFrame, uids: pd.DataFrame, name: str, split_day: int) -> float:
    """Share of rows after split_day whose ID already appeared on or before it (no labels)."""
    before = set(uids.loc[df["day"] <= split_day, name])
    after = uids.loc[df["day"] > split_day, name]
    return float(after.isin(before).mean() * 100)


def choose_variant(table: pd.DataFrame, min_rows_in_repeat: float = 50.0) -> str:
    """Highest purity among repeat IDs, provided most rows sit in repeat IDs.

    Without the coverage floor a variant that splits everyone into singletons would win.
    """
    eligible = table.loc[CANDIDATES]
    eligible = eligible[eligible["rows in repeat ids %"] >= min_rows_in_repeat]
    if eligible.empty:
        raise ValueError("no candidate meets the coverage floor")
    return str(eligible["repeat ids that are pure %"].idxmax())


def _md_table(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = ["| variant | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, row in table.iterrows():
        cells = [f"{int(v):,}" if c == "ids" else f"{v:,.2f}" for c, v in row.items()]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(df: pd.DataFrame, uids: pd.DataFrame) -> tuple[str, str]:
    window = df["day"] <= SELECTION_LAST_DAY
    table = compare_variants(df[window], uids[window])
    chosen = choose_variant(table)
    carry = {
        name: history_carryover(df[window], uids[window], name, split_day=120)
        for name in VARIANTS
    }
    ids_full = {name: int(uids[name].nunique()) for name in VARIANTS}

    lines = [
        "# Customer ID (uid) reconstruction report",
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.data.uid`. Do not edit by hand.",
        "",
        "IEEE-CIS has no customer ID, so one is reconstructed. Every label-based number below "
        f"uses **days 1–{SELECTION_LAST_DAY} only** (train + validation); the test period "
        f"(days {SELECTION_LAST_DAY + 1}–182) is not used for this choice.",
        "",
        "## Variants",
        "",
        "| variant | definition |",
        "|---|---|",
        "| `card1` | card1 (coarse reference, not a candidate) |",
        "| `uid_card` | card1 + card2 + card3 + card5 |",
        "| `uid` | card1 + addr1 + (day − D1) |",
        "| `uid2` | card1 + addr1 + (day − D1) + P_emaildomain |",
        "",
        "Missing components are kept as an explicit `NA` token, so every row gets an ID. "
        "`complete key %` is the share of rows where every component is present.",
        "",
        f"## Comparison (days 1–{SELECTION_LAST_DAY}, {int(window.sum()):,} rows)",
        "",
        _md_table(table),
        "",
        "How to read it:",
        "",
        "- **repeat ids that are pure %**: among IDs with 2+ transactions, the share whose "
        "transactions are all fraud or all legitimate. Higher means an ID looks more like one person.",
        "- **rows in mixed ids %**: rows sitting in IDs that mix fraud and legitimate transactions "
        "(lower is better).",
        "- **fraud rows in all-fraud repeat ids %**: of fraud rows in repeat IDs, the share in IDs "
        "that are entirely fraud.",
        "- **rows in repeat ids %**: how much history the ID provides; splitting everyone into "
        "singletons would give perfect purity and no history.",
        "",
        "## History carry-over (no labels)",
        "",
        "Share of rows on days 121–150 whose ID already appeared on days 1–120, i.e. how often "
        "the validation period can use history built in training:",
        "",
        "| variant | carry-over % | IDs over all 182 days |",
        "|---|---|---|",
        *[f"| `{n}` | {carry[n]:.2f} | {ids_full[n]:,} |" for n in VARIANTS],
        "",
        "## Provisional choice",
        "",
        f"Rule: among `uid_card`, `uid` and `uid2`, the highest **repeat ids that are pure %** "
        f"with **rows in repeat ids %** ≥ 50. Result: **`{chosen}`** (PROVISIONAL; see "
        "research/decisions.md). This choice is a stated limitation in every paper.",
        "",
    ]
    return "\n".join(lines), chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merged", type=Path, default=MERGED_PATH)
    parser.add_argument("--out", type=Path, default=UID_PATH)
    parser.add_argument("--report", type=Path, default=RESEARCH_DIR / "uid_report.md")
    args = parser.parse_args()
    df = load_merged(args.merged, columns=UID_INPUT_COLUMNS)
    uids = build_uids(df)
    uids.to_parquet(args.out, index=False)
    report, chosen = build_report(df, uids)
    args.report.write_text(report, encoding="utf-8")
    print(f"wrote {args.out} and {args.report}; provisional choice: {chosen}")


if __name__ == "__main__":
    main()
