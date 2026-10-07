"""Phase 1.1 data report: computed from data/interim/merged.parquet, written as markdown.

Run:  python -m vaultic.data.report   (writes research/data_report.md)
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import pandas as pd

from vaultic.data.load import load_merged
from vaultic.paths import MERGED_PATH, RESEARCH_DIR

BLOCK_DAYS = 30  # "month" = 30-day block counted from the first day in the data (no calendar)

COLUMN_GROUPS = {
    "card1-6": r"^card\d$",
    "addr1-2": r"^addr\d$",
    "dist1-2": r"^dist\d$",
    "email domains": r"^[PR]_emaildomain$",
    "C1-C14 (counts)": r"^C\d+$",
    "D1-D15 (time deltas)": r"^D\d+$",
    "M1-M9 (matches)": r"^M\d+$",
    "V1-V339 (Vesta features)": r"^V\d+$",
    "id_01-id_38 (identity)": r"^id_\d+$",
    "DeviceType/DeviceInfo": r"^Device",
}


def add_block(df: pd.DataFrame) -> pd.Series:
    return (df["day"] - df["day"].min()) // BLOCK_DAYS


def block_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.assign(block=add_block(df)).groupby("block")
    return pd.DataFrame(
        {
            "first day": g["day"].min(),
            "last day": g["day"].max(),
            "days": g["day"].nunique(),
            "rows": g.size(),
            "fraud rate %": g["isFraud"].mean() * 100,
            "has_identity %": g["has_identity"].mean() * 100,
        }
    )


def missing_by_group(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, pattern in COLUMN_GROUPS.items():
        cols = [c for c in df.columns if re.match(pattern, c)]
        if not cols:
            continue
        rates = df[cols].isna().mean() * 100
        rows[name] = {
            "columns": len(cols),
            "mean missing %": rates.mean(),
            "min %": rates.min(),
            "max %": rates.max(),
        }
    table = pd.DataFrame(rows).T
    table["columns"] = table["columns"].astype(int)
    return table


def _md(table: pd.DataFrame, index_name: str) -> str:
    table = table.reset_index().rename(columns={"index": index_name})
    header = "| " + " | ".join(map(str, table.columns)) + " |"
    sep = "|" + "---|" * len(table.columns)
    formatters = []
    for col in table.columns:
        if pd.api.types.is_integer_dtype(table[col]):
            formatters.append(lambda v: f"{int(v):,}")
        elif pd.api.types.is_float_dtype(table[col]):
            formatters.append(lambda v: f"{v:,.2f}")
        else:
            formatters.append(str)
    lines = [header, sep]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(f(v) for f, v in zip(formatters, row)) + " |")
    return "\n".join(lines)


def build_report(df: pd.DataFrame, source: Path = MERGED_PATH) -> str:
    n = len(df)
    fraud = int(df["isFraud"].sum())
    days = df["day"]
    amt = df["TransactionAmt"]
    blocks = block_table(df)
    for col in ["first day", "last day", "days", "rows"]:
        blocks[col] = blocks[col].astype(int)
    product = df.groupby("ProductCD", observed=True)["isFraud"].agg(["size", "mean"])
    product = pd.DataFrame(
        {"rows": product["size"].astype(int), "fraud rate %": product["mean"] * 100}
    )

    lines = [
        "# IEEE-CIS data report",
        "",
        f"Generated {date.today().isoformat()} by `python -m vaultic.data.report` from "
        f"`{source.name}`. Do not edit by hand.",
        "",
        "Data: anonymized e-commerce transactions released by Vesta for the IEEE-CIS competition "
        "(training files only; the Kaggle test files have no public labels and are not used).",
        "",
        "## Overview",
        "",
        "| Fact | Value |",
        "|---|---|",
        f"| Transactions | {n:,} |",
        f"| Columns after merge | {df.shape[1]:,} (incl. added `has_identity`, `day`) |",
        f"| Fraud | {fraud:,} ({fraud / n * 100:.3f}%) |",
        f"| Rows with identity data | {int(df['has_identity'].sum()):,} "
        f"({df['has_identity'].mean() * 100:.2f}%) |",
        f"| TransactionDT range (s) | {df['TransactionDT'].min():,} – {df['TransactionDT'].max():,} |",
        f"| Day range (TransactionDT // 86400) | {days.min()} – {days.max()} "
        f"({days.nunique()} distinct days) |",
        f"| TransactionID unique | {df['TransactionID'].is_unique} |",
        f"| Sorted by TransactionDT | {df['TransactionDT'].is_monotonic_increasing} |",
        f"| TransactionAmt min / median / p99 / max | {amt.min():,.3f} / {amt.median():,.2f} / "
        f"{amt.quantile(0.99):,.2f} / {amt.max():,.2f} |",
        f"| Transactions per day min / median / max | {days.value_counts().min():,} / "
        f"{int(days.value_counts().median()):,} / {days.value_counts().max():,} |",
        "",
        "TransactionDT is seconds from an unknown reference point, so there are no calendar "
        "dates or local hours. `day` counts days from that reference.",
        "",
        f"## Per {BLOCK_DAYS}-day block",
        "",
        f"Blocks are {BLOCK_DAYS}-day windows counted from the first day in the data; the last "
        "block can be partial.",
        "",
        _md(blocks, "block"),
        "",
        "## Missing values by column group",
        "",
        _md(missing_by_group(df), "group"),
        "",
        "## ProductCD",
        "",
        _md(product, "ProductCD"),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merged", type=Path, default=MERGED_PATH)
    parser.add_argument("--out", type=Path, default=RESEARCH_DIR / "data_report.md")
    args = parser.parse_args()
    report = build_report(load_merged(args.merged), source=args.merged)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
