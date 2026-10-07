"""Phase 1.1: load the IEEE-CIS training files, merge, order by time, reduce dtypes.

Run:  python -m vaultic.data.load            (full data -> data/interim/merged.parquet)
      python -m vaultic.data.load --nrows 50000 --out data/interim/merged_sample.parquet

Raw CSVs in data/raw/ are only ever read. Only the Kaggle *training* files are used; the
Kaggle test files have no public labels and are never used for evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vaultic.paths import MERGED_PATH, RAW_DIR

SECONDS_PER_DAY = 86_400
TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

# Kept exact: uid reconstruction and amount features need exact keys and exact cents.
EXACT_DTYPES = {
    "TransactionID": "int32",
    "TransactionDT": "int32",
    "TransactionAmt": "float64",
    "isFraud": "int8",
}


def _column_dtypes(path: Path, probe_rows: int = 20_000) -> dict[str, str]:
    """float32 for numeric columns, object for strings, exact dtypes for key columns."""
    probe = pd.read_csv(path, nrows=probe_rows)
    dtypes = {}
    for col, dtype in probe.dtypes.items():
        if col in EXACT_DTYPES:
            dtypes[col] = EXACT_DTYPES[col]
        elif pd.api.types.is_object_dtype(dtype):
            dtypes[col] = "object"
        else:
            dtypes[col] = "float32"
    return dtypes


def read_csv_reduced(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=_column_dtypes(path), nrows=nrows)


def merge_and_order(transactions: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Left-join identity on TransactionID, add has_identity and day, sort by TransactionDT."""
    if transactions["TransactionID"].duplicated().any():
        raise ValueError("duplicate TransactionID in transaction file")
    if identity["TransactionID"].duplicated().any():
        raise ValueError("duplicate TransactionID in identity file")

    df = transactions.merge(identity, on="TransactionID", how="left", indicator=True)
    df["has_identity"] = (df.pop("_merge") == "both").astype("int8")
    df["day"] = (df["TransactionDT"] // SECONDS_PER_DAY).astype("int16")
    # TransactionID breaks ties so the order is fully deterministic.
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort")
    df = df.reset_index(drop=True)

    for col in df.columns[[pd.api.types.is_object_dtype(t) for t in df.dtypes]]:
        df[col] = df[col].astype("category")
    return df


def build_merged(
    raw_dir: Path = RAW_DIR, out_path: Path = MERGED_PATH, nrows: int | None = None
) -> pd.DataFrame:
    """Read both raw files, merge and order them, and write the result as parquet.

    With nrows, only the first nrows transactions (file order) are read; identity rows are
    read in full and joined to whichever transactions are present.
    """
    transactions = read_csv_reduced(raw_dir / TRANSACTION_FILE, nrows=nrows)
    identity = read_csv_reduced(raw_dir / IDENTITY_FILE)
    df = merge_and_order(transactions, identity)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def load_merged(path: Path = MERGED_PATH, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--out", type=Path, default=MERGED_PATH)
    args = parser.parse_args()
    df = build_merged(out_path=args.out, nrows=args.nrows)
    mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"wrote {args.out}: {len(df):,} rows x {df.shape[1]} columns, {mb:,.0f} MB in memory")


if __name__ == "__main__":
    main()
