"""Check that the raw CSVs in use are the ones recorded by DVC (data/raw/*.dvc).

Run:  python -m vaultic.data.verify

Useful wherever the CSVs come from somewhere else (e.g. Kaggle's read-only input folder):
compares the md5 of each file in RAW_DIR with the md5 in the repo's DVC pointer files.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

from vaultic.paths import RAW_DIR, REPO_ROOT

POINTER_DIR = REPO_ROOT / "data" / "raw"


def md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - DVC's content hash, not security
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(raw_dir: Path = RAW_DIR, pointer_dir: Path = POINTER_DIR) -> dict[str, str]:
    """name -> 'ok' | 'missing' | 'MISMATCH'."""
    status = {}
    for pointer in sorted(pointer_dir.glob("*.dvc")):
        out = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]
        path = raw_dir / out["path"]
        if not path.exists():
            status[out["path"]] = "missing"
        else:
            status[out["path"]] = "ok" if md5(path) == out["md5"] else "MISMATCH"
    return status


def main() -> None:
    status = verify()
    for name, state in status.items():
        print(f"{state:8s} {name}")
    sys.exit(0 if status and all(s == "ok" for s in status.values()) else 1)


if __name__ == "__main__":
    main()
