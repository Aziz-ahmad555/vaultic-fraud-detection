"""Free-space guard for heavy jobs (D239): drives C and E each keep at least 10 GB free.

    python tools/disk_space.py            prints free space; exit code 1 if a drive is below 10 GB

Run it before every heavy job: the full test suite (tests/conftest.py also calls it), builds,
benchmarks, parity or model runs, image pulls. Windows only; elsewhere it checks nothing.
"""

from __future__ import annotations

import os
import shutil
import sys

MIN_FREE_GB = 10.0
DRIVES = ("C:\\", "E:\\")


def free_gb(drive: str) -> float:
    return shutil.disk_usage(drive).free / 1024**3


def problems(min_gb: float = MIN_FREE_GB, drives=DRIVES, usage=free_gb) -> list[str]:
    """One message per drive below `min_gb` free (an absent drive is reported too)."""
    out = []
    for d in drives:
        try:
            gb = usage(d)
        except OSError:
            out.append(f"{d} is not available")
            continue
        if gb < min_gb:
            out.append(f"{d} has {gb:.1f} GB free, below the {min_gb:g} GB floor (D239)")
    return out


def main() -> int:
    if os.name != "nt":
        print("not Windows: no drive check")
        return 0
    for d in DRIVES:
        try:
            print(f"{d} {free_gb(d):.1f} GB free")
        except OSError:
            print(f"{d} not available")
    bad = problems()
    for p in bad:
        print("STOP:", p)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
