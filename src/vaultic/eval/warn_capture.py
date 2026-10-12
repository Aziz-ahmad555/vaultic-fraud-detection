"""Capture the warnings a run raises instead of silencing them.

The Phase 2 chain ran Python with `-W ignore`, which hid B1's lbfgs ConvergenceWarning (D61).
Runs now record every warning: grouped by (category, message, origin) with a count, re-printed
to stderr so the chain log keeps them, and summarised in metrics.json. Categories in FLAGGED
point at a result that may be wrong (an optimiser that stopped early, a metric that is
undefined, a numerical overflow) and are flagged in the run summary.

Warnings raised inside joblib/loky worker processes do not reach the parent process and are not
captured; warnings from threads are.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

FLAGGED = ("ConvergenceWarning", "FitFailedWarning", "UndefinedMetricWarning", "RuntimeWarning")
IGNORED = ("DeprecationWarning", "PendingDeprecationWarning", "FutureWarning")
MAX_MESSAGE = 300


class Captured:
    """Grouped warnings of one run (filled while the context is open)."""

    def __init__(self) -> None:
        self.groups: dict[tuple[str, str, str], int] = {}

    def add(self, w: warnings.WarningMessage) -> None:
        category = w.category.__name__
        message = " ".join(str(w.message).split())[:MAX_MESSAGE]
        origin = f"{_module(w.filename)}:{w.lineno}"
        key = (category, message, origin)
        self.groups[key] = self.groups.get(key, 0) + 1

    def summary(self) -> dict:
        """JSON-ready: counts, the flagged categories, and every group (ignored ones by count)."""
        groups = [
            {"category": c, "message": m, "origin": o, "count": n}
            for (c, m, o), n in sorted(self.groups.items())
        ]
        kept = [g for g in groups if g["category"] not in IGNORED]
        flagged: dict[str, int] = {}
        for g in kept:
            if g["category"] in FLAGGED:
                flagged[g["category"]] = flagged.get(g["category"], 0) + g["count"]
        return {
            "n_warnings": sum(g["count"] for g in groups),
            "n_ignored": sum(g["count"] for g in groups if g["category"] in IGNORED),
            "flagged": flagged,
            "groups": kept,
        }

    def flag_text(self) -> str:
        """'' when nothing is flagged, else e.g. 'WARNINGS: ConvergenceWarning x5'."""
        flagged = self.summary()["flagged"]
        if not flagged:
            return ""
        return "WARNINGS: " + ", ".join(f"{c} x{n}" for c, n in sorted(flagged.items()))

    def write_log(self, path: Path) -> None:
        lines = [f"{n:>6} x {c} at {o}: {m}" for (c, m, o), n in sorted(self.groups.items())]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _module(filename: str) -> str:
    """A machine-independent origin: the path from site-packages or src/ on, else the file name."""
    parts = Path(filename).parts
    for anchor in ("site-packages", "src"):
        if anchor in parts:
            return "/".join(parts[parts.index(anchor) + 1 :])
    return Path(filename).name


@contextmanager
def capture_warnings(echo: bool = True) -> Iterator[Captured]:
    """Record every warning raised in the block (each occurrence, not just the first)."""
    captured = Captured()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield captured
        finally:
            for w in caught:
                captured.add(w)
    if echo:
        for (c, m, o), n in sorted(captured.groups.items()):
            if c not in IGNORED:
                print(f"[warning x{n}] {c} at {o}: {m}", file=sys.stderr)
