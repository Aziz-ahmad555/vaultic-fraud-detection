"""Pre-commit hook: refuse to commit data, databases, model binaries or secrets."""

import re
import sys

FORBIDDEN = re.compile(
    r"(\.(csv|parquet|db|sqlite3?|joblib|pkl|pt|pth|ckpt)$)|(^|/)\.env$|(^|/)data/raw/(?!\.gitkeep$)",
    re.IGNORECASE,
)


def main(paths: list[str]) -> int:
    bad = [p for p in paths if FORBIDDEN.search(p.replace("\\", "/"))]
    for p in bad:
        print(f"refusing to commit {p}: data, databases, models and .env stay out of git")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
