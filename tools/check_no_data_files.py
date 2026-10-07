"""Pre-commit hook: refuse to commit data, databases, model binaries or secrets."""

import re
import sys

FORBIDDEN = re.compile(
    # data/raw may only hold .gitkeep, DVC pointer files (*.dvc) and DVC's .gitignore
    r"(\.(csv|parquet|db|sqlite3?|joblib|pkl|pt|pth|ckpt)$)|(^|/)\.env$"
    r"|(^|/)data/raw/(?!(\.gitkeep|\.gitignore|[^/]+\.dvc)$)",
    re.IGNORECASE,
)


def main(paths: list[str]) -> int:
    bad = [p for p in paths if FORBIDDEN.search(p.replace("\\", "/"))]
    for p in bad:
        print(f"refusing to commit {p}: data, databases, models and .env stay out of git")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
