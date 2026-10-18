"""Markdown files contain no control characters (a backslash escape such as \\t or \\f in a Windows
path, interpreted by a script, once turned 'E:\\dev-cache\\testlab' into a tab)."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _markdown_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.split()  # fmt: skip
        return [ROOT / f for f in out]
    except (OSError, subprocess.CalledProcessError):  # no git: every markdown file outside deps
        return [p for p in ROOT.rglob("*.md") if "node_modules" not in p.parts]


def test_markdown_files_have_no_control_characters():
    bad = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.split("\n"), 1):
            hits = sorted({hex(ord(c)) for c in line if ord(c) < 32 and c != "\r"})
            if hits:
                bad.append(f"{path.relative_to(ROOT)}:{n} {hits}")
    assert not bad, "control characters in markdown:\n" + "\n".join(bad)
