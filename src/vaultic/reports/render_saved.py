"""Re-render reports from their saved JSON, without refitting or rescoring anything (used by
`make paper-results`, reports/paper_results.py).

Run:  python -m vaultic.reports.render_saved final           E10-E12 (research/tables/final_*)
      python -m vaultic.reports.render_saved d95             research/tables/fusion_d95.md
      python -m vaultic.reports.render_saved f3-diagnosis    research/f3_collapse_diagnosis.md
      python -m vaultic.reports.render_saved f3-sensitivity  research/f3_log_sensitivity.md
      python -m vaultic.reports.render_saved e5-drift        research/e5_graph_drift.md
      python -m vaultic.reports.render_saved hold-sensitivity research/hold_sensitivity.md

Each report is rebuilt by the same writer that made it, from the JSON that run saved next to it.
A section whose heading ends "(added by hand, not generated)" is kept as it is in the current
file. E10-E12 keep the final run's generation date (the content is that run's), so a re-render
reproduces the committed files.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from vaultic.paths import RESEARCH_DIR

TABLES = RESEARCH_DIR / "tables"
HAND_MARK = "(added by hand, not generated)"


def keep_hand_section(path: Path, write) -> None:
    """Run `write()` (which rewrites `path`), then put back the file's hand-written section."""
    old = path.read_text("utf-8") if path.exists() else ""
    write()
    if HAND_MARK not in old:
        return
    new = path.read_text("utf-8")
    head = old[old.rindex("\n## ", 0, old.index(HAND_MARK)) + 1 :]
    cut = new.rfind("\n## ", 0, new.index(HAND_MARK)) + 1 if HAND_MARK in new else len(new)
    path.write_text(new[:cut].rstrip("\n") + "\n\n" + head, encoding="utf-8")


def _tuples(records: list[dict], prefix: str) -> list[dict]:
    """JSON stores the comparison cells (diff, low, high, p, ...) as lists; the writers expect
    tuples."""
    return [{k: tuple(v) if k.startswith(prefix) and isinstance(v, list) else v
             for k, v in r.items()} for r in records]  # fmt: skip


def render_final() -> None:
    from vaultic.fusion import final_run
    from vaultic.paths import REPO_ROOT

    r = json.loads((TABLES / "final_results.json").read_text("utf-8"))
    meta = r["meta"]
    run_dir = REPO_ROOT / "experiments" / "runs" / "EXP-200-final"
    completed = [json.loads((d / "metrics.json").read_text("utf-8")).get("completed")
                 for d in sorted(run_dir.iterdir()) if (d / "metrics.json").exists()]  # fmt: skip
    completed = [c for c in completed if c]
    day = date.fromisoformat(completed[-1][:10]) if completed else date.today()

    class _RunDate(date):
        @classmethod
        def today(cls):
            return day

    final_run.date = _RunDate  # the reports carry the final run's date
    final_run.write_reports(TABLES, "final_", meta, pd.DataFrame(r["e10_methods"]),
                            pd.DataFrame(r["e10_comparisons"]),
                            pd.DataFrame(_tuples(r["e10_subgroups"], "MVAF - ")), r["h2"],
                            pd.DataFrame(r["e11_drops"]), pd.DataFrame(r["e11_comparisons"]),
                            r["e12"], r["phase8_gate"])  # fmt: skip


def render_d95() -> None:
    from vaultic.fusion.d95_compare import report_lines

    out = TABLES / "fusion_d95.md"
    d = json.loads(out.with_suffix(".json").read_text("utf-8"))
    old = out.read_text("utf-8") if out.exists() else ""
    generated = old.split("Generated ", 1)[1][:10] if "Generated " in old else None
    runtime = next((x for x in old.splitlines() if x.startswith("Runtime ")),
                   "Runtime: not stored in the JSON.")  # fmt: skip
    subs = _tuples(d["subgroups"], "MVAF - ")
    # evaluation rows = cold start + with history (the two groups partition the rows)
    parts = [s for s in subs if s["subgroup"] in ("cold start", "with history")]
    n_rows, n_frauds = sum(s["rows"] for s in parts), sum(s["frauds"] for s in parts)
    lines = report_lines(d["gate_counts"], d["views"], d["fusion"], d["per_seed"], d["tuned"],
                         subs, n_rows, n_frauds, runtime, generated=generated)  # fmt: skip
    out.write_text("\n".join(lines), encoding="utf-8")


def _render_with(module, out_md: Path, out_json: Path) -> None:
    r = json.loads(out_json.read_text("utf-8"))
    keep_hand_section(out_md, lambda: module.write_md(r))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", choices=["final", "d95", "f3-diagnosis", "f3-sensitivity",
                                           "e5-drift", "hold-sensitivity", "gate-maturity"])  # fmt: skip
    args = parser.parse_args()
    if args.report == "final":
        render_final()
    elif args.report == "d95":
        render_d95()
    elif args.report == "f3-diagnosis":
        from vaultic.fusion import f3_diagnosis as m

        _render_with(m, m.OUT_MD, m.OUT_JSON)
    elif args.report == "f3-sensitivity":
        from vaultic.fusion import f3_sensitivity as m

        _render_with(m, m.OUT_MD, m.OUT_JSON)
    elif args.report == "gate-maturity":
        from vaultic.fusion import gate_maturity_sensitivity as m

        _render_with(m, m.OUT_MD, m.OUT_JSON)
    elif args.report == "hold-sensitivity":
        from vaultic.trust import hold_sensitivity as m

        _render_with(m, m.OUT_MD, m.OUT_JSON)
    else:
        from vaultic.reports import e5_graph_drift as m

        _render_with(m, m.OUT_MD, m.OUT_JSON)
    print(f"rendered {args.report}")


if __name__ == "__main__":
    main()
