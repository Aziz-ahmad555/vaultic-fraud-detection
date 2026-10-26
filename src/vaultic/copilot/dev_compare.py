"""Prompt-variant comparison on the copilot's DEV set (D224c): design work, never on E20 cases.

Dev set: the 5 demo cases (cache.pick_demo_cases) and 5 cases fixed in D224c (default_rng(224):
2 holds, 3 step-ups among the frozen alerts, demo cases excluded). Per case: one summary (as of the
decision time) and the question "Why was this transaction flagged?" at the case's own time, L = 30,
Test Lab off: 20 generations per variant, one generation each, no re-runs.

Selection rule (fixed in D224c before any generation): the highest kept-fact rate (kept facts /
facts written, pooled; an answer with no readable item counts as one written and removed fact);
ties: fewer removed facts, then lower total seconds. All variants are reported as they came out.
The chosen variant's 5 demo summaries become the cached demo summaries (the old file is kept).

Run (Ollama running, from the repository root):
    python -m vaultic.copilot.dev_compare [--bundle PATH]
Writes VAULTIC_CACHE_DIR/copilot/dev_compare_<stamp>.json (drive E, never committed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from vaultic.copilot.cache import NOTE, _demo_app, case_time, pick_demo_cases, summary_entry

EXTRA_DEV = (3566631, 3555114, 3522679, 3542087, 3510484)  # D224c: 2 holds, 3 step-ups
WHY = "Why was this transaction flagged?"
L = 30


def dev_cases(frame, actions) -> list[tuple[str, int]]:
    return pick_demo_cases(frame, actions) + [("dev (D224c)", t) for t in EXTRA_DEV]


def tally(rows: list[dict]) -> dict:
    kept = sum(len(r["kept"]) for r in rows)
    removed = sum(len(r["removed"]) for r in rows)
    written = kept + removed
    return {"generations": len(rows), "written": written, "kept": kept, "removed": removed,
            "kept_rate": kept / written if written else 0.0,
            "not_available": sum(1 for r in rows if not r["kept"]),
            "invalid_json": sum(1 for r in rows if any(x["why"] == "answer not valid JSON"
                                                        for x in r["removed"])),
            "seconds": round(sum(r["seconds"] for r in rows), 1)}  # fmt: skip


def choose(summary: dict[str, dict]) -> str:
    """D224c: highest kept-fact rate; ties: fewer removed, then fewer seconds."""
    return min(summary, key=lambda v: (-summary[v]["kept_rate"], summary[v]["removed"],
                                       summary[v]["seconds"]))  # fmt: skip


def run_variant(app, llm, variant: str, log=print) -> list[dict]:
    from vaultic.copilot.service import Copilot

    cop = Copilot(app.state.read, llm, variant=variant)
    frame = app.state.read.frame
    rows = []
    for rule, tid in dev_cases(frame, app.state.read.actions):
        t = case_time(frame, tid)
        s = summary_entry(cop, tid, t, L, rule)
        rows.append({**s, "task": "summary"})
        prep = cop.start("question", tid, WHY, t, L)
        f = cop.answer(prep)
        rows.append({"tid": tid, "rule": rule, "task": "why flagged", "variant": variant,
                     "answer": f["answer"], "kept": f["kept"], "removed": f["removed"],
                     "draft": prep.draft, "seconds": f["seconds"]})  # fmt: skip
        for r in rows[-2:]:
            log(f"{variant} {r['task']:<11} {tid}: {r['seconds']} s, kept {len(r['kept'])}, "
                f"removed {len(r['removed'])}")  # fmt: skip
    return rows


def recheck_row(cop, row: dict) -> dict:
    """The stored draft of one generation checked again with the current checker: the evidence is
    rebuilt exactly as when it was generated (a summary's must match its stored evidence hash), and
    no model is called (D224d)."""
    from vaultic.copilot.check import check_facts
    from vaultic.copilot.rules import max_facts

    t = case_time(cop.read.frame, row["tid"])
    kind, q = ("summary", "") if row["task"] == "summary" else ("question", WHY)
    prep = cop.start(kind, row["tid"], q, t, L)
    if row.get("evidence_hash") and row["evidence_hash"] != prep.evidence.digest:
        raise RuntimeError(f"evidence of {row['tid']} ({row['task']}) differs from the stored hash")
    res = check_facts(row.get("draft") or "", prep.evidence.by_handle, max_facts(row["variant"]))
    return {**row, "answer": res.answer, "kept": res.kept, "removed": res.removed}


def recheck(app, path: Path) -> dict:
    """Re-check every stored dev-set draft (all variants) with the current checker."""
    from vaultic.copilot.check import CHECKER_VERSION
    from vaultic.copilot.llm import ScriptedLLM
    from vaultic.copilot.service import Copilot

    stored = json.loads(path.read_text("utf-8"))
    rows = {v: [recheck_row(Copilot(app.state.read, ScriptedLLM(""), variant=v), r) for r in rs]
            for v, rs in stored["rows"].items()}  # fmt: skip
    summary = {v: tally(rs) for v, rs in rows.items()}
    return {"rule": "D224c", "checker": CHECKER_VERSION, "source": path.name,
            "model": stored["model"], "summary_before": stored["summary"], "summary": summary,
            "chosen_before": stored["chosen"], "chosen": choose(summary), "rows": rows}  # fmt: skip


def recheck_cache(app, path: Path) -> list[dict]:
    """The cached demo summaries re-checked from their stored drafts and stamped with the checker
    version (D224d (5)); the drafts, model and generation times are unchanged."""
    from vaultic.copilot.check import CHECKER_VERSION
    from vaultic.copilot.llm import ScriptedLLM
    from vaultic.copilot.service import Copilot

    out = []
    for e in json.loads(path.read_text("utf-8"))["entries"]:
        cop = Copilot(app.state.read, ScriptedLLM(""), variant=e["variant"])
        r = recheck_row(cop, {**e, "task": "summary"})
        out.append({k: v for k, v in r.items() if k != "task"} | {"checker": CHECKER_VERSION})
    return out


def main(argv=None) -> None:
    from vaultic.copilot.llm import OllamaLLM
    from vaultic.copilot.rules import VARIANTS
    from vaultic.copilot.service import cache_file

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bundle", type=Path, default=None)
    p.add_argument("--recheck", type=Path, default=None,
                   help="re-check a stored dev_compare_<stamp>.json and the cached demo summaries "
                        "with the current checker (no model call)")  # fmt: skip
    a = p.parse_args(argv)
    if a.recheck is not None:
        app = _demo_app(a.bundle)
        data = recheck(app, a.recheck)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = a.recheck.with_name(f"dev_recheck_{data['checker']}_{stamp}.json")
        out.write_text(json.dumps(data, indent=1), "utf-8")
        cache = cache_file()
        entries = recheck_cache(app, cache)
        cache.rename(cache.with_name(f"demo_summaries_before_{data['checker']}_{stamp}.json"))
        cache.write_text(json.dumps({"note": NOTE, "entries": entries}, indent=1), "utf-8")
        for v in data["summary"]:
            print(v, "before", data["summary_before"][v], "\n   now", data["summary"][v])
        print(
            f"chosen before {data['chosen_before']}, now {data['chosen']}; wrote {out} and {cache}"
        )
        return
    llm = OllamaLLM()
    llm.check()
    app = _demo_app(a.bundle)
    results = {v: run_variant(app, llm, v) for v in VARIANTS}
    summary = {v: tally(rows) for v, rows in results.items()}
    best = choose(summary)
    out_dir = cache_file().parent
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"dev_compare_{stamp}.json"
    out.write_text(json.dumps({"rule": "D224c", "model": llm.model, "summary": summary,
                               "chosen": best, "rows": results}, indent=1), "utf-8")  # fmt: skip
    demo = {t for _, t in pick_demo_cases(app.state.read.frame, app.state.read.actions)}
    entries = [{k: v for k, v in r.items() if k != "task"} for r in results[best]
               if r["task"] == "summary" and r["tid"] in demo]  # fmt: skip
    cache = cache_file()
    if cache.exists():
        cache.rename(cache.with_name(f"demo_summaries_before_d224c_{stamp}.json"))
    cache.write_text(json.dumps({"note": NOTE, "entries": entries}, indent=1), "utf-8")
    for v, s in summary.items():
        print(v, s)
    print(f"chosen: {best}; wrote {out} and {cache}")


if __name__ == "__main__":
    main(sys.argv[1:])
