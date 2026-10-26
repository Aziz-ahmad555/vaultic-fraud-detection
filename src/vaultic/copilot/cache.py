"""Cached demo summaries (D224 (9)): a fixed list of demo cases, picked by rule from the demo bundle,
summarised once by the local model and stored with the model, the generation time, the evidence
hash and the checker result. The service serves an entry only when the request's evidence hash
matches, labelled "cached, generated at <time> by <model>", never as live.

Run (Ollama running, from the repository root; writes VAULTIC_CACHE_DIR/copilot/demo_summaries.json,
drive E, never committed):
    python -m vaultic.copilot.cache [--bundle PATH] [--L 30] [--out PATH]

The summaries are of the evidence at each case's own decision time (D224 (9)), with the Test Lab
off (view-level reasons); with the Test Lab on, the evidence differs and the summary is generated
live.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]


def pick_demo_cases(frame: pd.DataFrame, actions) -> list[tuple[str, int]]:
    """(rule, TransactionID) for the demo cases, in a fixed order, without repeats."""
    acts = np.asarray(actions, dtype=object)
    f = frame.assign(_action=acts)
    rules = [
        ("highest-risk hold", f[f["_action"] == "hold"], "p"),
        ("hold without a graph score", f[(f["_action"] == "hold") & f["p_graph"].isna()], "p"),
        ("cold-start step-up", f[(f["_action"] == "step_up") & (f["ctx_hist_n_past"] == 0)], "p"),
        ("step-up, uncertain set, highest disagreement",
         f[(f["_action"] == "step_up") & (f["conformal_set"] == "uncertain")], "disagreement"),
        ("highest-risk step-up", f[f["_action"] == "step_up"], "p"),
    ]  # fmt: skip
    out, seen = [], set()
    for name, sub, col in rules:
        sub = sub[~sub["TransactionID"].isin(seen)]
        if len(sub):
            tid = int(
                sub.sort_values([col, "TransactionDT"], ascending=[False, True])[
                    "TransactionID"
                ].iloc[0]
            )
            out.append((name, tid))
            seen.add(tid)
    return out


def _demo_app(bundle: Path | None):
    spec = importlib.util.spec_from_file_location(
        "demo_server", REPO / "services/api/demo_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.create_app(bundle)


NOTE = ("cached demo summaries (D224): local language model, evidence at the case's decision time, "
        "Test Lab off; not a research number")  # fmt: skip


def case_time(frame: pd.DataFrame, tid: int) -> float:
    return float(frame.loc[frame["TransactionID"] == tid, "TransactionDT"].iloc[0])


def summary_entry(cop, tid: int, t: float, L: int, rule: str) -> dict:
    """One summary generated now, as a cache entry (the rules hash of the copilot's variant)."""
    from vaultic.copilot.rules import rules_digest

    prep = cop.start("summary", tid, "", t, L)
    final = cop.answer(prep)
    return {"tid": tid, "rule": rule, "L": L, "as_of": t, "variant": cop.variant,
            "evidence_hash": prep.evidence.digest, "rules_hash": rules_digest(cop.variant),
            "evidence_tokens": prep.evidence.tokens, "model": cop.llm.model,
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "answer": final["answer"], "kept": final["kept"], "removed": final["removed"],
            "draft": prep.draft, "seconds": final["seconds"]}  # fmt: skip


def build(app, llm, L: int = 30, log=print, variant: str | None = None) -> dict:
    from vaultic.copilot.service import Copilot

    # no audit: an offline generation step, not an analyst
    cop = Copilot(app.state.read, llm, variant=variant)
    frame = app.state.read.frame
    entries = []
    for rule, tid in pick_demo_cases(frame, app.state.read.actions):
        e = summary_entry(cop, tid, case_time(frame, tid), L, rule)
        entries.append(e)
        log(f"{rule}: transaction {tid}, {e['seconds']} s, kept {len(e['kept'])}, "
            f"removed {len(e['removed'])}")  # fmt: skip
    return {"note": NOTE, "entries": entries}


def main(argv=None) -> None:
    from vaultic.copilot.llm import OllamaLLM
    from vaultic.copilot.service import cache_file

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bundle", type=Path, default=None)
    p.add_argument("--L", type=int, default=30)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    llm = OllamaLLM()
    llm.check()
    out = a.out or cache_file()
    data = build(_demo_app(a.bundle), llm, a.L)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1), "utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
