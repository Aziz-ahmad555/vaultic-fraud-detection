"""E20 addendum (D225p): descriptive tables added AFTER the final analysis and its independent review.
Nothing here is pre-registered, tested or used for any decision; the frozen analysis module
(vaultic.eval.e20_analysis, a9d2f34) is used unchanged, only its loader and helpers are called.

- per-system confusion tables, judge (rows) vs Aziz (columns): pass 1 by A shown / A removed / B / C,
  pass 2 by A / B;
- A's empty normal items per question: emptied by the checker (facts drafted, all removed) vs nothing
  drafted;
- a check that Aziz's pass-2 units are exactly the drawn pass-2 sample (the frozen module checks pass 1).

    python -m vaultic.eval.e20_addendum
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from vaultic.eval import e20_analysis as an
from vaultic.eval.e20 import lf_sha256

OUT = an.TABLES / "addendum"


def per_system_confusion(d: dict) -> dict:
    p1: dict[str, tuple[list, list]] = {}
    for r in d["human1"]:
        uid, n = r["unit id"], int(r["claim no"])
        k = d["key"][uid]
        group = f"A {k['status']}" if k["system"] == "A" else k["system"]
        a, b = p1.setdefault(group, ([], []))
        a.append(d["judge1"][uid]["labels"][n - 1])
        b.append(r["label"])
    p2: dict[str, tuple[list, list]] = {}
    for r in d["human2"]:
        a, b = p2.setdefault(r["system"], ([], []))
        a.append(d["judge2"][r["unit id"]]["accurate"])
        b.append(r["accurate"])
    return {
        "pass1": {g: {"n": len(a), "confusion": an.confusion(a, b, an.P1_LABELS)}
                  for g, (a, b) in sorted(p1.items())},
        "pass2": {g: {"n": len(a), "confusion": an.confusion(a, b, an.P2_LABELS)}
                  for g, (a, b) in sorted(p2.items())},
    }  # fmt: skip


def empty_a_items(d: dict) -> dict:
    """A's normal items with no shown fact, per question: emptied by the checker vs nothing drafted."""
    out = {}
    for r in d["items"]:
        if r["system"] != "A" or r["slot"] == "ADV":
            continue
        e = out.setdefault(r["slot"], {"items": 0, "empty": 0, "emptied_by_checker": 0,
                                       "nothing_drafted": 0, "facts_removed_in_emptied_items": 0})  # fmt: skip
        e["items"] += 1
        if not r.get("shown"):
            e["empty"] += 1
            if r.get("removed"):
                e["emptied_by_checker"] += 1
                e["facts_removed_in_emptied_items"] += len(r["removed"])
            else:
                e["nothing_drafted"] += 1
    total = {k: sum(v[k] for v in out.values()) for k in next(iter(out.values()))}
    return {"per_question": dict(sorted(out.items())), "total": total}


def pass2_sample_check(d: dict) -> dict:
    drawn = {u for v in d["sample"]["pass2"]["units"].values() for u in v}
    labelled = {r["unit id"] for r in d["human2"]}
    return {"drawn": len(drawn), "labelled": len(labelled), "equal": drawn == labelled,
            "missing": sorted(drawn - labelled), "extra": sorted(labelled - drawn)}  # fmt: skip


def addendum(d: dict) -> dict:
    return {
        "decision": "D225p", "status": "descriptive only; added after the final analysis and its review; "
        "not pre-registered, not tested, not used for any decision",
        "per_system_confusion": per_system_confusion(d), "a_empty_items": empty_a_items(d),
        "pass2_sample_check": pass2_sample_check(d),
    }  # fmt: skip


def markdown(r: dict) -> str:
    L = ["## Per-system confusion (rows judge, columns Aziz)", ""]
    for p, cats in (("pass1", an.P1_LABELS), ("pass2", an.P2_LABELS)):
        for g, v in r["per_system_confusion"][p].items():
            L += [f"{p}, {g} (n = {v['n']}):", "", "| | " + " | ".join(f"human {c}" for c in cats) + " |",
                  "|---" * (len(cats) + 1) + "|"]  # fmt: skip
            L += [f"| {row} | " + " | ".join(str(x) for x in v["confusion"][row].values()) + " |"
                  for row in v["confusion"]]  # fmt: skip
            L.append("")
    e = r["a_empty_items"]
    L += ["## A's empty normal items, per question", "",
          "| question | items | empty | emptied by the checker | nothing drafted | facts removed in emptied items |",
          "|---|---|---|---|---|---|"]  # fmt: skip
    for q, v in [*e["per_question"].items(), ("total", e["total"])]:
        L.append(f"| {q} | {v['items']} | {v['empty']} | {v['emptied_by_checker']} | {v['nothing_drafted']} | "
                 f"{v['facts_removed_in_emptied_items']} |")  # fmt: skip
    c = r["pass2_sample_check"]
    L += [
        "",
        f"Pass-2 sample check: {c['labelled']} labelled units, {c['drawn']} drawn; equal: {c['equal']}.",
    ]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args(argv)
    run_id = json.loads((an.TABLES / "run_manifest.json").read_text("utf-8"))["run_id"]
    try:
        d = an.load_inputs(a.run_dir or an.E20_ROOT / run_id, an.TABLES)
    except an.AnalysisRefused as e:
        raise SystemExit(f"E20 addendum refused: {e}") from None
    r = addendum(d)
    r["provenance"] = {"code_sha256": lf_sha256(Path(__file__)),
                       "frozen_analysis_sha256": lf_sha256(Path(an.__file__)),
                       "inputs_sha256": d["hashes"], "finished": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "e20_addendum.json").write_text(json.dumps(r, indent=1) + "\n", "utf-8", newline="\n")
    (a.out / "e20_addendum.md").write_text(markdown(r), "utf-8", newline="\n")
    print(f"wrote {a.out / 'e20_addendum.json'} and {a.out / 'e20_addendum.md'}")


if __name__ == "__main__":
    main()
