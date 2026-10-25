"""The copilot's read-only tools (D224 (1)) and the trimmed evidence they render (D224 (3)).

Every tool reads through the demo backend's read-only accessor (app.state.read: the endpoints' own
functions, so the same as-of cut and label-delay curtain, and the frozen E12 decisions, never
what-if) or, for live reasons, the Test Lab engine. None of them writes anything. A tool returns
evidence items: a stable evidence id (for the UI), the tool name and one short line of text. The
renderer gives the lines per-turn handles [E1]..[En] within a token budget.

Strings that come from the data go through rules.clean (no brackets or newlines, at most 60
characters, instruction-like text withheld): tool results are data, never commands (D224 (6)).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vaultic.copilot.rules import clean
from vaultic.copilot.similar import similar_cases

DAY = 86400
VIEWS = ("tabular", "behavioral", "temporal", "graph", "anomaly")
CHARS_PER_TOKEN = 3  # conservative: digits are separate tokens for Qwen2.5
BUDGET = {"question": 700, "summary": 1000}
VIEW_LEVEL = ("view-level reasons (gate weight x view logit); feature reasons not available: the "
              "Test Lab is off")  # fmt: skip
LIVE = "live frozen-system reasons (Test Lab, parity-gated)"


@dataclass
class Item:
    evidence_id: str
    tool: str
    text: str


@dataclass
class Ctx:
    read: Any  # the demo backend's read-only accessor (app.state.read)
    tid: int
    time: float  # as-of
    L: int
    case: dict  # the transaction as the case page receives it
    testlab: Any = None  # the Test Lab service, or None when it is not mounted
    flags: list = field(default_factory=list)  # instruction-like strings withheld from the data
    meta: dict = field(default_factory=dict)  # e.g. the reasons source

    @property
    def uid(self) -> str:
        return str(self.case["entities"]["uid"])

    @property
    def case_time(self) -> float:
        return float(self.case["row"]["TransactionDT"])


def tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def clock(t: float) -> str:
    s = int(t) % DAY
    return f"day {int(t // DAY)} at {s // 3600:02d}:{s % 3600 // 60:02d}"


def label_text(tx: dict, L: int) -> str:
    lab = tx["row"].get("label")
    if lab is None:
        return f"outcome label not known yet (label delay {L} days)"
    return (
        f"outcome label: {'fraud' if lab == 1 else 'legit'} (known after the {L}-day label delay)"
    )


def yes_no(x) -> str:
    return "not available" if x is None else ("yes" if x else "no")


def num(x, fmt: str = "{}") -> str:
    return "not available" if x is None else fmt.format(x)


# ---- the tools ---------------------------------------------------------------------------------
def get_case(ctx: Ctx) -> list[Item]:
    c, d, x = ctx.case, ctx.case["decision"], ctx.case["inputs"]
    tid = ctx.tid
    head = (f"transaction {tid}: amount {c['amount']:.2f}, product {clean(c['product'], ctx.flags)}, "
            f"{clock(ctx.case_time)}; decision {d['action']} (frozen cost rule); calibrated fraud "
            f"probability {d['p']:.3f}; expected loss {d['expected_loss']:.2f}; conformal set "
            f"{d['conformal_set']}; view disagreement {d['disagreement']:.3f}.")  # fmt: skip
    cust = (f"customer {clean(ctx.uid, ctx.flags)}: {x['historyCount']} earlier transactions in "
            f"total; usual amount {num(x['usualAmount'], '{:.2f}')}; device seen before "
            f"{yes_no(x['deviceKnown'])}; transactions in the last 24 hours "
            f"{num(x['velocity24h'])}; other customers on the same card in the last 7 days "
            f"{num(x['sharedEntities'])}.")  # fmt: skip
    return [Item(f"case:{tid}/decision", "get_case", head),
            Item(f"case:{tid}/customer", "get_case", cust),
            Item(f"case:{tid}/label", "get_case", label_text(c, ctx.L) + ".")]  # fmt: skip


def _live_reasons(ctx: Ctx) -> dict | None:
    """Live reasons from the Test Lab engine, or None when it is not ready. The engine is called
    directly with record_latency=False so the Test Lab's own latency figures stay untouched."""
    svc = ctx.testlab
    if svc is None or getattr(svc, "state", None) != "ready":
        return None
    e = svc._ready()
    p = svc._position(ctx.tid)
    with e.lock:
        return e.score_stored(p, record_latency=False)


def get_reasons(ctx: Ctx) -> list[Item]:
    tid = ctx.tid
    try:
        live = _live_reasons(ctx)
        why_not = None
    except Exception as e:  # reported, never hidden: the view-level reasons are used instead
        live, why_not = None, f"{type(e).__name__}"
    if live is not None:
        ctx.meta["reasons_source"] = LIVE
        match = live.get("matches_frozen_run")
        out = [Item(f"case:{tid}/reasons", "get_reasons",
                    f"reasons source: {LIVE}; live risk {live['risk']:.3f}; matches the frozen run "
                    f"{yes_no(match)}.")]  # fmt: skip
        for i, r in enumerate(live.get("reasons", [])[:5], 1):
            out.append(Item(f"case:{tid}/feature:{i}", "get_reasons",
                            f"feature reason {i}: {clean(r['text'], ctx.flags, 110)} "
                            f"({r['points']:+.1f} points)."))  # fmt: skip
        views = ", ".join(f"{r['view']} {r['points']:+.1f}" for r in live.get("view_reasons", []))
        if views:
            out.append(Item(f"case:{tid}/view-contributions", "get_reasons",
                            f"view contributions in points: {views}."))  # fmt: skip
        return out
    ctx.meta["reasons_source"] = "view-level"
    note = view_level_note(ctx, why_not)
    out = [Item(f"case:{tid}/reasons", "get_reasons", f"reasons source: {note}.")]
    w, row = ctx.case["weights"], ctx.case["row"]
    for r in ctx.case.get("reasons", []):
        v = r.get("view")
        if v not in VIEWS:
            continue
        pts = float(r["points"])
        out.append(Item(f"case:{tid}/view-reason:{v}", "get_reasons",
                        f"{v} view {'raises' if pts > 0 else 'lowers'} the risk by {abs(pts):.1f} "
                        f"points (gate weight {w[v]:.2f}, view probability "
                        f"{num(row.get(f'p_{v}'), '{:.3f}')})."))  # fmt: skip
    return out


def view_level_note(ctx: Ctx, why_not: str | None) -> str:
    state = getattr(ctx.testlab, "state", None)  # D224d: "loading" / "refused" / "error", not "off"
    return (VIEW_LEVEL.replace("is off", f"failed ({why_not})") if why_not is not None else
            VIEW_LEVEL.replace("is off", f"is {state}") if ctx.testlab is not None else VIEW_LEVEL)  # fmt: skip


def reason_facts(ctx: Ctx, k: int = 5) -> tuple[str, list[dict]]:
    """(source label, top reasons as {"text", "points"}) for the page's key facts (D226): the
    same live / view-level choice as get_reasons, as data instead of evidence lines."""
    try:
        live, why_not = _live_reasons(ctx), None
    except Exception as e:
        live, why_not = None, f"{type(e).__name__}"
    if live is not None:
        rs = [{"text": clean(r["text"], None, 110), "points": float(r["points"])}
              for r in live.get("reasons", [])[:k]]  # fmt: skip
        return LIVE, rs
    rs = [{"text": clean(r.get("text", ""), None, 110), "points": float(r["points"])}
          for r in ctx.case.get("reasons", [])[:k]]  # fmt: skip
    return view_level_note(ctx, why_not), rs


def get_view_scores(ctx: Ctx) -> list[Item]:
    row, w = ctx.case["row"], ctx.case["weights"]
    out = []
    for v in VIEWS:
        p = row.get(f"p_{v}")
        text = (f"view {v}: missing (no score; masked, never filled)" if p is None else
                f"view {v}: calibrated score {p:.3f}, gate weight {w[v]:.2f}")  # fmt: skip
        out.append(Item(f"case:{ctx.tid}/view:{v}", "get_view_scores", text + "."))
    return out


def network_ids(ctx: Ctx) -> set[str]:
    """Every entity value within two links of the case's customer, plus the case's own ids."""
    net = ctx.read.network(f"uid:{ctx.uid}", ctx.time, ctx.L)
    ids = {str(ctx.tid), ctx.uid} | {str(v) for v in ctx.case["entities"].values() if v is not None}
    return {i.lower() for i in ids | {n["label"] for n in net["nodes"]}}


def get_graph(ctx: Ctx) -> list[Item]:
    net = ctx.read.network(f"uid:{ctx.uid}", ctx.time, ctx.L)
    nodes = {n["id"]: n for n in net["nodes"]}
    degree = Counter(e["target"] for e in net["edges"])
    kinds = Counter(n["type"] for n in net["nodes"])
    fraud_nodes = sum(1 for n in net["nodes"] if n["knownFraud"] > 0)
    parts = ", ".join(
        f"{k} {kinds[k]}" for k in ("uid", "card", "email", "address", "device") if kinds[k]
    )
    out = [Item(f"graph:uid:{ctx.uid}/summary", "get_graph",
                f"network within two links of customer {clean(ctx.uid, ctx.flags)}: "
                f"{len(nodes)} entities ({parts.replace('uid', 'customers')}); entities with known "
                f"fraud (labels known at this time): {fraud_nodes}; ring detection not yet built; "
                f"e-mail domains and billing regions are not expanded.")]  # fmt: skip
    for kind in ("card", "email", "address", "device"):
        val = ctx.case["entities"].get(kind)
        n = nodes.get(f"{kind}:{val}") if val is not None else None
        if n is None:
            continue
        out.append(Item(f"graph:{kind}:{val}", "get_graph",
                        f"this case's {kind} {clean(val, ctx.flags)}: {n['transactions']} "
                        f"transactions, {n['knownFraud']} known fraud, linked to "
                        f"{degree[n['id']]} customers here."))  # fmt: skip
    others = [n for n in net["nodes"] if n["type"] == "uid" and n["label"] != ctx.uid]
    others.sort(key=lambda n: (-n["knownFraud"], -n["transactions"], n["label"]))
    for n in others[:3]:
        out.append(Item(f"graph:uid:{n['label']}", "get_graph",
                        f"linked customer {clean(n['label'], ctx.flags)}: {n['transactions']} "
                        f"transactions, {n['knownFraud']} known fraud."))  # fmt: skip
    return out


def get_customer_history(ctx: Ctx, n: int = 5) -> list[Item]:
    rows = [r for r in ctx.read.history(ctx.uid, ctx.time, ctx.L)
            if float(r["row"]["TransactionDT"]) < ctx.case_time]  # fmt: skip
    total = ctx.case["inputs"]["historyCount"]
    head = (
        f"customer {clean(ctx.uid, ctx.flags)}: {total} earlier transactions in total (all periods)"
    )
    if not rows:
        return [Item(f"customer:{ctx.uid}/history", "get_customer_history",
                     head + "; none of them in the test-period data shown here.")]  # fmt: skip
    acts = Counter(r["decision"]["action"] for r in rows)
    known = sum(1 for r in rows if r["row"].get("label") == 1)
    amounts = [r["amount"] for r in rows]
    out = [Item(f"customer:{ctx.uid}/history", "get_customer_history",
                f"{head}; {len(rows)} of them in the test-period data shown here; their decisions: "
                f"{', '.join(f'{a} {k}' for a, k in sorted(acts.items()))}; known fraud among them: "
                f"{known}; median amount {float(np.median(amounts)):.2f}.")]  # fmt: skip
    for r in rows[::-1][:n]:  # newest first
        t = int(r["row"]["TransactionID"])
        out.append(Item(f"tx:{t}", "get_customer_history",
                        f"earlier transaction {t}: {clock(r['row']['TransactionDT'])}, amount "
                        f"{r['amount']:.2f}, decision {r['decision']['action']}, "
                        f"{label_text(r, ctx.L)}."))  # fmt: skip
    return out


def get_similar_cases(ctx: Ctx, k: int = 5) -> list[Item]:
    """Aggregate counts only (D224d (4)): no other customer's transaction id or per-case label."""
    frame, actions = ctx.read.frame, ctx.read.actions
    ids = frame["TransactionID"].to_numpy()
    pos = int(np.flatnonzero(ids == ctx.tid)[0])
    found = similar_cases(frame, actions, pos, min(ctx.case_time, ctx.time), k)
    act = ctx.case["decision"]["action"]
    labels = Counter(ctx.read.transaction(int(ids[p]), ctx.time, ctx.L)["row"].get("label")
                     for p, _ in found)  # fmt: skip
    return [Item(f"case:{ctx.tid}/similar", "get_similar_cases",
                 f"similar earlier cases (by calibrated view scores, same frozen decision {act}; "
                 f"not by investigation outcome): {len(found)} found; matured labels only: "
                 f"{labels[1]} fraud, {labels[0]} legit, {labels[None]} not known yet.")]  # fmt: skip


def get_counterfactual(ctx: Ctx) -> list[Item]:
    return [Item(f"case:{ctx.tid}/counterfactual", "get_counterfactual",
                 "counterfactual for this case: not available (the stored counterfactuals are kept "
                 "per alert index without a transaction id; the illustrative what-if model is not "
                 "the evaluated system and is not used).")]  # fmt: skip


TOOLS = {"case": get_case, "reasons": get_reasons, "views": get_view_scores, "graph": get_graph,
         "history": get_customer_history, "similar": get_similar_cases,
         "counterfactual": get_counterfactual}  # fmt: skip


# ---- trimmed evidence ----------------------------------------------------------------------------
@dataclass
class Evidence:
    lines: list[dict]  # {"handle", "evidence_id", "tool", "text"}
    dropped: int  # items left out to stay within the budget

    @property
    def by_handle(self) -> dict[str, str]:
        return {x["handle"]: x["text"] for x in self.lines}

    @property
    def block(self) -> str:
        return "\n".join(f"[{x['handle']}] {x['text']}" for x in self.lines)

    @property
    def tokens(self) -> int:
        return tokens(self.block)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.block.encode("utf-8")).hexdigest()


def render(groups: list[list[Item]], budget: int) -> Evidence:
    """The first item of every tool always; then the rest in order while the budget allows."""
    keep = [[g[0]] if g else [] for g in groups]
    used = sum(tokens(i.text) + 2 for g in keep for i in g)
    dropped = 0
    for gi, g in enumerate(groups):
        for item in g[1:]:
            cost = tokens(item.text) + 2
            if used + cost <= budget:
                keep[gi].append(item)
                used += cost
            else:
                dropped += 1
    lines = [{"handle": f"E{n}", "evidence_id": i.evidence_id, "tool": i.tool, "text": i.text}
             for n, i in enumerate((i for g in keep for i in g), 1)]  # fmt: skip
    return Evidence(lines, dropped)
