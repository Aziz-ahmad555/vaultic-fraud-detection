"""Deterministic citation checker (D224 (4)): runs on the finished answer, never on a draft.

A sentence is kept only if
  - it cites at least one handle ([E1], [E2, E3] ...) and every handle it cites was given this turn;
  - every number in it appears in the cited lines: exactly, or rounded to the sentence's decimals
    (integers below 10 must match exactly); "x%" also matches a probability x / 100;
  - every id-like token in it (letters with digits, kind:value, e-mail domains) appears in the
    cited lines;
  - every decision word in it (allow, monitor, step-up, hold, block) appears in the cited lines;
  - it does not claim that something was changed (a decision, label, threshold or setting);
  - a claim that the case is fraud or legitimate is backed by a cited line with a known outcome label.
Everything else is removed and counted with its reason; with nothing left the answer is NOT_AVAILABLE.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

NOT_AVAILABLE = "Not available from the case evidence."
KNOWN_LABEL = "outcome label: "  # get_case writes "outcome label: fraud (known ...)" when matured

_CITE = re.compile(r"\[([^\[\]]*)\]")
_HANDLE = re.compile(r"^E\d+$")
_RANGE = re.compile(r"\[E(\d+)\]\s*[-–]\s*\[E(\d+)\]")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SENTENCE = re.compile(r".+?(?:[.!?](?:\s*\[[^\[\]]*\])*(?=\s|$)|$)", re.S)
_TOKEN = re.compile(r"[\w$%@:./,+-]+")
_NUMBER = re.compile(r"^\$?\d+(?:\.\d+)?%?$")
_GROUPED = re.compile(r"^\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?$")
_DOMAIN = re.compile(
    r"^[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|edu|gov|mx|fr|de|uk|es|jp|co|io)$", re.I
)
_KIND = re.compile(r"^(?:uid|card|email|address|device|customer|tx|case):\S+$", re.I)
_CHANGE = re.compile(
    r"\b(?:i|we|copilot|assistant)\s+(?:have\s+|has\s+|just\s+)?(?:changed|updated|set|marked|"
    r"overrode|overridden|approved|released|unblocked|blocked|relabell?ed|cleared|dismissed|lowered|"
    r"raised|reset|allowed)\b|"
    r"\b(?:has|have|had|was|were|is\s+now|been)\s+(?:been\s+)?(?:changed|updated|overridden|overruled|"
    r"relabell?ed|re-?labell?ed|marked\s+as|switched|reset|lowered|raised)\b|"
    r"\b(?:decision|label|threshold|setting)s?\s+(?:is|are)\s+now\b",
    re.I,
)
_ACTION = re.compile(r"\b(allow|monitor|step[ _-]?up|hold|block)\b", re.I)
_LABEL_CLAIM = re.compile(
    r"\b(?:is|was|are|were|confirmed|labell?ed|marked)\s+(?:as\s+|a\s+|an\s+)?(?:confirmed\s+)?"
    r"(?:fraud|fraudulent|legit|legitimate|genuine|not\s+fraud)\b",
    re.I,
)


@dataclass
class CheckResult:
    answer: str  # the kept sentences, or NOT_AVAILABLE
    kept: list[dict] = field(default_factory=list)  # {"text", "citations"}
    removed: list[dict] = field(default_factory=list)  # {"text", "why"}

    def summary(self) -> dict:
        return {"kept": len(self.kept), "removed": len(self.removed)}


def sentences(text: str) -> list[str]:
    """Split on newlines, then on . ! ? before a space (a citation right after the stop stays with
    its sentence; decimals such as 0.35 are not split). List bullets are dropped."""
    out = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip()
        out += [m.group(0).strip() for m in _SENTENCE.finditer(line) if m.group(0).strip()]
    return out


def _parts(token: str) -> list[str]:
    return [p for p in re.split(r"[:/-]", token) if p]


def _numbers(text: str) -> list[str]:
    """Numbers written as numbers: letter-free tokens, split at : / - (times, ranges)."""
    out = []
    for tok in _TOKEN.findall(_CITE.sub(" ", text)):
        tok = tok.strip(".,:;+")
        if not tok or re.search(r"[A-Za-z]", tok):
            continue
        for p in _parts(tok) if not _GROUPED.match(tok) else [tok]:
            p = p.strip(".,+")
            if _GROUPED.match(p) or _NUMBER.match(p):
                out.append(p)
    return out


def _value(num: str) -> tuple[float, int, bool]:
    pct = num.endswith("%")
    core = num.strip("$%").replace(",", "")
    decimals = len(core.split(".")[1]) if "." in core else 0
    return float(core), decimals, pct


def _ids(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(_CITE.sub(" ", text)):
        tok = tok.strip(".,:;+-")
        letters, digits = re.search(r"[A-Za-z]", tok), re.search(r"\d", tok)
        if (letters and digits) or _DOMAIN.match(tok) or _KIND.match(tok):
            out.append(tok)
    return out


def number_supported(num: str, evidence_numbers: list[float]) -> bool:
    v, d, pct = _value(num)
    for e in evidence_numbers:
        if abs(e - v) < 1e-9:
            return True
        if (d >= 1 or abs(e) >= 10) and round(e, d) == v:
            return True
        if pct and 0 <= e <= 1 and round(e * 100, d) == v:
            return True
    return False


def _actions(text: str) -> set[str]:
    """The decision words of a text, normalised ("step up", "step-up" -> "step_up")."""
    return {re.sub(r"[ _-]", "_", a.lower()).replace("stepup", "step_up") for a in
            _ACTION.findall(_CITE.sub(" ", text))}  # fmt: skip


def id_supported(token: str, cited_lower: str) -> bool:
    """The token (for kind:value, its value) as a whole word of the cited lines."""
    tok = token.split(":", 1)[1] if _KIND.match(token) else token
    return re.search(r"(?<![\w-])" + re.escape(tok.lower()) + r"(?![\w-])", cited_lower) is not None


def check(answer: str, evidence: dict[str, str]) -> CheckResult:
    """answer: the model's finished text; evidence: handle ("E1") -> the line given this turn."""
    res = CheckResult(answer=NOT_AVAILABLE)
    # "[E8]-[E10]" cites E8, E9 and E10
    answer = _RANGE.sub(lambda m: "[" + ", ".join(f"E{i}" for i in range(int(m.group(1)),
                        int(m.group(2)) + 1)) + "]" if int(m.group(2)) - int(m.group(1)) < 20
                        else m.group(0), answer)  # fmt: skip
    for s in sentences(answer):
        if s.strip(" .").lower() == NOT_AVAILABLE.strip(" .").lower():
            continue  # the instructed phrase claims nothing: neither kept nor counted as removed
        cites = [h for c in _CITE.findall(s) for h in re.split(r"[,;\s]+", c) if h]
        if cites and not re.sub(r"<\s*E\d+\s*>|[\s.,;:]", "", _CITE.sub("", s)):
            why = "handles listed apart from their sentences"  # D224b: nothing to tie them to
        else:
            why = verdict(s, cites, evidence)
        if why:
            res.removed.append({"text": s, "why": why})
        else:
            res.kept.append({"text": s, "citations": list(dict.fromkeys(cites))})
    if res.kept:
        res.answer = " ".join(k["text"] for k in res.kept)
    return res


def verdict(s: str, cites: list[str], evidence: dict[str, str]) -> str | None:
    """Why a sentence with these citations is removed, or None when it is kept."""
    if not cites:
        return "no citation"
    if any(not _HANDLE.match(h) or h not in evidence for h in cites):
        return "cites evidence that was not given"
    cited = " ".join(evidence[h] for h in dict.fromkeys(cites))
    ev_nums = [_value(n)[0] for n in _numbers(cited)]
    low = cited.lower()
    if _CHANGE.search(s):
        return "claims a change (the copilot cannot change anything)"
    if bad := [n for n in _numbers(s) if not number_supported(n, ev_nums)]:
        return f"number not in the cited evidence: {bad[0]}"
    if bad := [t for t in _ids(s) if not id_supported(t, low)]:
        return f"id not in the cited evidence: {bad[0]}"
    if bad := [a for a in _actions(s) if a not in _actions(cited)]:
        return f"decision not in the cited evidence: {bad[0]}"
    if _LABEL_CLAIM.search(s) and KNOWN_LABEL not in low:
        return "claims an outcome label that the cited evidence does not state"
    return None


# ---- structured output (D224c) -------------------------------------------------------------------
_ITEM = re.compile(
    r'\{\s*"fact"\s*:\s*("(?:[^"\\]|\\.)*")\s*,\s*"evidence"\s*:\s*("(?:[^"\\]|\\.)*")\s*\}'
)


def parse_facts(draft: str) -> tuple[list[dict], bool]:
    """The items of a structured answer and whether it was valid JSON. Invalid JSON (e.g. cut off
    at the token limit) keeps only the complete items that can be read."""
    try:
        obj = json.loads(draft)
        items = obj.get("facts") if isinstance(obj, dict) else obj
        if not isinstance(items, list):
            return [], False
        return [{"fact": str(i.get("fact", "")), "evidence": str(i.get("evidence", ""))}
                for i in items if isinstance(i, dict)], True  # fmt: skip
    except (ValueError, AttributeError):
        return [{"fact": json.loads(f), "evidence": json.loads(e)}
                for f, e in _ITEM.findall(draft)], False  # fmt: skip


def check_facts(draft: str, evidence: dict[str, str], max_facts: int = 5) -> CheckResult:
    """Each fact is checked against its one cited line (handles written inside the fact text are
    ignored); prose is rendered by code: '<fact> [Ek].' per kept fact. An empty list is
    NOT_AVAILABLE; output with no readable item counts as one removed fact."""
    res = CheckResult(answer=NOT_AVAILABLE)
    items, valid = parse_facts(draft)
    if not items and not valid:
        res.removed.append({"text": draft[:300], "why": "answer not valid JSON"})
        return res
    for n, item in enumerate(items):
        fact = " ".join(_CITE.sub(" ", item["fact"]).split()).rstrip(" .")
        if not fact or fact.lower() == NOT_AVAILABLE.strip(" .").lower():
            continue
        text = f"{fact} [{item['evidence']}]."
        why = (
            "more facts than allowed"
            if n >= max_facts
            else verdict(text, [item["evidence"]], evidence)
        )
        if why:
            res.removed.append({"text": text, "why": why})
        else:
            res.kept.append({"text": text, "citations": [item["evidence"]]})
    if res.kept:
        res.answer = " ".join(k["text"] for k in res.kept)
    return res
