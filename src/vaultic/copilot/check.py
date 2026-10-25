"""Deterministic citation checker (D224 (4), D224a, D224d): runs on the finished answer, never on
a draft.

A sentence (or a structured fact, D224c) is kept only if
  - it cites at least one handle and every handle it cites was given this turn;
  - every number in it is a whole value of the cited lines (a clock time such as 03:15 is one value;
    its digits never support other numbers), exactly or rounded to the sentence's decimals (integers
    below 10 exactly; "x%" also matches a probability x / 100), AND it belongs to a field whose name
    shares a non-generic word with the sentence ("fraud probability 0.350" is removed when 0.350 is
    the view disagreement); ids of 5 or more digits and clock times need only appear as whole values;
  - every id in it (letters with digits, digits with "_" or ".", kind:value, e-mail domains and other
    dotted names) appears as a whole word of the cited lines;
  - a decision word in any form (allowed, held, stepped up, monitoring, blocked ...) cites the case's
    own decision line and matches its decision; about another transaction, it cites that
    transaction's line, names its id and matches its decision;
  - it does not claim that something was changed (first person or passive: updated, set to, marked,
    dismissed, escalated, closed, overridden ...);
  - a claim that the case is fraud or legitimate ("definitely fraud", "not legitimate", "not fraud"
    ...) cites the case's own label line, which states a known label of the same polarity;
  - a mention of fraud cites a line that mentions fraud.
Not checked (measured in E20): negation, comparisons and numbers written as words.
Everything else is removed and counted with its reason; with nothing left the answer is NOT_AVAILABLE.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

CHECKER_VERSION = "D224d"  # part of the cached-summary key (D224d (5))
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
_CLOCK = re.compile(r"^\d{1,2}:\d{2}$")
_DOTTED = re.compile(r"^[a-z0-9-]+(?:\.[a-z0-9-]+)+$", re.I)
_KIND = re.compile(r"^(?:uid|card|email|address|device|customer|tx|case):\S+$", re.I)
_SEGMENT = re.compile(r";|,(?=\s)|\(|\)")

# decision words in every form -> the five actions
_ACTION = re.compile(
    r"\b(?:(allow(?:s|ed|ing)?)|(block(?:s|ed|ing)?)|(hold(?:s|ing)?|held)|"
    r"(step[ _-]?ups?|steps?\s+up|stepped\s+up|stepping\s+up)|(monitor(?:s|ed|ing)?))\b",
    re.I,
)
_ACTION_NAMES = ("allow", "block", "hold", "step_up", "monitor")
_CASE_DECISION = re.compile(r"^transaction (\d+): .*?\bdecision (\w+)")
_OTHER_DECISION = re.compile(r"\btransaction (\d+): .*?\bdecision (\w+)")

_CHANGE = re.compile(
    r"\b(?:i|we|i've|we've|i'll|we'll|copilot|assistant)\b[^.]{0,30}?\b(?:changed|updated|set|marked|"
    r"overrode|overridden|approved|released|unblocked|relabell?ed|cleared|dismissed|escalated|closed|"
    r"lowered|raised|reset|edited|modified|saved|deleted|removed|reopened|resolved|flagged)\b|"
    r"\b(?:has|have|had|was|were|is|are|been|got|gets|get)\s+(?:(?:now|been|just|already|also)\s+)*"
    r"(?:changed|updated|set\s+to|marked|overridden|overruled|relabell?ed|re-?labell?ed|switched|reset|"
    r"lowered|raised|dismissed|escalated|closed|cleared|approved|released|unblocked|reopened|edited|"
    r"modified|deleted|removed|resolved)\b|"
    r"\b(?:decision|label|threshold|setting|status)s?\s+(?:is|are)\s+now\b",
    re.I,
)
_VERB = (r"(?:is|was|are|were|be|been|being|it's|it\s+is|that's|as|labell?ed|marked|confirmed|deemed|"
         r"considered|classified\s+as)")  # fmt: skip
_ADV = (r"(?:(?:a|an|definitely|clearly|certainly|confirmed|actually|likely|probably|indeed|really|"
        r"truly|surely|obviously|most\s+likely|almost\s+certainly)\s+)*")  # fmt: skip
_FRAUD_CLAIM = re.compile(
    rf"\b{_VERB}\s+{_ADV}(?:fraud|fraudulent)\b|\bnot\s+(?:a\s+)?(?:legit\w*|genuine)\b|"
    r"\b(?:definitely|clearly|certainly|confirmed)\s+(?:a\s+)?fraud(?:ulent)?\b",
    re.I,
)
_LEGIT_CLAIM = re.compile(
    rf"\b{_VERB}\s+{_ADV}(?:legit\w*|genuine)\b|\bnot\s+(?:a\s+)?(?:case\s+of\s+)?fraud(?:ulent)?\b",
    re.I,
)
_FRAUD_WORD = re.compile(r"\bfraud\w*\b", re.I)
_CASE_LABEL = re.compile(r"^outcome label(?:: (fraud|legit))?")
_OTHER_LABEL = re.compile(r"\btransaction (\d+): .*?\boutcome label: (fraud|legit)")

# field binding of numbers
_STOP = set(
    "the and with for its his her their them this that these those are was were has have had been "
    "being from into than then there here which who what when where all any per out not but very also "
    "only about over under some each both such more most less few into onto upon while whose".split()
)
_GENERIC = {"fraud", "transaction", "customer", "case", "view", "evidence", "line", "known"}
_SYNONYM = {"risk": "prob", "probability": "prob", "score": "prob", "likelihood": "prob",
            "percent": "prob", "chance": "prob"}  # fmt: skip


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


# ---- value tokens ----------------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    return [t.strip(".,:;+") for t in _TOKEN.findall(_CITE.sub(" ", text)) if t.strip(".,:;+")]


def values(text: str) -> list[tuple[str, object]]:
    """Whole value tokens of a text: ("num", "12.5") / ("clock", minutes) / ("id", token)."""
    out = []
    for tok in _tokens(text):
        if re.search(r"[A-Za-z]", tok) or not re.search(r"\d", tok):
            continue
        if _CLOCK.match(tok):
            h, m = tok.split(":")
            out.append(("clock", int(h) * 60 + int(m)))
        elif _NUMBER.match(tok) or _GROUPED.match(tok):
            out.append(("num", tok))
        else:  # digits with "_", several "." or "-" / "/": an id, matched exactly
            out.append(("id", tok))
    return out


def _value(num: str) -> tuple[float, int, bool]:
    pct = num.endswith("%")
    core = num.strip("$%").replace(",", "")
    decimals = len(core.split(".")[1]) if "." in core else 0
    return float(core), decimals, pct


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


def _words(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z]+", _CITE.sub(" ", text).lower()):
        if len(w) < 3 or w in _STOP:
            continue
        w = w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        out.add(_SYNONYM.get(w, w))
    return out


def number_problem(s: str, cited: list[str]) -> str | None:
    """The first number of the sentence that is not a whole value of a matching field, or None."""
    segments = [(_words(seg), values(seg)) for line in cited for seg in _SEGMENT.split(line)]
    all_values = [v for _, vs in segments for v in vs]
    fact_words = _words(s)
    for kind, val in values(s):
        if kind == "clock":
            if ("clock", val) not in all_values:
                return f"time not in the cited evidence: {val // 60:02d}:{val % 60:02d}"
            continue
        if kind == "id":
            if ("id", val) not in all_values:
                return f"id not in the cited evidence: {val}"
            continue
        core = val.strip("$%").replace(",", "")
        if re.fullmatch(r"\d{5,}", core):  # a transaction or customer id: whole value only
            if not any(
                k == "num" and v.strip("$%").replace(",", "") == core for k, v in all_values
            ):
                return f"id not in the cited evidence: {val}"
            continue
        ok = False
        for seg_words, seg_values in segments:
            nums = [_value(v)[0] for k, v in seg_values if k == "num"]
            if not number_supported(val, nums):
                continue
            need = seg_words - _GENERIC
            if (need & fact_words) if need else (not seg_words or seg_words & fact_words):
                ok = True
                break
        if not ok:
            found = any(number_supported(val, [_value(v)[0]]) for k, v in all_values if k == "num")
            return (f"number belongs to another field of the cited evidence: {val}" if found
                    else f"number not in the cited evidence: {val}")  # fmt: skip
    return None


def _ids(text: str) -> list[str]:
    out = []
    for tok in _tokens(text):
        tok = tok.strip("-")
        letters, digits = re.search(r"[A-Za-z]", tok), re.search(r"\d", tok)
        dotted = _DOTTED.match(tok) and re.search(r"\.[a-z]{2,}$", tok, re.I)
        if (letters and digits) or dotted or _KIND.match(tok):
            out.append(tok)
    return out


def id_supported(token: str, cited_lower: str) -> bool:
    """The token (for kind:value, its value) as a whole word of the cited lines."""
    tok = token.split(":", 1)[1] if _KIND.match(token) else token
    return (
        re.search(r"(?<![\w.-])" + re.escape(tok.lower()) + r"(?![\w-]|\.\w)", cited_lower)
        is not None
    )


# ---- decisions and labels ------------------------------------------------------------------------
def actions(text: str) -> set[str]:
    """The decision words of a text in any form, as the five action names."""
    out = set()
    for m in _ACTION.finditer(_CITE.sub(" ", text)):
        out.add(next(n for n, g in zip(_ACTION_NAMES, m.groups(), strict=True) if g))
    return out


def decision_problem(s: str, cited: list[str]) -> str | None:
    words = actions(s)
    if not words:
        return None
    fact_ids = {v for k, v in values(s) if k == "num"}
    for a in words:
        ok = False
        for line in cited:
            m = _CASE_DECISION.match(line)
            if m and m.group(2) == a:
                ok = True
            o = _OTHER_DECISION.search(line)
            if not m and o and o.group(2) == a and o.group(1) in fact_ids:
                ok = True
        if not ok:
            case = any(_CASE_DECISION.match(line) for line in cited)
            return (f"decision not in the cited evidence: {a}" if case else
                    f"decision word must cite the case's own decision line: {a}")  # fmt: skip
    return None


def label_problem(s: str, cited: list[str]) -> str | None:
    fraud, legit = bool(_FRAUD_CLAIM.search(s)), bool(_LEGIT_CLAIM.search(s))
    if fraud and legit:
        return "ambiguous outcome label claim"
    if fraud or legit:
        want = "fraud" if fraud else "legit"
        fact_ids = {v for k, v in values(s) if k == "num"}
        for line in cited:  # another transaction's line: only when the fact names its id
            o = _OTHER_LABEL.search(line)
            if o and not _CASE_LABEL.match(line) and o.group(1) in fact_ids:
                return None if o.group(2) == want else "label claim disagrees with the cited label"
        labels = [m for line in cited if (m := _CASE_LABEL.match(line))]
        if not labels:
            return "a label claim must cite the case's own label line"
        known = [m.group(1) for m in labels if m.group(1)]
        if not known:
            return "claims an outcome label that is not known yet"
        if known[0] != want:
            return "label claim disagrees with the cited label"
        return None
    if _FRAUD_WORD.search(s) and not any(_FRAUD_WORD.search(line) for line in cited):
        return "mentions fraud but the cited evidence does not"
    return None


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
    cited = [evidence[h] for h in dict.fromkeys(cites)]
    low = " ".join(cited).lower()
    if _CHANGE.search(s):
        return "claims a change (the copilot cannot change anything)"
    if why := label_problem(s, cited):
        return why
    if why := number_problem(s, cited):
        return why
    if bad := [t for t in _ids(s) if not id_supported(t, low)]:
        return f"id not in the cited evidence: {bad[0]}"
    return decision_problem(s, cited)


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
