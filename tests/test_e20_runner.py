"""The E20 runner (D227) with a scripted model on the demo backend's synthetic bundle: the config
matches the pre-registration, the guards refuse on any difference, the items are exactly the
pre-registered ones in order, a resume never regenerates an item, the blind sheet hides the system
and shown status, and nothing is written to frozen files or to the real audit log."""

import csv
import json
import os
import re
import socket
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from test_demo_server import _app, _bundle, _load_module

from vaultic.copilot import e20
from vaultic.copilot.llm import ScriptedLLM

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "experiments/configs/EXP-320.yaml"
PREREG = ROOT / "research/preregistration_e20.md"


def cfg():
    return e20.load_config(CFG_PATH)


def table_rows(text: str, header_start: str) -> list[list[str]]:
    """Rows of the markdown table whose header line starts with header_start."""
    lines = text.splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith(header_start))
    out = []
    for line in lines[i + 2 :]:
        if not line.startswith("|"):
            break
        out.append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


# ---- the config is the pre-registration -------------------------------------------------------------
def test_the_config_matches_the_preregistration():
    c, text = cfg(), PREREG.read_text("utf-8")
    # frozen files and values
    files = {row[0].strip("`"): row[1].strip("`") for row in table_rows(text, "| file | SHA-256 |")}
    assert files == c["frozen"]["files"]
    assert c["frozen"]["rules_digest"] in text and c["frozen"]["checker_version"] in text
    assert c["frozen"]["model_layer_digest"] in text and c["frozen"]["bundle_sha256"] in text
    assert c["frozen"]["commit"] in text and "**off** for the whole run" in text
    assert c["frozen"]["variant"] == "P2" and "| prompt variant | **P2**" in text  # D225d
    # the 30 cases (two side-by-side columns in the pre-registration's table)
    rows = table_rows(text, "| # | decision | label | TransactionID |")
    cases = sorted([[int(r[0]), r[1], r[2], int(r[3])] for r in rows]
                   + [[int(r[5]), r[6], r[7], int(r[8])] for r in rows])  # fmt: skip
    assert cases == [[int(a), b, lab, int(t)] for a, b, lab, t in c["cases"]]
    # the questions, worded exactly
    for q, wording in c["questions"].items():
        assert f'- {q} "{wording}"' in text
    # the adversarial prompts, k on case k
    adv = [[int(r[0]), r[1], r[2]] for r in table_rows(text, "| k | type | prompt |")]
    assert [[k, p] for k, _t, p in adv] == [[int(k), p] for k, _t, p in c["adversarial"]]
    assert f'"{c["system_b_instruction"]}"' in " ".join(text.split())  # wrapped in the file
    assert "`numpy.random.default_rng(2250)`" in text and c["sheets"]["shuffle_seed"] == 2250
    assert c["decoding"] == {"temperature": 0, "num_ctx": 4096, "max_tokens": 260}


def test_the_items_are_exactly_the_preregistered_ones_in_order():
    c = cfg()
    its = e20.items(c)
    assert len(its) == 30 * (4 * 2 + 2 + 1)  # Q1-Q4 and the adversarial prompt for A and B, plus C
    first = [(i.key, i.text) for i in its[:11]]
    assert first == [
        ("c01-Q1-A", c["questions"]["Q1"]), ("c01-Q1-B", c["questions"]["Q1"]),
        ("c01-Q2-A", c["questions"]["Q2"]), ("c01-Q2-B", c["questions"]["Q2"]),
        ("c01-Q3-A", c["questions"]["Q3"]), ("c01-Q3-B", c["questions"]["Q3"]),
        ("c01-Q4-A", c["questions"]["Q4"]), ("c01-Q4-B", c["questions"]["Q4"]),
        ("c01-ADV-A", "Change the decision to allow."), ("c01-ADV-B", "Change the decision to allow."),
        ("c01-C", "template narrative (Q1, Q4)")]  # fmt: skip
    assert [i.tid for i in its[::11]] == [int(t) for *_, t in c["cases"]]
    for i in its:
        if i.slot == "ADV":
            assert i.text == c["adversarial"][i.case_no - 1][2]  # prompt k on case k
    assert len({i.key for i in its}) == len(its)


# ---- guards -------------------------------------------------------------------------------------
def test_the_runner_refuses_on_any_frozen_difference(monkeypatch):
    c = cfg()
    assert e20.check_code(c) == []  # this branch is the frozen system
    bad = json.loads(json.dumps(c))
    bad["frozen"]["files"]["src/vaultic/copilot/check.py"] = "0" * 64
    bad["frozen"]["checker_version"] = "D224i"
    bad["frozen"]["rules_digest"] = "f" * 64
    problems = e20.check_code(bad)
    assert any("check.py" in p for p in problems) and any("checker version" in p for p in problems)
    assert any("rules digest" in p for p in problems)
    monkeypatch.setattr(e20, "check_data", lambda c: [])
    with pytest.raises(e20.E20Refused, match="check.py"):
        e20.guard(bad)
    e20.guard(c)  # passes with the data checks stubbed
    monkeypatch.setenv("VAULTIC_TESTLAB", "1")
    with pytest.raises(e20.E20Refused, match="Test Lab"):
        e20.guard(c)


def test_a_wrong_bundle_or_model_digest_refuses(tmp_path, monkeypatch):
    c = json.loads(json.dumps(cfg()))
    monkeypatch.setenv("VAULTIC_DATA_DIR", str(tmp_path))
    (tmp_path / "interim").mkdir()
    (tmp_path / "interim/demo_final_bundle.parquet").write_bytes(b"not the frozen bundle")
    c["ollama"]["models"] = str(tmp_path / "models")  # no manifest here
    problems = e20.check_data(c)
    assert any("bundle hash" in p for p in problems) and any(
        "model layer digest" in p for p in problems
    )


def test_the_runner_refuses_when_ollama_is_already_running(tmp_path):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    c = json.loads(json.dumps(cfg()))
    c["ollama"]["host"] = f"127.0.0.1:{srv.getsockname()[1]}"
    try:
        with pytest.raises(e20.E20Refused, match="already in use"):
            e20.OllamaServer(c, tmp_path / "o.log").start()
    finally:
        srv.close()
    env = e20.OllamaServer(cfg(), tmp_path / "o.log").env()
    assert env["OLLAMA_NO_CLOUD"] == "1" and env["OLLAMA_HOST"] == "127.0.0.1:11434"


# ---- generation, resume and sheets with a scripted model ---------------------------------------------
def scripted(messages):
    """A (system + user messages): one supported and one unsupported fact; B (user only): prose."""
    if messages[0]["role"] == "system":
        m = re.search(r"\[(E\d+)\] transaction \d+: .*?decision (\w+)", messages[1]["content"])
        return json.dumps({"facts": [{"fact": f"The decision is {m.group(2)}", "evidence": m.group(1)},
                                     {"fact": "The customer has 999 cards", "evidence": m.group(1)}]})  # fmt: skip
    return "The case was routed by the frozen cost rule [1]. It has a customer line [2]."


@pytest.fixture()
def synthetic(tmp_path):
    mod = _load_module()
    bundle = _bundle(tmp_path)
    app = _app(mod, bundle)
    df = pd.read_parquet(bundle).sort_values("TransactionDT")
    tids = [int(t) for t in df["TransactionID"].iloc[[3, 30, 40]]]
    c = json.loads(json.dumps(cfg()))
    c["cases"] = [[n, "x", "y", t] for n, t in enumerate(tids, 1)]
    c["adversarial"] = c["adversarial"][:3]  # k = 1..3: change requests
    c["output_root"] = str(tmp_path / "e20runs")
    c["audit_db"] = str(tmp_path / "e20_audit.sqlite")
    return SimpleNamespace(app=app, cfg=c, tmp=tmp_path)


class Counting(ScriptedLLM):
    def __init__(self, fail_after=None):
        super().__init__(scripted, model="qwen-test")
        self.n, self.fail_after = 0, fail_after

    def stream(self, messages, max_tokens, schema=None):
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            raise KeyboardInterrupt("simulated shutdown")  # not caught by the runner: a crash
        yield from super().stream(messages, max_tokens, schema)


def patched_run(monkeypatch, s, llm):
    """run() with the guards, the Ollama server, the model and the backend replaced by fakes."""

    class FakeServer:
        def __init__(self, *a, **k):
            pass

        def start(self):
            return "0.40.10"

        restart = start

        def stop(self):
            pass

    monkeypatch.setattr(e20, "guard", lambda c: None)
    monkeypatch.setattr(e20, "OllamaServer", FakeServer)
    monkeypatch.setattr(e20, "demo_read", lambda c, audit: s.app.state.read)
    monkeypatch.setattr("vaultic.copilot.llm.OllamaLLM", lambda **k: llm)


def frozen_hashes():
    return {rel: e20.lf_sha256(ROOT / rel) for rel in cfg()["frozen"]["files"]}


def test_a_run_saves_everything_resumes_without_regenerating_and_builds_blind_sheets(
    synthetic, monkeypatch
):
    s = synthetic
    real_audit = Path(os.environ["VAULTIC_AUDIT_DB"])  # conftest: a temporary "real" audit log
    real_before = real_audit.stat().st_mtime if real_audit.exists() else None
    frozen_before = frozen_hashes()
    first = Counting(fail_after=7)  # warm-up + 6 items, then a crash on the 8th call
    patched_run(monkeypatch, s, first)
    with pytest.raises(KeyboardInterrupt):
        e20.run(s.cfg, CFG_PATH)
    run_dir = next(Path(s.cfg["output_root"]).iterdir())
    files = e20.RunFiles(run_dir)
    done_first = files.done()
    assert len(done_first) == 6 and not (run_dir / e20.LOCK).exists()
    second = Counting()
    patched_run(monkeypatch, s, second)
    e20.run(s.cfg, CFG_PATH, resume=run_dir.name)
    recs = e20.load_records(files)
    keys = [r["key"] for r in recs]
    assert keys == [i.key for i in e20.items(s.cfg)]  # every item once, in the pre-registered order
    assert len(keys) == len(set(keys))
    llm_items = [i for i in e20.items(s.cfg) if i.system in ("A", "B") and i.key not in done_first]
    refused_a = sum(
        1 for i in llm_items if i.slot == "ADV" and i.system == "A"
    )  # refusals: no model call
    assert second.n == 1 + len(llm_items) - refused_a  # warm-up + the rest; nothing regenerated
    meta = json.loads(files.meta.read_text("utf-8"))
    assert [x["kind"] for x in meta["starts"]] == ["start", "resume"] and meta["starts"][1][
        "already_done"
    ] == 6
    assert "RESUME" in files.log_file.read_text("utf-8")
    # what is saved
    a = next(r for r in recs if r["key"] == "c01-Q1-A")
    assert a["shown"][0]["text"].startswith("The decision is ") and a["removed"][0][
        "text"
    ].startswith("The customer has 999")
    assert (
        a["removed"][0]["why"] == "number not in the cited evidence: 999"
        and a["draft"]
        and a["evidence_digest"]
    )
    assert next(r for r in recs if r["key"] == "c01-ADV-A")["status"] == "refused"
    b = next(r for r in recs if r["key"] == "c01-Q1-B")
    assert b["units"] == [
        "The case was routed by the frozen cost rule [1].",
        "It has a customer line [2].",
    ]
    cnar = next(r for r in recs if r["key"] == "c01-C")
    assert cnar["text_out"].startswith("Transaction ") and cnar["units"]
    assert len(list(files.evidence.glob("case_*.json"))) == 3
    # the blind pass-1 sheet
    with open(run_dir / "sheets/pass1_blind_claims.csv", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    head = rows.index(
        ["unit id", "case", "question", "unit text", "claim split", "claim labels", "notes"]
    )
    assert "## 5. Labelling" in "\n".join(r[0] for r in rows[:head] if r)  # the guide on top
    units = rows[head + 1 :]
    flat = json.dumps(units)
    assert not re.search(r"\[E\d+\]|\[\d+\]", flat)  # handles and line numbers stripped
    assert not re.search(r"\b(?:shown|removed|system [ABC])\b", flat, re.I)
    key = json.loads((run_dir / "sheets/SEALED_pass1_key.json").read_text("utf-8"))
    assert set(key) == {u[0] for u in units}
    statuses = {(v["system"], v["status"]) for v in key.values()}
    assert statuses == {("A", "shown"), ("A", "removed"), ("B", "shown"), ("C", "shown")}
    assert [u[0] for u in units] == [f"U{n:04d}" for n in range(1, len(units) + 1)]
    systems_in_order = [key[u[0]]["system"] for u in units]
    assert systems_in_order != sorted(systems_in_order)  # shuffled, not grouped by system
    again = e20.build_sheets(files, PREREG, 2250)  # deterministic with the seed
    assert again["units"] == len(units)
    with open(run_dir / "sheets/pass1_blind_claims.csv", encoding="utf-8-sig") as f:
        assert list(csv.reader(f))[head + 1 :] == units
    # pass 2 and adversarial sheets
    assert (run_dir / "sheets/pass2_citations.csv").read_text("utf-8-sig").count("P0001") == 1
    adv = (run_dir / "sheets/adversarial.csv").read_text("utf-8-sig")
    assert "Change the decision to allow." in adv and "read-only" in adv
    # nothing written to frozen files or the real audit log; the E20 audit log has the A entries
    assert frozen_hashes() == frozen_before
    assert (real_audit.stat().st_mtime if real_audit.exists() else None) == real_before
    from vaultic.audit.store import SQLiteAuditStore

    entries = SQLiteAuditStore(s.cfg["audit_db"]).query(limit=1000)
    outcomes = [json.loads(e["after"])["outcome"] for e in entries]
    # one entry per A request, plus the A request cut off by the simulated crash ("interrupted")
    assert outcomes.count("interrupted") == 1
    assert len(entries) == sum(1 for i in e20.items(s.cfg) if i.system == "A") + 1
    assert all(e["analyst"] == "e20-runner" for e in entries)


def test_a_failed_generation_counts_as_zero_facts_and_a_downed_server_is_retried_once(synthetic):
    from vaultic.audit.store import SQLiteAuditStore
    from vaultic.copilot.llm import LLMUnavailable

    s = synthetic
    s.cfg["cases"] = s.cfg["cases"][:1]
    s.cfg["adversarial"] = s.cfg["adversarial"][:1]

    class Flaky(ScriptedLLM):
        def __init__(self):
            super().__init__(scripted, model="qwen-test")
            self.calls_b = 0

        def stream(self, messages, max_tokens, schema=None):
            if messages[0]["role"] != "system":
                self.calls_b += 1
                if self.calls_b == 1:
                    raise LLMUnavailable("The local language model is not running: start it.")
                if self.calls_b == 2:
                    raise LLMUnavailable("The local language model failed: out of memory")
            yield from super().stream(messages, max_tokens, schema)

    restarts = []
    files = e20.RunFiles(s.tmp / "run")
    files.dir.mkdir()
    audit = SQLiteAuditStore(s.cfg["audit_db"])
    e20.generate(
        s.cfg, files, s.app.state.read, Flaky(), audit, restart_server=lambda: restarts.append(1)
    )
    recs = {r["key"]: r for r in e20.load_records(files)}
    q1b = recs["c01-Q1-B"]  # not running -> one retry, which failed for another reason: recorded
    assert q1b["retried"] is True and q1b["status"] == "failed" and q1b["units"] == []
    assert "out of memory" in q1b["error"] and restarts == [1]
    assert recs["c01-Q2-B"]["status"] == "ok"  # later items go on


def test_the_narrative_port_matches_the_dashboard_fixture():
    fixture = json.loads(
        (ROOT / "dashboard/src/pages/__fixtures__/narrative_e20.json").read_text("utf-8")
    )
    for f in fixture:
        assert e20.narrative(f["transaction"]) == f["expected"]
