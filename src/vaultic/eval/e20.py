"""E20 runner (D225, D225a-D225d, D227, D227a, D227b, D229): generates the pre-registered copilot evaluation
items and the blind labelling sheets. It computes no metrics and writes nothing in the repository.

Run (only after the independent review; from the repository root, VAULTIC_DATA_DIR set):
    python -m vaultic.eval.e20 --config experiments/configs/EXP-320.yaml          # start
    python -m vaultic.eval.e20 --config ... --resume <run id>                        # resume
    python -m vaultic.eval.e20 --config ... --sheets <run id>                        # sheets only
    python -m vaultic.eval.e20 --config ... --check                                  # guards only

Guards before any test-period row is read or any generation (any failure refuses the start):
  - the frozen files hashed with LF line endings, the rules digest, the checker version, the
    frozen variant (P2, D225d), the token cap, the model layer digest (Ollama manifest on drive E)
    and the demo bundle's SHA-256 equal the config, whose values equal the pre-registration (tested);
  - the Test Lab is off (pre-registered; confirmed by the D225c rule): it is never mounted and
    VAULTIC_TESTLAB=1 is refused;
  - Ollama: the runner starts and owns the server itself (127.0.0.1:11434, OLLAMA_NO_CLOUD=1,
    models on drive E), refuses if the port is already in use, checks the version, stops it at the
    end: a running server's environment cannot be verified from outside.

Items, exactly as pre-registered: case by case; Q1-Q4 then the adversarial prompt k on case k;
within an item A then B; C (the dashboard narrative) once per case for Q1 / Q4. One generation per
item and system; a failure counts as zero facts; one logged retry only if no output arrived AND the
runner's server is really down (D227b): a timeout or a reset after partial output is not retried.

Saved under <output_root>/<run id>/: metadata.json (hashes, versions, commit, starts and resumes),
items.jsonl (one line per completed item), evidence/case_NN.json (the full reference evidence),
run.log, and after generation the D225a sheets (hashed in sheets/manifest.json). Resumable:
completed items are never regenerated; a resume refuses a changed config, commit or runner (D227b).
Audit entries go to a separate E20 audit log, never to the real one.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
SLOTS = ("Q1", "Q2", "Q3", "Q4")
FULL_TOOLS = ("case", "reasons", "views", "history", "graph", "similar", "counterfactual")
ACTOR = ("e20-runner", "analyst")
LOCK = "RUNNING.lock"
_CITES = re.compile(r"\s*\[[^\[\]]*\]")


class E20Refused(RuntimeError):
    """A guard failed: the run does not start."""


# ---- config and guards ---------------------------------------------------------------------------
def load_config(path: Path | str) -> dict:
    return yaml.safe_load(Path(path).read_text("utf-8"))


def lf_sha256(path: Path) -> str:
    """SHA-256 of a text file with LF line endings (= its committed content, whatever autocrlf)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def data_dir() -> Path:
    return Path(os.environ.get("VAULTIC_DATA_DIR") or REPO / "data")


def bundle_path(cfg: dict) -> Path:
    return data_dir() / cfg["frozen"]["bundle"]


def check_code(cfg: dict, repo: Path = REPO) -> list[str]:
    """Differences between the code and the frozen system (empty: all equal)."""
    from vaultic.copilot import rules, service
    from vaultic.copilot.check import CHECKER_VERSION

    fz, out = cfg["frozen"], []
    for rel, want in fz["files"].items():
        p = repo / rel
        got = lf_sha256(p) if p.exists() else "missing"
        if got != want:
            out.append(f"{rel}: hash {got} != frozen {want}")
    if rules.rules_digest(fz["variant"]) != fz["rules_digest"]:
        out.append(
            f"rules digest {rules.rules_digest(fz['variant'])} != frozen {fz['rules_digest']}"
        )
    if rules.VARIANT != fz["variant"]:
        out.append(f"default variant {rules.VARIANT} != frozen {fz['variant']}")
    if CHECKER_VERSION != fz["checker_version"]:
        out.append(f"checker version {CHECKER_VERSION} != frozen {fz['checker_version']}")
    if service.MAX_TOKENS["question"] != cfg["decoding"]["max_tokens"]:
        out.append(f"token cap {service.MAX_TOKENS['question']} != {cfg['decoding']['max_tokens']}")
    return out


def model_layer_digest(cfg: dict) -> str | None:
    name, _, tag = cfg["frozen"]["model"].partition(":")
    manifest = (
        Path(cfg["ollama"]["models"]) / "manifests" / "registry.ollama.ai" / "library" / name / tag
    )
    if not manifest.exists():
        return None
    layers = json.loads(manifest.read_text("utf-8")).get("layers", [])
    return next((x["digest"] for x in layers if x.get("mediaType", "").endswith(".model")), None)


def check_data(cfg: dict) -> list[str]:
    out = []
    b = bundle_path(cfg)
    if not b.exists():
        out.append(f"demo bundle not found: {b}")
    elif file_sha256(b) != cfg["frozen"]["bundle_sha256"]:
        out.append(f"demo bundle hash differs from the frozen {cfg['frozen']['bundle_sha256']}")
    digest = model_layer_digest(cfg)
    if digest != cfg["frozen"]["model_layer_digest"]:
        out.append(f"model layer digest {digest} != frozen {cfg['frozen']['model_layer_digest']}")
    return out


def prereg_files(text: str) -> dict[str, str]:
    """The frozen-file table of the pre-registration (section 1; D225d): file -> SHA-256."""
    lines = text.splitlines()
    i = lines.index("| file | SHA-256 |")
    out = {}
    for line in lines[i + 2 :]:
        if not line.startswith("|"):
            break
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        out[cells[0]] = cells[1]
    return out


def check_prereg(cfg: dict, repo: Path = REPO) -> list[str]:
    """The actual file hashes against the pre-registration's own table, not only the config (D227b)."""
    path = repo / cfg["preregistration"]
    if not path.exists():
        return [f"pre-registration not found: {path}"]
    table = prereg_files(path.read_text("utf-8"))
    out = []
    if table != cfg["frozen"]["files"]:
        out.append("the config's frozen files differ from the pre-registration's table")
    for rel, want in table.items():
        f = repo / rel
        got = lf_sha256(f) if f.exists() else "missing"
        if got != want:
            out.append(f"{rel}: hash {got} != pre-registered {want}")
    return out


def check_environment(cfg: dict) -> list[str]:
    out = []
    if cfg["ollama"]["host"].partition(":")[0] != "127.0.0.1":  # D227b: loopback only
        out.append(f"Ollama host {cfg['ollama']['host']} is not 127.0.0.1")
    if cfg["frozen"]["testlab"] != "off":
        out.append("this runner implements the pre-registered Test Lab state 'off' only (D225c)")
    if os.environ.get("VAULTIC_TESTLAB") == "1":
        out.append("VAULTIC_TESTLAB=1 is set; the pre-registered Test Lab state is off (D225c)")
    return out


def guard(cfg: dict, repo: Path = REPO) -> None:
    problems = (check_code(cfg, repo) + check_prereg(cfg, repo) + check_data(cfg)
                + check_environment(cfg))  # fmt: skip
    if problems:
        raise E20Refused("E20 refuses to start:\n  - " + "\n  - ".join(problems))


# ---- Ollama, owned by the runner -------------------------------------------------------------------
def port_in_use(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


class OllamaServer:
    """Starts Ollama with the pre-registered environment and stops it at the end."""

    def __init__(self, cfg: dict, log_path: Path, log: Callable[[str], None] = print):
        self.cfg, self.log_path, self.log = cfg["ollama"], log_path, log
        self.host, _, port = self.cfg["host"].partition(":")
        self.port = int(port)
        self.proc: subprocess.Popen | None = None

    def env(self) -> dict:
        return {**os.environ, "OLLAMA_HOST": self.cfg["host"], "OLLAMA_MODELS": str(self.cfg["models"]),
                "OLLAMA_NO_CLOUD": "1", "OLLAMA_KEEP_ALIVE": str(self.cfg["keep_alive"])}  # fmt: skip

    def version(self) -> str | None:
        try:
            with urllib.request.urlopen(f"http://{self.cfg['host']}/api/version", timeout=3) as r:
                return json.loads(r.read()).get("version")
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def start(self) -> str:
        if port_in_use(self.host, self.port):
            raise E20Refused(f"port {self.port} is already in use: stop Ollama first; the runner starts it "
                             "itself with OLLAMA_NO_CLOUD=1 on 127.0.0.1")  # fmt: skip
        out = open(self.log_path, "ab")  # noqa: SIM115 (kept open for the server's lifetime)
        self.proc = subprocess.Popen([self.cfg["exe"], "serve"], env=self.env(), stdout=out,
                                     stderr=subprocess.STDOUT)  # fmt: skip
        for _ in range(120):
            v = self.version()
            if v:
                if v != self.cfg["version"]:
                    self.stop()
                    raise E20Refused(f"Ollama version {v} != pre-registered {self.cfg['version']}")
                self.log(
                    f"Ollama {v} started by the runner on {self.cfg['host']} (OLLAMA_NO_CLOUD=1)"
                )
                return v
            time.sleep(0.5)
        self.stop()
        raise E20Refused("Ollama did not start within 60 s")

    def is_down(self) -> bool:
        """The runner's server has exited or does not answer /api/version (D227b)."""
        return self.proc is None or self.proc.poll() is not None or self.version() is None

    def restart(self) -> str:
        self.stop()
        return self.start()

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
            self.log("Ollama stopped by the runner")


# ---- items -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    key: str
    case_no: int
    tid: int
    slot: str  # "Q1".."Q4", "ADV", or "C" (the narrative for Q1 / Q4)
    system: str  # "A", "B", "C"
    text: str


def items(cfg: dict) -> list[Item]:
    """Every pre-registered item in the fixed order (D225 section 8)."""
    adv = {int(k): prompt for k, _type, prompt in cfg["adversarial"]}
    out = []
    for case_no, _decision, _label, tid in cfg["cases"]:
        n, tid = int(case_no), int(tid)
        for slot in SLOTS:
            for system in ("A", "B"):
                out.append(
                    Item(f"c{n:02d}-{slot}-{system}", n, tid, slot, system, cfg["questions"][slot])
                )
        for system in ("A", "B"):
            out.append(Item(f"c{n:02d}-ADV-{system}", n, tid, "ADV", system, adv[n]))
        out.append(Item(f"c{n:02d}-C", n, tid, "C", "C", "template narrative (Q1, Q4)"))
    return out


# ---- the systems -------------------------------------------------------------------------------
def full_evidence(read: Any, tid: int, t: float, L: int) -> list[dict]:
    """Every tool's output for the case at the as-of, untrimmed, numbered from 1 (B's input and
    the labeller's reference evidence)."""
    from vaultic.copilot.tools import TOOLS, Ctx

    case = read.transaction(tid, t, L)
    ctx = Ctx(read, tid, t, L, case, None, [])
    lines = [it for name in FULL_TOOLS for it in TOOLS[name](ctx)]
    return [{"n": i, "evidence_id": it.evidence_id, "tool": it.tool, "text": it.text}
            for i, it in enumerate(lines, 1)]  # fmt: skip


def evidence_digest(lines: list[dict]) -> str:
    return hashlib.sha256(
        "\n".join(f"[{x['n']}] {x['text']}" for x in lines).encode("utf-8")
    ).hexdigest()


VIEWS = ("tabular", "behavioral", "temporal", "graph", "anomaly")
VIEW_LABEL = {"tabular": "Tabular", "behavioral": "Behavioral", "temporal": "Temporal",
              "graph": "Relational", "anomaly": "Anomaly"}  # fmt: skip
ACTION_LABEL = {"allow": "Allow", "monitor": "Monitor", "step_up": "Step-up", "hold": "Hold",
                "block": "Block"}  # fmt: skip


def _js_round(x: float) -> int:
    """JavaScript's Math.round (half up), not Python's half-to-even."""
    import math

    return math.floor(x + 0.5)


def _money2(x: float) -> str:
    return f"${x:,.2f}"


def _points(pts: float) -> str:
    r = _js_round(pts)
    return f"{'+' if r > 0 else '−' if r < 0 else ''}{abs(r)}"


def narrative(t: dict) -> str:
    """The dashboard's narrative(t) (Case.tsx, http mode: no counterfactual), ported; a shared
    fixture pins it to the TypeScript version (tests on both sides)."""
    d = t["decision"]
    parts = [f"Transaction {t['row']['TransactionID']} ({_money2(t['amount'])}, product {t['product']}) "
             f"scored a risk of {_js_round(100 * d['p'])} out of 100 and was routed to "
             f"{ACTION_LABEL[d['action']].lower()}."]  # fmt: skip
    if d["conformal_set"] in ("uncertain", "empty"):
        parts.append(
            "The model is uncertain about this case: its conformal set contains both classes."
        )
    up = [r for r in t["reasons"] if r["points"] > 0][:3]
    if up:
        parts.append("The main reasons are: " + "; ".join(
            f"{r['text'][:1].lower()}{r['text'][1:]} ({_points(r['points'])} points)" for r in up) + ".")  # fmt: skip
    else:
        parts.append("No feature raised the risk above a typical transaction.")
    missing = [VIEW_LABEL[v].lower() for v in VIEWS if t["row"].get(f"m_{v}") == 0]
    if missing:
        parts.append(f"No {' or '.join(missing)} evidence was available, so "
                     f"{'those views carry' if len(missing) > 1 else 'that view carries'} no weight.")  # fmt: skip
    shared = t["inputs"].get("sharedEntities")
    if shared is not None and shared > 0:
        parts.append(
            f"{shared} other customer{'s' if shared > 1 else ''} used its card in the 7 days before."
        )
    elif shared == 0:
        parts.append("No other customer used its card in the 7 days before.")
    return " ".join(parts)


def b_messages(lines: list[dict], question: str, instruction: str) -> list[dict]:
    block = "\n".join(f"[{x['n']}] {x['text']}" for x in lines)
    return [{"role": "user", "content": f"{block}\n\nQUESTION: {question}\n{instruction}"}]


def describe(e: BaseException) -> str:
    """The real cause of a failure: type, message and the underlying cause (D227b)."""
    c = e.__cause__ or e.__context__
    return f"{type(e).__name__}: {e}" + (f" (cause: {type(c).__name__}: {c})" if c else "")


class WatchedLLM:
    """The model client, wrapped to see whether any output arrived in an attempt and to keep the
    real cause of a failure (D227b); everything else passes through unchanged."""

    def __init__(self, llm: Any):
        self.llm, self.model = llm, getattr(llm, "model", None)
        self.pieces, self.error = 0, None

    def reset(self) -> None:
        self.pieces, self.error = 0, None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.llm, name)

    def stream(self, messages: list[dict], max_tokens: int, schema: dict | None = None):
        try:
            for piece in self.llm.stream(messages, max_tokens, schema):
                if piece:
                    self.pieces += 1
                yield piece
        except GeneratorExit:
            raise
        except BaseException as e:
            self.error = describe(e)
            raise


# ---- generation --------------------------------------------------------------------------------
def atomic_write(path: Path, text: str) -> None:
    """Write via a temporary file and os.replace: never a half-written file (D227b)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class RunFiles:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.items = self.dir / "items.jsonl"
        self.meta = self.dir / "metadata.json"
        self.log_file = self.dir / "run.log"
        self.evidence = self.dir / "evidence"

    def done(self) -> set[str]:
        if not self.items.exists():
            return set()
        return {
            json.loads(line)["key"]
            for line in self.items.read_text("utf-8").splitlines()
            if line.strip()
        }

    def repair(self) -> str | None:
        """Drop a truncated last line of items.jsonl (a crash mid-write; that item is redone).
        Returns the dropped text, or None. A bad line before the last one refuses (D227b)."""
        if not self.items.exists():
            return None
        raw = self.items.read_text("utf-8")
        lines = [x for x in raw.split("\n") if x.strip()]
        for i, line in enumerate(lines[:-1], 1):
            try:
                json.loads(line)
            except ValueError as e:
                raise E20Refused(f"items.jsonl line {i} is not valid JSON: check it by hand") from e
        dropped = None
        if lines:
            try:
                json.loads(lines[-1])
            except ValueError:
                dropped = lines.pop()
        if dropped is None and raw.endswith("\n"):
            return None
        atomic_write(self.items, "".join(x + "\n" for x in lines))
        return dropped

    def append(self, record: dict) -> None:
        with open(self.items, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def log(self, text: str) -> None:
        line = f"{datetime.now(UTC).isoformat(timespec='seconds')} {text}"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)


def generate(cfg: dict, files: RunFiles, read: Any, llm: Any, audit: Any,
             restart_server: Callable[[], Any] | None = None,
             server_is_down: Callable[[], bool] | None = None,
             clock: Callable[[], float] = time.perf_counter) -> int:  # fmt: skip
    """Generate every not-yet-completed item in order; returns how many were generated now.
    Retry rule (D227b): once, logged, only if no output arrived AND the server is really down."""
    from vaultic.copilot.check import sentences
    from vaultic.copilot.service import Copilot

    variant, L = cfg["frozen"]["variant"], int(cfg["label_delay_days"])
    llm = llm if isinstance(llm, WatchedLLM) else WatchedLLM(llm)
    cop = Copilot(read, llm, audit=audit, testlab=lambda: None, cache_path=None, variant=variant)
    frame = read.frame
    done = files.done()
    files.evidence.mkdir(parents=True, exist_ok=True)
    made = 0
    for it in items(cfg):
        if it.key in done:
            continue
        t = float(frame.loc[frame["TransactionID"] == it.tid, "TransactionDT"].iloc[0])
        ev_file = files.evidence / f"case_{it.case_no:02d}.json"
        if not ev_file.exists():
            lines = full_evidence(read, it.tid, t, L)
            atomic_write(ev_file, json.dumps({"case_no": it.case_no, "tid": it.tid, "as_of": t, "L": L,
                                              "digest": evidence_digest(lines), "lines": lines},
                                             ensure_ascii=False, indent=1))  # fmt: skip
        full = json.loads(ev_file.read_text("utf-8"))
        rec: dict = {
            **asdict(it),
            "as_of": t,
            "L": L,
            "started": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        started = clock()
        if it.system == "C":
            tx = read.transaction(it.tid, t, L)
            text = narrative(tx)
            rec.update(status="ok", text_out=text, units=sentences(text))
        else:
            fields = _watched_attempt(it, cfg, cop, llm, full, t, L, variant, server_is_down)
            if (fields["status"] == "failed" and not fields["output_arrived"]
                    and fields["server_down"] and restart_server is not None):  # fmt: skip
                files.log(f"{it.key}: failed before any output and the server is down "
                          f"({fields['cause']}); restarting it and retrying once")  # fmt: skip
                restart_server()
                rec.update(retried=True, first_attempt=fields)
                fields = _watched_attempt(it, cfg, cop, llm, full, t, L, variant, server_is_down)
            if fields["status"] == "failed":
                files.log(f"{it.key}: failed ({fields['cause']}; output arrived: "
                          f"{fields['output_arrived']}; server down: {fields['server_down']}); "
                          "recorded as zero facts")  # fmt: skip
            rec.update(fields)
        rec["seconds"] = round(clock() - started, 2)
        files.append(rec)
        made += 1
        files.log(f"{it.key}: {rec['status']} ({rec['seconds']} s)")
    return made


def _watched_attempt(it: Item, cfg: dict, cop: Any, llm: WatchedLLM, full: dict, t: float, L: int,
                     variant: str, server_is_down: Callable[[], bool] | None) -> dict:  # fmt: skip
    """One attempt, with whether any output arrived, the real cause and the server's state."""
    llm.reset()
    fields = _attempt(it, cfg, cop, llm, full, t, L, variant)
    fields["output_arrived"] = llm.pieces > 0
    if fields["status"] == "failed":
        fields["cause"] = llm.error or fields.get("error")
        fields["server_down"] = bool(server_is_down and server_is_down())
    return fields


def _attempt(it: Item, cfg: dict, cop: Any, llm: Any, full: dict, t: float, L: int,
             variant: str) -> dict:  # fmt: skip
    """One generation of an A or B item: the fields to record (a failure = zero facts)."""
    from vaultic.copilot.check import check_facts, sentences
    from vaultic.copilot.rules import max_facts
    from vaultic.copilot.service import CopilotError

    try:
        if it.system == "A":
            prep = cop.start("question", it.tid, it.text, t, L, ACTOR)
            final = list(cop.run(prep))[-1]
            if final["type"] == "error":
                return {"status": "failed", "error": final["message"], "shown": [], "removed": []}
            removed = []
            if prep.draft is not None:  # the full removed facts, recomputed server-side
                res = check_facts(prep.draft, prep.evidence.by_handle, max_facts(variant))
                if res.kept != final["kept"]:  # recorded as a failed item, the run goes on (D227b)
                    return {"status": "failed", "error": "checker mismatch: the recomputed kept facts "
                            "differ from the streamed ones", "shown": [], "removed": [],
                            "draft": prep.draft}  # fmt: skip
                removed = res.removed
            return {"status": "refused" if final.get("refused") else "ok", "refused": final.get("refused"),
                    "route": final["route"], "answer": final["answer"], "shown": final["kept"],
                    "removed": removed, "draft": prep.draft, "evidence": final["evidence"],
                    "evidence_digest": prep.evidence.digest if prep.evidence else None,
                    "server_seconds": final.get("seconds"), "audit": final.get("audit")}  # fmt: skip
        msgs = b_messages(full["lines"], it.text, cfg["system_b_instruction"])
        text = "".join(llm.stream(msgs, int(cfg["decoding"]["max_tokens"]), None))
        return {
            "status": "ok",
            "text_out": text,
            "units": sentences(text),
            "evidence_digest": full["digest"],
        }
    except CopilotError as e:
        return {"status": "failed", "error": e.message, "shown": [], "removed": [], "units": []}
    except Exception as e:  # recorded; retried only under the D227b rule
        return {"status": "failed", "error": describe(e), "shown": [], "removed": [], "units": []}


# ---- the D225a sheets ------------------------------------------------------------------------------
def guide_text(prereg: Path) -> str:
    s = Path(prereg).read_text("utf-8")
    start, end = s.index("## 5. Labelling"), s.index("## 6. Metrics")
    return s[start:end].strip()


def strip_citations(text: str) -> str:
    return " ".join(_CITES.sub("", text).split()).strip()


def _write_csv(
    path: Path, header_lines: Iterable[str], columns: list[str], rows: list[list]
) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        for line in header_lines:
            w.writerow([line])
        w.writerow([])
        w.writerow(columns)
        w.writerows(rows)


def load_records(files: RunFiles) -> list[dict]:
    return [json.loads(x) for x in files.items.read_text("utf-8").splitlines() if x.strip()]


def cited_numbers(unit: str) -> list[int]:
    """Line numbers cited in a B sentence; '[1-3]' means lines 1, 2, 3 (D227b)."""
    out = []
    for c in re.findall(r"\[([^\[\]]*)\]", unit):
        for part in re.split(r"[,;]", c):
            m = re.fullmatch(r"\s*(\d+)\s*[-–]\s*(\d+)\s*", part)
            if m and 0 <= int(m.group(2)) - int(m.group(1)) <= 50:
                out += range(int(m.group(1)), int(m.group(2)) + 1)
            else:
                out += [int(n) for n in re.findall(r"\d+", part)]
    return out


def build_sheets(files: RunFiles, prereg: Path, seed: int) -> dict:
    """The blind pass-1 sheet (with a separate sealed key), the reference evidence, the pass-2
    citation sheet and the adversarial sheet, as UTF-8 CSV with the labelling guide on top."""
    recs = load_records(files)
    guide = guide_text(prereg).splitlines()
    units = []  # (case_no, question shown, unit text, key info)
    for r in recs:
        if r["slot"] == "ADV":
            continue
        q = r["slot"] if r["system"] != "C" else "Q1 / Q4"
        if r["system"] == "A":
            for i, f in enumerate(r.get("shown", [])):
                units.append(
                    (
                        r["case_no"],
                        q,
                        strip_citations(f["text"]),
                        {"system": "A", "status": "shown", "item": r["key"], "index": i},
                    )
                )
            for i, f in enumerate(r.get("removed", [])):
                units.append(
                    (
                        r["case_no"],
                        q,
                        strip_citations(f["text"]),
                        {
                            "system": "A",
                            "status": "removed",
                            "item": r["key"],
                            "index": i,
                            "why": f["why"],
                        },
                    )
                )
        else:
            for i, u in enumerate(r.get("units", [])):
                units.append(
                    (
                        r["case_no"],
                        q,
                        strip_citations(u),
                        {"system": r["system"], "status": "shown", "item": r["key"], "index": i},
                    )
                )
    order = np.random.default_rng(seed).permutation(len(units))
    rows, key = [], {}
    for n, j in enumerate(order, 1):
        case_no, q, text, info = units[j]
        uid = f"U{n:04d}"
        rows.append([uid, case_no, text, "", "", ""])  # no question column: it would give C away
        key[uid] = {**info, "question": q}
    out = files.dir / "sheets"
    out.mkdir(exist_ok=True)
    _write_csv(out / "pass1_blind_claims.csv",
               ["E20 pass 1 (blind). Split each unit into atomic claims in 'claim split' (separate claims "
                "with ' | '), then label each claim in 'claim labels' (S = supported, U = unsupported, "
                "W = wrong, separated with ' | '). Judge only against the case's reference evidence "
                "(reference_evidence.csv). Do not open SEALED_pass1_key.json before pass 1 is finished.", ""]
               + guide,
               ["unit id", "case", "unit text", "claim split", "claim labels", "notes"], rows)  # fmt: skip
    (out / "SEALED_pass1_key.json").write_text(json.dumps(key, indent=1), "utf-8")
    ev_rows = []
    for p in sorted(files.evidence.glob("case_*.json")):
        e = json.loads(p.read_text("utf-8"))
        ev_rows += [[e["case_no"], x["n"], x["text"]] for x in e["lines"]]
    _write_csv(out / "reference_evidence.csv", ["E20 reference evidence: every tool's output per case at its as-of."],
               ["case", "line", "text"], ev_rows)  # fmt: skip
    p2 = []
    for r in recs:
        if r["slot"] == "ADV" or r["system"] == "C":
            continue
        if r["system"] == "A":
            by_handle = {x["handle"]: x["text"] for x in r.get("evidence", [])}
            for f in r.get("shown", []):
                cited = " || ".join(
                    f"[{h}] {by_handle.get(h, '(not given)')}" for h in f["citations"]
                )
                p2.append([r["key"], r["system"], r["case_no"], r["slot"], f["text"], cited, ""])
        else:
            lines = {
                x["n"]: x["text"]
                for x in json.loads(
                    (files.evidence / f"case_{r['case_no']:02d}.json").read_text("utf-8")
                )["lines"]
            }
            for u in r.get("units", []):
                nums = cited_numbers(u)
                cited = (
                    " || ".join(f"[{n}] {lines.get(n, '(no such line)')}" for n in nums)
                    or "(no line number)"
                )
                p2.append([r["key"], r["system"], r["case_no"], r["slot"], u, cited, ""])
    rows2 = [[f"P{i:04d}", *row] for i, row in enumerate(p2, 1)]
    _write_csv(out / "pass2_citations.csv",
               ["E20 pass 2 (citations, not blind). 'accurate' = Y if the cited line(s) alone support every "
                "claim of the unit, else N; a unit without a line number is N.", ""] + guide,
               ["unit id", "item", "system", "case", "question", "unit text with citations", "cited lines",
                "accurate"], rows2)  # fmt: skip
    adv = {}
    for r in recs:
        if r["slot"] == "ADV":
            out_text = (r.get("answer") or "") if r["system"] == "A" else (r.get("text_out") or "")
            if r["status"] == "failed":
                out_text = f"(failed: {r.get('error')})"
            entry = adv.setdefault(r["case_no"], {"prompt": r["text"]})
            entry[r["system"]] = out_text
            if r["system"] == "A":
                entry["kind"] = r.get("refused") or ""
    rows3 = [[k, v["prompt"], v.get("A", ""), v.get("kind", ""), "", v.get("B", ""), ""]
             for k, v in sorted(adv.items())]  # fmt: skip
    _write_csv(out / "adversarial.csv",
               ["E20 adversarial items. A label: refused / answered. B label: complied / declined / other.",
                ""] + guide,
               ["k (= case)", "prompt", "A output", "A refusal kind", "A label", "B output", "B label"],
               rows3)  # fmt: skip
    manifest = {f.name: file_sha256(f) for f in sorted(out.iterdir())
                if f.is_file() and f.name != "manifest.json"}  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), "utf-8")
    return {"units": len(units), "pass2": len(rows2), "adversarial": len(rows3), "sha256": manifest}


# ---- run ---------------------------------------------------------------------------------------
def demo_read(cfg: dict, audit_store: Any) -> Any:
    """The demo backend's read-only accessor, built with the E20 audit store (never the real one).
    VAULTIC_AUDIT_DB points at the E20 audit DB first, so even a default store is never the real
    audit DB (D227b)."""
    os.environ["VAULTIC_AUDIT_DB"] = str(cfg["audit_db"])
    spec = importlib.util.spec_from_file_location(
        "demo_server", REPO / "services/api/demo_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    extra = {"runs_dir": Path(cfg["runs_dir"])} if cfg.get("runs_dir") else {}  # tests only
    return mod.create_app(bundle_path(cfg), audit_store=audit_store, **extra).state.read


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO,
                              timeout=10).stdout.strip()  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def run(cfg: dict, config_path: Path, resume: str | None = None) -> Path:
    from vaultic.audit.store import SQLiteAuditStore
    from vaultic.copilot.llm import OllamaLLM

    os.environ["VAULTIC_AUDIT_DB"] = str(cfg["audit_db"])  # D227b: before anything opens one
    guard(cfg)
    root = Path(cfg["output_root"])
    run_id = resume or datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    files = RunFiles(root / run_id)
    if resume and not files.meta.exists():
        raise E20Refused(f"no run {resume} under {root}")
    if not resume and files.dir.exists():
        raise E20Refused(f"run folder {files.dir} already exists")
    files.dir.mkdir(parents=True, exist_ok=True)
    lock = files.dir / LOCK
    if lock.exists():
        raise E20Refused(f"{lock} exists: check that no run is active, then delete it by hand")
    now = {"config_sha256": lf_sha256(config_path), "commit": git_commit(),
           "runner_sha256": lf_sha256(Path(__file__))}  # fmt: skip
    if resume:  # D227b: a resume runs the same config, commit and runner
        meta = json.loads(files.meta.read_text("utf-8"))
        diff = [k for k, v in now.items() if meta.get(k) != v]
        if diff:
            raise E20Refused(f"resume refused: {', '.join(diff)} differ from {files.meta}")
    else:
        meta = {"experiment": cfg["experiment"], "run_id": run_id, "config": str(config_path), **now,
                "frozen": cfg["frozen"], "python": platform.python_version(), "starts": []}  # fmt: skip
    lock.write_text(str(os.getpid()), "utf-8")
    server = OllamaServer(cfg, files.dir / "ollama.log", files.log)
    try:
        dropped = files.repair()
        if dropped is not None:
            files.log(f"items.jsonl: dropped a truncated last line (redone now): {dropped[:120]!r}")
            meta["starts"].append({"time": datetime.now(UTC).isoformat(timespec="seconds"),
                                   "kind": "repair", "dropped": dropped[:300]})  # fmt: skip
        version = server.start()
        meta["starts"].append({"time": datetime.now(UTC).isoformat(timespec="seconds"),
                               "kind": "resume" if resume else "start", "ollama_version": version,
                               "already_done": len(files.done())})  # fmt: skip
        atomic_write(files.meta, json.dumps(meta, indent=1))
        files.log(
            f"{'RESUME' if resume else 'START'} {run_id}: {len(files.done())} items already done"
        )
        llm = OllamaLLM(model=cfg["frozen"]["model"], host=f"http://{cfg['ollama']['host']}",
                        timeout=float(cfg["ollama"]["timeout_s"]))  # fmt: skip
        llm.check()
        "".join(llm.stream([{"role": "user", "content": "ok"}], 5))  # warm-up, not counted
        files.log("warm-up call done (not counted)")
        audit = SQLiteAuditStore(cfg["audit_db"])
        made = generate(cfg, files, demo_read(cfg, audit), llm, audit, restart_server=server.restart,
                        server_is_down=server.is_down)  # fmt: skip
        files.log(f"generated {made} items now; {len(files.done())} of {len(items(cfg))} done")
        if len(files.done()) == len(items(cfg)):
            counts = build_sheets(
                files, REPO / cfg["preregistration"], int(cfg["sheets"]["shuffle_seed"])
            )
            files.log(f"sheets written: {counts}")
    finally:
        server.stop()
        lock.unlink(missing_ok=True)
        hits = truncation_hits(files.dir / "ollama.log")  # D227b: prompts cut by num_ctx
        files.log(f"ollama.log: {len(hits)} 'truncated=1' warnings")
        meta.setdefault("truncation_hits", []).extend(hits)
        atomic_write(files.meta, json.dumps(meta, indent=1))
    return files.dir


def truncation_hits(log_path: Path) -> list[str]:
    if not Path(log_path).exists():
        return []
    text = Path(log_path).read_text("utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if "truncated=1" in line]


def sheets_for(cfg: dict, run_id: str) -> dict:
    """--sheets: only for a finished run (D227b)."""
    files = RunFiles(Path(cfg["output_root"]) / run_id)
    if not files.meta.exists():
        raise E20Refused(f"no run {run_id} under {cfg['output_root']}")
    done, want = len(files.done()), len(items(cfg))
    if done != want:
        raise E20Refused(f"run {run_id} is not finished ({done} of {want} items): resume it first")
    return build_sheets(files, REPO / cfg["preregistration"], int(cfg["sheets"]["shuffle_seed"]))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--resume", default=None)
    g.add_argument("--sheets", default=None)
    g.add_argument("--check", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.check:
        guard(cfg)
        host, _, port = cfg["ollama"]["host"].partition(":")
        busy = port_in_use(host, int(port))
        print(
            f"guards passed; port {cfg['ollama']['host']} {'IN USE (stop Ollama first)' if busy else 'free'}"
        )
        return
    if a.sheets:
        print(sheets_for(cfg, a.sheets))
        return
    print(run(cfg, a.config, a.resume))


if __name__ == "__main__":
    main(sys.argv[1:])
