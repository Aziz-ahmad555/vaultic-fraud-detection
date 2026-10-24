"""CPU speed test of local models served by Ollama (D222, D223). Not a research number.

Run (Ollama running on 127.0.0.1:11434, Test Lab off, machine idle):
    python tools/llm_speed_test.py qwen2.5:7b-instruct-q4_K_M [more models] [--prompts b_full,b_trim]

Per model: one warm-up call (model load, reported separately), then REPEATS runs of each prompt.
Every run starts with a fresh run id and fresh evidence values, so Ollama's prompt cache can never
serve a repeat (D223: the first version repeated identical prompts and its medians were mostly
cached runs). Prompts:
  a_short    a short question without evidence
  b_full     a typical copilot turn with the full evidence block (about 1,800 tokens), 160 tokens out
  b_trim     the same question with only the evidence it needs (about 500-700 tokens), 160 out (D223)
  c_summary  a case summary from the full evidence (about 2,900 tokens), up to 320 out
plus 10 tool-call trials (does the model return a valid call of get_case?). Reads Ollama's timings
(prompt and generation tokens per second, load time) and the wall time; memory is read from the
llama-server runner process after each run (Windows; None elsewhere). Writes
VAULTIC_CACHE_DIR/copilot/speed_test_<stamp>.json (default E:/dev-cache), never committed. Standard
library only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

HOST = "http://127.0.0.1:11434"
REPEATS = 3
TOOL_TRIALS = 10
CACHE = Path(os.environ.get("VAULTIC_CACHE_DIR") or "E:/dev-cache") / "copilot"
RUNNER = "llama-server"  # the process that holds the loaded model (Ollama 0.40)

RULES = (
    "You are an investigation assistant for fraud analysts. Answer ONLY from the evidence below. "
    "End every sentence with the evidence id in square brackets, e.g. [case:1/decision]. If the "
    "evidence does not contain the answer, say 'not available'. Never change a decision or label."
)
VIEWS = ["tabular", "behavioral", "temporal", "graph", "anomaly"]
QUESTION = "Question: Why was this transaction flagged? Answer in at most 5 sentences."


def evidence(rows: int, rng: random.Random) -> str:
    """A case's evidence block with fresh ids and values (never the same text twice)."""
    tid, uid = rng.randint(3_000_000, 3_600_000), rng.randint(1, 250_000)
    head = (f"[case:{tid}/decision] transaction {tid}, amount {rng.uniform(5, 900):.2f}, product "
            f"{rng.choice('WCRHS')}, day {rng.randint(151, 182)} at {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}, "
            f"risk {rng.randint(1, 99)} of 100, E12 action {rng.choice(['step-up', 'hold'])}, conformal set "
            f"{rng.choice(['fraud', 'uncertain'])}, view disagreement {rng.uniform(0, 0.4):.2f}.\n"
            f"[case:{tid}/customer] customer {uid} has {rng.randint(0, 80)} earlier transactions, first seen "
            f"day {rng.randint(1, 150)}.\n")  # fmt: skip
    body = []
    for i in range(rows):
        v = VIEWS[i % 5]
        body.append(f"[case:{tid}/view:{v}:{i}] view {v}: calibrated score {rng.random():.3f}, gate weight "
                    f"{rng.random():.2f}, available yes; reason: the {v} view "
                    f"{rng.choice(['raises', 'lowers'])} the risk because of its inputs observed before the "
                    f"transaction time.\n")  # fmt: skip
    return head + "".join(body)


def messages(name: str, rng: random.Random) -> tuple[list, dict]:
    run = f"[run:{uuid.uuid4().hex}] "  # first tokens differ every run: no cached prefix
    if name == "a_short":
        return [
            {
                "role": "user",
                "content": run + "In one sentence, what is a conformal prediction set?",
            }
        ], {"num_predict": 60}
    if name == "b_full":
        return ([{"role": "system", "content": run + RULES},
                 {"role": "user", "content": "Evidence:\n" + evidence(28, rng) + "\n" + QUESTION}], {"num_predict": 160})  # fmt: skip
    if name == "b_trim":  # only what "why flagged" needs: decision, customer, five views (D223)
        return ([{"role": "system", "content": run + RULES},
                 {"role": "user", "content": "Evidence:\n" + evidence(5, rng) + "\n" + QUESTION}], {"num_predict": 160})  # fmt: skip
    if name == "c_summary":
        return ([{"role": "system", "content": run + RULES},
                 {"role": "user", "content": "Evidence:\n" + evidence(48, rng) + "\nWrite a case summary for the case file "
                  "in about 10 sentences."}], {"num_predict": 320})  # fmt: skip
    raise ValueError(name)


TOOL = {"type": "function", "function": {
    "name": "get_case", "description": "Read one case's decision, amount, product and time.",
    "parameters": {"type": "object", "properties": {"id": {"type": "integer", "description": "TransactionID"}},
                   "required": ["id"]}}}  # fmt: skip


def runner_memory_gb() -> float | None:
    """Working set of the llama-server runner (Windows), None if not available."""
    if os.name != "nt":
        return None
    cmd = f"(Get-Process -Name {RUNNER} -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum"
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return round(int(out.stdout.strip()) / 1024**3, 2) if out.stdout.strip() else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def chat(model: str, msgs: list, options: dict | None = None, tools: list | None = None) -> dict:
    body = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "options": {"temperature": 0, **(options or {})},
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{HOST}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    out["wall_s"] = time.perf_counter() - started
    return out


def rates(r: dict) -> dict:
    ns = 1e9
    return {"prompt_tokens": r.get("prompt_eval_count", 0), "gen_tokens": r.get("eval_count", 0),
            "prompt_tok_s": r.get("prompt_eval_count", 0) / max(r.get("prompt_eval_duration", 1) / ns, 1e-9),
            "gen_tok_s": r.get("eval_count", 0) / max(r.get("eval_duration", 1) / ns, 1e-9),
            "load_s": r.get("load_duration", 0) / ns, "wall_s": r["wall_s"],
            "runner_gb": runner_memory_gb()}  # fmt: skip


def tool_trial(model: str) -> bool:
    msg = [
        {"role": "user", "content": f"[run:{uuid.uuid4().hex}] Use the tool to read case 3527980."}
    ]
    calls = chat(model, msg, {"num_predict": 80}, [TOOL]).get("message", {}).get("tool_calls") or []
    return any(c.get("function", {}).get("name") == "get_case"
               and str(c.get("function", {}).get("arguments", {}).get("id")) == "3527980" for c in calls)  # fmt: skip


def run(model: str, prompts: list[str], rng: random.Random, tools: bool) -> dict:
    print(f"== {model}: warm-up (model load)", flush=True)
    out = {"warmup": rates(chat(model, *messages("a_short", rng))), "prompts": {}}
    for name in prompts:
        reps = []
        for i in range(REPEATS):
            reps.append(rates(chat(model, *messages(name, rng))))
            r = reps[-1]
            print(f"   {name} run {i + 1}: wall {r['wall_s']:.1f} s, prompt {r['prompt_tok_s']:.1f} tok/s, gen "
                  f"{r['gen_tok_s']:.2f} tok/s ({r['prompt_tokens']} in, {r['gen_tokens']} out), runner "
                  f"{r['runner_gb']} GB", flush=True)  # fmt: skip
        out["prompts"][name] = {"runs": reps, "median_wall_s": statistics.median(x["wall_s"] for x in reps),
                                "max_wall_s": max(x["wall_s"] for x in reps)}  # fmt: skip
    if tools:
        started = time.perf_counter()
        ok = [tool_trial(model) for _ in range(TOOL_TRIALS)]
        out["tool_calls"] = {"valid": sum(ok), "trials": TOOL_TRIALS,
                             "mean_wall_s": (time.perf_counter() - started) / TOOL_TRIALS}  # fmt: skip
        print(f"   tool calls: {sum(ok)}/{TOOL_TRIALS} valid", flush=True)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("models", nargs="+")
    p.add_argument("--prompts", default="a_short,b_full,b_trim,c_summary")
    p.add_argument("--no-tools", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    result = {"machine": {"cpu": platform.processor(), "cores": os.cpu_count(), "python": platform.python_version()},
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "repeats": REPEATS, "cache_free": True, "models": {}}  # fmt: skip
    for m in a.models:
        result["models"][m] = run(m, a.prompts.split(","), rng, not a.no_tools)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"speed_test_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=1), "utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
