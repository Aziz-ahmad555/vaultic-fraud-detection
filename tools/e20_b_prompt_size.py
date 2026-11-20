"""B prompt size for E20 (D227b, D227c): builds the 30 system-B prompts from the frozen code exactly as
the runner does (full_evidence + b_messages, the case's own as-of, L = 30, the longest pre-registered
question or adversarial prompt) and writes research/tables/e20/b_prompt_size.json. Only the evidence
text is used: no outcome label or other column is read into the table. No model is called; Ollama
0.40.10 has no token-count endpoint, so tokens are a conservative ceil(chars / 3) estimate.

Run from the repository root, VAULTIC_DATA_DIR set:
    python tools/e20_b_prompt_size.py [--config experiments/configs/EXP-320.yaml]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHAT_TEMPLATE_ALLOWANCE = 30  # tokens of Qwen's chat template around one user message (generous)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO / "experiments/configs/EXP-320.yaml")
    p.add_argument("--out", type=Path, default=REPO / "research/tables/e20/b_prompt_size.json")
    a = p.parse_args(argv)
    from vaultic.audit.store import SQLiteAuditStore
    from vaultic.eval import e20

    cfg = e20.load_config(a.config)
    os.environ["VAULTIC_AUDIT_DB"] = str(cfg["audit_db"])  # the E20 audit DB, never the real one
    read = e20.demo_read(cfg, SQLiteAuditStore(cfg["audit_db"]))
    frame = read.frame
    prompts = list(cfg["questions"].values()) + [prompt for _, _, prompt in cfg["adversarial"]]
    longest = max(prompts, key=len)
    L = int(cfg["label_delay_days"])
    num_ctx, cap = int(cfg["decoding"]["num_ctx"]), int(cfg["decoding"]["max_tokens"])
    rows = []
    for case_no, _decision, _label, tid in cfg["cases"]:  # decision / label not used
        t = float(frame.loc[frame["TransactionID"] == int(tid), "TransactionDT"].iloc[0])
        lines = e20.full_evidence(read, int(tid), t, L)
        chars = len(e20.b_messages(lines, longest, cfg["system_b_instruction"])[0]["content"])
        rows.append({"case": int(case_no), "TransactionID": int(tid), "evidence_lines": len(lines),
                     "prompt_chars": chars, "tokens_est": math.ceil(chars / 3)})  # fmt: skip
    worst = max(r["tokens_est"] for r in rows)
    total = worst + CHAT_TEMPLATE_ALLOWANCE + cap
    out = {"rule": "D227b / D227c", "config": str(a.config.relative_to(REPO)).replace("\\", "/"),
           "estimate": "tokens_est = ceil(prompt_chars / 3) (conservative; no token-count endpoint)",
           "longest_prompt_used": longest, "num_ctx": num_ctx, "output_cap": cap,
           "chat_template_allowance": CHAT_TEMPLATE_ALLOWANCE, "max_tokens_est": worst,
           "max_total_with_allowance_and_cap": total, "margin_vs_num_ctx": num_ctx - total,
           "stop_line_tokens": 3800, "any_over_stop_line": worst > 3800, "cases": rows}  # fmt: skip
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n", "utf-8")
    print(f"max {worst} tokens (est.), with allowance and cap {total} of {num_ctx}; wrote {a.out}")


if __name__ == "__main__":
    main()
