"""POST-HOC / EXPLORATORY sensitivity analysis: F3 with a bounded history count (D119).

Run:  python -m vaultic.fusion.f3_sensitivity     (needs the inputs cached by f3_diagnosis)

Not a pre-registered experiment and not a replacement result. After the EXP-200-final final run
the F3 diagnosis (D117 / D118) traced F3's test collapse to the raw context input
ctx_hist_n_past. This asks how F3 does when that one input is log-transformed and capped at its
training maximum: x -> log1p(min(x, max over the gate rows)). Everything else is F3 as in the
final run: the D95-tuned C (not re-tuned), seeds 0-4, the same gate rows (fit + tune), the same
other inputs. F3-log is evaluated on the test rows with the E10 metrics (its cost threshold
chosen on the calibrate_fused rows, as for every method) and compared, paired on the same 1,000
bootstrap resamples, with the pre-registered F3 and MVAF (F4 and B5 for reference; their scores
and thresholds are the stored final-run ones). P-values are unadjusted and exploratory. Writes
research/f3_log_sensitivity.md and research/tables/f3_log_sensitivity.json; changes no reported
number.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from vaultic.fusion.f3_diagnosis import CACHE, CONTEXT, FINAL_RUN, SEEDS, subgroup_masks
from vaultic.paths import MERGED_PATH, RESEARCH_DIR

OUT_MD = RESEARCH_DIR / "f3_log_sensitivity.md"
OUT_JSON = RESEARCH_DIR / "tables" / "f3_log_sensitivity.json"
HIST = CONTEXT.index("ctx_hist_n_past")


def bounded_history(context: np.ndarray, cap: float) -> np.ndarray:
    """The context with ctx_hist_n_past -> log1p(min(count, cap)); other columns unchanged."""
    out = np.asarray(context, dtype=np.float64).copy()
    out[:, HIST] = np.log1p(np.minimum(out[:, HIST], cap))
    return out


def fit_f3(views, context, y) -> list:
    from vaultic.fusion.baselines import make_fusion

    tuned = json.loads((RESEARCH_DIR / "tables" / "fusion_d95.json").read_text("utf-8"))["tuned"]
    return [make_fusion("F3", seed=s, **tuned["F3"]["params"]).fit(views, y, context)
            for s in SEEDS]  # fmt: skip


def main() -> None:
    from vaultic.eval.metrics import choose_cost_threshold, pr_auc
    from vaultic.fusion.final_eval import e10_table

    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} missing: run python -m vaultic.fusion.f3_diagnosis first")
    d = dict(np.load(CACHE))
    cap = float(d["gate_context"][:, HIST].max())

    pre = fit_f3(d["gate_views"], d["gate_context"], d["gate_y"])  # pre-registered F3
    log = fit_f3(d["gate_views"], bounded_history(d["gate_context"], cap), d["gate_y"])
    ctx_t, ctx_c = bounded_history(d["test_context"], cap), bounded_history(d["cal_context"], cap)
    seeds_pre = [m.predict_proba(d["test_views"], d["test_context"]) for m in pre]
    seeds_log = [m.predict_proba(d["test_views"], ctx_t) for m in log]
    s_log = np.mean(seeds_log, axis=0)
    s_log_cal = np.mean([m.predict_proba(d["cal_views"], ctx_c) for m in log], axis=0)

    stored = pd.read_parquet(FINAL_RUN / "predictions.parquet").set_index("TransactionID")
    ids, y = d["test_ids"], d["test_y"]
    scores = {"F3-log (post-hoc)": s_log}
    scores.update(
        {m: stored[f"score_{m}"].reindex(ids).to_numpy() for m in ("F3", "MVAF", "F4", "B5")}
    )
    if np.max(np.abs(scores["F3"] - np.mean(seeds_pre, axis=0))) > 1e-9:
        raise RuntimeError(
            "the refitted pre-registered F3 does not match the stored final-run scores"
        )

    amt = pd.read_parquet(MERGED_PATH, columns=["TransactionID", "TransactionAmt"])
    amt = amt.set_index("TransactionID")["TransactionAmt"]
    a_test = amt.reindex(ids).to_numpy(float)
    a_cal = amt.reindex(d["cal_ids"]).to_numpy(float)
    final_meta = json.loads((RESEARCH_DIR / "tables" / "final_results.json").read_text("utf-8"))[
        "meta"
    ]
    thresholds = {m: final_meta["thresholds"][m] for m in ("F3", "MVAF", "F4", "B5")}
    thresholds["F3-log (post-hoc)"] = choose_cost_threshold(d["cal_y"], s_log_cal, a_cal)
    per_seed = {"F3-log (post-hoc)": seeds_log, "F3": seeds_pre}
    methods, comps = e10_table(y, scores, per_seed, a_test, thresholds,
                               reference="F3-log (post-hoc)", n_boot=1000, seed=0)  # fmt: skip

    masks = subgroup_masks(d["test_views"], d["test_context"])
    subs = [{"subgroup": g, "rows": int(m.sum()), "frauds": int(y[m].sum()),
             **{k: float(pr_auc(y[m], s[m])) for k, s in scores.items()}}
            for g, m in masks.items() if 0 < y[m].sum() < m.sum()]  # fmt: skip
    day = d["test_day"]
    blocks = []
    for first in range(int(day.min()), int(day.max()) + 1, 7):
        m = (day >= first) & (day < first + 7)
        blocks.append({"days": f"{first}-{min(first + 6, int(day.max()))}",
                       **{k: float(pr_auc(y[m], scores[k][m]))
                          for k in ("F3-log (post-hoc)", "F3", "MVAF")}})  # fmt: skip
    coef = float(np.mean([m.model_[-1].coef_[0][len(d["gate_views"][0]) + HIST] for m in log]))
    result = {"generated": date.today().isoformat(), "post_hoc": True, "cap": cap,
              "f3_log_coef_hist": coef, "methods": methods.to_dict("records"),
              "comparisons": comps.to_dict("records"), "subgroups": subs, "blocks": blocks}  # fmt: skip
    OUT_JSON.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    write_md(result)
    print(f"wrote {OUT_MD} and {OUT_JSON}")


def write_md(r: dict) -> None:
    lines = [
        "# F3 with a bounded history count: sensitivity analysis (POST-HOC / EXPLORATORY)", "",
        "**POST-HOC / EXPLORATORY, not a replacement result.** Not pre-registered; designed after "
        "the EXP-200-final final run and the F3 diagnosis (research/f3_collapse_diagnosis.md, "
        "D117 / D118). The pre-registered F3 and every number in research/tables/final_e10.md - "
        "final_e12.md stand as reported. P-values below are unadjusted and exploratory. "
        f"Generated {r['generated']} by `python -m vaultic.fusion.f3_sensitivity` (D119).", "",
        f"F3-log = F3 with ctx_hist_n_past replaced by log1p(min(count, {r['cap']:.0f})), "
        f"{r['cap']:.0f} being the count's maximum on the gate rows; same D95-tuned C (not "
        "re-tuned), seeds 0-4, gate rows (fit + tune) and other inputs as the pre-registered F3; "
        "cost threshold chosen on the calibrate_fused rows. F3, MVAF, F4 and B5 are the stored "
        "final-run scores and thresholds. Test period, 92,427 rows, 3,213 frauds.", "",
        "| method | PR-AUC [95% CI] | recall@1%FPR [95% CI] | cost ($) [95% CI] |", "|---|---|---|---|",
    ]  # fmt: skip
    for m in r["methods"]:
        lines.append(f"| {m['method']} | {m['pr_auc']:.4f} [{m['pr_auc_ci_low']:.4f}, "
                     f"{m['pr_auc_ci_high']:.4f}] | {m['recall_at_1pct_fpr']:.4f} "
                     f"[{m['recall_at_1pct_fpr_ci_low']:.4f}, {m['recall_at_1pct_fpr_ci_high']:.4f}] | "
                     f"{m['cost']:,.0f} [{m['cost_ci_low']:,.0f}, {m['cost_ci_high']:,.0f}] |")  # fmt: skip
    lines += ["", "Paired comparisons (F3-log minus method, same 1,000 resamples; for cost, positive "
              "= F3-log costs more; unadjusted p, exploratory):", "",
              "| metric | comparison | diff | 95% CI | p |", "|---|---|---|---|---|"]  # fmt: skip
    for c in r["comparisons"]:
        fmt = ",.0f" if c["metric"] == "cost" else "+.4f"
        lines.append(
            f"| {c['metric']} | {c['comparison']} | {c['diff']:{fmt}} | "
            f"[{c['ci_low']:{fmt}}, {c['ci_high']:{fmt}}] | {c['p_value']:.3f} |"
        )
    names = [k for k in r["subgroups"][0] if k not in ("subgroup", "rows", "frauds")]
    lines += ["", "## PR-AUC by subgroup", "",
              "| subgroup | rows | frauds | " + " | ".join(names) + " |",
              "|---|---|---|" + "---|" * len(names)]  # fmt: skip
    for s in r["subgroups"]:
        lines.append(f"| {s['subgroup']} | {s['rows']} | {s['frauds']} | "
                     + " | ".join(f"{s[k]:.4f}" for k in names) + " |")  # fmt: skip
    lines += ["", "## PR-AUC by 7-day block", "", "| days | F3-log (post-hoc) | F3 | MVAF |",
              "|---|---|---|---|"]  # fmt: skip
    for b in r["blocks"]:
        lines.append(
            f"| {b['days']} | {b['F3-log (post-hoc)']:.4f} | {b['F3']:.4f} | " f"{b['MVAF']:.4f} |"
        )
    lines += ["", "## Reading (added by hand, not generated)", "", "_To be written._", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
