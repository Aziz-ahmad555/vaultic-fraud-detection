"""The scenario suite (Phase B): every scenario x seeds 0-4 x intensities low / medium / high.

    python -m vaultic.scenarios.suite            # writes E:/dev-cache/scenarios/suite/

Each run is saved (summary.json, rows.parquet) and the suite summary aggregates, per scenario,
intensity and threshold set (D196 primary, D195 sensitivity) and per decision mode (rules only,
model only, rules + model):

  recall    alerted synthetic fraud rows as a count over all seeds, with the per-seed min-max
            (alert = step-up or higher, D196 / D159) and the same for hard actions (hold or block)
  FPR       on the real legitimate replay rows: mean over seeds with min-max, alert and hard
  budgets   D191 checks, the number of seeds that pass
  bust-out  campaigns alerted before the first cash-out (D190), build-up alerts

Synthetic scenario - not the evaluation. Single-seed runs elsewhere are examples only.
"""

from __future__ import annotations

import json
import time

import numpy as np

from vaultic.scenarios import LABEL

SEEDS = (0, 1, 2, 3, 4)
MODES = ("rules_only", "model_only", "rules_plus_model")


def suite_dir(generator: str):
    """Where the suite of a generator version is written (dev cache, never the repository)."""
    from vaultic.live.store import cache_root

    return cache_root() / "scenarios" / f"suite_gen_{generator}"


def aggregate(runs: list[dict]) -> dict:
    """runs: summaries of one scenario x intensity (one per seed)."""
    out = {}
    for set_name in runs[0]["results"]:
        res = [r["results"][set_name] for r in runs]
        n = [x["n_synthetic_fraud"] for x in res]
        entry = {"n_seeds": len(runs), "synthetic_fraud_rows": int(sum(n))}
        for mode in MODES:
            m = [x[mode] for x in res]
            row = {}
            for kind, key, fkey in (("alert", "alerts_synthetic", "fpr_legit"),
                                    ("hard", "hard_alerts_synthetic", "hard_fpr_legit")):  # fmt: skip
                hits = [x[key] for x in m]
                fpr = [x[fkey] for x in m]
                row[kind] = {"recall_count": f"{sum(hits)}/{sum(n)}",
                             "recall_per_seed_min_max": [
                                 f"{hits[i]}/{n[i]}" for i in (int(np.argmin(np.array(hits) / n)),
                                                              int(np.argmax(np.array(hits) / n)))],
                             "fpr_mean": float(np.mean(fpr)),
                             "fpr_min_max": [float(min(fpr)), float(max(fpr))]}  # fmt: skip
            su = [x["stepup_only_fpr_legit"] for x in m]
            row["stepup_only_fpr_mean"] = float(np.mean(su))
            row["stepup_only_fpr_min_max"] = [float(min(su)), float(max(su))]
            if "alerted_before_first_cash_out" in m[0]:
                row["campaigns_alerted_before_cash_out"] = (
                    f"{sum(x['alerted_before_first_cash_out'] for x in m)}/{len(m)}"
                )
                row["build_up_alerts"] = (f"{sum(x['build_up_alerts'] for x in m)}/"
                                          f"{sum(x['build_up_rows'] for x in m)}")  # fmt: skip
            entry[mode] = row
        added = [x["added_fpr"] for x in res]
        entry["added_fpr_max"] = float(max(added))
        entry["budget_added_fpr_ok_seeds"] = (
            f"{sum(x['budget_added_fpr_ok'] for x in res)}/{len(res)}"
        )
        entry["budget_rules_only_ok_seeds"] = (
            f"{sum(x['budget_rules_only_ok'] for x in res)}/{len(res)}"
        )
        out[set_name] = entry
    return out


def markdown(summary: dict) -> str:
    lines = [f"# Attack scenario suite (generator {summary.get('generator', 'v1')}) - {LABEL}", "",
             "Seeds 0-4 per cell. Recall: alerted synthetic fraud rows, total over seeds "
             "(per-seed min - max). FPR: real legitimate replay rows, mean (min - max). Alert = "
             "step-up or higher (D196); hard = hold or block (extra reporting). A step-up is an "
             "extra customer check (friction), not a block; the FPR is split into step-up only and "
             "hold/block. 'D196 without R3 (post hoc)' was chosen after seeing the results (D201) "
             "and is a sensitivity row only.", ""]  # fmt: skip
    for sc, by_int in summary["scenarios"].items():
        lines.append(f"## {sc}")
        note = by_int.get("note")
        if note:
            lines.append(f"\n{note}\n")
        for inten, agg in by_int.items():
            if inten == "note":
                continue
            for set_name, e in agg.items():
                lines.append(f"\n**{inten}, {set_name}** - budgets: added FPR ok "
                             f"{e['budget_added_fpr_ok_seeds']} seeds (max {e['added_fpr_max']:+.4f}), "
                             f"rules only <= 2% {e['budget_rules_only_ok_seeds']}\n")  # fmt: skip
                lines.append("| mode | recall (alert) | FPR (alert) | FPR step-up only |"
                             " recall (hard) | FPR hold/block | extra |")  # fmt: skip
                lines.append("|---|---|---|---|---|---|---|")
                for mode in MODES:
                    r = e[mode]
                    a, h = r["alert"], r["hard"]
                    extra = ""
                    if "campaigns_alerted_before_cash_out" in r:
                        extra = (f"alerted before cash-out {r['campaigns_alerted_before_cash_out']};"
                                 f" build-up alerts {r['build_up_alerts']}")  # fmt: skip
                    lines.append(
                        f"| {mode} | {a['recall_count']} ({' - '.join(a['recall_per_seed_min_max'])})"
                        f" | {a['fpr_mean']:.4f} ({a['fpr_min_max'][0]:.4f} - {a['fpr_min_max'][1]:.4f})"
                        f" | {r['stepup_only_fpr_mean']:.4f} ({r['stepup_only_fpr_min_max'][0]:.4f} -"
                        f" {r['stepup_only_fpr_min_max'][1]:.4f})"
                        f" | {h['recall_count']} ({' - '.join(h['recall_per_seed_min_max'])})"
                        f" | {h['fpr_mean']:.4f} ({h['fpr_min_max'][0]:.4f} - {h['fpr_min_max'][1]:.4f})"
                        f" | {extra} |")  # fmt: skip
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    from vaultic.scenarios.generators import BY_VERSION, INTENSITIES, MULE_DEVICE_NOTE
    from vaultic.scenarios.run import load_all, run_scenario, save

    p = argparse.ArgumentParser(description=f"scenario suite ({LABEL})")
    p.add_argument("--generator", default="v2", choices=sorted(BY_VERSION))
    p.add_argument("--scenarios", nargs="*", default=list(BY_VERSION["v2"]))
    a = p.parse_args()
    generators = BY_VERSION[a.generator]
    root = suite_dir(a.generator)
    started = time.perf_counter()
    system, store, stored = load_all()
    print(f"loaded in {time.perf_counter() - started:.0f} s", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    summary = {"label": LABEL, "generator": a.generator, "seeds": list(SEEDS), "scenarios": {}}
    for name in a.scenarios:
        by_int = {}
        if name == "mule_ring":
            by_int["note"] = MULE_DEVICE_NOTE
        for inten in INTENSITIES:
            runs = []
            for seed in SEEDS:
                sc = generators[name](store, seed=seed, intensity=inten)
                res = run_scenario(store, system, stored, sc, log=lambda m: None)
                save(res, root / f"{name}_s{seed}_{inten}")
                runs.append({k: v for k, v in res.items() if k not in ("rows", "traces")})
                print(f"[{time.perf_counter() - started:6.0f}s] {name} {inten} seed {seed}: "
                      f"{res['n_synthetic']} synthetic, {res['n_legit_live']} legit live, "
                      f"{res['seconds']} s", flush=True)  # fmt: skip
            by_int[inten] = aggregate(runs)
        summary["scenarios"][name] = by_int
        (root / "summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
        (root / "summary.md").write_text(markdown(summary), "utf-8")
    print(markdown(summary))
    print(f"done in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
