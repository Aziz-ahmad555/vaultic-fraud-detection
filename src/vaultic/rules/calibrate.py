"""Choose each rule's threshold on validation and measure its effect there (D192, D195).

    python -m vaultic.rules.calibrate          # writes rules/thresholds_v1.json

Validation only (days 128-150 from the split file, real rows). Test-period rows are dropped
right after loading and never read further. For each rule the threshold is the most permissive
value (the one that fires most often) keeping that rule's own false-positive rate at or below
MAX_FPR: FPR = legitimate validation rows the rule fires on / all legitimate validation rows
(a missing input never fires). Candidates are the values the rule takes on legitimate rows.
If the rule cannot stay within budget at any of them, it fires on no legitimate row (threshold
just above their largest value).

The effect of each rule at its threshold (hits, precision, recall, FPR) and of all rules together
("rules only") is written next to the thresholds. Validation numbers, not the evaluation.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd

from vaultic.rules.engine import (
    ACTIONS,
    RULES,
    SENSITIVITY_FILE,
    SEVERITY,
    THRESHOLDS_FILE,
    rule_values,
)

MAX_FPR = 0.005
COLUMNS = {
    "base_features": ["uid_n_1h", "uid_n_24h"],
    "behavioral": ["ent_uids_7d_device", "hist_n_past", "new_device", "amt_ratio_median"],
    "graph": ["g_fraud_rate_card", "g_fraud_rate_device"],
}


def most_permissive(values: np.ndarray, legit: np.ndarray, max_fpr: float = MAX_FPR) -> float:
    """Smallest threshold t with (#legit rows with value >= t) / #legit <= max_fpr."""
    n_legit = int(legit.sum())
    v = values[legit]
    v = np.sort(v[np.isfinite(v)])
    allowed = int(np.floor(max_fpr * n_legit))
    if len(v) == 0:
        return float("inf")
    cands = np.unique(v)
    # count of legit values >= c, for each candidate c (v sorted ascending)
    counts = len(v) - np.searchsorted(v, cands, side="left")
    ok = cands[counts <= allowed]
    if len(ok):
        return float(ok.min())
    return float(np.nextafter(v.max(), np.inf))


def effect(fired: np.ndarray, y: np.ndarray) -> dict:
    tp = int((fired & (y == 1)).sum())
    fp = int((fired & (y == 0)).sum())
    return {"hits": int(fired.sum()), "true_positives": tp, "false_positives": fp,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(int((y == 1).sum()), 1),
            "fpr": fp / max(int((y == 0).sum()), 1)}  # fmt: skip


def load_validation() -> pd.DataFrame:
    from vaultic.data.splits import load_splits
    from vaultic.paths import FEATURES_DIR, MERGED_PATH

    splits = load_splits()
    variant = splits.uid_variant
    df = pd.read_parquet(MERGED_PATH, columns=["TransactionID", "TransactionDT", "TransactionAmt",
                                                "isFraud"])  # fmt: skip
    for stem, cols in COLUMNS.items():
        f = pd.read_parquet(FEATURES_DIR / f"{stem}_{variant}.parquet",
                            columns=["TransactionID", *cols])  # fmt: skip
        if not np.array_equal(f["TransactionID"].to_numpy(), df["TransactionID"].to_numpy()):
            raise ValueError(f"{stem} features are not aligned with the data")
        df[cols] = f[cols].to_numpy()
    day = (df["TransactionDT"] // 86400).to_numpy()
    keep = (day >= splits.validation.first) & (day <= splits.validation.last)
    out = df[keep].reset_index(drop=True)
    out.attrs["days"] = (splits.validation.first, splits.validation.last)
    return out


def calibrate(val: pd.DataFrame, max_fpr: float = MAX_FPR) -> dict:
    y = val["isFraud"].to_numpy()
    legit = y == 0
    rules, worst = {}, np.zeros(len(val), dtype=int)
    for rule in RULES:
        v = rule_values(val, rule)
        th = most_permissive(v, legit, max_fpr)
        fired = np.isfinite(v) & (v >= th)
        worst = np.where(fired, np.maximum(worst, SEVERITY[rule.action]), worst)
        rules[rule.rule_id] = {"name": rule.name, "inputs": list(rule.inputs),
                               "condition": rule.condition, "action": rule.action,
                               "threshold": th, "validation": effect(fired, y)}  # fmt: skip
    alert = worst >= SEVERITY["step_up"]
    return {
        "rules": rules,
        "rules_only": effect(alert, y),
        "actions_on_validation": {a: int((worst == SEVERITY[a]).sum()) for a in ACTIONS},
    }


def model_validation_actions() -> pd.DataFrame:
    """Model-only E12 actions (D159) on the validation rows, from the stage-2 development refit
    (stage2/system.py, which reproduces the stored validation scores, D179). Days 147-150 are the
    rows the E12 layer was fitted on (calibrate_fused), so the model is in-sample there.
    Cached under the dev cache (not committed)."""
    import joblib

    from vaultic.live.store import cache_root
    from vaultic.stage2.system import dev_system_path, load_parts, require_no_test

    out = cache_root() / "scenarios" / "model_validation_actions.parquet"
    if out.exists():
        return pd.read_parquet(out)
    parts = load_parts(dev=True)
    df, features = parts["df"], parts["features"]
    val = parts["splits"].validation.contains(df["day"].to_numpy())
    require_no_test(parts, df["day"].to_numpy()[val])
    d = df[val].reset_index(drop=True)
    f = features[val].reset_index(drop=True)
    system = joblib.load(dev_system_path("dev"))
    sc = system.score(f, system.context(d, f))
    action = system.decide(sc["risk"], sc["p"], d["TransactionAmt"].to_numpy(float))
    res = pd.DataFrame({"TransactionID": d["TransactionID"].to_numpy(), "e12_action": action})
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out, index=False)
    return res


def thresholds_at(val: pd.DataFrame, cap: float) -> dict[str, float]:
    legit = val["isFraud"].to_numpy() == 0
    return {r.rule_id: most_permissive(rule_values(val, r), legit, cap) for r in RULES}


def combined(val: pd.DataFrame, th: dict[str, float], model_alert: np.ndarray) -> dict:
    """Rules-only and rules + model alerts at thresholds th (an alert = step-up or higher)."""
    y = val["isFraud"].to_numpy()
    worst = np.zeros(len(val), dtype=int)
    per_rule = {}
    for rule in RULES:
        v = rule_values(val, rule)
        fired = np.isfinite(v) & (v >= th[rule.rule_id])
        worst = np.where(fired, np.maximum(worst, SEVERITY[rule.action]), worst)
        per_rule[rule.rule_id] = effect(fired, y)
    rules_alert = worst >= SEVERITY["step_up"]
    return {"per_rule": per_rule, "rules_only": effect(rules_alert, y),
            "model_only": effect(model_alert, y),
            "rules_plus_model": effect(rules_alert | model_alert, y)}  # fmt: skip


def common_cap(val: pd.DataFrame, model_alert: np.ndarray, start: float = 0.005,
               stop: float = 0.001, step: float = 0.0001, rules_budget: float = 0.02,
               added_budget: float = 0.005) -> tuple[float | None, list[dict]]:  # fmt: skip
    """D196: the largest common per-rule FPR cap c (start, start - step, ..., stop) with
    rules-only validation FPR <= rules_budget and FPR(rules + model) - FPR(model only) <=
    added_budget. Returns (c or None, the search log)."""
    log = []
    n = int(round((start - stop) / step))
    for i in range(n + 1):
        cap = round(start - i * step, 6)
        th = thresholds_at(val, cap)
        c = combined(val, th, model_alert)
        added = c["rules_plus_model"]["fpr"] - c["model_only"]["fpr"]
        ok = c["rules_only"]["fpr"] <= rules_budget and added <= added_budget
        log.append({"cap": cap, "rules_only_fpr": c["rules_only"]["fpr"],
                    "added_fpr": added, "ok": bool(ok)})  # fmt: skip
        if ok:
            return cap, log
    return None, log


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="rule thresholds on validation")
    parser.add_argument("--d196", action="store_true",
                        help="common per-rule cap meeting the D191 budgets (thresholds_v2.json)")  # fmt: skip
    if parser.parse_args().d196:
        return main_d196()
    val = load_validation()
    out = calibrate(val)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()  # fmt: skip
    data = {"version": 1, "decision": "D192, D195", "max_rule_fpr": MAX_FPR,
            "validation_days": list(val.attrs["days"]), "n_rows": len(val),
            "n_fraud": int(val["isFraud"].sum()), "code_commit": commit,
            "note": "validation numbers (real rows), not the evaluation; chosen before any scenario",
            **out}  # fmt: skip
    SENSITIVITY_FILE.write_text(json.dumps(data, indent=2) + "\n", "utf-8")
    for rid, r in out["rules"].items():
        e = r["validation"]
        print(
            f"{rid} {r['name']:<44} th={r['threshold']:<10.6g} hits={e['hits']:<6} "
            f"prec={e['precision']:.3f} rec={e['recall']:.4f} fpr={e['fpr']:.4f}"
        )
    e = out["rules_only"]
    print(
        f"rules only: hits={e['hits']} prec={e['precision']:.3f} rec={e['recall']:.4f} "
        f"fpr={e['fpr']:.4f}"
    )


def sensitivity_without_fit_days(val: pd.DataFrame, model_alert: np.ndarray) -> dict:
    """D196 sensitivity (reported only, never used): the same cap search on validation days
    before the E12 layer's fit days (calibrate_fused, 147-150), where the model is out of sample."""
    from vaultic.data.splits import load_splits

    first_fit = load_splits().calibrate_fused.first
    keep = (val["TransactionDT"] // 86400).to_numpy() < first_fit
    sub = val[keep].reset_index(drop=True)
    cap, log = common_cap(sub, model_alert[keep])
    out = {"days": [int(val.attrs["days"][0]), int(first_fit) - 1], "n_rows": len(sub),
           "n_fraud": int(sub["isFraud"].sum()), "common_rule_fpr_cap": cap, "search": log}  # fmt: skip
    if cap is not None:
        th = thresholds_at(sub, cap)
        c = combined(sub, th, model_alert[keep])
        out.update({"thresholds": th, "per_rule": c["per_rule"], "rules_only": c["rules_only"],
                    "model_only": c["model_only"], "rules_plus_model": c["rules_plus_model"]})  # fmt: skip
    return out


def main_d196() -> None:
    from vaultic.trust.decision import STEP_UP

    val = load_validation()
    m = model_validation_actions()
    pos = pd.Index(m["TransactionID"].to_numpy()).get_indexer(val["TransactionID"].to_numpy())
    if (pos < 0).any() or len(m) != len(val):
        raise ValueError("model actions do not cover the validation rows")
    model_alert = m["e12_action"].to_numpy()[pos] >= STEP_UP
    cap, log = common_cap(val, model_alert)
    for r in log:
        print(
            f"cap={r['cap']:.4f} rules_only_fpr={r['rules_only_fpr']:.4f} "
            f"added={r['added_fpr']:.4f} ok={r['ok']}"
        )
    if cap is None:
        raise SystemExit("no common cap >= 0.1% meets both budgets (D196): stop and report")
    th = thresholds_at(val, cap)
    c = combined(val, th, model_alert)
    sens = sensitivity_without_fit_days(val, model_alert)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()  # fmt: skip
    rules = {r.rule_id: {"name": r.name, "inputs": list(r.inputs), "condition": r.condition,
                         "action": r.action, "threshold": th[r.rule_id],
                         "validation": c["per_rule"][r.rule_id]} for r in RULES}  # fmt: skip
    data = {"version": 2, "decision": "D196 (amends D195; D195 thresholds kept as sensitivity)",
            "common_rule_fpr_cap": cap, "validation_days": list(val.attrs["days"]),
            "n_rows": len(val), "n_fraud": int(val["isFraud"].sum()), "code_commit": commit,
            "note": "validation numbers (real rows), not the evaluation; chosen before any scenario",
            "rules": rules, "rules_only": c["rules_only"], "model_only": c["model_only"],
            "rules_plus_model": c["rules_plus_model"],
            "added_fpr_over_model": c["rules_plus_model"]["fpr"] - c["model_only"]["fpr"],
            "search": log, "sensitivity_without_e12_fit_days": sens}  # fmt: skip
    THRESHOLDS_FILE.write_text(json.dumps(data, indent=2) + "\n", "utf-8")
    print(json.dumps({k: v for k, v in data.items() if k != "search"}, indent=1))


if __name__ == "__main__":
    main()
