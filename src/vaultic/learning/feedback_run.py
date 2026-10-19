"""EXP-310 runner (D210, D211): the design run EXP-310-dev.

    python -m vaultic.learning.feedback_run experiments/configs/EXP-310-dev.yaml

Days 1-120 only. For every delay L and seed s: the champion (B5 configuration, seed s, fitted on
days 1-56 with the stored 30-day inputs, as the frozen B5 was) serves from day 61 with the arm's
inputs; its E12 thresholds and P2 reference are chosen on days 57-60 with the arm's inputs; all
policies are simulated in one pass (learning/feedback.py). Outputs under
experiments/runs/EXP-310-dev/<timestamp>/ (git-ignored): config.yaml, metrics.json, events.json,
served.parquet, report.md. EXP-310-final is refused here (Aziz's go-ahead and the final-run
guards first).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vaultic.learning.feedback import POLICIES, Data, evaluate, make_model, simulate


def fit_b5(params: dict, seed: int, X, y):
    from vaultic.views.tabular import make_model as make_estimator

    return make_estimator("xgboost", params, seed).fit(X, y)


def weekly_pr_auc(datas, served_by_seed, policies, first, last, width=7) -> list[dict]:
    from vaultic.eval.metrics import pr_auc

    out = []
    for lo in range(first, last + 1, width):
        hi = min(lo + width - 1, last)
        for pol in policies:
            vals = []
            for s, f in served_by_seed.items():
                g = f[(f["policy"] == pol) & (f["day"] >= lo) & (f["day"] <= hi)]
                y = datas[s].y[g["row"].to_numpy()]
                vals.append(pr_auc(y, g["p"].to_numpy()) if 0 < y.sum() < len(y) else np.nan)
            out.append({"policy": pol, "days": [lo, hi], "pr_auc": float(np.nanmean(vals))})
    return out


def main(argv=None) -> None:
    from vaultic.learning.feedback_inputs import (
        DEV_LAST_DAY,
        inputs_for_delay,
        load,
        require_last_day,
    )
    from vaultic.live.store import cache_root
    from vaultic.paths import CONFIG_DIR, REPO_ROOT

    argv = argv if argv is not None else sys.argv[1:]
    cfg_path = Path(argv[0]) if argv else CONFIG_DIR / "EXP-310-dev.yaml"
    cfg = yaml.safe_load(cfg_path.read_text("utf-8"))
    if cfg.get("status") != "dev":
        raise SystemExit(
            "only the design run (status dev) runs here; EXP-310-final needs Aziz's go-ahead"
        )
    params = yaml.safe_load((CONFIG_DIR / f"{cfg['model_config']}.yaml").read_text("utf-8"))[
        "model"
    ]["params"]
    started = time.perf_counter()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / "experiments" / "runs" / cfg["id"] / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), "utf-8")

    def log(msg: str) -> None:
        line = f"[{time.perf_counter() - started:7.0f}s] {msg}"
        print(line, flush=True)
        with open(out_dir / "run.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    loaded = load("dev")
    df = loaded["df"]
    require_last_day(df["day"].to_numpy(), DEV_LAST_DAY)
    log(f"loaded {len(df):,} rows, days {df['day'].min()}-{df['day'].max()}")
    day = df["day"].to_numpy()
    f_lo, f_hi = cfg["champion"]["fit_days"]
    c_lo, c_hi = cfg["champion"]["calibration_days"]
    fit_rows = np.flatnonzero((day >= f_lo) & (day <= f_hi))
    cal_rows = np.flatnonzero((day >= c_lo) & (day <= c_hi))
    X30 = inputs_for_delay(loaded, 30)
    y = df["isFraud"].to_numpy()
    # the recompute path must reproduce the stored 30-day columns before it is trusted at L != 30
    from vaultic.learning.feedback_inputs import label_derived_at
    from vaultic.views.definitions import LABEL_DERIVED

    re30 = label_derived_at(df, loaded["features"]["uid"], 30, loaded["splits"].train.contains(day))
    worst = max(float(np.nanmax(np.abs(re30[c].to_numpy(float) - X30[c].to_numpy(float))))
                for c in LABEL_DERIVED)  # fmt: skip
    same_nan = all(np.array_equal(np.isnan(re30[c].to_numpy(float)), np.isnan(X30[c].to_numpy(float)))
                   for c in LABEL_DERIVED)  # fmt: skip
    log(f"recompute check at L=30: max |diff| {worst:.2e}, same missing pattern {same_nan}")
    if worst > 1e-6 or not same_nan:
        raise SystemExit("the recomputed label-derived columns do not reproduce the stored ones")
    champions = {}
    for s in cfg["seeds"]:
        log(f"champion seed {s}: fitting on days {f_lo}-{f_hi}")
        champions[s] = fit_b5(params, s, X30.iloc[fit_rows], y[fit_rows])

    policies = tuple(cfg["policies"])
    assert set(policies) <= set(POLICIES)
    metrics, all_events, served_frames = {}, [], []
    cache = cache_root() / "feedback"
    for L in cfg["delays"]:
        X = inputs_for_delay(loaded, L, cache_dir=cache)
        log(f"L={L}: inputs ready")
        datas, served_by_seed = {}, {}
        for s in cfg["seeds"]:
            data = Data(df["TransactionDT"].to_numpy(), y, df["TransactionAmt"].to_numpy(), X, L)
            champ = make_model(f"champion_s{s}", champions[s], data, fit_rows, cal_rows, f_hi)
            served, events = simulate(data, champ, cfg["deploy_day"], cfg["end_day"],
                                      partial(fit_b5, params, s), policies,
                                      log=lambda m, s=s: log(f"seed {s}: {m}"))  # fmt: skip
            for e in events:
                e["seed"] = s
            all_events.extend(events)
            datas[s], served_by_seed[s] = data, served
            served_frames.append(served.assign(delay=L, seed=s))
            accepted = sum(e.get("accepted", False) for e in events if e["kind"] == "attempt")
            log(f"L={L} seed {s}: {len(events)} events, {accepted} promotions")
        lo, hi = cfg["evaluation_days"]
        metrics[f"L{L}"] = {"evaluation": evaluate(datas, served_by_seed, (lo, hi), policies,
                                                   n_boot=cfg["bootstrap"]["n"]),
                            "weekly": weekly_pr_auc(datas, served_by_seed, policies, lo, hi)}  # fmt: skip
        log(f"L={L}: evaluated")
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float), "utf-8")
        (out_dir / "events.json").write_text(json.dumps(all_events, indent=1, default=str), "utf-8")
    pd.concat(served_frames, ignore_index=True).to_parquet(out_dir / "served.parquet", index=False)
    (out_dir / "report.md").write_text(report(cfg, metrics, all_events), "utf-8")
    log(f"done in {time.perf_counter() - started:.0f} s; outputs in {out_dir}")


def report(cfg: dict, metrics: dict, events: list[dict]) -> str:
    lines = [f"# {cfg['id']} - design run (training period only; no hypothesis numbers)", "",
             f"Days {cfg['deploy_day']}-{cfg['end_day']}, champion fitted on days "
             f"{cfg['champion']['fit_days']}, seeds {cfg['seeds']}. PR-AUC = seed mean of the served "
             "scores (95% CI over rows); cost = E12 total cost (seed mean); vs P0 = paired "
             "bootstrap difference.", ""]  # fmt: skip
    for key, m in metrics.items():
        lines += [f"## {key}", "", "| policy | PR-AUC (95% CI) | vs P0 (95% CI) | cost | cost vs P0 (95% CI) | attempts | promotions |",
                  "|---|---|---|---|---|---|---|"]  # fmt: skip
        L = int(key[1:])
        for pol, e in m["evaluation"].items():
            att = [
                x
                for x in events
                if x["kind"] == "attempt" and x["policy"] == pol and x["delay"] == L
            ]
            prom = sum(x.get("accepted", False) for x in att)
            vs = e.get("vs_P0")
            cv = e.get("cost_vs_P0")
            lines.append(
                f"| {pol} | {e['pr_auc']:.4f} ({e['pr_auc_ci'][0]:.4f}-{e['pr_auc_ci'][1]:.4f}) | "
                + (f"{vs['diff']:+.4f} ({vs['ci_low']:+.4f} to {vs['ci_high']:+.4f})" if vs else "-")
                + f" | {e['cost']:,.0f} | "
                + (f"{cv['diff']:+,.0f} ({cv['ci_low']:+,.0f} to {cv['ci_high']:+,.0f})" if cv else "-")
                + f" | {len(att)} | {prom} |")  # fmt: skip
        reasons = {}
        for x in events:
            if x["kind"] == "attempt" and x["delay"] == L:
                reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
        alarms = sum(1 for x in events if x["kind"] == "alarm" and x["delay"] == L)
        lines += ["", f"Attempt outcomes (all policies, seeds): {reasons}; alarms: {alarms}.", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
