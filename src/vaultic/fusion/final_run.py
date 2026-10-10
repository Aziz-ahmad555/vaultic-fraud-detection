"""The pre-registered Paper 2 final experiments E10, E11 and E12 (D101; details D102).

Run:  python -m vaultic.fusion.final_run --mode dev --smoke     pipeline check, validation only
      python -m vaultic.fusion.final_run --mode final            THE final run (test period)

Pipeline exactly as in D95 (no changes to views, gate or baselines):
  - view table on the cross-fitted plan; the fixed fold's ONE set of view models (tuned configs
    EXP-V-<view>) predicts validation and, in final mode only, the test period (D77);
  - per-view calibrators on calibrate_views (144-146); everything fitted on fused scores
    (thresholds, fused calibration, conformal, routing lambdas) on calibrate_fused (147-150);
  - MVAF and F3-F7 with the hyperparameters tuned in D95 (read from fusion_d95.json, not
    re-tuned), F1 / F2 as in D95, each fitted with seeds 0-4 on the gate rows (fit + tune);
  - B5 (EXP-009) and F0 (EXP-F0-inner, the D95 version) refitted here with their frozen
    hyperparameters and seeds 0-4 on days 1-120, so E11 can remove views from their inputs; in
    final mode the refitted B5 is checked against the stored Phase 2 B5 test predictions.

Evaluation rows: final mode = the test period (days 151-182), calibration rows = 147-150.
Dev mode (validation only, a pipeline check, never a result): evaluation rows = days 149-150,
calibration rows = 147-148, on a 1-in-10 uid sample with --smoke; reports go to
E:\\dev-cache\\tmp\\final_dev, not to research/.

Refit check (validation only):  python -m vaultic.fusion.final_run --mode refit-check
  refits B5 and F0 on the full days 1-120 (no view models) and compares them with the stored
  EXP-009 / EXP-F0-inner validation predictions (correlation, max |diff|). A --smoke dev run
  cannot make this check (its models see 1 in 10 uids); a full dev run makes it on its rows.

Final mode refuses to run unless research/frozen_final.md matches every frozen file, the
working tree is clean and HEAD is the frozen commit (only the freeze record and the two logs
may differ), and a second final run needs --rerun-reason (FINAL-RERUN rule, logged in
research/decisions.md). Every E10-E12 setting is read from EXP-200-final.yaml and checked
against the D101 / D102 values (EXPECTED) at startup; SHA-256 hashes of the data and feature
files and of the stored B5 / F0 predictions are recorded in metrics.json (D108).

Start record (D109): right after the guards and BEFORE any test data is scored or evaluated
(build_table, which loads the data and fits and scores every model, runs only after it), the
final run
writes experiments/runs/EXP-200-final/<ts>/metrics.json {"mode": "final", "status": "started"}
and a FINAL-STARTED line in research/experiment_log.md, and commits and pushes that line; if
the commit or push fails it stops. The guard counts a "started" run as a final run, so a crash
cannot be silently re-run. At the end metrics.json is completed (status "completed").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vaultic.eval.metrics import choose_cost_threshold
from vaultic.paths import CONFIG_DIR, REPO_ROOT, RESEARCH_DIR, RUNS_DIR

CONFIG = CONFIG_DIR / "EXP-200-final.yaml"
FROZEN = RESEARCH_DIR / "frozen_final.md"
DEV_OUT = Path("E:/dev-cache/tmp/final_dev")
GATES = ("MVAF", "F1", "F2", "F3", "F4", "F5", "F6", "F7")
E11_CONDITIONS = ("tabular", "behavioral", "temporal", "graph", "anomaly", "tabular only")
# D101 / D102 values of every setting the run reads from EXP-200-final.yaml (D108)
EXPECTED = {
    ("seeds",): [0, 1, 2, 3, 4],
    ("bootstrap", "n"): 1000,
    ("bootstrap", "seed"): 0,
    ("e10", "reference"): "MVAF",
    ("e10", "methods"): ["MVAF", "F0", "B5", "F1", "F2", "F3", "F4", "F5", "F6", "F7"],
    ("e10", "metrics"): ["pr_auc", "recall_at_1pct_fpr", "cost"],
    ("e10", "subgroup_versus"): ["F3", "F4"],
    ("e10", "h2_alpha"): 0.05,
    ("e11", "conditions"): list(E11_CONDITIONS),
    ("e11", "versus"): ["F3", "F4"],
    ("e12_methods",): ["MVAF", "F3", "B5"],
    ("e12", "conformal_targets"): [0.90, 0.95],
    ("e12", "ece_bins"): 15,
    ("e12", "min_isotonic_frauds"): 100,
    ("e12", "adaptive"): {"gamma": 0.05, "block_days": 7, "label_delay_days": 30},
    ("e12", "routing_k_per_day"): [50, 100, 200, 500],
    ("e12", "lambda_grid"): [0.0, 0.25, 0.5, 1.0, 2.0],
    ("e12", "phase8_gate_tolerance"): 0.02,
    ("e12", "sweep"): {"c_fp": [2, 5, 10, 20, 50], "step_up_success_rate": [0.7, 0.9, 1.0],
                       "legit_step_up_friction": [0.0, 1.0, 5.0], "c_step": [0.10, 0.50, 2.00]},
}  # fmt: skip
PUSH_TIMEOUT_S = 120
LOG_FILES = {"research/frozen_final.md", "research/experiment_log.md", "research/decisions.md"}


def check_settings(cfg: dict) -> None:
    """Every E10-E12 setting in the config must equal its D101 / D102 value (D108)."""
    from vaultic.eval.metrics import ECE_BINS
    from vaultic.fusion.final_eval import metric_fns

    bad = []
    for path, want in EXPECTED.items():
        got = cfg
        for k in path:
            got = got.get(k) if isinstance(got, dict) else None
        if got != want:
            bad.append(f"{'.'.join(path)} = {got!r} (D101/D102: {want!r})")
    if list(metric_fns(np.zeros(1), {})) != cfg["e10"]["metrics"]:
        bad.append("the E10 metric functions differ from e10.metrics")
    if ECE_BINS != cfg["e12"]["ece_bins"]:
        bad.append(
            f"ECE_BINS = {ECE_BINS} in eval/metrics.py, config says {cfg['e12']['ece_bins']}"
        )
    if bad:
        raise ValueError(
            "EXP-200-final settings differ from the pre-registration: " + "; ".join(bad)
        )


def check_git(frozen: Path, repo: Path = REPO_ROOT) -> str:
    """Final mode: a clean working tree whose HEAD is the commit recorded in the freeze record;
    only the freeze record itself and the two logs may differ from it (they are written after
    freezing: the record is committed after it is made, FINAL-STARTED / FINAL-RERUN lines are
    appended by the run). Returns HEAD."""

    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True)

    dirty = git("status", "--porcelain").stdout.strip()
    if dirty:
        raise RuntimeError(f"the working tree is not clean; commit or remove first:\n{dirty}")
    found = re.search(r"at git commit `([0-9a-f]{40})`", frozen.read_text("utf-8"))
    if not found:
        raise RuntimeError(f"{frozen} records no git commit")
    frozen_commit, head = found.group(1), git("rev-parse", "HEAD").stdout.strip()
    if git("merge-base", "--is-ancestor", frozen_commit, head).returncode != 0:
        raise RuntimeError(f"HEAD {head[:12]} does not descend from the frozen commit "
                           f"{frozen_commit[:12]}")  # fmt: skip
    changed = set(git("diff", "--name-only", frozen_commit, head).stdout.split())
    if changed - LOG_FILES:
        raise RuntimeError(f"files changed since the frozen commit {frozen_commit[:12]}: "
                           f"{sorted(changed - LOG_FILES)}; re-freeze (logged) first")  # fmt: skip
    return head


def record_start(run_dir: Path, head: str, experiment_log: Path, extra_logs=(),
                 repo: Path = REPO_ROOT, push: bool = True) -> None:  # fmt: skip
    """D109: mark the final run as started, committed and pushed; test data is not scored or
    evaluated before FINAL-STARTED.

    Any failing git step (commit, or a push that fails or hangs: network, auth) raises, so the
    run stops before any test data is scored or evaluated. The local start record stays (metrics.json "started"
    and the local commit), so the guard still counts the attempt; push it by hand before a
    logged re-run."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # never wait for a credential prompt

    def git(*a):
        try:
            r = subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, env=env,
                               timeout=PUSH_TIMEOUT_S)  # fmt: skip
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"git {' '.join(a)} timed out after {PUSH_TIMEOUT_S} s; the final "
                               "run stops before any test data is scored or evaluated") from e  # fmt: skip
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(a)} failed; the final run stops before any test data "
                               "is scored or evaluated:"
                               f"\n{r.stderr}")  # fmt: skip
        return r

    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(
        {"mode": "final", "status": "started", "git_commit": head,
         "started": datetime.now().isoformat(timespec="seconds")}, indent=1), "utf-8")  # fmt: skip
    rel = run_dir.resolve().relative_to(repo.resolve()).as_posix()
    with open(experiment_log, "a", encoding="utf-8") as f:
        f.write(f"| {run_dir.parent.name} | {date.today().isoformat()} | FINAL-STARTED: E10-E12 "
                f"final run started at commit `{head[:12]}` (D101); test data not scored or evaluated yet | run "
                f"`{rel}` | — |\n")  # fmt: skip
    logs = [experiment_log, *extra_logs]
    git("add", *[Path(p).resolve().relative_to(repo.resolve()).as_posix() for p in logs])
    git("commit", "-q", "-m", f"FINAL-STARTED {run_dir.parent.name} ({run_dir.name})")
    if push:
        git("push", "-q", "origin", "HEAD")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def input_hashes(paths) -> dict[str, str]:
    out = {}
    for p in paths:
        p = Path(p)
        try:
            name = p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            name = p.as_posix()
        out[name] = file_sha256(p)
    return out


def frozen_files(cfg: dict) -> list[Path]:
    """Every file the final run depends on; hashed into research/frozen_final.md."""
    files = [CONFIG, CONFIG_DIR / "splits.yaml", CONFIG_DIR / "v_columns.yaml",
             CONFIG_DIR / f"{cfg['b5_config']}.yaml", CONFIG_DIR / f"{cfg['f0_config']}.yaml",
             REPO_ROOT / cfg["fusion_params"]]  # fmt: skip
    files += [CONFIG_DIR / f"EXP-V-{v}.yaml" for v in cfg["views"]]
    return files


def _rows_subset(rows, mask):
    from vaultic.fusion.data import Rows

    return Rows(*(getattr(rows, f)[mask] for f in ("ids", "views", "context", "y", "day")))


def _params(p: dict) -> dict:
    return {k: tuple(v) if k == "hidden" else v for k, v in p.items()}


def _gate_scores(both, cal, ev, tuned: dict, seeds,
                 conditions=E11_CONDITIONS) -> tuple[dict, dict, dict, dict]:  # fmt: skip
    """Fused scores on calibration / evaluation rows (5-seed mean), per-seed evaluation scores,
    and the E11 evaluation scores with each view removed."""
    from vaultic.fusion.baselines import make_fusion
    from vaultic.fusion.final_eval import VIEWS
    from vaultic.fusion.tune_fusion import fit_seeds

    s_cal, s_ev, per_seed, removed = {}, {}, {}, {c: {} for c in conditions}
    for name in GATES:
        if name in tuned:
            models = fit_seeds(name, _params(tuned[name]["params"]), both, seeds)
        else:
            models = [
                make_fusion(name, seed=s).fit(both.views, both.y, both.context) for s in seeds
            ]
        s_cal[name] = np.mean([m.predict_proba(cal.views, cal.context) for m in models], axis=0)
        preds = [m.predict_proba(ev.views, ev.context) for m in models]
        s_ev[name], per_seed[name] = np.mean(preds, axis=0), preds
        for cond in conditions:
            v = ev.views.copy()
            if cond == "tabular only":
                v[:, [j for j, n in enumerate(VIEWS) if n != "tabular"]] = np.nan
            else:
                v[:, VIEWS.index(cond)] = np.nan
            removed[cond][name] = np.mean([m.predict_proba(v, ev.context) for m in models], axis=0)
    return s_cal, s_ev, per_seed, removed


def _view_columns(parts) -> dict[str, list[str]]:
    from vaultic.views.definitions import LABEL_DERIVED

    cols = parts["cols"]
    return {"tabular": parts["tab_cols"], "behavioral": cols["behavioral"] + list(LABEL_DERIVED),
            "graph": cols["graph"], "anomaly": cols["anomaly"], "temporal": cols["sequence"]}  # fmt: skip


def _feature_model_scores(parts, cfg_id: str, columns: list[str], cal_ids, ev_ids, seeds,
                          removals: dict[str, list[str]], device: str = "cpu"):  # fmt: skip
    """B5 / F0 refitted on days 1-120 with the config's frozen hyperparameters; scores on
    calibration and evaluation rows (5-seed mean), per seed, and with columns removed (NaN)."""
    from vaultic.views.tabular import make_model

    df, features, splits = parts["df"], parts["features"], parts["splits"]
    params = yaml.safe_load((CONFIG_DIR / f"{cfg_id}.yaml").read_text("utf-8"))["model"]["params"]
    train = splits.train.contains(df["day"].to_numpy())
    X = features[columns]
    pos = pd.Index(df["TransactionID"].to_numpy())
    pc, pe = pos.get_indexer(cal_ids), pos.get_indexer(ev_ids)
    if (pc < 0).any() or (pe < 0).any():
        raise ValueError("some evaluation rows are missing from the data")
    Xc, Xe = X.iloc[pc], X.iloc[pe]
    y_tr = df["isFraud"].to_numpy()[train]
    s_cal, preds, removed = [], [], {c: [] for c in removals}
    for s in seeds:
        model = make_model("xgboost", params, s, device).fit(X[train], y_tr)
        s_cal.append(model.predict_proba(Xc)[:, 1])
        preds.append(model.predict_proba(Xe)[:, 1])
        for cond, drop_cols in removals.items():
            Xr = Xe.copy()
            Xr.loc[:, [c for c in drop_cols if c in Xr.columns]] = np.nan
            removed[cond].append(model.predict_proba(Xr)[:, 1])
    return (np.mean(s_cal, axis=0), np.mean(preds, axis=0), preds,
            {c: np.mean(v, axis=0) for c, v in removed.items()})  # fmt: skip


def main() -> None:
    from vaultic.eval.run import _guard_final_rerun
    from vaultic.fusion.data import fusion_split
    from vaultic.fusion.dev_compare import build_table
    from vaultic.fusion.final_eval import (
        e10_subgroups,
        e10_table,
        e11_drops,
        h2_verdict,
        refit_check,
        subgroup_masks,
    )
    from vaultic.fusion.tune_fusion import concat_rows
    from vaultic.reports.phase2_summary import check_freeze
    from vaultic.trust.final_trust import phase8_gate, trust_report
    from vaultic.views.definitions import LABEL_DERIVED
    from vaultic.views.orchestrate import CONTEXT, calibrate_views

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["dev", "freeze", "refit-check", "final"], required=True)
    parser.add_argument("--smoke", action="store_true", help="dev only: 1 in 10 uids")
    parser.add_argument("--rerun-reason", default=None)
    parser.add_argument("--n-boot", type=int, default=None, help="dev only: fewer resamples")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text("utf-8"))
    check_settings(cfg)  # every mode, so nothing is frozen or run with other settings (D108)
    if args.mode == "freeze":  # hash every file the final run depends on (D101)
        from vaultic.reports.phase2_summary import write_freeze

        write_freeze(frozen_files(cfg), FROZEN,
                     title="Frozen configs for the Paper 2 final runs (E10-E12, D101)")  # fmt: skip
        print(f"wrote {FROZEN}")
        return
    if args.mode == "refit-check":
        _refit_check_mode(cfg, build_table)
        return
    final = args.mode == "final"
    if final:
        if args.smoke or args.n_boot:
            raise ValueError("the final run uses all rows and the pre-registered bootstrap size")
        if not FROZEN.exists() or check_freeze(FROZEN):
            raise RuntimeError(f"{FROZEN} is missing or a frozen file changed: "
                               f"{check_freeze(FROZEN) if FROZEN.exists() else 'no record'}")  # fmt: skip
        head = check_git(FROZEN)
        # a run whose metrics.json says mode final (also "started") counts as a final run
        _guard_final_rerun(cfg["id"], RUNS_DIR, args.rerun_reason, RESEARCH_DIR / "decisions.md")
        run_dir = RUNS_DIR / cfg["id"] / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        record_start(run_dir, head, RESEARCH_DIR / "experiment_log.md",
                     extra_logs=[RESEARCH_DIR / "decisions.md"], repo=REPO_ROOT)  # fmt: skip
    n_boot = args.n_boot or cfg["bootstrap"]["n"]
    boot_seed = cfg["bootstrap"]["seed"]
    seeds = tuple(cfg["seeds"])
    e10, e11, e12 = cfg["e10"], cfg["e11"], cfg["e12"]
    if not final:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True).stdout.strip()  # fmt: skip
    started = time.perf_counter()

    table, parts = build_table(args.smoke, d95=True, final=final, return_parts=True)
    if not final and ((table["role"] == "test").any() or (table.get("split") == "test").any()):
        raise RuntimeError("a test-period row reached a development run")
    if parts["splits"].label_delay_days != e12["adaptive"]["label_delay_days"]:
        raise ValueError("splits.yaml label delay differs from e12.adaptive.label_delay_days")
    table, _ = calibrate_views(table)
    split = fusion_split(table)
    both = concat_rows(split.fit, split.tune)
    if final:
        cal, ev = split.calibrate, split.test
    else:
        cal = _rows_subset(split.calibrate, split.calibrate.day <= 148)
        ev = _rows_subset(split.calibrate, split.calibrate.day >= 149)
    tuned = json.loads((REPO_ROOT / cfg["fusion_params"]).read_text("utf-8"))["tuned"]

    s_cal, s_ev, per_seed, removed = _gate_scores(both, cal, ev, tuned, seeds, e11["conditions"])
    vcols = _view_columns(parts)
    b5_removals = {"tabular": vcols["tabular"], "behavioral": list(LABEL_DERIVED),
                   "tabular only": list(LABEL_DERIVED)}  # fmt: skip
    keep = set(vcols["tabular"])
    f0_cols = _f0_columns(parts)
    f0_removals = {v: vcols[v] for v in ("tabular", "behavioral", "temporal", "graph", "anomaly")}
    f0_removals["tabular only"] = [c for c in f0_cols if c not in keep]
    for name, cid, columns, removals in (("B5", cfg["b5_config"], parts["b5_cols"], b5_removals),
                                         ("F0", cfg["f0_config"], f0_cols, f0_removals)):  # fmt: skip
        sc, se, ps, rem = _feature_model_scores(
            parts, cid, columns, cal.ids, ev.ids, seeds, removals
        )
        s_cal[name], s_ev[name], per_seed[name] = sc, se, ps
        for cond, s in rem.items():
            removed[cond][name] = s

    df = parts["df"]
    amt = df.set_index("TransactionID")["TransactionAmt"]
    a_cal, a_ev = amt.reindex(cal.ids).to_numpy(float), amt.reindex(ev.ids).to_numpy(float)
    thresholds = {m: choose_cost_threshold(cal.y, s_cal[m], a_cal) for m in s_ev}
    if set(e10["methods"]) != set(s_ev):
        raise RuntimeError(f"methods scored {sorted(s_ev)} differ from e10.methods")
    scores = {m: s_ev[m] for m in e10["methods"]}
    methods, comps = e10_table(ev.y, scores, per_seed, a_ev, thresholds,
                               reference=e10["reference"], n_boot=n_boot, seed=boot_seed)  # fmt: skip
    masks = subgroup_masks(ev.context[:, CONTEXT.index("ctx_hist_n_past")],
                           ev.context[:, CONTEXT.index("ctx_has_identity")],
                           (~np.isnan(ev.views)).astype(int))  # fmt: skip
    subs = e10_subgroups(ev.y, scores, masks, reference=e10["reference"],
                         versus=tuple(e10["subgroup_versus"]), n_boot=n_boot)  # fmt: skip
    h2 = h2_verdict(comps, subs, alpha=e10["h2_alpha"])
    drops, drop_comps = e11_drops(ev.y, scores, removed, reference=e10["reference"],
                                  versus=tuple(e11["versus"]), n_boot=n_boot, seed=boot_seed)  # fmt: skip

    dis = table.drop_duplicates("TransactionID").set_index("TransactionID")["disagreement"]
    trust = {}
    for m in cfg["e12_methods"]:
        d_cal = dis.reindex(cal.ids).to_numpy(float) if m != "B5" else None
        d_ev = dis.reindex(ev.ids).to_numpy(float) if m != "B5" else None
        trust[m] = trust_report(
            {"y": cal.y, "p": s_cal[m], "amount": a_cal, "day": cal.day, "d": d_cal},
            {"y": ev.y, "p": s_ev[m], "amount": a_ev, "day": ev.day, "d": d_ev},
            label_delay_days=e12["adaptive"]["label_delay_days"],
            n_boot=n_boot, settings=e12, seed=boot_seed,
        )  # fmt: skip
    gate = phase8_gate(trust, tolerance=e12["phase8_gate_tolerance"])

    # the refitted B5 / F0 must reproduce the stored predictions on the same rows: B5 on test
    # (final), B5 and F0 on validation (full dev run); a smoke run cannot (1 in 10 uids)
    if args.smoke:
        sanity = {"skipped": "smoke run: B5 / F0 trained on a 1-in-10 uid sample; use "
                             "--mode refit-check"}  # fmt: skip
    elif final:
        sanity = {"B5 (test)": refit_check(_stored(cfg["b5_run"], "test"), ev.ids, s_ev["B5"])}
    else:
        sanity = {f"{m} (validation)": refit_check(_stored(cfg[run], "validation"), ev.ids, s_ev[m])
                  for m, run in (("B5", "b5_run"), ("F0", "f0_run"))}  # fmt: skip

    out_dir = (RESEARCH_DIR / "tables") if final else DEV_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "final_" if final else "final_dev_"
    meta = {"mode": args.mode, "smoke": args.smoke, "n_boot": n_boot, "seeds": list(seeds),
            "evaluation_rows": int(len(ev.y)), "evaluation_frauds": int(ev.y.sum()),
            "calibration_rows": int(len(cal.y)), "calibration_frauds": int(cal.y.sum()),
            "gate_rows": int(len(both.y)), "thresholds": thresholds, "refit_check": sanity,
            "git_commit": head,
            "input_hashes": input_hashes([*parts["inputs"],
                                          REPO_ROOT / cfg["b5_run"] / "predictions.parquet",
                                          REPO_ROOT / cfg["f0_run"] / "predictions.parquet"]),
            "runtime_s": round(time.perf_counter() - started)}  # fmt: skip
    write_reports(out_dir, prefix, meta, methods, comps, subs, h2, drops, drop_comps, trust, gate)
    if final:  # run_dir was created by record_start; complete its metrics.json
        pd.DataFrame({"TransactionID": ev.ids, "day": ev.day, "label": ev.y,
                      **{f"score_{m}": s for m, s in scores.items()}}).to_parquet(
            run_dir / "predictions.parquet", index=False)  # fmt: skip
        start = json.loads((run_dir / "metrics.json").read_text("utf-8"))
        (run_dir / "metrics.json").write_text(json.dumps(
            {**start, "mode": "final", "status": "completed",
             "completed": datetime.now().isoformat(timespec="seconds"), **meta, "h2": h2,
             "phase8_gate": gate}, indent=1, default=str), "utf-8")  # fmt: skip
        (run_dir / "config.yaml").write_text(CONFIG.read_text("utf-8"), "utf-8")
    print(
        f"wrote {out_dir} ({prefix}*); H2: {h2['verdict']}; "
        f"Phase 8 gate passed: {gate['passed']}"
    )


def _stored(run: str, split: str) -> pd.Series:
    stored = pd.read_parquet(REPO_ROOT / run / "predictions.parquet")
    return stored[stored["split"] == split].set_index("TransactionID")["score"]


def _refit_check_mode(cfg: dict, build_table) -> None:
    """B5 and F0 refitted on the full training period vs their stored validation predictions."""
    from vaultic.fusion.final_eval import refit_check

    parts = build_table(False, d95=True, parts_only=True)
    df, splits = parts["df"], parts["splits"]
    val = df.loc[splits.validation.contains(df["day"].to_numpy()), "TransactionID"].to_numpy()
    out = {}
    for m, cid, run, columns in (("B5", cfg["b5_config"], cfg["b5_run"], parts["b5_cols"]),
                                 ("F0", cfg["f0_config"], cfg["f0_run"], _f0_columns(parts))):  # fmt: skip
        _, s, _, _ = _feature_model_scores(parts, cid, columns, val, val, tuple(cfg["seeds"]), {})
        out[m] = refit_check(_stored(run, "validation"), val, s)
        print(m, out[m], flush=True)
    DEV_OUT.mkdir(parents=True, exist_ok=True)
    (DEV_OUT / "final_dev_refit_check.json").write_text(json.dumps(out, indent=1), "utf-8")


def _f0_columns(parts) -> list[str]:
    vcols = _view_columns(parts)
    return list(dict.fromkeys(parts["b5_cols"] + vcols["behavioral"] + vcols["graph"]
                              + vcols["anomaly"] + vcols["temporal"]))  # fmt: skip


def _ci(r, k):
    return f"{r[k]:.4f} [{r[k + '_ci_low']:.4f}, {r[k + '_ci_high']:.4f}]"


def write_reports(out_dir, prefix, meta, methods, comps, subs, h2, drops, drop_comps, trust, gate):
    banner = ("" if meta["mode"] == "final" else
              "**PIPELINE CHECK on validation (days 149-150), not a result.**\n\n")  # fmt: skip
    head = (f"Generated {date.today().isoformat()} by `python -m vaultic.fusion.final_run --mode "
            f"{meta['mode']}`. Pre-registered in D101 (details D102). Evaluation rows: "
            f"{meta['evaluation_rows']} ({meta['evaluation_frauds']} frauds); thresholds and all "
            f"calibration on {meta['calibration_rows']} rows ({meta['calibration_frauds']} frauds); "
            f"{len(meta['seeds'])} seeds; {meta['n_boot']} paired bootstrap resamples.")  # fmt: skip
    lines = [f"# E10: fusion comparison ({meta['mode']})", "", banner + head, "",
             "| method | PR-AUC [95% CI] | recall@1%FPR [95% CI] | cost ($) [95% CI] |",
             "|---|---|---|---|"]  # fmt: skip
    for r in methods.to_dict("records"):
        lines.append(f"| {r['method']} | {_ci(r, 'pr_auc')} | {_ci(r, 'recall_at_1pct_fpr')} | "
                     f"{r['cost']:.0f} [{r['cost_ci_low']:.0f}, {r['cost_ci_high']:.0f}] |")  # fmt: skip
    lines += ["", "Comparisons (MVAF − method; for cost, positive = MVAF costs more). Holm within "
              "each metric. Primary: MVAF − F3 PR-AUC.", "",
              "| metric | comparison | diff | 95% CI | p | p (Holm) | relative % | per-seed mean diff | Cohen's d (paired seeds) |",
              "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in comps.to_dict("records"):
        rel = r.get("relative_pct", np.nan)
        dd = r.get("cohens_d_paired", np.nan)
        sm = r.get("seed_mean_diff", np.nan)
        lines.append(f"| {r['metric']} | {r['comparison']} | {r['diff']:+.4f} | "
                     f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['p_value']:.3f} | "
                     f"{r['p_holm']:.3f} | {'' if pd.isna(rel) else f'{rel:+.1f}'} | "
                     f"{'' if pd.isna(sm) else f'{sm:+.4f}'} | "
                     f"{'' if pd.isna(dd) else f'{dd:+.2f}'} |")  # fmt: skip
    mcols = [c for c in subs.columns if c not in ("subgroup", "rows", "frauds")
             and not c.startswith("MVAF - ") and not c.endswith(("_ci_low", "_ci_high"))]  # fmt: skip
    vcols = [c for c in subs.columns if c.startswith("MVAF - ")]
    lines += ["", "## Subgroups (PR-AUC)", "",
              "| subgroup | rows | frauds | " + " | ".join(mcols + vcols) + " |",
              "|---|---|---|" + "---|" * (len(mcols) + len(vcols))]  # fmt: skip
    for r in subs.to_dict("records"):
        vals = ["" if pd.isna(r.get(c, np.nan)) else
                f"{r[c]:.4f} [{r[c + '_ci_low']:.4f}, {r[c + '_ci_high']:.4f}]" for c in mcols]  # fmt: skip
        ds = ["" if not isinstance(r.get(c), tuple) else
              f"{r[c][0]:+.4f} [{r[c][1]:+.4f}, {r[c][2]:+.4f}], p (Holm) {r[c][4]:.3f}"
              for c in vcols]  # fmt: skip
        lines.append(f"| {r['subgroup']} | {r['rows']} | {r['frauds']} | "
                     + " | ".join(vals + ds) + " |")  # fmt: skip
    lines += [
        "",
        "Per-method subgroup PR-AUC with a bootstrap 95% CI. Subgroup comparisons: paired "
        "bootstrap; Holm across every comparison in this table (D104).",
    ]
    lines += ["", f"**H2 (D101):** beats F3 / F4 overall: {h2['beats_overall']}; larger margin on "
              f"missing-view rows: {h2['larger_margin_on_missing_views']} → "
              f"**{h2['verdict'].upper()}**.", ""]  # fmt: skip
    for name, chk in (meta.get("refit_check") or {}).items():
        if name == "skipped":
            lines += [f"Refit check skipped: {chk}.", ""]
        else:
            lines += [f"Refitted {name} vs stored predictions: max |diff| {chk['max_abs_diff']:.2e}, "
                      f"correlation {chk['corr']:.6f} ({chk['n_compared']} rows compared, "
                      f"{chk['n_missing']} missing or not finite).", ""]  # fmt: skip
    (out_dir / f"{prefix}e10.md").write_text("\n".join(lines), encoding="utf-8")

    lines = [f"# E11: robustness to missing views ({meta['mode']})", "", banner + head, "",
             "| condition | method | PR-AUC full | PR-AUC removed | drop [95% CI] |",
             "|---|---|---|---|---|"]  # fmt: skip
    for r in drops.to_dict("records"):
        lines.append(f"| {r['condition']} | {r['method']} | {r['pr_auc_full']:.4f} | "
                     f"{r['pr_auc_removed']:.4f} | {r['drop']:+.4f} [{r['ci_low']:+.4f}, "
                     f"{r['ci_high']:+.4f}] |")  # fmt: skip
    lines += ["", "Primary: MVAF's drop minus F3's / F4's (negative = MVAF degrades less); Holm "
              "across the five single-view conditions.", "",
              "| condition | comparison | diff | 95% CI | p | p (Holm) |", "|---|---|---|---|---|---|"]  # fmt: skip
    for r in drop_comps.to_dict("records"):
        ph = "" if pd.isna(r.get("p_holm", np.nan)) else f"{r['p_holm']:.3f}"
        lines.append(f"| {r['condition']} | {r['comparison']} | {r['diff']:+.4f} | "
                     f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['p_value']:.3f} | {ph} |")  # fmt: skip
    (out_dir / f"{prefix}e11.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    from vaultic.trust.final_trust import plot_reliability

    fig_dir = (RESEARCH_DIR / "figures") if meta["mode"] == "final" else out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# E12: trust layer ({meta['mode']})", "", banner + head, ""]
    for m, r in trust.items():
        c = r["calibration"]
        fig = fig_dir / f"{prefix}e12_reliability_{m}.png"
        plot_reliability(r, f"{m}: reliability, evaluation rows ({meta['mode']})", fig)
        lines += [f"## {m}", "", f"Reliability diagram: `{fig.as_posix()}`.", "",
                  f"Calibrator: {r['calibrator']['method']}. ECE {c['ece_before']:.4f} → "
                  f"{c['ece_after']:.4f}; Brier {c['brier_before']:.4f} → {c['brier_after']:.4f}.", "",
                  "| conformal | coverage | legit | fraud | uncertain sets | within 2 pts |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
        for t, cv in r["conformal"].items():
            lines.append(f"| Mondrian {t} | {cv['coverage']:.4f} | {cv['coverage_legit']:.4f} | "
                         f"{cv['coverage_fraud']:.4f} | {cv['share_uncertain']:.4f} | "
                         f"{'yes' if cv['within_tolerance'] else 'no'} |")  # fmt: skip
        for t, ad in r["adaptive"].items():
            blocks = ", ".join(f"{b['first_day']}-{b['last_day']}: {b['coverage']:.3f}"
                               for b in ad["blocks"])  # fmt: skip
            lines.append(f"| adaptive {t} (L = {ad['label_delay_days']}) | "
                         f"{ad['overall']['coverage']:.4f} | | | | blocks: {blocks} |")  # fmt: skip
        dflt = r["decision"]["default"]
        lines += ["", f"Decision engine, default costs: total cost ${dflt['evaluation_cost']:,.0f} "
                  f"(sensitivity sweep: {len(r['decision'].get('sensitivity', []))} settings, in the "
                  "JSON).", "", "| K/day | policy | fraud value caught | fraud caught | recall@K | "
                  "$ saved / review |", "|---|---|---|---|---|---|"]  # fmt: skip
        for p in r["routing"]["policies"]:
            lines.append(f"| {p['K']} | {p['policy']} | {p['fraud_value_caught']:,.0f} | "
                         f"{p['fraud_caught']} | {p['recall_at_k']:.4f} | "
                         f"{p['money_saved_per_review']:.1f} |")  # fmt: skip
        lines += [
            "",
            "H5 (fraud value, day-block bootstrap; Holm across this method's H5 tests, " "D104):",
            "",
        ]
        lines += [f"- K = {h['K']}: {h['comparison']} {h['fraud_value_diff']:+,.0f} "
                  f"[{h['ci_low']:+,.0f}, {h['ci_high']:+,.0f}], p = {h['p_value']:.3f}, "
                  f"p (Holm) = {h['p_holm']:.3f}"
                  for h in r["routing"]["h5"]]  # fmt: skip
        lines.append("")
    lines += [f"**Phase 8 exit gate** (Mondrian coverage within 2 points of target): "
              f"{gate['per_method']} → **{'PASSED' if gate['passed'] else 'NOT PASSED'}**.", ""]  # fmt: skip
    (out_dir / f"{prefix}e12.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / f"{prefix}results.json").write_text(json.dumps({
        "meta": meta, "e10_methods": methods.to_dict("records"),
        "e10_comparisons": comps.to_dict("records"), "e10_subgroups": subs.to_dict("records"),
        "h2": h2, "e11_drops": drops.to_dict("records"),
        "e11_comparisons": drop_comps.to_dict("records"), "e12": trust, "phase8_gate": gate,
    }, indent=1, default=str), encoding="utf-8")  # fmt: skip


if __name__ == "__main__":
    main()
