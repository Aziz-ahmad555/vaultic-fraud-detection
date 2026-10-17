"""E24: compute cost per model (D121 section 4; details D158 (3), D162). Reporting only, no test.

  hardware    CPU model, physical / logical cores, RAM, OS, Python (recorded automatically)
  training    wall-clock of one seed's fit (seed 0) with the frozen settings, 3 repetitions,
              median; feature models on days 1-120 (validation-only refits, no test rows),
              fusion methods on the gate rows (fit + tune) of the D95 view table
  inference   wall-clock per 1,000 validation rows (feature models: validation days 128-150;
              fusion: calibrate_fused 147-150), median of 5 repetitions
  tuning      from the tuning logs (research/tuning_<id>.md, total minutes over both weighting
              arms); where no log records it, trials x the measured fit time of the tuned
              setting, labelled as estimated (fusion: 50 trials; B1: its 7-value C grid)
  carbon      a labelled ESTIMATE: energy = wall-clock under load x the CPU's TDP (no PUE,
              laptop), x the grid carbon intensity (Ember, Pakistan, 2024; D158 (3))

Models: B1-B6, F0, the five views, F1-F7, MVAF, the ladder's F4 rungs 1-4 and the anomaly-score
models behind the anomaly view (global IF, autoencoder, per-uid IF; D168).
"""

from __future__ import annotations

import json
import os
import platform
import re
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from vaultic.paths import RESEARCH_DIR

FIT_REPEATS = 3
PREDICT_REPEATS = 5
BASELINES = {"B1": "EXP-012", "B2": "EXP-002", "B3": "EXP-011", "B4": "EXP-013", "B5": "EXP-009",
             "B6": "EXP-010", "F0": "EXP-F0-inner"}  # fmt: skip
TUNING_LOGS = {"B3": "B3", "B4": "B4", "B5": "B5", "F0": "F0-inner", "tabular": "V-tabular",
               "behavioral": "V-behavioral", "temporal": "V-temporal", "graph": "V-graph",
               "anomaly": "V-anomaly"}  # fmt: skip
ESTIMATED_TRIALS = {"B1": 7, "MVAF": 50, "F3": 50, "F4": 50, "F5": 50, "F6": 50, "F7": 50}
UNTUNED = (
    "B2",
    "B6",
    "F1",
    "F2",
    "anomaly: global IF",
    "anomaly: autoencoder",
    "anomaly: per-uid IF",
)  # B2 defaults, B6 FYP-1, F1 / F2 unsearched, anomaly D68


def hardware() -> dict:
    """CPU, cores, RAM and OS of this machine."""
    cpu = platform.processor()
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")  # fmt: skip
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    out = {"cpu": cpu, "logical_cores": os.cpu_count(), "os": platform.platform(),
           "python": platform.python_version()}  # fmt: skip
    try:
        import psutil

        out["physical_cores"] = psutil.cpu_count(logical=False)
        out["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        out["physical_cores"] = out["ram_gb"] = None
    return out


def time_model(make: Callable[[], object], fit: Callable[[object], object],
               predict: Callable[[object], object], n_predict_rows: int,
               fit_repeats: int = FIT_REPEATS, predict_repeats: int = PREDICT_REPEATS) -> dict:  # fmt: skip
    """Median fit seconds over fresh models, and median inference ms per 1,000 rows."""
    fits, model = [], None
    for _ in range(fit_repeats):
        model = make()
        t0 = time.perf_counter()
        fit(model)
        fits.append(time.perf_counter() - t0)
    preds = []
    for _ in range(predict_repeats):
        t0 = time.perf_counter()
        predict(model)
        preds.append(time.perf_counter() - t0)
    return {"fit_seconds": statistics.median(fits), "fit_seconds_all": fits,
            "inference_ms_per_1000": statistics.median(preds) / n_predict_rows * 1e6,
            "predict_rows": int(n_predict_rows)}  # fmt: skip


def tuning_minutes(log_id: str, research: Path = RESEARCH_DIR) -> float | None:
    """Total tuning minutes in research/tuning_<log_id>.md (sum over the weighting arms)."""
    path = research / f"tuning_{log_id}.md"
    if not path.exists():
        return None
    text = path.read_text("utf-8")
    lines = text.splitlines()
    head = next((i for i, line in enumerate(lines) if "total minutes" in line), None)
    if head is None:
        return None
    cols = [c.strip() for c in lines[head].strip("|").split("|")]
    j = cols.index("total minutes")
    total = 0.0
    for line in lines[head + 2 :]:
        if not line.startswith("|"):
            break
        cell = [c.strip() for c in line.strip("|").split("|")][j]
        total += float(re.sub(r"[^\d.]", "", cell))
    return total


def tuning_entry(name: str, fit_seconds: float, research: Path = RESEARCH_DIR) -> dict:
    if name in UNTUNED:
        return {"tuning_hours": 0.0, "tuning_source": "not tuned"}
    key = name.removeprefix("view: ")  # view rows are named "view: <view>" (D171)
    if key in TUNING_LOGS:
        minutes = tuning_minutes(TUNING_LOGS[key], research)
        if minutes is not None:
            return {"tuning_hours": minutes / 60,
                    "tuning_source": f"research/tuning_{TUNING_LOGS[key]}.md"}  # fmt: skip
    trials = ESTIMATED_TRIALS.get(name.split(" ")[0], 50)  # ladder rungs: F4's 50 trials
    return {"tuning_hours": trials * fit_seconds / 3600,
            "tuning_source": f"estimated: {trials} trials x measured fit time"}  # fmt: skip


def carbon(wall_seconds: float, tdp_watts: float, grams_per_kwh: float) -> dict:
    """Labelled estimate: wall-clock under load x TDP x grid intensity (no PUE)."""
    kwh = wall_seconds * tdp_watts / 3.6e6
    return {"energy_kwh": kwh, "co2_kg": kwh * grams_per_kwh / 1000,
            "note": "estimate, not a measurement: wall-clock under load x CPU TDP x grid carbon "
                    "intensity; no PUE (laptop)"}  # fmt: skip


def summarise(rows: list[dict], settings: dict) -> dict:
    """Compute hours (training + tuning) and the carbon estimate over every model."""
    train_s = sum(r["fit_seconds"] for r in rows)
    tune_h = sum(r["tuning_hours"] for r in rows)
    wall = train_s + tune_h * 3600
    grid = settings.get("grid_g_co2_per_kwh")
    return {"training_hours": train_s / 3600, "tuning_hours": tune_h,
            "compute_hours": wall / 3600,
            "carbon": carbon(wall, settings["cpu_tdp_watts"], grid) if grid else
            {"energy_kwh": wall * settings["cpu_tdp_watts"] / 3.6e6, "co2_kg": None,
             "note": "grid carbon intensity not entered yet (Ember, Pakistan, 2024; D158 (3))"}}  # fmt: skip


# ---- the real models ---------------------------------------------------------------------


def run(settings: dict, ladder_params: dict | None, log=print) -> dict:
    """Time every model on the real data (training period / gate rows, validation rows)."""
    import pandas as pd

    from vaultic.eval.run import load_config, load_inputs
    from vaultic.features.sets import design_matrix, drop_features
    from vaultic.fusion.baselines import make_fusion
    from vaultic.fusion.data import fusion_split
    from vaultic.fusion.dev_compare import d95_views
    from vaultic.fusion.final_run import GATES, _params
    from vaultic.fusion.tune_fusion import concat_rows
    from vaultic.paths import CONFIG_DIR
    from vaultic.stage2.system import FUSION_PARAMS, VIEW_TABLE, load_parts
    from vaultic.views.orchestrate import VIEWS, calibrate_views
    from vaultic.views.plan import crossfit_plan
    from vaultic.views.tabular import make_model

    rows = []

    def record(name, kind, timing):
        rows.append({"model": name, "kind": kind, **timing,
                     **tuning_entry(name, timing["fit_seconds"])})  # fmt: skip
        log(f"{name}: fit {timing['fit_seconds']:.1f} s, "
            f"{timing['inference_ms_per_1000']:.2f} ms / 1,000 rows")  # fmt: skip

    from vaultic.data.splits import load_splits

    splits = load_splits()
    for name, cid in BASELINES.items():
        cfg = load_config(CONFIG_DIR / f"{cid}.yaml")
        df, base, _ = load_inputs(cfg, splits)
        keep = df["day"].to_numpy() <= splits.validation.last  # no test rows
        df = df[keep].reset_index(drop=True)
        base = None if base is None else base[keep].reset_index(drop=True)
        X = drop_features(design_matrix(df, base, cfg["features"]), cfg.get("drop_features"))
        y = df["isFraud"].to_numpy()
        part = splits.assign(df["day"])
        tr, va = part == "train", part == "validation"
        m = cfg["model"]
        record(name, "baseline", time_model(
            lambda m=m: make_model(m["name"], m.get("params", {}), 0),
            lambda model, X=X, y=y, tr=tr: model.fit(X[tr], y[tr]),
            lambda model, X=X, va=va: model.predict_proba(X[va]), int(va.sum())))  # fmt: skip
        del df, base, X

    parts = load_parts(dev=True)
    df, features = parts["df"], parts["features"]
    fixed = next(f for f in crossfit_plan(parts["splits"]) if f.name == "fixed")
    tr = fixed.train_rows(df["day"].to_numpy(), df["TransactionDT"].to_numpy())
    va = parts["splits"].validation.contains(df["day"].to_numpy())
    X_tr = features.loc[tr].reset_index(drop=True)
    X_va = features.loc[va].reset_index(drop=True)
    y_tr = df["isFraud"].to_numpy()[tr]
    for v in VIEWS:
        record(f"view: {v}", "view", time_model(
            lambda v=v: d95_views(parts["tab_cols"], parts["cols"])[v],
            lambda view: view.fit(X_tr, y_tr), lambda view: view.predict(X_va), int(va.sum())))  # fmt: skip

    table, _ = calibrate_views(pd.read_parquet(VIEW_TABLE))
    split = fusion_split(table)
    both, cal = concat_rows(split.fit, split.tune), split.calibrate
    tuned = json.loads(FUSION_PARAMS.read_text("utf-8"))["tuned"]
    for name in GATES:
        params = _params(tuned[name]["params"]) if name in tuned else {}
        record(name, "fusion", time_model(
            lambda name=name, params=params: make_fusion(name, seed=0, **params),
            lambda model: model.fit(both.views, both.y, both.context),
            lambda model: model.predict_proba(cal.views, cal.context), len(cal.y)))  # fmt: skip
    for rung, info in (ladder_params or {}).items():
        cols = info["views"]
        idx = [VIEWS.index(v) for v in cols]
        record(f"F4 rung {rung}", "ladder", time_model(
            lambda info=info: make_fusion("F4", seed=0, **info["params"]),
            lambda model, idx=idx: model.fit(both.views[:, idx], both.y, both.context),
            lambda model, idx=idx: model.predict_proba(cal.views[:, idx], cal.context),
            len(cal.y)))  # fmt: skip
    return {"hardware": hardware(), "settings": settings, "models": rows,
            "summary": summarise(rows, settings)}  # fmt: skip


def time_anomaly(log=print) -> list[dict]:
    """D168: the anomaly-score models (views/anomaly.py), fitted as anomaly_build fitted the
    model that scored every row after the training period (last forward-chained cut: legit rows
    of days 1-120, at most 150,000 sampled, autoencoder 50 iterations, seed 0); each component
    timed separately; inference on the validation rows."""
    from vaultic.stage2.system import ANOMALY, anomaly_fit_rows, load_parts
    from vaultic.views.anomaly import (
        AutoencoderAnomaly,
        GlobalIsolationForest,
        PerUidIsolationForest,
    )
    from vaultic.views.definitions import LABEL_DERIVED

    parts = load_parts(dev=True)
    df, features, splits = parts["df"], parts["features"], parts["splits"]
    day = df["day"].to_numpy()
    cols = [c for c in parts["b5_cols"] if c not in LABEL_DERIVED]
    fit = anomaly_fit_rows(day, splits.train.first, splits.train.last, ANOMALY["block_days"],
                           ANOMALY["max_fit_rows"], ANOMALY["seed"])  # fmt: skip
    X = features[cols].to_numpy(dtype=float)
    y = df["isFraud"].to_numpy()
    uid = features["uid"].to_numpy()
    n_past = features["uid_n_past"].to_numpy()
    va = splits.validation.contains(day)
    Xf, yf, uf = X[fit], y[fit], uid[fit]
    seed = ANOMALY["seed"]
    models = (
        ("anomaly: global IF", lambda: GlobalIsolationForest(seed=seed),
         lambda m: m.fit(Xf, yf), lambda m: m.transform(X[va])),
        ("anomaly: autoencoder",
         lambda: AutoencoderAnomaly(seed=seed, max_iter=ANOMALY["ae_max_iter"]),
         lambda m: m.fit(Xf, yf), lambda m: m.transform(X[va])),
        ("anomaly: per-uid IF", lambda: PerUidIsolationForest(seed=seed),
         lambda m: m.fit(Xf, uf), lambda m: m.transform(X[va], uid[va], n_past[va])),
    )  # fmt: skip
    rows = []
    for name, make, do_fit, do_pred in models:
        t = time_model(make, do_fit, do_pred, int(va.sum()))
        rows.append({"model": name, "kind": "anomaly-score model", **t,
                     **tuning_entry(name, t["fit_seconds"])})  # fmt: skip
        log(f"{name}: fit {t['fit_seconds']:.1f} s, "
            f"{t['inference_ms_per_1000']:.2f} ms / 1,000 rows")  # fmt: skip
    return rows


def write_report(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e24_compute.json").write_text(json.dumps(result, indent=1, default=str), "utf-8")
    hw, s, sm = result["hardware"], result["settings"], result["summary"]
    lines = ["# E24: compute cost per model", "",
             "Generated by `python -m vaultic.stage2.e24` (D121 section 4). Validation-only refits; "
             "no test data.", "",
             f"Hardware: {hw['cpu']}, {hw.get('physical_cores')} physical / {hw['logical_cores']} "
             f"logical cores, {hw.get('ram_gb')} GB RAM, {hw['os']}, Python {hw['python']}.", "",
             "| model | kind | fit (s, median of 3) | inference (ms / 1,000 rows, median of 5) | "
             "tuning (h) | tuning source |", "|---|---|---|---|---|---|"]  # fmt: skip
    for r in result["models"]:
        lines.append(f"| {r['model']} | {r['kind']} | {r['fit_seconds']:.1f} | "
                     f"{r['inference_ms_per_1000']:.2f} | {r['tuning_hours']:.2f} | "
                     f"{r['tuning_source']} |")  # fmt: skip
    c = sm["carbon"]
    lines += ["", f"Compute hours: training {sm['training_hours']:.2f} + tuning "
                  f"{sm['tuning_hours']:.2f} = **{sm['compute_hours']:.2f} h**.", ""]  # fmt: skip
    energy = (f"Energy = wall-clock under load x TDP {s['cpu_tdp_watts']} W ({s['tdp_source']}); "
              "no PUE (laptop).")  # fmt: skip
    if c["co2_kg"] is None:
        lines.append(f"**Carbon:** pending, {c['energy_kwh']:.3f} kWh estimated; {c['note']}. "
                     + energy)  # fmt: skip
    else:
        lines.append(f"**Carbon (estimate, not a measurement):** {c['energy_kwh']:.3f} kWh x "
                     f"{s['grid_g_co2_per_kwh']} gCO2/kWh = **{c['co2_kg']:.3f} kg CO2**. {energy} "
                     f"Grid intensity: {s['grid_source']}, accessed {s['grid_accessed']}.")  # fmt: skip
    path = out_dir / "e24_compute.md"
    path.write_text("\n".join(lines) + "\n", "utf-8")
    return path


def main() -> None:
    """python -m vaultic.stage2.e24 --settings experiments/configs/EXP-224.yaml --out DIR"""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, default=None, help="ladder tuning JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--add-anomaly",
        type=Path,
        default=None,
        help="existing e24_compute.json: add the anomaly-score models (D168)",
    )
    parser.add_argument(
        "--resummarise",
        type=Path,
        default=None,
        help="existing e24_compute.json: recompute the summary with --settings",
    )
    args = parser.parse_args()
    settings = yaml.safe_load(args.settings.read_text("utf-8"))["e24"]
    started = time.perf_counter()

    def log(m):
        print(f"[{time.perf_counter() - started:6.0f}s] {m}", flush=True)

    existing = args.add_anomaly or args.resummarise
    if existing:
        result = json.loads(existing.read_text("utf-8"))
        if args.add_anomaly:
            keep = [r for r in result["models"] if r["kind"] != "anomaly-score model"]
            result["models"] = keep + time_anomaly(log)
        for r in result["models"]:  # tuning entries recomputed from the recorded fit times
            r.update(tuning_entry(r["model"], r["fit_seconds"]))
        result["settings"] = settings
        result["summary"] = summarise(result["models"], settings)
    else:
        ladder = json.loads(args.ladder.read_text("utf-8"))["rungs"] if args.ladder else None
        result = run(settings, ladder, log=log)
    print(f"wrote {write_report(result, args.out)}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
