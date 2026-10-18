"""Per-row diagnosis of a scenario's synthetic rows (Phase B; synthetic scenario - not the evaluation).

    python -m vaultic.scenarios.diagnose --scenario card_testing --seed 0 --intensity medium

For every synthetic row: MVAF risk, E12 action, conformal set, per-view calibrated scores and gate
weights, the velocity inputs, and the top 5 feature attributions (TreeSHAP of the view models
weighted by the gate, as the Test Lab shows them). Two checks separate model behaviour from a
feature bug:

  offline rebuild   the customer's real history plus ALL synthetic rows run through the offline
                    builders in one batch (features/pipeline.build_features,
                    features/behavioral.build_behavioral); every base and customer-behavioural
                    column must equal the per-row online features from the overlay
  training range    for the velocity inputs, the training-period distribution (days 1-120) and
                    the training fraud rate per value, to see whether the synthetic values are
                    inside what the model saw
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from vaultic.scenarios import LABEL

VELOCITY = ("uid_n_1h", "uid_n_24h", "uid_amt_sum_1h", "uid_secs_since_prev", "vel_n_1h",
            "vel_n_24h", "uid_n_past")  # fmt: skip


def offline_rebuild_check(store, scenario, frames: pd.DataFrame) -> dict:
    from vaultic.features.behavioral import build_behavioral
    from vaultic.features.pipeline import INPUT_COLUMNS, build_features

    u = int(scenario.uids[0])
    t0 = int(scenario.rows["TransactionDT"].min())
    hist = store.before(store.uid_rows.get(u), t0)
    rows = pd.concat([store.df.iloc[hist], scenario.rows], ignore_index=True)
    uid = pd.Series(np.r_[store.uid[hist], scenario.uids])
    base = build_features(rows[INPUT_COLUMNS], uid, store.encoders, store.splits.label_delay_days)
    beh = build_behavioral(rows, uid)
    n = len(scenario.rows)
    off = pd.concat([base.iloc[-n:].reset_index(drop=True),
                     beh.iloc[-n:].reset_index(drop=True)], axis=1)  # fmt: skip
    off = off.loc[:, ~off.columns.duplicated()]
    cols = [c for c in off.columns if c in frames.columns and c != "TransactionID"
            and not c.startswith("ent_uids_")]  # fmt: skip
    worst, col = 0.0, None
    for c in cols:
        a = off[c].to_numpy(float)
        b = frames[c].to_numpy(float)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            return {"passed": False, "column": c, "reason": "missing pattern differs"}
        ok = ~np.isnan(a)
        d = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0
        if d > worst:
            worst, col = d, c
    return {"passed": worst <= 1e-4, "columns_compared": len(cols), "max_abs_diff": worst,
            "worst_column": col}  # fmt: skip


def training_reference(store, columns) -> dict:
    tr = store.splits.train.contains(store.df["day"].to_numpy())
    y = store.df["isFraud"].to_numpy()[tr]
    out = {}
    for c in columns:
        if c not in store.features:
            continue
        v = store.features[c].to_numpy(float)[tr]
        ok = np.isfinite(v)
        q = np.percentile(v[ok], [50, 90, 99, 99.9]).tolist() if ok.any() else []
        out[c] = {"p50_p90_p99_p999": q, "max": float(v[ok].max()) if ok.any() else None}
        if c == "uid_n_1h":
            bins = [0, 1, 2, 3, 4, 5, 6, 8, 11, 16, np.inf]
            b = pd.cut(v[ok], bins, right=False)
            g = pd.DataFrame({"bin": b, "y": y[ok]}).groupby("bin", observed=False)["y"]
            out[c]["fraud_rate_by_value"] = {str(k): {"rows": int(n), "fraud_rate": float(r)}
                                             for k, n, r in zip(g.size().index, g.size(),
                                                                g.mean().fillna(0), strict=True)}  # fmt: skip
    return out


def diagnose(store, system, scenario) -> dict:
    from vaultic.live.engine import LiveEngine
    from vaultic.live.features import online_features
    from vaultic.scenarios.overlay import with_synthetic

    overlay = with_synthetic(store, scenario.rows, scenario.uids, log=lambda m: None)
    eng = LiveEngine(overlay, system)
    per_row, frames = [], []
    for i in range(len(scenario.rows)):
        q = scenario.rows.iloc[[i]]
        u = int(scenario.uids[i])
        r = eng.score(q, u, explain=True, record_latency=False)
        f = online_features(overlay, system, q, u)
        frames.append(f)
        per_row.append({
            "charge": i + 1, "amount": float(q["TransactionAmt"].iloc[0]),
            "risk": r["risk"], "calibrated": r["calibrated"], "action": r["decision"],
            "conformal_set": r["conformal_set"], "disagreement": r["disagreement"],
            "views": {v["view"]: {"p": v["p"], "weight": v["weight"]} for v in r["views"]},
            "velocity": {c: float(f[c].iloc[0]) for c in VELOCITY if c in f},
            "top5": [{"feature": x["feature"], "points": x["points"],
                      "value": float(f[x["feature"]].iloc[0]) if x["feature"] in f else None}
                     for x in r["reasons"][:5]],
        })  # fmt: skip
    frames = pd.concat(frames, ignore_index=True)
    top = sorted({x["feature"] for r in per_row for x in r["top5"]} | set(VELOCITY))
    return {"label": LABEL, "scenario": scenario.name, "seed": scenario.seed,
            "intensity": scenario.intensity, "rows": per_row,
            "offline_rebuild": offline_rebuild_check(store, scenario, frames),
            "training_reference": training_reference(store, top)}  # fmt: skip


def main() -> None:
    import argparse

    from vaultic.live.engine import load_system
    from vaultic.live.store import build_store, cache_root
    from vaultic.scenarios.generators import GENERATORS, INTENSITY
    from vaultic.scenarios.run import default_day

    p = argparse.ArgumentParser(description=f"diagnose a scenario's synthetic rows ({LABEL})")
    p.add_argument("--scenario", default="card_testing", choices=sorted(GENERATORS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--intensity", default="medium", choices=sorted(INTENSITY))
    a = p.parse_args()
    system = load_system(log=lambda m: None)
    store = build_store(log=lambda m: None, days=[default_day(a.seed)])
    sc = GENERATORS[a.scenario](store, seed=a.seed, intensity=a.intensity)
    out = diagnose(store, system, sc)
    path = cache_root() / "scenarios" / "results" / f"{a.scenario}_s{a.seed}_{a.intensity}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "diagnosis.json").write_text(json.dumps(out, indent=1), "utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
