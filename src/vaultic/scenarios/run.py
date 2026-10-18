"""Run one scenario through the Test Lab engine (Phase B) and score it three ways.

    python -m vaultic.scenarios.run --scenario card_testing --seed 0 --intensity medium

Replay = every real LEGITIMATE transaction of the scenario's day(s) (real fraud rows are left out,
so the false-positive rate is measured on known-legitimate traffic) plus the synthetic rows.

  stored (parity-verified)  a legitimate row sharing no customer, card or device with a synthetic
                            row, or earlier than the first synthetic row: its stored offline
                            features scored by the frozen system; its MVAF risk must equal the
                            stored EXP-200-final score (D189 score tolerance) or the run stops (D193)
  scored live               every synthetic row and every other legitimate row: online features
                            from the history overlay (synthetic rows in its history), then the
                            frozen system

For each row: the model-only E12 action, the rules-only action (with the fired rules) and rules +
model (the more severe). An alert is step-up or higher. Reported per threshold set (D196 primary,
D195 sensitivity): recall on the synthetic fraud rows, FPR on the legitimate replay rows, the
added FPR of rules + model over model only, and the D191 budget checks. Every output carries the
label 'synthetic scenario - not the evaluation' and is written under the dev cache only.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from vaultic.rules.engine import (
    ACTIONS,
    RULES,
    SENSITIVITY_FILE,
    SEVERITY,
    THRESHOLDS_FILE,
    load_thresholds,
    rule_values,
)
from vaultic.scenarios import LABEL, require_no_synthetic

STORED, LIVE = "stored (parity-verified)", "scored live"
FPR_ADDED_BUDGET, RULES_ONLY_BUDGET = 0.005, 0.02  # D191


def threshold_sets() -> dict[str, dict[str, float]]:
    return {"D196 (primary)": load_thresholds(THRESHOLDS_FILE),
            "D195 (sensitivity)": load_thresholds(SENSITIVITY_FILE)}  # fmt: skip


def model_actions(system, rows: pd.DataFrame, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(E12 action codes, MVAF risk) exactly as LiveEngine.score computes them."""
    rows, frame = rows.reset_index(drop=True), frame.reset_index(drop=True)
    ctx = system.context(rows, frame)
    s = system.score(frame, ctx, system.availability(frame))
    return system.decide(s["risk"], s["p"], rows["TransactionAmt"].to_numpy(float)), s["risk"]


def rule_actions(frame: pd.DataFrame, th: dict[str, float]) -> tuple[np.ndarray, list[list[str]]]:
    worst = np.zeros(len(frame), dtype=int)
    fired_by_row = [[] for _ in range(len(frame))]
    for rule in RULES:
        v = rule_values(frame, rule)
        fired = np.isfinite(v) & (v >= th[rule.rule_id])
        worst = np.where(fired, np.maximum(worst, SEVERITY[rule.action]), worst)
        for i in np.flatnonzero(fired):
            fired_by_row[i].append(rule.rule_id)
    return worst, fired_by_row


def affected(store, scenario, positions: np.ndarray) -> np.ndarray:
    """Replay rows sharing a customer, card or device with a synthetic row, at or after the first
    synthetic row (earlier rows cannot see it)."""
    from vaultic.features.behavioral import ENTITIES, _entity_key

    t_first = int(scenario.rows["TransactionDT"].min())
    hit = np.isin(store.uid[positions], scenario.uids)
    for name in ("card", "device"):
        keys = {k for k in _entity_key(scenario.rows, ENTITIES[name]) if k is not None}
        hit |= np.isin(store.entity_keys[name][positions].astype(object), list(keys))
    return hit & (store.time[positions] >= t_first)


def run_scenario(store, system, stored_scores: pd.DataFrame, scenario, log=print) -> dict:
    from vaultic.live.features import online_features
    from vaultic.live.parity import SCORE_TOLERANCE
    from vaultic.scenarios.overlay import with_synthetic

    started = time.perf_counter()
    days = np.unique(scenario.rows["day"].to_numpy())
    df = store.df
    y = df["isFraud"].to_numpy()
    pos = np.flatnonzero(np.isin(df["day"].to_numpy(), days) & (y == 0))
    require_no_synthetic(df["TransactionID"].to_numpy()[pos])
    live_mask = affected(store, scenario, pos)
    stored_pos, live_pos = pos[~live_mask], pos[live_mask]
    log(f"replay: {len(pos)} legitimate rows ({len(stored_pos)} stored, {len(live_pos)} live), "
        f"{len(scenario.rows)} synthetic rows")  # fmt: skip

    # stored legitimate rows: offline features, verified against the stored scores
    frame_s = store.features.iloc[stored_pos].reset_index(drop=True)
    rows_s = df.iloc[stored_pos].reset_index(drop=True)
    act_s, risk_s = model_actions(system, rows_s, frame_s)
    ref = stored_scores.loc[rows_s["TransactionID"].to_numpy(), "score_MVAF"].to_numpy(float)
    diff = float(np.max(np.abs(ref - risk_s))) if len(ref) else 0.0
    if diff > SCORE_TOLERANCE:
        raise RuntimeError(f"stored rows do not reproduce the stored scores (max diff {diff})")

    # live rows: synthetic rows and affected legitimate rows, with the overlay
    overlay = with_synthetic(store, scenario.rows, scenario.uids, log=log)
    live_rows = pd.concat([scenario.rows, df.iloc[live_pos]], ignore_index=True)
    live_uid = np.concatenate([scenario.uids, store.uid[live_pos]])
    live_syn = np.r_[np.ones(len(scenario.rows), bool), np.zeros(len(live_pos), bool)]
    live_lab = np.r_[scenario.labels, np.zeros(len(live_pos), int)]
    order = np.argsort(live_rows["TransactionDT"].to_numpy(), kind="stable")
    live_rows, live_uid = live_rows.iloc[order].reset_index(drop=True), live_uid[order]
    live_syn, live_lab = live_syn[order], live_lab[order]
    frames = [online_features(overlay, system, live_rows.iloc[[i]], int(live_uid[i]))
              for i in range(len(live_rows))]  # fmt: skip
    frame_l = pd.concat(frames, ignore_index=True)
    overlay_check(store, scenario, live_rows, live_syn, frame_l)
    act_l, risk_l = model_actions(system, live_rows, frame_l)

    rows = pd.concat([live_rows, rows_s], ignore_index=True)
    frame = pd.concat([frame_l, frame_s], ignore_index=True)
    frame["TransactionAmt"] = rows["TransactionAmt"].to_numpy(float)
    table = pd.DataFrame({
        "TransactionID": rows["TransactionID"].to_numpy(),
        "TransactionDT": rows["TransactionDT"].to_numpy(),
        "amount": rows["TransactionAmt"].to_numpy(float),
        "synthetic": np.r_[live_syn, np.zeros(len(rows_s), bool)],
        "label": np.r_[live_lab, np.zeros(len(rows_s), int)],
        "source": [LIVE] * len(live_rows) + [STORED] * len(rows_s),
        "risk": np.r_[risk_l, risk_s],
        "model_action": np.r_[act_l, act_s].astype(int),
    })  # fmt: skip
    results = {}
    for set_name, th in threshold_sets().items():
        r_act, fired = rule_actions(frame, th)
        both = np.maximum(r_act, table["model_action"].to_numpy())
        results[set_name] = summarise(table, r_act, both)
        if set_name.startswith("D196"):
            table["rule_action"] = r_act
            table["rules_fired"] = [",".join(f) for f in fired]
            table["combined_action"] = both
    for c in ("model_action", "rule_action", "combined_action"):
        table[c] = [ACTIONS[a] for a in table[c]]
    table = table.sort_values("TransactionDT").reset_index(drop=True)
    return {"label": LABEL, "scenario": scenario.name, "seed": scenario.seed,
            "intensity": scenario.intensity, "days": [int(d) for d in days],
            "info": scenario.info, "n_synthetic": int(len(scenario.rows)),
            "n_legit_replay": int(len(pos)), "n_legit_stored": int(len(stored_pos)),
            "n_legit_live": int(len(live_pos)), "stored_score_max_diff": diff,
            "seconds": round(time.perf_counter() - started, 1), "results": results,
            "rows": table}  # fmt: skip


def overlay_check(store, scenario, live_rows, live_syn, frame) -> None:
    """Each synthetic row's 1-hour customer count must equal the stored rows of that customer in
    the hour before it plus the earlier synthetic rows of that customer in that hour."""
    t_syn = scenario.rows["TransactionDT"].to_numpy(np.int64)
    for i in np.flatnonzero(live_syn):
        t = int(live_rows["TransactionDT"].iloc[i])
        u = int(scenario.uids[np.flatnonzero(t_syn == t)[0]])
        prev = store.before(store.uid_rows.get(u), t)
        n_stored = int((store.time[prev] >= t - 3600).sum())
        same = scenario.uids == u
        n_syn = int(((t_syn < t) & (t_syn >= t - 3600) & same).sum())
        got = float(frame["uid_n_1h"].iloc[i])
        if got != n_stored + n_syn:
            raise RuntimeError(f"overlay check failed at t={t}: uid_n_1h {got} != "
                               f"{n_stored} stored + {n_syn} synthetic")  # fmt: skip


def summarise(table: pd.DataFrame, rule_act: np.ndarray, both: np.ndarray) -> dict:
    step = SEVERITY["step_up"]
    syn = table["synthetic"].to_numpy() & (table["label"].to_numpy() == 1)
    legit = ~table["synthetic"].to_numpy()
    model = table["model_action"].to_numpy() >= step
    rules, comb = rule_act >= step, both >= step

    def rate(mask, alert):
        return float(alert[mask].mean()) if mask.any() else None

    out = {}
    for name, alert in (("rules_only", rules), ("model_only", model), ("rules_plus_model", comb)):
        first = np.flatnonzero(alert[syn])
        out[name] = {"recall_synthetic": rate(syn, alert), "fpr_legit": rate(legit, alert),
                     "alerts_synthetic": int(alert[syn].sum()),
                     "alerts_legit": int(alert[legit].sum()),
                     "first_alerted_synthetic_row": int(first[0]) + 1 if len(first) else None}  # fmt: skip
    added = out["rules_plus_model"]["fpr_legit"] - out["model_only"]["fpr_legit"]
    out["added_fpr"] = added
    out["budget_added_fpr_ok"] = bool(added <= FPR_ADDED_BUDGET)
    out["budget_rules_only_ok"] = bool(out["rules_only"]["fpr_legit"] <= RULES_ONLY_BUDGET)
    return out


def default_day(seed: int) -> int:
    """The day card_testing draws for this seed (its generator's first draw)."""
    from vaultic.data.splits import load_splits

    s = load_splits()
    return int(np.random.default_rng(seed).integers(int(s.test.first), int(s.test.last) + 1))


def main() -> None:
    import argparse

    from vaultic.live.engine import load_system
    from vaultic.live.store import build_store, cache_root
    from vaultic.paths import REPO_ROOT
    from vaultic.scenarios.generators import GENERATORS, INTENSITY
    from vaultic.stage2.final import E10_RUN

    p = argparse.ArgumentParser(description=f"run one attack scenario ({LABEL})")
    p.add_argument("--scenario", default="card_testing", choices=sorted(GENERATORS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--intensity", default="medium", choices=sorted(INTENSITY))
    a = p.parse_args()
    day = default_day(a.seed)
    t0 = time.perf_counter()
    system = load_system()
    store = build_store(days=[day])
    stored = pd.read_parquet(REPO_ROOT / E10_RUN / "predictions.parquet").set_index("TransactionID")
    print(f"loaded in {time.perf_counter() - t0:.0f} s", flush=True)
    sc = GENERATORS[a.scenario](store, seed=a.seed, intensity=a.intensity)
    res = run_scenario(store, system, stored, sc)
    out = cache_root() / "scenarios" / "results" / f"{a.scenario}_s{a.seed}_{a.intensity}"
    out.mkdir(parents=True, exist_ok=True)
    res["rows"].to_parquet(out / "rows.parquet", index=False)
    summary = {k: v for k, v in res.items() if k != "rows"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
    print(json.dumps(summary, indent=1))
    print(res["rows"][res["rows"]["source"] == LIVE].to_string(max_rows=80))


if __name__ == "__main__":
    main()
