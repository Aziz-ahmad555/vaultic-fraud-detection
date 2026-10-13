"""Stage-2 experiments: development runs (validation only) and the final procedures that
stage2/final.py calls after its guards (D121; D158-D164).

Development (validation only, outputs to E:/dev-cache/stage2, never results):
  python -m vaultic.stage2.run ladder-tune          F4 tuned per rung 1-4 on the inner split ->
                                                    research/tables/stage2_ladder_tuned.json
  python -m vaultic.stage2.run ladder-dev           pipeline check: calibrate 147-148, eval 149-150
  python -m vaultic.stage2.run cf-grid              counterfactual settings on 200 validation
                                                    alerts -> research/tables/stage2_cf_setting.json
                                                    (run in .venv-dice)
  python -m vaultic.stage2.run e15-dev [--quick]    pipeline check on calibrate_fused alerts
                                                    (run in .venv-dice)
  python -m vaultic.stage2.run e23-dev              pipeline check on validation (B5, F0 only)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date

import numpy as np
import pandas as pd

from vaultic.paths import REPO_ROOT, RESEARCH_DIR
from vaultic.stage2.system import DEV_DIR, VIEW_TABLE

LADDER_TUNED = RESEARCH_DIR / "tables" / "stage2_ladder_tuned.json"
CF_SETTING = RESEARCH_DIR / "tables" / "stage2_cf_setting.json"
CF_GRID_ALERTS = 200
DICE_TRAIN_ROWS = 20_000
POSTHOC = "binary"  # DiCE's post-hoc sparsity search (D164)


def _clock():
    started = time.perf_counter()
    return lambda m: print(f"[{time.perf_counter() - started:7.0f}s] {m}", flush=True)


def _git_head() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def _stored_e10() -> pd.DataFrame:
    """The EXP-200-final test scores. Final procedures only (after FINAL-STARTED)."""
    from vaultic.stage2.final import E10_RUN

    return pd.read_parquet(REPO_ROOT / E10_RUN / "predictions.parquet").set_index("TransactionID")


def _e10_thresholds() -> dict:
    final = json.loads((RESEARCH_DIR / "tables" / "final_results.json").read_text("utf-8"))
    return final["meta"]["thresholds"]


def _write(out_dir, name: str, obj) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(out_dir / f"{name}.csv", index=False)
    else:
        (out_dir / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str), "utf-8")


# ---- E23 -----------------------------------------------------------------------------------


def _e23_inputs(ids):
    from vaultic.data.load import load_merged
    from vaultic.data.splits import load_splits
    from vaultic.paths import FEATURES_DIR, MERGED_PATH

    splits = load_splits()
    m = load_merged(MERGED_PATH, columns=["TransactionID", "day", "ProductCD", "DeviceType",
                                          "P_emaildomain"])  # fmt: skip
    train_emails = m.loc[splits.train.contains(m["day"].to_numpy()), "P_emaildomain"]
    raw = m.set_index("TransactionID").reindex(ids).reset_index()
    beh = pd.read_parquet(FEATURES_DIR / f"behavioral_{splits.uid_variant}.parquet",
                          columns=["TransactionID", "hist_n_past"])  # fmt: skip
    hist = beh.set_index("TransactionID")["hist_n_past"].reindex(ids).to_numpy()
    return raw, train_emails, hist


def _e23(ids, y, scores, thresholds, cfg, out_dir, label: str) -> dict:
    from vaultic.stage2.e23 import attribute_groups, e23_tables

    raw, train_emails, hist = _e23_inputs(ids)
    groups = attribute_groups(raw, train_emails, hist)
    t = e23_tables(y, scores, groups, thresholds, cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"])
    for k in ("groups", "gaps", "mvaf_minus_b5"):
        _write(out_dir, f"e23_{k}", t[k])
    meta = {"label": label, "rows": int(len(y)), "frauds": int(np.sum(y)),
            "thresholds": t["thresholds"], "methods": list(scores)}  # fmt: skip
    _write(out_dir, "e23_meta", meta)
    return meta


def e23_final(cfg, run_dir, head) -> dict:
    from vaultic.stage2.final import OUTPUT_DIR

    stored = _stored_e10()
    ids, y = stored.index.to_numpy(), stored["label"].to_numpy()
    scores = {m: stored[f"score_{m}"].to_numpy() for m in cfg["methods"]}
    th = {m: _e10_thresholds()[m] for m in cfg["methods"]}
    return _e23(ids, y, scores, th, cfg, OUTPUT_DIR, "IEEE-CIS test period, EXP-200-final scores")


def e23_dev() -> None:
    """Pipeline check on validation (stored EXP-009 B5 and EXP-F0-inner F0 validation scores;
    thresholds chosen on calibrate_fused, evaluated on all validation rows). Not a result."""
    from vaultic.eval.metrics import choose_cost_threshold
    from vaultic.fusion.final_run import _stored
    from vaultic.stage2.final import load_config

    cfg = load_config("EXP-223-final")
    b5 = _stored("experiments/runs/EXP-009/20261008-211052-888539", "validation")
    f0 = _stored("experiments/runs/EXP-F0-inner/20261009-213740-324011", "validation")
    from vaultic.data.load import load_merged
    from vaultic.paths import MERGED_PATH

    m = load_merged(MERGED_PATH, columns=["TransactionID", "day", "isFraud", "TransactionAmt"])
    m = m[(m["day"] >= 128) & (m["day"] <= 150)].set_index("TransactionID")
    ids = m.index.intersection(b5.index).intersection(f0.index).to_numpy()
    y, amt, day = (m.loc[ids, c].to_numpy() for c in ("isFraud", "TransactionAmt", "day"))
    scores = {"B5": b5.reindex(ids).to_numpy(), "F0": f0.reindex(ids).to_numpy()}
    cal = day >= 147
    th = {k: choose_cost_threshold(y[cal], s[cal], amt[cal]) for k, s in scores.items()}
    meta = _e23(ids, y, scores, th, {**cfg, "bootstrap": {"n": 200, "seed": 0}}, DEV_DIR / "e23",
                "DEVELOPMENT pipeline check on validation; not a result")  # fmt: skip
    print(json.dumps(meta, indent=1, default=str))


# ---- ladder ---------------------------------------------------------------------------------


def _d95_split(table=None):
    from vaultic.fusion.data import fusion_split
    from vaultic.fusion.tune_fusion import concat_rows
    from vaultic.views.orchestrate import calibrate_views

    table = pd.read_parquet(VIEW_TABLE) if table is None else table
    table, _ = calibrate_views(table)
    split = fusion_split(table)
    return split, concat_rows(split.fit, split.tune)


def ladder_tune() -> None:
    from vaultic.stage2.ladder import TRIALS, tune_rungs

    log = _clock()
    split, _ = _d95_split()
    if len(split.test):
        raise RuntimeError("the D95 view table holds test rows")
    tuned = tune_rungs(split, TRIALS, 0, log)
    out = {"generated": date.today().isoformat(), "git_commit": _git_head(),
           "note": "F4 tuned per ladder rung on the inner split of the D95 gate rows (fit -> "
                   "tune, validation only), 50 Optuna trials, TPE seed 0 (D121 section 2)",
           "rungs": tuned}  # fmt: skip
    LADDER_TUNED.write_text(json.dumps(out, indent=1), "utf-8")
    print(f"wrote {LADDER_TUNED}")


def _ladder_scores(both, cal, ev, tuned, seeds, e10_f4, contexts=None):
    """Rungs 1-4 (tuned) and rung 5 (E10's F4 settings) fitted with seeds."""
    from vaultic.stage2.ladder import NEW_RUNGS, fit_rung_scores, rung_name

    s_cal, s_ev, per_seed = {}, {}, {}
    for k in (*NEW_RUNGS, 5):
        params = tuned[str(k)]["params"] if k != 5 else e10_f4
        s_cal[rung_name(k)], s_ev[rung_name(k)], per_seed[rung_name(k)] = fit_rung_scores(
            both, cal, ev, params, k, seeds, contexts)  # fmt: skip
    return s_cal, s_ev, per_seed


def _ladder_tables(ev, scores, per_seed, amount, thresholds, cfg, out_dir, prefix):
    from vaultic.fusion.final_eval import subgroup_masks
    from vaultic.stage2.ladder import ladder_comparisons, rung_metrics, rung_subgroups
    from vaultic.views.orchestrate import CONTEXT

    n, seed = cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"]
    steps, refs = ladder_comparisons(ev.y, scores, per_seed, amount, thresholds, n, seed)
    metrics = rung_metrics(ev.y, scores, amount, thresholds, n, seed)
    masks = subgroup_masks(ev.context[:, CONTEXT.index("ctx_hist_n_past")],
                           ev.context[:, CONTEXT.index("ctx_has_identity")],
                           (~np.isnan(ev.views)).astype(int))  # fmt: skip
    subs = rung_subgroups(ev.y, scores, masks)
    for name, t in (("steps", steps), ("references", refs), ("metrics", metrics),
                    ("subgroups", subs)):  # fmt: skip
        _write(out_dir, f"{prefix}{name}", t)
    return steps


def _ladder_run(both, cal, ev, amount_cal, amount_ev, cfg, out_dir, refs: dict, log,
                stored_f4=None) -> dict:  # fmt: skip
    """Primary ladder + bounded-count sensitivity. refs: name -> (scores, threshold)."""
    from vaultic.eval.metrics import choose_cost_threshold
    from vaultic.fusion.final_eval import refit_check
    from vaultic.fusion.final_run import _params
    from vaultic.stage2.ladder import bounded_context, rung_name
    from vaultic.stage2.system import FUSION_PARAMS
    from vaultic.views.orchestrate import CONTEXT

    seeds = tuple(cfg["seeds"])
    tuned = json.loads(LADDER_TUNED.read_text("utf-8"))["rungs"]
    e10_f4 = _params(json.loads(FUSION_PARAMS.read_text("utf-8"))["tuned"]["F4"]["params"])
    meta = {}
    s_cal, s_ev, per_seed = _ladder_scores(both, cal, ev, tuned, seeds, e10_f4)
    log("ladder rungs fitted")
    th = {m: choose_cost_threshold(cal.y, s_cal[m], amount_cal) for m in s_cal}
    if stored_f4 is not None:  # final: rung 5 must reproduce E10's F4, then the stored scores
        from vaultic.stage2.system import refit_complete

        meta["rung5_reproduction"] = refit_check(stored_f4, ev.ids, s_ev[rung_name(5)])
        if not refit_complete(meta["rung5_reproduction"], len(ev.ids)):  # D173
            raise RuntimeError(f"rung 5 does not reproduce E10's F4: {meta['rung5_reproduction']}")
        s_ev[rung_name(5)] = stored_f4.reindex(ev.ids).to_numpy()
        # D174: rung 5 is E10's F4, so its cost threshold must be E10's F4 threshold
        meta["rung5_threshold_diff"] = abs(th[rung_name(5)] - _e10_thresholds()["F4"])
        if meta["rung5_threshold_diff"] > 1e-6:
            raise RuntimeError(f"rung 5's threshold {th[rung_name(5)]} differs from E10's F4 "
                               f"threshold {_e10_thresholds()['F4']}")  # fmt: skip
    scores = dict(s_ev)
    for name, (s, t) in refs.items():
        scores[name], th[name] = s, t
    steps = _ladder_tables(ev, scores, per_seed, amount_ev, th, cfg, out_dir, "ladder_")
    meta["thresholds"] = th
    meta["h1"] = steps[steps["metric"] == "pr_auc"][["step", "diff", "p_holm",
                                                     "h1_supported"]].to_dict("records")  # fmt: skip
    # bounded-count sensitivity (D117-D119): not in the H1 family
    j = CONTEXT.index("ctx_hist_n_past")
    gate_max = float(np.nanmax(both.context[:, j]))
    ctx = tuple(bounded_context(r.context, gate_max) for r in (both, cal, ev))
    b_cal, b_ev, b_seed = _ladder_scores(both, cal, ev, tuned, seeds, e10_f4, ctx)
    b_th = {m: choose_cost_threshold(cal.y, b_cal[m], amount_cal) for m in b_cal}
    for name, (s, t) in refs.items():
        b_ev[name], b_th[name] = s, t
    _ladder_tables(ev, b_ev, b_seed, amount_ev, b_th, cfg, out_dir, "ladder_bounded_")
    meta["bounded_gate_max"] = gate_max
    log("bounded-count sensitivity done")
    return meta


def ladder_dev() -> None:
    """Pipeline check: calibration rows 147-148, evaluation rows 149-150. Not a result."""
    from vaultic.data.load import load_merged
    from vaultic.fusion.final_run import _rows_subset, _stored
    from vaultic.paths import MERGED_PATH
    from vaultic.stage2.final import load_config

    log = _clock()
    cfg = {**load_config("EXP-210-final"), "bootstrap": {"n": 200, "seed": 0}}
    split, both = _d95_split()
    cal = _rows_subset(split.calibrate, split.calibrate.day <= 148)
    ev = _rows_subset(split.calibrate, split.calibrate.day >= 149)
    amt = load_merged(MERGED_PATH, columns=["TransactionID", "TransactionAmt"]).set_index(
        "TransactionID")["TransactionAmt"]  # fmt: skip
    from vaultic.eval.metrics import choose_cost_threshold

    b5 = _stored("experiments/runs/EXP-009/20261008-211052-888539", "validation")
    s_b5_cal, s_b5 = b5.reindex(cal.ids).to_numpy(), b5.reindex(ev.ids).to_numpy()
    a_cal, a_ev = amt.reindex(cal.ids).to_numpy(float), amt.reindex(ev.ids).to_numpy(float)
    refs = {"B5": (s_b5, choose_cost_threshold(cal.y, s_b5_cal, a_cal))}
    meta = _ladder_run(both, cal, ev, a_cal, a_ev, cfg, DEV_DIR / "ladder", refs, log)
    meta["label"] = "DEVELOPMENT pipeline check (calibrate 147-148, evaluate 149-150); not a result"
    _write(DEV_DIR / "ladder", "ladder_meta", meta)
    print(json.dumps(meta, indent=1, default=str))


def ladder_final(cfg, run_dir, head) -> dict:
    from vaultic.data.load import load_merged
    from vaultic.fusion.dev_compare import build_table
    from vaultic.fusion.tune_fusion import concat_rows
    from vaultic.paths import MERGED_PATH
    from vaultic.stage2.final import OUTPUT_DIR

    log = _clock()
    table = build_table(False, d95=True, final=True)
    log("view table built (final)")
    split, _ = _d95_split(table)
    both, cal, ev = concat_rows(split.fit, split.tune), split.calibrate, split.test
    amt = load_merged(MERGED_PATH, columns=["TransactionID", "TransactionAmt"]).set_index(
        "TransactionID")["TransactionAmt"]  # fmt: skip
    a_cal, a_ev = amt.reindex(cal.ids).to_numpy(float), amt.reindex(ev.ids).to_numpy(float)
    stored = _stored_e10()
    th = _e10_thresholds()
    refs = {m: (stored[f"score_{m}"].reindex(ev.ids).to_numpy(), th[m]) for m in ("MVAF", "B5")}
    meta = _ladder_run(both, cal, ev, a_cal, a_ev, cfg, OUTPUT_DIR, refs, log,
                       stored_f4=stored["score_F4"])  # fmt: skip
    meta.update({"evaluation_rows": int(len(ev.y)), "evaluation_frauds": int(ev.y.sum()),
                 "git_commit": head})  # fmt: skip
    _write(OUTPUT_DIR, "ladder_meta", meta)
    return meta


# ---- E15 ------------------------------------------------------------------------------------


def _alert_rows(system, parts, rows_mask, kind: str):
    """Positions (in parts' frames) of the alerts of `kind` among rows_mask."""
    from vaultic.stage2.system import require_no_test

    df, features = parts["df"], parts["features"]
    pos = np.flatnonzero(rows_mask)
    require_no_test(parts, df["day"].to_numpy()[pos])
    frame = features.loc[pos].reset_index(drop=True)
    if kind == "B5":
        s, _ = system.b5(frame)
        return pos[s >= system.b5_threshold]
    ctx = system.context(df.loc[pos].reset_index(drop=True), frame)
    sc = system.score(frame, ctx)
    amount = df["TransactionAmt"].to_numpy(float)[pos]
    return pos[system.is_alert(sc["risk"], sc["p"], amount)]


def _sample(pos: np.ndarray, n: int, seed: int) -> np.ndarray:
    """D160: one permutation (seed); its first n positions (all if fewer)."""
    rng = np.random.default_rng(seed)
    return pos[rng.permutation(len(pos))][:n]


def _cf_problem(system, parts, pos, kind):
    from vaultic.stage2.amount import AmountHistory
    from vaultic.stage2.counterfactuals import Problem, ProductCodes, ProductHistory

    df, features = parts["df"], parts["features"]
    uid = features["uid"].to_numpy()
    frame = features.loc[pos].reset_index(drop=True)
    rows = df.loc[pos].reset_index(drop=True)
    ctx = system.context(rows, frame)
    codes = ProductCodes(df, features, system.encoder)
    return Problem(system, kind, rows, frame, ctx, system.availability(frame),
                   AmountHistory(df, uid, pos), ProductHistory(df, uid, pos), codes)  # fmt: skip


def _dice_train(parts, seed: int = 0) -> tuple[pd.DataFrame, float, np.ndarray]:
    """DiCE's data frame (D158: 20,000 training-period rows, seed 0; the outcome column is the
    fraud label, which the random method does not use), the training MAD of the amount and the
    training amounts."""
    df, splits = parts["df"], parts["splits"]
    train = df[splits.train.contains(df["day"].to_numpy())]
    amounts = train["TransactionAmt"].to_numpy(float)
    pick = np.random.default_rng(seed).choice(len(train), DICE_TRAIN_ROWS, replace=False)
    t = train.iloc[np.sort(pick)]
    frame = pd.DataFrame({"TransactionAmt": t["TransactionAmt"].to_numpy(float),
                          "ProductCD": t["ProductCD"].astype(str).to_numpy(),
                          "alert": t["isFraud"].to_numpy(int)})  # fmt: skip
    mad = float(np.median(np.abs(amounts - np.median(amounts))))
    return frame, mad, amounts


def cf_grid() -> None:
    """The D158 grid on 200 MVAF alerts of calibrate_fused (147-150), then the selection rule."""
    import joblib

    from vaultic.stage2.counterfactuals import choose_setting, grid_settings, run_setting
    from vaultic.stage2.system import dev_system_path, load_parts

    log = _clock()
    system = joblib.load(dev_system_path())
    parts = load_parts(dev=True)
    day = parts["df"]["day"].to_numpy()
    alerts = _alert_rows(system, parts, (day >= 147) & (day <= 150), "MVAF")
    pos = _sample(alerts, CF_GRID_ALERTS, 0)
    log(f"{len(alerts)} MVAF alerts in 147-150; {len(pos)} sampled")
    problem = _cf_problem(system, parts, pos, "MVAF")
    train, mad, amounts = _dice_train(parts)
    results, summary = [], []
    for setting in grid_settings():
        r = run_setting(problem, train, {**setting, "posthoc": POSTHOC}, mad, amounts, 0, log)
        results.append((setting, r))
        summary.append({**setting, "validity": float(r["validity"].mean()),
                        "any_valid": float(r["any_valid"].mean()),
                        "proximity": float(np.nanmean(r["proximity"])) if r["proximity"].notna().any() else None,
                        "sparsity": float(np.nanmean(r["sparsity"])) if r["sparsity"].notna().any() else None})  # fmt: skip
        log(f"{setting}: validity {summary[-1]['validity']:.3f}")
        _write(DEV_DIR / "cf_grid", "cf_grid_partial", summary)
    chosen = choose_setting(results)
    out = {"generated": date.today().isoformat(), "git_commit": _git_head(),
           "note": "D158 (2) grid on 200 MVAF alerts (seed 0) of calibrate_fused, days 147-150, "
                   "validation only; selection: highest mean validity, ties within 0.01 -> lower "
                   "proximity, fewer changed inputs, cheaper setting. Validity boundary D159; "
                   "varied inputs D163; DiCE post-hoc sparsity 'binary' (D164)",
           "alerts_available": int(len(alerts)), "alerts": int(len(pos)),
           "chosen": {**chosen, "posthoc": POSTHOC}, "grid": summary}  # fmt: skip
    CF_SETTING.write_text(json.dumps(out, indent=1), "utf-8")
    print(f"wrote {CF_SETTING}: {chosen}")


def _e15(system, parts, eval_mask, perm_mask, cfg, out_dir, log, n_main, n_cf, quick=False):
    """Everything in E15 for both systems on the alerts among eval_mask."""
    from vaultic.eval.metrics import pr_auc  # noqa: F401  (imported for clarity of scope)
    from vaultic.stage2.amount import AmountHistory, with_amount
    from vaultic.stage2.counterfactuals import run_setting
    from vaultic.stage2.e15 import (
        B5Explained,
        MvafExplained,
        faithfulness,
        mean_ci,
        permutation_importance,
        seed_consistency,
        stability,
        view_deletion,
    )

    df, features = parts["df"], parts["features"]
    uid = features["uid"].to_numpy()
    seed, n_boot = cfg["sample_seed"], cfg["bootstrap"]["n"]
    setting = json.loads(CF_SETTING.read_text("utf-8"))["chosen"]
    train, mad, amounts = _dice_train(parts)
    perm_pos = np.flatnonzero(perm_mask)
    if quick:
        perm_pos = perm_pos[:: max(1, len(perm_pos) // 2000)]
    perm_frame = features.loc[perm_pos].reset_index(drop=True)
    perm_ctx = system.context(df.loc[perm_pos].reset_index(drop=True), perm_frame)
    perm_avail = system.availability(perm_frame)
    perm_y = df["isFraud"].to_numpy()[perm_pos]
    out = {}
    for expl in (MvafExplained(system), B5Explained(system)):
        kind = expl.name
        alerts = _alert_rows(system, parts, eval_mask, kind)
        pos = _sample(alerts, n_main, seed)
        frame = features.loc[pos].reset_index(drop=True)
        ctx = system.context(df.loc[pos].reset_index(drop=True), frame)
        avail = system.availability(frame)
        log(f"{kind}: {len(alerts)} alerts, {len(pos)} sampled")
        imp = permutation_importance(expl, perm_frame, perm_ctx, perm_avail, perm_y,
                                     1 if quick else cfg["permutation_repeats"], seed, log)  # fmt: skip
        res = {"alerts_available": int(len(alerts)), "alerts": int(len(pos)),
               "permutation_importance_top10": [expl.columns[j] for j in np.argsort(-imp)[:10]]}  # fmt: skip
        res.update(faithfulness(expl, frame, ctx, avail, system.medians, imp, n_boot, seed, log))
        if kind == "MVAF":
            res["view_deletion"] = view_deletion(system, frame, ctx, avail, seed)
        attr0 = expl.attributions(frame, ctx, avail)
        hist = AmountHistory(df, uid, pos)
        perturbed = {}
        for f in cfg["perturbations"]:
            fr, cx = with_amount(system, frame, ctx, hist, frame["TransactionAmt"].to_numpy() * f)
            perturbed[f] = expl.attributions(fr, cx, avail)
        res["stability"] = stability(expl, attr0, perturbed)
        res["seed_consistency"] = seed_consistency(expl.attributions(frame, ctx, avail, True),
                                                   cfg["seed_top"])  # fmt: skip
        log(f"{kind}: stability and seeds done")
        cf_pos = pos[:n_cf]  # nested sample (D160)
        problem = _cf_problem(system, parts, cf_pos, kind)
        cf = run_setting(problem, train, setting, mad, amounts, seed, log)
        _write(out_dir, f"e15_{kind}_counterfactuals", cf)
        res["counterfactuals"] = {k: mean_ci(cf[k].to_numpy(float), n_boot, seed)
                                  for k in ("validity", "any_valid", "loose_validity", "proximity",
                                            "sparsity", "actionable_share")}  # fmt: skip
        res["counterfactuals"]["setting"] = setting
        v = res["counterfactuals"]["validity"]
        res["counterfactuals"]["exit_criterion_met"] = bool(
            kind == "MVAF" and v["mean"] > cfg["validity_target"])  # fmt: skip
        out[kind] = res
        _write(out_dir, "e15_results", out)
        log(f"{kind} done")
    gate = {"H6a_MVAF_supported": out["MVAF"]["hypotheses"]["H6a"]["supported"],
            "MVAF_validity": out["MVAF"]["counterfactuals"]["validity"]["mean"],
            "validity_target": cfg["validity_target"]}  # fmt: skip
    gate["passed"] = bool(
        gate["H6a_MVAF_supported"] and gate["MVAF_validity"] > gate["validity_target"]
    )
    out["phase9_gate"] = gate
    _write(out_dir, "e15_results", out)
    return out


def e15_dev(quick: bool) -> None:
    """Pipeline check on calibrate_fused (147-150) alerts. Not a result."""
    import joblib

    from vaultic.stage2.final import load_config
    from vaultic.stage2.system import dev_system_path, load_parts

    log = _clock()
    cfg = {**load_config("EXP-215-final"), "bootstrap": {"n": 200, "seed": 0}}
    system = joblib.load(dev_system_path())
    parts = load_parts(dev=True)
    day = parts["df"]["day"].to_numpy()
    cal = (day >= 147) & (day <= 150)
    out = _e15(system, parts, cal, cal, cfg, DEV_DIR / "e15", log, 60 if quick else 200,
               10 if quick else 50, quick)  # fmt: skip
    print(json.dumps(out["phase9_gate"], indent=1))


def e15_final(cfg, run_dir, head) -> dict:
    from vaultic.stage2.counterfactuals import VARY
    from vaultic.stage2.final import OUTPUT_DIR
    from vaultic.stage2.system import check_test, check_validation, fit_system, load_parts

    # D178: the frozen config's varied inputs must be the ones the code varies (D163), checked
    # before anything is loaded (an explicit check, not an assert that python -O would strip)
    if tuple(cfg["cf_vary"]) != VARY:
        raise RuntimeError(f"cf_vary {cfg['cf_vary']} differs from the code's VARY {VARY}")
    log = _clock()
    parts = load_parts(dev=False)
    system = fit_system(parts, log=log)
    from vaultic.fusion.final_run import _stored

    val = check_validation(system, parts, b5_stored=_stored(
        "experiments/runs/EXP-009/20261008-211052-888539", "validation"))  # fmt: skip
    test = check_test(system, parts, _stored_e10())
    log(f"refit checks: validation {val['passed']}, test {test['passed']}")
    if not (val["passed"] and test["passed"]):
        _write(OUTPUT_DIR, "e15_refit_checks", {"validation": val, "test": test})
        raise RuntimeError("the refitted system does not reproduce the final run; stopping")
    day = parts["df"]["day"].to_numpy()
    test_rows = day >= parts["splits"].test.first
    cal = (day >= 147) & (day <= 150)
    out = _e15(system, parts, test_rows, cal, cfg, OUTPUT_DIR, log, cfg["alerts"],
               cfg["counterfactual_alerts"])  # fmt: skip
    _write(OUTPUT_DIR, "e15_refit_checks", {"validation": val, "test": test})
    return {"phase9_gate": out["phase9_gate"], "refit_checks": {"validation": val, "test": test},
            "git_commit": head}  # fmt: skip


FINAL = {"EXP-223-final": e23_final, "EXP-210-final": ladder_final, "EXP-215-final": e15_final}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["ladder-tune", "ladder-dev", "cf-grid", "e15-dev",
                                            "e23-dev"])  # fmt: skip
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    {"ladder-tune": ladder_tune, "ladder-dev": ladder_dev, "cf-grid": cf_grid,
     "e15-dev": lambda: e15_dev(args.quick), "e23-dev": e23_dev}[args.command]()  # fmt: skip


if __name__ == "__main__":
    main()
