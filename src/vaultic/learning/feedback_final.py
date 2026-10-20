"""EXP-310-final: freeze, guards, parity, resume (D210, D211, D213, D214). NOT run without Aziz's
go-ahead.

    python -m vaultic.learning.feedback_final --freeze                write research/frozen_feedback.md
    python -m vaultic.learning.feedback_final                         ONE final run (test period)
    python -m vaultic.learning.feedback_final --resume <run folder>   resume an interrupted run

Order: (1) the guards of the earlier final runs (stage2.final / fusion.final_run): every frozen
setting equals its pre-registered value, the freeze record matches every frozen file, data file
(the stage-2 system cache with the champion models included) and library version, clean tree
descending from the frozen commit, not behind origin, package location, an O_EXCL lock, one final
run, and the FINAL-STARTED row committed and pushed BEFORE any test row is read; (2) parity: each
seed's champion booster matches the hash in the freeze record and the mean of the five seed models
reproduces the stored EXP-200-final score_B5 on every test row within 1e-6; on failure the run is
aborted (run.log, experiment log, metrics.json status "aborted", FINAL-ABORTED pushed) and cannot
be resumed; (3) the simulation, one delay at a time, each delay written to <run>/delay_L<L>/ with a
DONE marker.

Resume (D214): the same guards as a start, under the same lock; the run's recorded git_commit must
be an ancestor of HEAD with only the logs changed since, and its FINAL-STARTED row must be on
origin; then a FINAL-RESUMED row is committed and pushed BEFORE any test row is read (refused if
the push fails). A resume continues the same run and is logged; it is not a re-run. Completed
delays are skipped; a partial delay is recomputed from scratch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import date, datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vaultic.paths import CONFIG_DIR, REPO_ROOT, RESEARCH_DIR, RUNS_DIR

EXP = "EXP-310-final"
FROZEN = RESEARCH_DIR / "frozen_feedback.md"
FREEZE_RECORD = "research/frozen_feedback.md"
EXPERIMENT_LOG = "research/experiment_log.md"
LOG_FILES = {FREEZE_RECORD, EXPERIMENT_LOG, "research/decisions.md"}
OUTPUTS = ("research/tables/feedback/",)
OUTPUT_DIR = RESEARCH_DIR / "tables" / "feedback"
PARITY_TOLERANCE = 1e-6
E10_RUN = "experiments/runs/EXP-200-final/20261011-002400-202994"
CHAMPION_HEADER = "## Champion B5 seed models (SHA-256 of each XGBoost booster, JSON)"
EXPECTED = {
    ("id",): EXP, ("status",): "frozen-final", ("venv",): ".venv", ("model_config",): "EXP-009",
    ("seeds",): [0, 1, 2, 3, 4], ("delays",): [0, 7, 30, 60], ("deploy_day",): 121,
    ("end_day",): 182, ("warmup_days",): [121, 150], ("evaluation_days",): [151, 182],
    ("champion", "source"): "frozen B5 (stage-2 system cache, D179)",
    ("champion", "fit_days"): [1, 120], ("champion", "calibration_days"): [147, 150],
    ("policies",): ["P0", "P1", "P2", "P3", "P1_ungated", "P1_monthly"],
    ("bootstrap", "n"): 1000, ("bootstrap", "seed"): 0, ("e10_run",): E10_RUN,
    ("parity_tolerance",): PARITY_TOLERANCE, ("holm_alpha",): 0.05,
}  # fmt: skip


def check_settings(cfg: dict) -> None:
    bad = []
    for path, want in EXPECTED.items():
        got = cfg
        for k in path:
            got = got.get(k) if isinstance(got, dict) else None
        if got != want:
            bad.append(f"{'.'.join(path)} = {got!r} (pre-registered: {want!r})")
    if bad:
        raise ValueError(f"{EXP} settings differ from the pre-registration: " + "; ".join(bad))


def system_cache() -> Path:
    from vaultic.live.engine import STAGE2_DEV_SYSTEM

    return Path(STAGE2_DEV_SYSTEM)


def frozen_files() -> list[Path]:
    learning = REPO_ROOT / "src" / "vaultic" / "learning"
    return [CONFIG_DIR / f"{EXP}.yaml", CONFIG_DIR / "splits.yaml", CONFIG_DIR / "v_columns.yaml",
            CONFIG_DIR / "EXP-009.yaml", RESEARCH_DIR / "preregistration_feedback.md",
            RESEARCH_DIR / "tables" / "final_results.json",
            *(learning / f for f in ("feedback.py", "feedback_inputs.py", "feedback_run.py",
                                     "feedback_final.py"))]  # fmt: skip


def data_files() -> list[Path]:
    """Every data file the final run reads, the champion models included (D214)."""
    from vaultic.data.uid import UID_PATH
    from vaultic.paths import FEATURES_DIR, MERGED_PATH

    return [MERGED_PATH, UID_PATH, *sorted(FEATURES_DIR.glob("*.parquet")),
            REPO_ROOT / E10_RUN / "predictions.parquet", system_cache()]  # fmt: skip


def booster_hashes(models) -> list[str]:
    """SHA-256 of each seed model's booster (JSON), independent of how it was pickled."""
    return [hashlib.sha256(bytes(m.get_booster().save_raw(raw_format="json"))).hexdigest()
            for m in models]  # fmt: skip


def read_champion_hashes(record: str) -> list[str]:
    if CHAMPION_HEADER not in record:
        raise RuntimeError("the freeze record holds no champion model hashes")
    section = record.split(CHAMPION_HEADER, 1)[1]
    return [
        h
        for _, h in sorted(
            (int(s), h) for s, h in re.findall(r"seed (\d+): `([0-9a-f]{64})`", section)
        )
    ]


def freeze() -> None:
    import joblib

    from vaultic.fusion.final_run import data_section, env_values
    from vaultic.reports.phase2_summary import write_freeze
    from vaultic.stage2.final import library_section_venvs

    check_settings(yaml.safe_load((CONFIG_DIR / f"{EXP}.yaml").read_text("utf-8")))
    write_freeze(frozen_files(), FROZEN, title="Frozen config for the feedback-loop final run "
                 "(EXP-310-final; D210, D211, D213, D214)")  # fmt: skip
    hashes = booster_hashes(joblib.load(system_cache()).b5_models)
    with open(FROZEN, "a", encoding="utf-8") as f:
        f.write(data_section(data_files(), env_values()))
        f.write(library_section_venvs())
        f.write(
            f"\n{CHAMPION_HEADER}\n\n"
            + "".join(f"- seed {s}: `{h}`\n" for s, h in enumerate(hashes))
        )
    print(f"wrote {FROZEN}")


# ---- guards -------------------------------------------------------------------------------
def guards(cfg: dict) -> tuple[str, str, str, dict]:
    """The checks a start and a resume both run (D214): settings, package location, freeze
    record (frozen files), libraries of the venv, clean tree from the frozen commit, origin, data
    hashes. Returns (freeze record text, HEAD, branch, data hashes)."""
    from vaultic.fusion.final_run import (
        check_data,
        check_git,
        check_origin,
        check_package_location,
        env_values,
    )
    from vaultic.reports.phase2_summary import check_freeze
    from vaultic.stage2.final import check_libraries_venv

    check_settings(cfg)
    check_package_location()
    if not FROZEN.exists() or check_freeze(FROZEN):
        raise RuntimeError(f"{FROZEN} is missing or a frozen file changed: "
                           f"{check_freeze(FROZEN) if FROZEN.exists() else 'no record'}")  # fmt: skip
    record = FROZEN.read_text("utf-8")
    check_libraries_venv(record, cfg["venv"])
    head = check_git(FROZEN, REPO_ROOT, log_files=LOG_FILES, freeze_record=FREEZE_RECORD,
                     outputs=OUTPUTS)  # fmt: skip
    branch = check_origin(REPO_ROOT)
    hashes = check_data(record, data_files(), env_values())
    return record, head, branch, hashes


def _lock():
    from vaultic.fusion.final_run import LOCK_NAME, exclusive_lock

    return exclusive_lock(RUNS_DIR / EXP / LOCK_NAME)


def start(cfg: dict, rerun_reason: str | None) -> tuple[Path, str]:
    """All guards, then FINAL-STARTED committed and pushed; no test row is read before."""
    from vaultic.eval.run import _guard_final_rerun, previous_final_runs
    from vaultic.fusion.final_run import log_guard, record_start

    with _lock():
        _, head, branch, hashes = guards(cfg)
        previous_final_runs(EXP, RUNS_DIR)
        _guard_final_rerun(EXP, RUNS_DIR, rerun_reason, RESEARCH_DIR / "decisions.md")
        log_guard(EXP, rerun_reason, branch, REPO_ROOT)
        run_dir = RUNS_DIR / EXP / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        record_start(run_dir, head, RESEARCH_DIR / "experiment_log.md",
                     extra_logs=[RESEARCH_DIR / "decisions.md"], repo=REPO_ROOT,
                     data_hashes=hashes, what="feedback loop (champion-challenger) final run",
                     prereg="D210", freeze_record=FREEZE_RECORD)  # fmt: skip
    return run_dir, head


def check_resume(run_dir: Path) -> dict:
    """Only an existing EXP-310-final run that started, was not aborted and did not complete."""
    meta_path = Path(run_dir) / "metrics.json"
    if not meta_path.exists():
        raise RuntimeError(f"{run_dir} is not a started {EXP} run (no metrics.json)")
    meta = json.loads(meta_path.read_text("utf-8"))
    status = meta.get("status")
    if status == "completed":
        raise RuntimeError(f"{run_dir} already completed: nothing to resume")
    if status == "aborted":
        raise RuntimeError(f"{run_dir} was aborted ({meta.get('reason')}): it cannot be resumed")
    if status != "started" or Path(run_dir).parent.name != EXP:
        raise RuntimeError(f"{run_dir} is not a started {EXP} run")
    return meta


def _git(*a, check=True):
    from vaultic.fusion.final_run import PUSH_TIMEOUT_S, _git_env

    r = subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
                       env=_git_env(), timeout=PUSH_TIMEOUT_S)  # fmt: skip
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(a)} failed:\n{r.stderr}")
    return r


def check_recorded_commit(recorded: str, head: str) -> None:
    """The run's start commit is an ancestor of HEAD and only the logs changed since (D214)."""
    if _git("merge-base", "--is-ancestor", recorded, head, check=False).returncode != 0:
        raise RuntimeError(
            f"the run's commit {recorded[:12]} is not an ancestor of HEAD {head[:12]}"
        )
    changed = set(_git("diff", "--name-only", "--no-renames", recorded, head).stdout.splitlines())
    if changed - LOG_FILES:
        raise RuntimeError(f"files other than the logs changed since the run started: "
                           f"{sorted(changed - LOG_FILES)}")  # fmt: skip


def started_on_origin(run_name: str, branch: str) -> bool:
    """The FINAL-STARTED row naming this run folder is in the experiment log on origin."""
    text = _git("show", f"origin/{branch}:{EXPERIMENT_LOG}", check=False).stdout or ""
    return any(line.startswith(f"| {EXP} |") and "FINAL-STARTED" in line and run_name in line
               for line in text.splitlines())  # fmt: skip


def record_event(run_dir: Path, kind: str, detail: str) -> None:
    """Append a FINAL-RESUMED / FINAL-ABORTED row to the experiment log, commit and push; raises
    if any git step fails (the caller stops). The text never contains the words the re-run guard
    counts (FINAL-STARTED, FINAL-RERUN), so a resume is not taken for a second run."""
    from vaultic.fusion.final_run import repo_relative

    assert kind in ("FINAL-RESUMED", "FINAL-ABORTED")
    assert "FINAL-STARTED" not in detail and "FINAL-RERUN" not in detail
    rel = repo_relative(run_dir, REPO_ROOT)
    with open(REPO_ROOT / EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(f"| {EXP} | {date.today().isoformat()} | {kind}: {detail} | run `{rel}` | — |\n")
    _git("add", EXPERIMENT_LOG)
    _git("commit", "-q", "-m", f"{kind} {EXP} ({Path(run_dir).name})")
    _git("push", "-q", "origin", "HEAD")


def resume(cfg: dict, run_dir: Path) -> Path:
    """Same guards as a start (D214), then FINAL-RESUMED pushed before any test row is read."""
    with _lock():
        _, head, branch, _ = guards(cfg)
        meta = check_resume(run_dir)
        check_recorded_commit(meta["git_commit"], head)
        if not started_on_origin(Path(run_dir).name, branch):
            raise RuntimeError(f"no FINAL-STARTED row for {Path(run_dir).name} on origin/{branch}: "
                               "refusing to resume")  # fmt: skip
        record_event(run_dir, "FINAL-RESUMED", f"resume of the same run (started at commit "
                     f"`{meta['git_commit'][:12]}`), logged; not a re-run")  # fmt: skip
        with open(Path(run_dir) / "resume.log", "a", encoding="utf-8") as f:
            f.write(f"resumed {datetime.now().isoformat(timespec='seconds')} at HEAD {head}\n")
    return Path(run_dir)


def abort(run_dir: Path, reason: str, log=print) -> None:
    """Parity failure (D214): logged everywhere, status 'aborted', FINAL-ABORTED pushed."""
    log(f"ABORTED: {reason}")
    meta_path = Path(run_dir) / "metrics.json"
    meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
    meta.update(
        status="aborted", reason=reason, aborted=datetime.now().isoformat(timespec="seconds")
    )
    meta_path.write_text(json.dumps(meta, indent=1, default=str), "utf-8")
    record_event(run_dir, "FINAL-ABORTED", f"run aborted before the simulation: {reason}")


# ---- parity -------------------------------------------------------------------------------
def parity(b5_models, X: pd.DataFrame, transaction_ids, stored: pd.Series,
           tolerance: float = PARITY_TOLERANCE, expected_hashes=None) -> dict:  # fmt: skip
    """Each seed's booster against the frozen hash (when given), then the mean of the five seed
    models vs the stored score_B5 (by TransactionID); raises on any difference or missing row."""
    if expected_hashes is not None:
        got = booster_hashes(b5_models)
        if got != list(expected_hashes):
            bad = [s for s, (a, b) in enumerate(zip(got, expected_hashes, strict=False)) if a != b]
            raise RuntimeError(
                f"parity failed: champion model hash differs for seed(s) {bad or 'count'}"
            )
    mean = np.mean([m.predict_proba(X)[:, 1] for m in b5_models], axis=0)
    ids = np.asarray(transaction_ids)
    missing = int(
        (~pd.Index(stored.index).isin(ids)).sum() + (~pd.Index(ids).isin(stored.index)).sum()
    )
    ref = stored.reindex(ids).to_numpy(float)
    diff = float(np.nanmax(np.abs(ref - mean))) if len(ids) else 0.0
    out = {"rows": int(len(ids)), "missing": missing, "max_abs_diff": diff, "tolerance": tolerance,
           "models_checked": expected_hashes is not None,
           "passed": bool(missing == 0 and np.isfinite(diff) and diff <= tolerance)}  # fmt: skip
    if not out["passed"]:
        raise RuntimeError(f"parity failed: {out}")
    return out


# ---- resume per completed delay -----------------------------------------------------------
def run_delays(run_dir: Path, delays, run_one, log=print) -> dict:
    """run_one(L) -> (served DataFrame, events list, metrics dict). Completed delays (DONE marker)
    are skipped; an unfinished delay folder is recomputed from scratch."""
    results = {}
    for L in delays:
        d = Path(run_dir) / f"delay_L{L}"
        if (d / "DONE").exists():
            log(f"L={L}: completed earlier, skipped (resume)")
            results[L] = json.loads((d / "metrics.json").read_text("utf-8"))
            continue
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():  # never reuse a partial delay
            f.unlink()
        served, events, metrics = run_one(L)
        served.to_parquet(d / "served.parquet", index=False)
        (d / "events.json").write_text(json.dumps(events, indent=1, default=str), "utf-8")
        (d / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float), "utf-8")
        (d / "DONE").write_text(datetime.now().isoformat(timespec="seconds"), "utf-8")
        results[L] = metrics
        log(f"L={L}: done")
    return results


# ---- the run ------------------------------------------------------------------------------
def simulate_delay(cfg, loaded, system, L, log) -> tuple[pd.DataFrame, list, dict]:
    from vaultic.learning.feedback import Data, evaluate, make_model, simulate
    from vaultic.learning.feedback_inputs import inputs_for_delay
    from vaultic.learning.feedback_run import fit_b5, weekly_pr_auc
    from vaultic.live.store import cache_root

    params = yaml.safe_load((CONFIG_DIR / f"{cfg['model_config']}.yaml").read_text("utf-8"))[
        "model"
    ]["params"]
    df = loaded["df"]
    day, y = df["day"].to_numpy(), df["isFraud"].to_numpy()
    f_lo, f_hi = cfg["champion"]["fit_days"]
    c_lo, c_hi = cfg["champion"]["calibration_days"]
    fit_rows = np.flatnonzero((day >= f_lo) & (day <= f_hi))
    cal_rows = np.flatnonzero((day >= c_lo) & (day <= c_hi))
    X = inputs_for_delay(loaded, L, cache_dir=cache_root() / "feedback")
    datas, served_by_seed, events, frames = {}, {}, [], []
    for s in cfg["seeds"]:
        data = Data(df["TransactionDT"].to_numpy(), y, df["TransactionAmt"].to_numpy(), X, L)
        champ = make_model(f"champion_s{s}", system.b5_models[s], data, fit_rows, cal_rows, f_hi)
        served, ev = simulate(data, champ, cfg["deploy_day"], cfg["end_day"],
                              partial(fit_b5, params, s), tuple(cfg["policies"]),
                              log=lambda m, s=s: log(f"seed {s}: {m}"))  # fmt: skip
        for e in ev:
            e["seed"] = s
        events += ev
        datas[s], served_by_seed[s] = data, served
        frames.append(served.assign(seed=s))
    lo, hi = cfg["evaluation_days"]
    pols = tuple(cfg["policies"])
    n, bs = cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"]
    metrics = {"evaluation": evaluate(datas, served_by_seed, (lo, hi), pols, n_boot=n, boot_seed=bs),
               "gated_vs_ungated": evaluate(datas, served_by_seed, (lo, hi), ("P1_ungated", "P1"),
                                            n_boot=n, reference="P1_ungated", boot_seed=bs),
               "weekly": weekly_pr_auc(datas, served_by_seed, pols, lo, hi),
               "warmup_days_121_150": evaluate(datas, served_by_seed, tuple(cfg["warmup_days"]), pols,
                                               n_boot=n, boot_seed=bs)}  # fmt: skip
    return pd.concat(frames, ignore_index=True), events, metrics


def hypotheses(results: dict, alpha: float = 0.05) -> dict:
    """H-FB1..4 from the per-delay metrics (D210 section 1; D214: H-FB3 descriptive, H-FB4 on the
    point estimate)."""
    from vaultic.eval.bootstrap import holm

    m30 = results[30]["evaluation"]
    gated = ("P1", "P2", "P3")
    p = [m30[g]["vs_P0"]["p_value"] for g in gated]
    adj = holm(p)
    h1 = {g: {**m30[g]["vs_P0"], "p_holm": a, "supported": bool(a < alpha and m30[g]["vs_P0"]["diff"] > 0)}
          for g, a in zip(gated, adj, strict=True)}  # fmt: skip
    h2 = {
        g: {**m30[g]["cost_vs_P0"], "cost_increase": bool(m30[g]["cost_vs_P0"]["ci_low"] > 0)}
        for g in gated
    }
    gains = {L: results[L]["evaluation"]["P1"]["vs_P0"]["diff"] for L in sorted(results)}
    order = [gains[L] for L in sorted(gains)]
    h4 = results[30]["gated_vs_ungated"]["P1"]["cost_vs_P1_ungated"]
    return {"H-FB1": {"per_policy": h1, "supported": any(v["supported"] for v in h1.values())},
            "H-FB2": {"per_policy": h2, "supported": not any(v["cost_increase"] for v in h2.values())},
            "H-FB3": {"kind": "descriptive (point estimates, no test)", "gains_by_delay": gains,
                      "supported": all(a >= b for a, b in zip(order, order[1:], strict=False))},
            "H-FB4": {"basis": "point estimate of gated minus ungated cost",
                      "gated_minus_ungated_cost": h4, "supported": bool(h4["diff"] <= 0)}}  # fmt: skip


def _load_system():
    import joblib

    return joblib.load(system_cache())


def _stored_scores() -> pd.Series:
    return pd.read_parquet(REPO_ROOT / E10_RUN / "predictions.parquet").set_index("TransactionID")[
        "score_B5"
    ]


def main(argv=None) -> None:
    import vaultic.learning.feedback_inputs as fi

    p = argparse.ArgumentParser(description=f"{EXP} (needs Aziz's go-ahead)")
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--rerun-reason")
    a = p.parse_args(argv)
    if a.freeze:
        return freeze()
    cfg = yaml.safe_load((CONFIG_DIR / f"{EXP}.yaml").read_text("utf-8"))
    if a.resume:
        run_dir = resume(cfg, a.resume)  # FINAL-RESUMED pushed before any test row
    else:
        run_dir, _ = start(cfg, a.rerun_reason)  # FINAL-STARTED pushed before any test row

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(Path(run_dir) / "run.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    loaded = fi.load("final", run_dir=run_dir)  # test rows are read from here on
    system = _load_system()
    df = loaded["df"]
    test = (df["day"] >= cfg["evaluation_days"][0]).to_numpy()
    try:
        expected = read_champion_hashes(FROZEN.read_text("utf-8"))
        X30 = fi.inputs_for_delay(loaded, 30)
        par = parity(system.b5_models, X30[test], df["TransactionID"].to_numpy()[test],
                     _stored_scores(), cfg["parity_tolerance"], expected)  # fmt: skip
    except Exception as e:
        abort(run_dir, str(e), log)
        raise SystemExit(f"{EXP} aborted: {e}") from e
    log(f"parity passed: {par}")
    (Path(run_dir) / "parity.json").write_text(json.dumps(par, indent=1), "utf-8")
    results = run_delays(
        run_dir, cfg["delays"], lambda L: simulate_delay(cfg, loaded, system, L, log), log
    )
    hyp = hypotheses(results, cfg["holm_alpha"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "exp310_final.json").write_text(json.dumps({"hypotheses": hyp, "per_delay": results},
                                                             indent=1, default=float), "utf-8")  # fmt: skip
    from vaultic.stage2.final import complete

    complete(Path(run_dir), cfg, {"hypotheses": hyp, "parity": par})
    log("completed")


if __name__ == "__main__":
    main()
