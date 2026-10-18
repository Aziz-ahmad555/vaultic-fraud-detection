"""POST-HOC diagnostics for the Phase B scenario results (D201, D202), chosen after seeing them.

    python -m vaultic.scenarios.posthoc      # writes docs/scenarios/posthoc_*.{md,json}

Real legitimate test-period rows only (days 151-182), from the stored offline features (opened
behind the post-hoc guard, D144, like the Test Lab). Nothing here is an evaluation number and
nothing is tuned on it; thresholds and model are fixed.

  per day      rules-only FPR per rule (R1-R6) and combined, for the D196 thresholds, D196
               without R3 (post hoc) and D195; and DeviceInfo 'Windows' 7-day distinct-customer
               count (max and median over the day's Windows rows) against R3's threshold
  new accounts the model's alert rates (D196 definition: step-up or higher; and hold/block) on
               REAL legitimate rows of young accounts (masked time-delta field D1 = 0-13; the uid
               reconstruction treats day - D1 as the account's start, as the synthetic build-up
               rows do, so these are the first 14 days) on the bust-out replay days, next to the bust-out build-up rows'
               alert rate; also the subset that is card-not-present (ProductCD W), and how many
               real rows have every masked M and V field missing like the synthetic build-up rows
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from vaultic.rules.engine import RULES, SEVERITY, rule_values

LABEL = "post-hoc diagnostic (chosen after seeing the scenario results) - not the evaluation"


def per_day(store, sets: dict[str, dict[str, float]]) -> list[dict]:
    from vaultic.rules.engine import BY_ID

    df, f = store.df, store.features
    day = df["day"].to_numpy()
    legit = df["isFraud"].to_numpy() == 0
    test = (day >= store.splits.test.first) & (day <= store.splits.test.last)
    frame = f.assign(TransactionAmt=df["TransactionAmt"].to_numpy(float))
    values = {r.rule_id: rule_values(frame, r) for r in RULES}
    win = (df["DeviceInfo"].astype(object) == "Windows").to_numpy()
    ent = f["ent_uids_7d_device"].to_numpy(float)
    r3 = sets["D196 (primary)"]["R3"]
    out = []
    for d in range(int(store.splits.test.first), int(store.splits.test.last) + 1):
        m = (day == d) & legit & test
        row = {"day": d, "legit_rows": int(m.sum())}
        for name, th in sets.items():
            worst = np.zeros(int(m.sum()), dtype=int)
            per = {}
            for rid, v in values.items():
                fired = np.isfinite(v[m]) & (v[m] >= th[rid])
                per[rid] = float(fired.mean())
                worst = np.where(fired, np.maximum(worst, SEVERITY[BY_ID[rid].action]), worst)
            row[name] = {"per_rule_fpr": per,
                         "rules_only_fpr": float((worst >= SEVERITY["step_up"]).mean()),
                         "rules_only_hold_block_fpr": float((worst >= SEVERITY["hold"]).mean())}  # fmt: skip
        w = (day == d) & win
        row["windows_7d_customers_max"] = float(np.nanmax(ent[w])) if w.any() else None
        row["windows_7d_customers_median"] = float(np.nanmedian(ent[w])) if w.any() else None
        row["windows_rows_at_or_above_r3"] = int((ent[w] >= r3).sum())
        row["r3_threshold"] = r3
        out.append(row)
    return out


def new_accounts(store, system, days: list[int], build_up: dict) -> dict:
    from vaultic.scenarios.run import model_actions

    df, f = store.df, store.features
    day = df["day"].to_numpy()
    d1 = df["D1"].to_numpy(float)
    legit = df["isFraud"].to_numpy() == 0
    on_days = np.isin(day, days) & legit
    young = on_days & (d1 >= 0) & (d1 <= 13)
    cnp = young & (df["ProductCD"].astype(object) == "W").to_numpy()
    mv = [c for c in df.columns if re.match(r"^(M|V)\d+$", c)]
    all_mv_missing = df[mv].isna().all(axis=1).to_numpy()
    pos = np.flatnonzero(on_days)
    act, _ = model_actions(system, df.iloc[pos], f.iloc[pos])
    a = pd.Series(act, index=pos)

    def rates(mask):
        p = np.flatnonzero(mask)
        x = a.loc[p].to_numpy()
        return {"rows": int(len(p)),
                "alert_rate": float((x >= SEVERITY["step_up"]).mean()) if len(p) else None,
                "hold_block_rate": float((x >= SEVERITY["hold"]).mean()) if len(p) else None}  # fmt: skip

    return {"days": [int(d) for d in days],
            "all_legit_on_days": rates(on_days),
            "young_accounts_d1_0_13": rates(young),
            "young_accounts_card_not_present_W": rates(cnp),
            "young_accounts_all_M_V_missing": rates(young & all_mv_missing),
            "bust_out_build_up": build_up}  # fmt: skip


def build_up_rates(suite_dir) -> tuple[list[int], dict]:
    """Bust-out build-up rows' model alert rates and replay days, from the suite's runs."""
    from vaultic.scenarios.suite import SEEDS

    days, alert, hard, n = set(), 0, 0, 0
    for inten in ("low", "medium", "high"):
        for s in SEEDS:
            run = suite_dir / f"bust_out_s{s}_{inten}"
            rows = pd.read_parquet(run / "rows.parquet")
            info = json.loads((run / "summary.json").read_text("utf-8"))
            days |= set(info["days"])
            b = rows[rows["synthetic"] & (rows["label"] == 0)]
            sev = b["model_action"].map(SEVERITY).to_numpy()
            alert += int((sev >= SEVERITY["step_up"]).sum())
            hard += int((sev >= SEVERITY["hold"]).sum())
            n += len(b)
    return sorted(days), {"rows": n, "alert_rate": alert / n, "hold_block_rate": hard / n,
                          "alerts": alert}  # fmt: skip


def markdown(daily: list[dict], acc: dict, set_names) -> str:
    lines = [f"# Post-hoc diagnostics - {LABEL}", "",
             "Real legitimate test-period rows, stored offline features; thresholds and model fixed. "
             "A step-up is an extra customer check (friction), not a block.", "",
             "## Rules-only false-positive rate per test day", "",
             "R3 = distinct customers on the DeviceInfo string in the previous 7 days >= threshold; "
             "IEEE-CIS has no true device ID; DeviceInfo is a model/OS string.", ""]  # fmt: skip
    p, s, o = set_names
    lines.append(
        f"| day | legit rows | R1 | R2 | R3 | R4 | R5 | R6 | rules only: {p} | {s} | {o} |"
        " Windows 7-day customers (max / median) | Windows rows >= R3 threshold |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in daily:
        pr = r[p]["per_rule_fpr"]
        lines.append(
            f"| {r['day']} | {r['legit_rows']} | " + " | ".join(f"{pr[k]:.4f}" for k in sorted(pr))
            + f" | {r[p]['rules_only_fpr']:.4f} | {r[s]['rules_only_fpr']:.4f} |"
            f" {r[o]['rules_only_fpr']:.4f} | {r['windows_7d_customers_max']:.0f} /"
            f" {r['windows_7d_customers_median']:.0f} | {r['windows_rows_at_or_above_r3']} |")  # fmt: skip
    lines += ["", f"R3 threshold (D196): {daily[0]['r3_threshold']:g}. Per-rule columns use the "
              "D196 thresholds.", "", "## Model alerts on real new accounts vs bust-out build-up",
              "", "| group | rows | alert rate (step-up or higher) | hold/block rate |",
              "|---|---|---|---|"]  # fmt: skip
    names = {"bust_out_build_up": "bust-out build-up rows (synthetic, labelled legitimate)",
             "all_legit_on_days": "all real legitimate rows, same replay days",
             "young_accounts_d1_0_13": "real legitimate young accounts (masked time-delta D1 0-13)",
             "young_accounts_card_not_present_W": "  of which card-not-present (ProductCD W)",
             "young_accounts_all_M_V_missing": "  of which every M and V field missing"}  # fmt: skip
    for k, label in names.items():
        x = acc[k]
        ar = "-" if x["alert_rate"] is None else f"{x['alert_rate']:.3f}"
        hr = "-" if x["hold_block_rate"] is None else f"{x['hold_block_rate']:.4f}"
        lines.append(f"| {label} | {x['rows']} | {ar} | {hr} |")
    lines += ["", f"Replay days: {acc['days'][0]}-{acc['days'][-1]}. D1 is a masked time-delta "
              "field; the uid reconstruction treats day - D1 as the account's start, as the "
              "synthetic build-up rows do.", ""]  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    from vaultic.live.store import cache_root
    from vaultic.paths import REPO_ROOT
    from vaultic.scenarios.run import POST_HOC_NO_R3, load_all, threshold_sets

    system, store, _ = load_all(log=lambda m: None)
    sets = threshold_sets()
    daily = per_day(store, sets)
    days, build = build_up_rates(cache_root() / "scenarios" / "suite")
    acc = new_accounts(store, system, days, build)
    out = REPO_ROOT / "docs" / "scenarios"
    out.mkdir(parents=True, exist_ok=True)
    data = {"label": LABEL, "per_day": daily, "new_accounts": acc}
    (out / "posthoc_diagnostics.json").write_text(json.dumps(data, indent=1), "utf-8")
    names = ("D196 (primary)", "D195 (sensitivity)", POST_HOC_NO_R3)
    (out / "posthoc_diagnostics.md").write_text(markdown(daily, acc, names), "utf-8")
    print(markdown(daily, acc, names))


if __name__ == "__main__":
    main()
