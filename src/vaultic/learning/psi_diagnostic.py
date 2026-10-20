"""P2 PSI diagnostic on the DESIGN days only (EXP-310-dev, days 61-120; never test days).

    python -m vaultic.learning.psi_diagnostic          # writes E:/dev-cache/feedback/psi_diagnostic.json

In EXP-310-dev P2 raised no alarm at any delay. P2 never promoted, so the scores it served are the
champion's. This recomputes, for seed 0 and every delay, the PSI that simulate() computes each day
(the champion's scores over its last 7 served days vs its reference scores, the threshold slice =
days 57-60) and compares the engine's own reference with an independent recomputation.
"""

from __future__ import annotations

import json

import numpy as np
import yaml


def main() -> None:
    from vaultic.drift.signals import score_drift
    from vaultic.learning.feedback import PSI_THRESHOLD, PSI_WINDOW_DAYS, Data, make_model
    from vaultic.learning.feedback_inputs import (
        DEV_LAST_DAY,
        inputs_for_delay,
        load,
        require_last_day,
    )
    from vaultic.learning.feedback_run import fit_b5
    from vaultic.live.store import cache_root
    from vaultic.paths import CONFIG_DIR

    cfg = yaml.safe_load((CONFIG_DIR / "EXP-310-dev.yaml").read_text("utf-8"))
    params = yaml.safe_load((CONFIG_DIR / "EXP-009.yaml").read_text("utf-8"))["model"]["params"]
    loaded = load("dev")
    df = loaded["df"]
    require_last_day(df["day"].to_numpy(), DEV_LAST_DAY)
    day, y = df["day"].to_numpy(), df["isFraud"].to_numpy()
    fit_rows = np.flatnonzero((day >= 1) & (day <= 56))
    cal_rows = np.flatnonzero((day >= 57) & (day <= 60))
    est = fit_b5(params, 0, inputs_for_delay(loaded, 30).iloc[fit_rows], y[fit_rows])
    out = {"seed": 0, "threshold": PSI_THRESHOLD, "window_days": PSI_WINDOW_DAYS, "delays": {}}
    for L in cfg["delays"]:
        X = inputs_for_delay(loaded, L, cache_dir=cache_root() / "feedback")
        data = Data(df["TransactionDT"].to_numpy(), y, df["TransactionAmt"].to_numpy(), X, L)
        champ = make_model("champion_s0", est, data, fit_rows, cal_rows, 56)
        ref_independent = est.predict_proba(X.iloc[cal_rows])[:, 1]
        days = list(range(cfg["deploy_day"], cfg["end_day"] + 1))
        daily = []
        for i, d in enumerate(days):
            if i + 1 < PSI_WINDOW_DAYS:
                continue
            window = days[i + 1 - PSI_WINDOW_DAYS : i + 1]
            rows = np.flatnonzero(np.isin(day, window))
            daily.append(
                {"day": d, "psi": float(score_drift(champ.reference, champ.score(data, rows)))}
            )
        psis = [x["psi"] for x in daily]
        out["delays"][f"L{L}"] = {
            "reference_rows": int(len(cal_rows)), "reference_days": [57, 60],
            "reference_matches_independent": bool(np.allclose(champ.reference, ref_independent)),
            "psi_max": float(max(psis)), "psi_median": float(np.median(psis)),
            "days_above_threshold": int(sum(p > PSI_THRESHOLD for p in psis)), "daily": daily}  # fmt: skip
        print(f"L={L}: PSI max {max(psis):.4f}, median {np.median(psis):.4f}, days > {PSI_THRESHOLD}: "
              f"{out['delays'][f'L{L}']['days_above_threshold']}", flush=True)  # fmt: skip
    path = cache_root() / "feedback" / "psi_diagnostic.json"
    path.write_text(json.dumps(out, indent=1), "utf-8")
    print(f"written {path}")


if __name__ == "__main__":
    main()
