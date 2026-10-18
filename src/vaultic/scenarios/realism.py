"""Generator realism check (Phase B, D203): masked-field missingness of synthetic rows vs real rows
of the same customer type.

    python -m vaultic.scenarios.realism --version v1     # docs/scenarios/realism_<version>.md/json

For every scenario (seeds 0-4 x low / medium / high) the synthetic rows are generated (not scored)
and compared with REAL rows of the same customer type:

  new customer       real rows of young accounts (masked time-delta D1 0-13)
  existing customer  real rows whose customer had at least 3 earlier rows

Per masked group (M1-M9, V1-V339, id_01-id_38, C1-C14, D2-D15): the share of rows with the whole
group missing and the mean share of the group's fields missing. Joint profile: the missingness
pattern over all M, V and id fields together; a synthetic row "has a profile no real row has"
when its exact pattern occurs in none of the real rows of its customer type (all days). The C
and D groups are reported alongside.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from vaultic.scenarios import LABEL

GROUPS = {
    "M": r"^M\d+$",
    "V": r"^V\d+$",
    "id": r"^id_\d+$",
    "C": r"^C\d+$",
    "D2-D15": r"^D(?:[2-9]|1[0-5])$",
}
NEW = ("mule_ring", "bust_out")


def columns(df: pd.DataFrame) -> dict[str, list[str]]:
    return {g: [c for c in df.columns if re.match(p, c)] for g, p in GROUPS.items()}


def group_stats(frame: pd.DataFrame, cols: dict[str, list[str]]) -> dict:
    out = {"rows": int(len(frame))}
    for g, cs in cols.items():
        miss = frame[cs].isna().to_numpy()
        out[g] = {"all_missing": float(miss.all(axis=1).mean()) if len(frame) else None,
                  "mean_missing": float(miss.mean()) if len(frame) else None}  # fmt: skip
    return out


def signatures(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    bits = np.packbits(frame[cols].isna().to_numpy(), axis=1)
    return np.array([b.tobytes() for b in bits], dtype=object)


def reference(store) -> dict:
    df = store.df
    d1 = df["D1"].to_numpy(float)
    n_past = np.nan_to_num(store.features["hist_n_past"].to_numpy(float))
    return {"new": (d1 >= 0) & (d1 <= 13), "existing": n_past >= 3}


def check(
    store, generators: dict, seeds=(0, 1, 2, 3, 4), intensities=("low", "medium", "high")
) -> dict:
    df = store.df
    cols = columns(df)
    joint = cols["M"] + cols["V"] + cols["id"]
    ref = reference(store)
    real_sigs = {k: set(signatures(df.loc[m], joint)) for k, m in ref.items()}
    out = {"label": LABEL, "real": {k: group_stats(df.loc[m], cols) for k, m in ref.items()},
           "scenarios": {}}  # fmt: skip
    for name, gen in generators.items():
        kind = "new" if name in NEW else "existing"
        rows = pd.concat([gen(store, seed=s, intensity=i).rows for s in seeds for i in intensities],
                         ignore_index=True)  # fmt: skip
        sig = signatures(rows, joint)
        unseen = np.array([s not in real_sigs[kind] for s in sig])
        st = group_stats(rows, cols)
        st["customer_type"] = kind
        st["joint_M_V_id_profile_unseen_in_real"] = float(unseen.mean())
        out["scenarios"][name] = st
    return out


def markdown(res: dict, version: str) -> str:
    groups = list(GROUPS)
    lines = [f"# Generator realism check ({version}) - {LABEL}", "",
             "Masked-field missingness of synthetic rows (seeds 0-4 x low / medium / high) vs real rows "
             "of the same customer type (new: young accounts, masked time-delta D1 0-13; existing: "
             "customer with >= 3 earlier rows). Cells: share of rows with the whole group missing / "
             "mean share of the group's fields missing. 'Unseen profile': share of synthetic rows whose "
             "exact missingness pattern over all M, V and id fields occurs in no real row of the type.",
             "", "| rows | n | " + " | ".join(groups) + " | unseen profile |",
             "|---|---|" + "---|" * len(groups) + "---|"]  # fmt: skip

    def cells(st):
        return " | ".join(
            f"{st[g]['all_missing']:.2f} / {st[g]['mean_missing']:.2f}" for g in groups
        )

    for k, st in res["real"].items():
        lines.append(f"| real {k} | {st['rows']:,} | {cells(st)} | - |")
    for name, st in res["scenarios"].items():
        lines.append(
            f"| {name} ({st['customer_type']}) | {st['rows']} | {cells(st)} |"
            f" {st['joint_M_V_id_profile_unseen_in_real']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    from vaultic.paths import REPO_ROOT
    from vaultic.scenarios.run import load_all

    p = argparse.ArgumentParser(description="generator realism check")
    p.add_argument("--version", default="v1", choices=("v1", "v2", "v3"))
    a = p.parse_args()
    _, store, _ = load_all(log=lambda m: None)
    from vaultic.scenarios.generators import BY_VERSION

    res = check(store, BY_VERSION[a.version])
    out = REPO_ROOT / "docs" / "scenarios"
    (out / f"realism_{a.version}.json").write_text(json.dumps(res, indent=1), "utf-8")
    (out / f"realism_{a.version}.md").write_text(markdown(res, a.version), "utf-8")
    print(markdown(res, a.version))


if __name__ == "__main__":
    main()
