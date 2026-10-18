"""Attack Lab service (Phase B): runs one scenario live through the Test Lab engine, and serves the
5-seed suite scorecard and the post-hoc diagnostics. Synthetic scenario - not the evaluation.

Uses the Test Lab's engine (its history store, frozen system and stored final-run scores) once
the parity gate has passed; one scenario runs at a time. With the model off (degraded mode) the
rules alone decide (rules/engine.combine with no model action).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from vaultic.scenarios import LABEL

TITLES = {
    "card_testing": "Card testing",
    "account_takeover": "Account takeover",
    "mule_ring": "Mule ring",
    "bust_out": "Bust-out",
    "mimicry": "Mimicry",
}
DESCRIPTIONS = {
    "card_testing": "A burst of small charges (1.00-9.99) on one existing customer's card, card-not-present, 15-90 s apart.",
    "account_takeover": "An established customer's account used from a device the customer never used, with a different email "
                        "domain and amounts 5-20 times the customer's median.",
    "mule_ring": "New cards (new customers) all transacting on one device within a day.",
    "bust_out": "A new card-not-present account builds up with ordinary purchases over several days (labelled legitimate), "
                "then cashes out with large purchases (labelled fraud).",
    "mimicry": "Fraud that copies the customer's own amounts, times of day and device.",
}  # fmt: skip
INTENSITY_MEANING = {
    "card_testing": {"low": "5 charges", "medium": "12 charges", "high": "25 charges"},
    "account_takeover": {"low": "1 transaction, 5x median", "medium": "2 transactions, 10x median",
                         "high": "3 transactions, 20x median"},
    "mule_ring": {"low": "5 cards", "medium": "10 cards", "high": "20 cards"},
    "bust_out": {"low": "3 build-up days, 3 cash-outs", "medium": "7 build-up days, 5 cash-outs",
                 "high": "14 build-up days, 8 cash-outs"},
    "mimicry": {"low": "1 transaction", "medium": "3 transactions", "high": "6 transactions"},
}  # fmt: skip
STEP_UP_NOTE = "A step-up is an extra customer check (friction), not a block."
NOTES: dict[str, list[str]] = {  # findings shown with each scenario (D198, D200, D202)
    "card_testing": [
        "The model's step-ups on 1-10 charges come only from its uncertainty set: the expected loss (probability x amount) of a "
        "small charge is a few cents, and late in a long burst the 1-hour count lowers the model's score (D198). Rule R1 covers that region."],
    "account_takeover": [
        "Rule R6 (new device + high amount) fires only on the first transaction on the new device: after it, the device is no "
        "longer new to the customer."],
    "mule_ring": [
        "Rule R3 (shared device, threshold 778 customers in 7 days) never fires on a ring of 5-20 cards. Rules-only catches come "
        "from R4 (blocklist) and depend on whether the seed's device already carried known fraud: 0 or all."],
    "bust_out": [
        "Pre-cash-out alerts are not evidence of bust-out detection: the model alerts on 41.9% of real legitimate young accounts "
        "(24.2% of all legitimate rows, same days) and on 83.8% of the synthetic build-up rows, whose every masked M and V field "
        "is missing - a profile no real row has (D202)."],
    "mimicry": ["Undetected by design: the fraud copies the customer's own amounts, times of day and device."],
}  # fmt: skip


class AttackInputError(ValueError):
    """A run request the Attack Lab cannot run (message shown to the user)."""


def _notes(name: str) -> list[str]:
    from vaultic.scenarios.generators import MULE_DEVICE_NOTE

    extra = {"mule_ring": [MULE_DEVICE_NOTE]}
    return extra.get(name, []) + NOTES.get(name, [])


class AttackLab:
    def __init__(self, testlab, suite_dir: Path | None = None, docs_dir: Path | None = None):
        from vaultic.live.store import cache_root
        from vaultic.paths import REPO_ROOT

        self.testlab = testlab
        self.suite_dir = suite_dir or cache_root() / "scenarios" / "suite"
        self.docs_dir = docs_dir or REPO_ROOT / "docs" / "scenarios"
        self.lock = threading.Lock()

    def scenarios(self) -> dict:
        from vaultic.scenarios.generators import INTENSITIES, SCENARIOS
        from vaultic.scenarios.run import threshold_sets

        return {"label": LABEL, "step_up_note": STEP_UP_NOTE, "intensities": list(INTENSITIES),
                "threshold_sets": list(threshold_sets()),
                "scenarios": [{"name": n, "title": TITLES[n], "description": DESCRIPTIONS[n],
                               "intensities": INTENSITY_MEANING[n], "notes": _notes(n)}
                              for n in SCENARIOS]}  # fmt: skip

    def run(self, spec: dict) -> dict:
        from vaultic.rules.engine import combine
        from vaultic.scenarios.generators import GENERATORS, INTENSITIES
        from vaultic.scenarios.run import run_scenario

        name = spec.get("scenario")
        if name not in GENERATORS:
            raise AttackInputError(f"scenario must be one of {sorted(GENERATORS)}")
        intensity = spec.get("intensity", "medium")
        if intensity not in INTENSITIES:
            raise AttackInputError(f"intensity must be one of {list(INTENSITIES)}")
        try:
            seed = int(spec.get("seed", 0))
        except (TypeError, ValueError) as e:
            raise AttackInputError("seed must be a whole number") from e
        if not 0 <= seed <= 9999:
            raise AttackInputError("seed must be between 0 and 9999")
        model_on = bool(spec.get("model", True))
        engine = self.testlab._ready()
        with self.lock, engine.lock:
            sc = GENERATORS[name](engine.store, seed=seed, intensity=intensity)
            res = run_scenario(engine.store, engine.system, engine.stored, sc, log=lambda m: None)
        rows = res["rows"]
        shown = rows[rows["synthetic"] | (rows["source"] == "scored live")].copy()
        decision = [combine(r, None if not model_on else m)["action"]
                    for r, m in zip(shown["rule_action"], shown["model_action"], strict=True)]  # fmt: skip
        shown["decision"] = decision
        replay = json.loads(shown.to_json(orient="records"))
        traces = res["traces"]
        for r in replay:
            r["trace"] = traces.get(int(r["TransactionID"]))
        return {"label": LABEL, "scenario": name, "title": TITLES[name], "seed": seed,
                "intensity": intensity, "model": "on" if model_on else "off (degraded mode: rules alone decide)",
                "days": res["days"], "info": _jsonable(res["info"]), "notes": _notes(name),
                "step_up_note": STEP_UP_NOTE, "n_synthetic": res["n_synthetic"],
                "n_legit_replay": res["n_legit_replay"], "n_legit_stored": res["n_legit_stored"],
                "n_legit_live": res["n_legit_live"], "stored_score_max_diff": res["stored_score_max_diff"],
                "seconds": res["seconds"], "results": res["results"], "replay": replay}  # fmt: skip

    def suite(self) -> dict:
        out = {"label": LABEL, "suite": None, "posthoc": None}
        s = self.suite_dir / "summary.json"
        if s.exists():
            out["suite"] = json.loads(s.read_text("utf-8"))
        p = self.docs_dir / "posthoc_diagnostics.json"
        if p.exists():
            out["posthoc"] = json.loads(p.read_text("utf-8"))
        return out


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
