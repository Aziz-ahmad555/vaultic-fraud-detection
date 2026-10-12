import { useEffect, useState } from "react";
import { api } from "../api";
import type { CostModel } from "../api/types";
import { parseField, toField } from "../lib/fields";
import { fmtThreshold } from "../lib/format";
import { useData } from "../lib/useData";
import { whatIfKey } from "../lib/whatIf";
import { useApp } from "../state/app";

const FIELDS: { key: keyof CostModel; label: string; step: number; max?: number }[] = [
  { key: "c_fp", label: "False positive cost ($)", step: 1 },
  { key: "c_rev", label: "Analyst review cost ($)", step: 1 },
  { key: "c_step", label: "Step-up (OTP) cost ($)", step: 0.1 },
  { key: "step_up_success_rate", label: "Step-up stops fraud (share)", step: 0.05, max: 1 },
  { key: "legit_step_up_friction", label: "Friction for a legitimate customer ($)", step: 0.5 },
  { key: "hold_success_rate", label: "Hold stops fraud (share; 1 = evaluated system)", step: 0.05, max: 1 },
];

/**
 * What-if cost mode (D122, real-data backend only): re-chooses the decision thresholds on the
 * calibration days for other cost settings and applies them to the test period. Illustrative,
 * never the evaluated system. Stateless (D128): the setting lives in this client and is sent with
 * every request; each response says whether it is what-if, and every page refetches on a change.
 */
export function WhatIfPanel() {
  const { toast, whatIf, setWhatIf } = useApp();
  // the status of the CURRENT setting: frozen costs for the evaluated system, else the re-chosen thresholds
  const status = useData(() => (api.whatIf ? api.whatIf(whatIf) : Promise.resolve(null)), [whatIfKey(whatIf)]);
  // fields as text: an emptied field keeps its current value instead of becoming 0 (D137)
  const [draft, setDraft] = useState<Record<keyof CostModel, string> | null>(null);
  useEffect(() => {
    if (status.data && !draft)
      setDraft(Object.fromEntries(FIELDS.map((f) => [f.key, toField(status.data!.cost[f.key] ?? 1)])) as Record<keyof CostModel, string>);
  }, [status.data, draft]);

  if (!api.whatIf || !status.data?.available) return null;

  const apply = async (cost: CostModel | null) => {
    try {
      if (cost) await api.whatIf!(cost); // validated by the backend before the pages use it
      setWhatIf(cost);
      toast(cost ? "What-if cost mode on (illustrative)" : "Back to the evaluated system");
    } catch (e) {
      toast(`Couldn't use these what-if costs. ${(e as Error).message}`, "error");
    }
  };
  // D155: thresholds from the response that produced them, only while it matches the setting shown
  const th = whatIf && whatIfKey(status.data.cost) === whatIfKey(whatIf) ? status.data.thresholds : null;

  return (
    <section className="pane">
      <div className="pane-head">
        <h2>What-if cost mode</h2>
        <span className="mock-note xs">Illustrative, not the evaluated system</span>
      </div>
      <div className="pane-body">
        <p className="muted small">
          Choose other costs: the decision thresholds are re-chosen on the calibration days (147–150) by minimising total cost, as in the research
          pipeline, and applied to the test period. The evaluated system is the frozen decision engine of the final run; switch back at any time.
          With a Hold that always stops the fraud (share 1) a Block can never cost less than a Hold, so Block appears only when held fraud can slip
          through (D123, D124). The setting is kept in this browser tab only and sent with each request.
        </p>
        {draft && (
          <form
            className="threshold-form"
            onSubmit={(e) => {
              e.preventDefault();
              const current = status.data!.cost;
              apply(Object.fromEntries(FIELDS.map((f) => [f.key, parseField(draft[f.key], current[f.key] ?? 1)])) as unknown as CostModel);
            }}
          >
            {FIELDS.map((f) => (
              <label key={f.key} className="field">
                <span>{f.label}</span>
                <input
                  type="number"
                  min={0}
                  max={f.max}
                  step={f.step}
                  value={draft[f.key]}
                  onChange={(e) => setDraft({ ...draft, [f.key]: e.target.value })}
                />
              </label>
            ))}
            <div className="actions" style={{ alignSelf: "end" }}>
              <button type="submit" className="btn btn-primary">Apply what-if costs</button>
              {whatIf && (
                <button type="button" className="btn" onClick={() => apply(null)}>
                  Back to the evaluated system
                </button>
              )}
            </div>
          </form>
        )}
        {th && (
          <p className="muted xs" style={{ marginTop: 8 }}>
            What-if thresholds (expected loss, $): monitor {fmtThreshold(th.t_monitor)}, step-up {fmtThreshold(th.t_step)}, hold {fmtThreshold(th.t_hold)},
            block {fmtThreshold(th.t_block)}; hold on view disagreement from {fmtThreshold(th.t_dis)}.
          </p>
        )}
      </div>
    </section>
  );
}
