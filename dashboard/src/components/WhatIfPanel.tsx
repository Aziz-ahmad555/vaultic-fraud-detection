import { useEffect, useState } from "react";
import { api } from "../api";
import type { CostModel } from "../api/types";
import { useData } from "../lib/useData";
import { useApp } from "../state/app";

const FIELDS: { key: keyof CostModel; label: string; step: number; max?: number }[] = [
  { key: "c_fp", label: "False positive cost ($)", step: 1 },
  { key: "c_rev", label: "Analyst review cost ($)", step: 1 },
  { key: "c_step", label: "Step-up (OTP) cost ($)", step: 0.1 },
  { key: "step_up_success_rate", label: "Step-up stops fraud (share)", step: 0.05, max: 1 },
  { key: "legit_step_up_friction", label: "Friction for a legitimate customer ($)", step: 0.5 },
];

/**
 * What-if cost mode (D122, real-data backend only): re-chooses the decision thresholds on the
 * calibration days for other cost settings and applies them to the test period. Illustrative,
 * never the evaluated system; every decision made in this mode is marked, and the top bar says so.
 */
export function WhatIfPanel({ onChange }: { onChange?: () => void }) {
  const { role, toast, setWhatIfActive, whatIfActive } = useApp();
  const status = useData(() => (api.whatIf ? api.whatIf() : Promise.resolve(null)), [whatIfActive]);
  const [draft, setDraft] = useState<CostModel | null>(null);
  useEffect(() => {
    if (status.data) setDraft(status.data.cost);
  }, [status.data]);

  if (!api.setWhatIf || !status.data?.available) return null;
  const set = api.setWhatIf;

  const apply = async (cost: CostModel | null) => {
    try {
      const w = await set(cost, role);
      setWhatIfActive(w.active);
      toast(w.active ? "What-if cost mode on (illustrative)" : "Back to the evaluated system");
      onChange?.();
    } catch (e) {
      toast(`Couldn't change the what-if cost mode. ${(e as Error).message}`, "error");
    }
  };

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
        </p>
        {draft && (
          <form
            className="threshold-form"
            onSubmit={(e) => {
              e.preventDefault();
              apply(draft);
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
                  onChange={(e) => setDraft({ ...draft, [f.key]: Number(e.target.value) })}
                />
              </label>
            ))}
            <div className="actions" style={{ alignSelf: "end" }}>
              <button type="submit" className="btn btn-primary">Apply what-if costs</button>
              {status.data.active && (
                <button type="button" className="btn" onClick={() => apply(null)}>
                  Back to the evaluated system
                </button>
              )}
            </div>
          </form>
        )}
        {status.data.active && status.data.thresholds && (
          <p className="muted xs" style={{ marginTop: 8 }}>
            What-if thresholds (expected loss, $): monitor {status.data.thresholds.t_monitor.toFixed(2)}, step-up {status.data.thresholds.t_step.toFixed(2)},
            hold {status.data.thresholds.t_hold.toFixed(2)}, block {status.data.thresholds.t_block === null ? "never" : status.data.thresholds.t_block.toFixed(2)}.
          </p>
        )}
      </div>
    </section>
  );
}
