import { useEffect, useState } from "react";
import { api } from "../api";
import type { ScoreInput, ScoreResult, View } from "../api/types";
import { VIEWS, VIEW_LABEL } from "../api/types";
import { DecisionBadge, SetBadge } from "../components/Badges";
import { Braid } from "../components/Braid";
import { ReasonList } from "../components/Reasons";
import { RiskGauge } from "../components/RiskGauge";
import { fmtMoney2 } from "../lib/format";

const START: ScoreInput = {
  amount: 120,
  usualAmount: 45,
  deviceKnown: true,
  velocity24h: 1,
  sharedEntities: 0,
  historyCount: 12,
  maskedC13: 1,
  available: { tabular: true, behavioral: true, temporal: true, graph: true, anomaly: true },
};

/** What-if scoring: every change re-scores through the same api.score the case page uses. */
export function Simulator() {
  const [x, setX] = useState<ScoreInput>(START);
  const [r, setR] = useState<ScoreResult | null>(null);
  useEffect(() => {
    let live = true;
    api.score(x).then((res) => live && setR(res));
    return () => {
      live = false;
    };
  }, [x]);
  const set = <K extends keyof ScoreInput>(k: K, v: ScoreInput[K]) => setX((s) => ({ ...s, [k]: v }));
  const setView = (v: View, on: boolean) => setX((s) => ({ ...s, available: { ...s.available, [v]: on } }));

  return (
    <div className="page">
      <div className="page-head">
        <h1>Scenario simulator</h1>
        <p className="muted small">Change one thing at a time and watch the risk, the braid and the reasons. Scored by the mock model.</p>
      </div>
      <div className="grid-sim">
        <section className="pane" aria-label="Inputs">
          <div className="pane-head"><h2>Transaction</h2><button type="button" className="btn" onClick={() => setX(START)}>Reset inputs</button></div>
          <div className="pane-body sim-inputs">
            <div className="cf-row">
              <label htmlFor="s-amt">Amount</label>
              <input id="s-amt" type="range" min={1} max={2000} step={1} value={x.amount} onChange={(e) => set("amount", Number(e.target.value))} aria-valuetext={fmtMoney2(x.amount)} />
              <span className="num">{fmtMoney2(x.amount)}</span>
            </div>
            <div className="cf-row">
              <label htmlFor="s-hist">History</label>
              <input id="s-hist" type="range" min={0} max={60} step={1} value={x.historyCount} onChange={(e) => { const n = Number(e.target.value); setX((s) => ({ ...s, historyCount: n, usualAmount: n ? (s.usualAmount ?? 45) : null })); }} />
              <span className="num">{x.historyCount} past</span>
            </div>
            <div className="cf-row">
              <span>Device</span>
              <div className="segmented span-2" role="group" aria-label="Device novelty">
                <button type="button" aria-pressed={x.deviceKnown === true} onClick={() => set("deviceKnown", true)}>Known</button>
                <button type="button" aria-pressed={x.deviceKnown === false} onClick={() => set("deviceKnown", false)}>New</button>
                <button type="button" aria-pressed={x.deviceKnown === null} onClick={() => set("deviceKnown", null)}>None</button>
              </div>
            </div>
            <div className="cf-row">
              <label htmlFor="s-vel">Velocity 24 h</label>
              <input id="s-vel" type="range" min={0} max={20} step={1} value={x.velocity24h} onChange={(e) => set("velocity24h", Number(e.target.value))} />
              <span className="num">{x.velocity24h}</span>
            </div>
            <div className="cf-row">
              <label htmlFor="s-sh">Shared entities</label>
              <input id="s-sh" type="range" min={0} max={10} step={1} value={x.sharedEntities} onChange={(e) => set("sharedEntities", Number(e.target.value))} />
              <span className="num">{x.sharedEntities}</span>
            </div>
            <fieldset className="views-fieldset">
              <legend className="small">Views available</legend>
              {VIEWS.map((v) => (
                <label key={v} className="field-inline">
                  <input type="checkbox" checked={x.available[v]} onChange={(e) => setView(v, e.target.checked)} />
                  <span style={{ color: "var(--ink)" }}>{VIEW_LABEL[v]}</span>
                </label>
              ))}
              {x.historyCount === 0 && <p className="muted xs">With no history, behavioral and temporal are missing whatever these switches say.</p>}
            </fieldset>
          </div>
        </section>
        <section className="pane" aria-label="Result" aria-live="polite">
          <div className="pane-head"><h2>Result</h2></div>
          {r && (
            <div className="pane-body">
              <div className="case-head">
                <RiskGauge p={r.p} />
                <div className="actions">
                  <DecisionBadge action={r.decision.action} />
                  <SetBadge set={r.decision.conformal_set} />
                  <span className="badge set-badge">Expected loss {fmtMoney2(r.decision.expected_loss)}</span>
                </div>
              </div>
              <div style={{ marginTop: 16 }}>
                <Braid size="large" weights={r.weights} scores={r.views} reasons={r.reasons} />
              </div>
              <h3 style={{ margin: "16px 0 8px" }}>Reasons</h3>
              <ReasonList reasons={r.reasons} limit={7} />
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
