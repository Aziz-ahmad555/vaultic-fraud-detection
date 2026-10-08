import { useState } from "react";
import { api } from "../api";
import { HeatMap, Legend, Lines } from "../components/Charts";
import { ErrorState, Loading } from "../components/States";
import { fmtMoney, fmtPct } from "../lib/format";
import { useData } from "../lib/useData";

/** Series colours for compared runs: Ink and Graphite plus dash patterns, never view colours. */
const RUN_STYLE = [
  { color: "var(--ink)", dashed: false },
  { color: "var(--graphite)", dashed: true },
  { color: "var(--ink)", dashed: true },
  { color: "var(--graphite)", dashed: false },
];

export function Lab() {
  const lab = useData(() => api.lab(), []);
  const [picked, setPicked] = useState<string[]>(["B5", "MVAF"]);
  const toggle = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : p.length >= 4 ? p : [...p, id]));

  if (lab.loading && !lab.data) return <div className="page"><Loading what="runs" /></div>;
  if (lab.error || !lab.data) return <div className="page"><ErrorState what="runs" error={lab.error ?? new Error("No data")} onRetry={lab.reload} /></div>;
  const { runs, ablation, sensitivity } = lab.data;
  const chosen = runs.filter((r) => picked.includes(r.id));
  const byId = new Map(runs.map((r) => [r.id, r]));

  return (
    <div className="page">
      <div className="page-head">
        <h1>Model lab</h1>
        <p className="mock-note small">Mock runs: synthetic curves and numbers to lay out the page. They are not results.</p>
      </div>
      <section className="pane">
        <div className="pane-head">
          <h2>Runs to compare</h2>
          <span className="muted small">Pick up to four</span>
        </div>
        <div className="pane-body run-picker" role="group" aria-label="Runs to compare">
          {runs.map((r) => (
            <label key={r.id} className="field-inline">
              <input type="checkbox" checked={picked.includes(r.id)} onChange={() => toggle(r.id)} disabled={!picked.includes(r.id) && picked.length >= 4} />
              <span style={{ color: "var(--ink)" }}>{r.name}</span>
            </label>
          ))}
        </div>
      </section>
      <div className="grid-2">
        <section className="pane">
          <div className="pane-head"><h2>Precision–recall curves <span className="muted small">mock numbers</span></h2></div>
          <div className="pane-body">
            <Lines
              xDomain={[0, 1]}
              yDomain={[0, 1]}
              xLabel="recall"
              yLabel="precision"
              series={chosen.map((r, i) => ({ name: r.name, ...RUN_STYLE[i], points: r.pr.map((p) => ({ x: p.recall, y: p.precision })) }))}
            />
            <Legend items={chosen.map((r, i) => ({ label: `${r.name}, PR-AUC ${r.prAuc.toFixed(3)}`, ...RUN_STYLE[i] }))} />
          </div>
        </section>
        <section className="pane">
          <div className="pane-head"><h2>Calibration <span className="muted small">mock numbers</span></h2></div>
          <div className="pane-body">
            <Lines
              xDomain={[0, 1]}
              yDomain={[0, 1]}
              diagonal
              xLabel="predicted"
              yLabel="observed"
              series={chosen.map((r, i) => ({ name: r.name, ...RUN_STYLE[i], points: r.calibration.map((c) => ({ x: c.predicted, y: c.observed })) }))}
            />
            <Legend items={[...chosen.map((r, i) => ({ label: r.name, ...RUN_STYLE[i] })), { label: "Perfect calibration", color: "var(--graphite)", dashed: true }]} />
          </div>
        </section>
      </div>
      <div className="grid-2">
        <section className="pane">
          <div className="pane-head"><h2>Ablation: adding views to B5 <span className="muted small">mock numbers</span></h2></div>
          <div className="table-wrap">
            <table className="data">
              <caption className="visually-hidden">Ablation table</caption>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col" className="num">PR-AUC</th>
                  <th scope="col" className="num">95% CI</th>
                  <th scope="col" className="num">Recall at 1% FPR</th>
                  <th scope="col" className="num">Cost</th>
                </tr>
              </thead>
              <tbody>
                {ablation.map((a) => {
                  const r = byId.get(a.runId)!;
                  return (
                    <tr key={a.step}>
                      <td>{a.step}</td>
                      <td className="num">{r.prAuc.toFixed(3)}</td>
                      <td className="num">{r.ciLow.toFixed(3)} to {r.ciHigh.toFixed(3)}</td>
                      <td className="num">{fmtPct(r.recallAt1)}</td>
                      <td className="num">{fmtMoney(r.cost)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
        <section className="pane">
          <div className="pane-head"><h2>Fusion: F1–F7 vs MVAF <span className="muted small">mock numbers</span></h2></div>
          <div className="table-wrap">
            <table className="data">
              <caption className="visually-hidden">Fusion comparison</caption>
              <thead>
                <tr>
                  <th scope="col">Method</th>
                  <th scope="col" className="num">PR-AUC</th>
                  <th scope="col" className="num">95% CI</th>
                  <th scope="col" className="num">Cost</th>
                </tr>
              </thead>
              <tbody>
                {runs.filter((r) => r.kind === "fusion").map((r) => (
                  <tr key={r.id}>
                    <td>{r.name}</td>
                    <td className="num">{r.prAuc.toFixed(3)}</td>
                    <td className="num">{r.ciLow.toFixed(3)} to {r.ciHigh.toFixed(3)}</td>
                    <td className="num">{fmtMoney(r.cost)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
      <section className="pane">
        <div className="pane-head">
          <h2>Cost sensitivity <span className="muted small">mock numbers</span></h2>
          <span className="muted small">Total cost by false-positive cost and step-up success rate (darker = costlier)</span>
        </div>
        <div className="pane-body">
          <HeatMap
            rows={[2, 5, 10, 20, 50]}
            cols={[0.7, 0.9, 1.0]}
            rowTitle="C_FP ($)"
            colTitle="step-up success"
            value={(r, c) => sensitivity.find((s) => s.c_fp === r && s.step_up_success_rate === c)?.cost ?? 0}
            format={(v) => fmtMoney(v)}
          />
        </div>
      </section>
    </div>
  );
}
