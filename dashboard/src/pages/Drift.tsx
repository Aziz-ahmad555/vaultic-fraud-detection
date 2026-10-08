import { api } from "../api";
import { HBars, Legend, Lines } from "../components/Charts";
import { Empty, ErrorState, Loading } from "../components/States";
import { fmtMoney } from "../lib/format";
import { useData } from "../lib/useData";
import { useApp } from "../state/app";

export function Drift() {
  const { asOf, setLabelDelay } = useApp();
  const d = useData(() => api.drift(asOf), [asOf.time, asOf.labelDelayDays]);
  const today = Math.floor(asOf.time / 86400);

  return (
    <div className="page">
      <div className="page-head">
        <h1>Drift monitor</h1>
        <div className="toolbar">
          <span className="small muted">Label delay</span>
          <div className="segmented" role="group" aria-label="Label delay">
            {[0, 7, 30, 60].map((L) => (
              <button key={L} type="button" aria-pressed={asOf.labelDelayDays === L} onClick={() => setLabelDelay(L)}>
                {L} days
              </button>
            ))}
          </div>
        </div>
      </div>
      <p className="muted small">
        Label-free signals (PSI, disagreement) see drift the same day; label-based detectors (ADWIN, Page-Hinkley) only see errors once labels arrive, L days
        later. Change the delay to see their alarms move.
      </p>
      {d.loading && !d.data && <Loading what="drift signals" />}
      {d.error && <ErrorState what="drift signals" error={d.error} onRetry={d.reload} />}
      {d.data && (
        <>
          <div className="grid-2">
            <section className="pane">
              <div className="pane-head">
                <h2>Feature drift (PSI, today vs reference)</h2>
              </div>
              <div className="pane-body">
                <HBars
                  rule={{ value: 0.2, label: "alert at 0.2" }}
                  bars={d.data.featurePsi.map((f) => ({ label: f.feature, value: f.psi, text: f.psi > 0.2 ? `${f.psi.toFixed(2)} alert` : f.psi.toFixed(2), color: f.psi > 0.2 ? "var(--ink)" : "var(--graphite)" }))}
                />
              </div>
            </section>
            <section className="pane">
              <div className="pane-head">
                <h2>Score and disagreement drift</h2>
              </div>
              <div className="pane-body">
                {d.data.scorePsi.length < 2 ? (
                  <Empty title="Not enough days yet">Move the as-of date forward to see a trend.</Empty>
                ) : (
                  <>
                    <Lines
                      xLabel="day"
                      series={[
                        { name: "Score PSI", color: "var(--ink)", points: d.data.scorePsi.map((p) => ({ x: p.day, y: p.value })) },
                        { name: "Mean disagreement", color: "var(--graphite)", dashed: true, points: d.data.disagreement.map((p) => ({ x: p.day, y: p.value })) },
                      ]}
                      markers={d.data.injected.map((i) => ({ x: i.day, label: "injected drift", kind: "dashed" as const }))}
                    />
                    <Legend items={[{ label: "Score PSI", color: "var(--ink)" }, { label: "Mean disagreement", color: "var(--graphite)", dashed: true }]} />
                  </>
                )}
              </div>
            </section>
          </div>
          <section className="pane">
            <div className="pane-head">
              <h2>Alarms and retraining</h2>
              <span className="muted small">Up to day {today}</span>
            </div>
            <div className="pane-body">
              <svg viewBox="0 0 900 96" width={900} height={96} style={{ maxWidth: "100%", height: "auto" }} role="img" aria-label={`Timeline: ${d.data.alarms.length} alarms, ${d.data.retrains.length} retraining attempts`} className="chart">
                {(() => {
                  const x = (day: number) => 20 + ((day - 128) / (150 - 128)) * 860;
                  return (
                    <>
                      <line x1={20} x2={880} y1={50} y2={50} stroke="var(--graphite)" />
                      {Array.from({ length: 23 }, (_, i) => 128 + i).map((day) => (
                        <text key={day} x={x(day)} y={92} textAnchor="middle" className="chart-tick">{day % 2 === 0 ? day : ""}</text>
                      ))}
                      {d.data.injected.map((i) => (
                        <g key={`inj${i.day}`}>
                          <line x1={x(i.day)} x2={x(i.day)} y1={20} y2={80} stroke="var(--ink)" strokeWidth={2} strokeDasharray="4 3" />
                          <text x={x(i.day) + 6} y={16} className="chart-tick">{i.kind}</text>
                        </g>
                      ))}
                      {d.data.alarms.map((a) => (
                        <g key={`${a.detector}${a.day}`}>
                          {a.detector === "PSI" || a.detector === "Disagreement" ? (
                            <circle cx={x(a.day)} cy={50} r={6} fill="var(--sheet)" stroke="var(--ink)" strokeWidth={2} />
                          ) : (
                            <rect x={x(a.day) - 6} y={44} width={12} height={12} transform={`rotate(45 ${x(a.day)} 50)`} fill="var(--ink)" />
                          )}
                          <text x={x(a.day)} y={a.detector === "PSI" || a.detector === "Disagreement" ? (a.detector === "PSI" ? 34 : 24) : a.detector === "ADWIN" ? 72 : 82} textAnchor="middle" className="chart-tick">{a.detector}</text>
                        </g>
                      ))}
                      {d.data.retrains.map((r) => (
                        <path key={`rt${r.day}`} d={`M${x(r.day)},58 l6,12 h-12 z`} fill={r.accepted ? "var(--ink)" : "var(--sheet)"} stroke="var(--ink)" />
                      ))}
                    </>
                  );
                })()}
              </svg>
              <ul className="timeline-legend m0 p0" style={{ listStyle: "none" }}>
                <li>Circle: label-free alarm (PSI, disagreement)</li>
                <li>Diamond: label-based alarm (ADWIN, Page-Hinkley)</li>
                <li>Triangle: retraining attempt (filled = challenger accepted)</li>
                <li>Dashed line: injected drift start</li>
              </ul>
            </div>
          </section>
          <section className="pane">
            <div className="pane-head">
              <h2>Champion–challenger history</h2>
              <span className="muted small">A challenger needs a significant PR-AUC gain (95% CI above 0) and no higher cost (D45)</span>
            </div>
            {d.data.retrains.length === 0 ? (
              <Empty title="No retraining attempts yet">They appear after a drift alarm. Move the as-of date past the injected drift on day 139.</Empty>
            ) : (
              <div className="table-wrap">
                <table className="data">
                  <caption className="visually-hidden">Retraining attempts</caption>
                  <thead>
                    <tr>
                      <th scope="col">Day</th>
                      <th scope="col">Trigger</th>
                      <th scope="col" className="num">PR-AUC champion</th>
                      <th scope="col" className="num">PR-AUC challenger</th>
                      <th scope="col" className="num">Gain 95% CI</th>
                      <th scope="col" className="num">Cost champion</th>
                      <th scope="col" className="num">Cost challenger</th>
                      <th scope="col">Outcome</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.data.retrains.map((r) => (
                      <tr key={r.day}>
                        <td className="num">{r.day}</td>
                        <td>{r.trigger === "alarm" ? "Drift alarm" : "Schedule"}</td>
                        <td className="num">{r.prChampion.toFixed(2)}</td>
                        <td className="num">{r.prChallenger.toFixed(2)}</td>
                        <td className="num">
                          {r.gainCiLow.toFixed(2)} to {r.gainCiHigh.toFixed(2)}
                        </td>
                        <td className="num">{fmtMoney(r.costChampion)}</td>
                        <td className="num">{fmtMoney(r.costChallenger)}</td>
                        <td>{r.accepted ? "Replaced the champion" : "Rejected: gain not significant"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
