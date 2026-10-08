import { api } from "../api";
import { ACTIONS, ACTION_LABEL, VIEWS, VIEW_LABEL } from "../api/types";
import { Columns, HBars, StackedBar } from "../components/Charts";
import { Empty, ErrorState, Loading } from "../components/States";
import { VIEW_VAR, riskVar } from "../lib/colors";
import { fmtClock, fmtInt, fmtMoney, fmtPct, risk } from "../lib/format";
import { useData } from "../lib/useData";
import { useApp } from "../state/app";

const DEC_FILL = (a: string) => `var(--dec-${a}-bg)`;
const DEC_INK = (a: string) => `var(--dec-${a}-fg)`;

export function Overview() {
  const { asOf } = useApp();
  const { data, error, loading, reload } = useData(() => api.daySummary(asOf), [asOf.time, asOf.labelDelayDays]);

  return (
    <div className="page">
      <div className="page-head">
        <h1>Today, day {Math.floor(asOf.time / 86400)}</h1>
        <p className="muted small">Up to {fmtClock(asOf.time, false)} on the dataset clock. Today's labels are not known yet, so money figures are expected values (risk × amount).</p>
      </div>
      {loading && !data && <Loading what="today's summary" />}
      {error && <ErrorState what="today's summary" error={error} onRetry={reload} />}
      {data && data.transactions === 0 && (
        <Empty title="No transactions yet today">Move the as-of time later in the day to see today's transactions.</Empty>
      )}
      {data && data.transactions > 0 && (
        <>
          <section className="figures" aria-label="Key figures">
            <div className="figure">
              <div className="muted small">Transactions</div>
              <div className="figure-value num">{fmtInt(data.transactions)}</div>
            </div>
            <div className="figure">
              <div className="muted small">Alerts</div>
              <div className="figure-value num">{fmtInt(data.alerts)}</div>
              <div className="figure-note">Step-up, hold and block</div>
            </div>
            <div className="figure">
              <div className="muted small">Fraud value stopped</div>
              <div className="figure-value num">{fmtMoney(data.expectedFraudStopped)}</div>
              <div className="figure-note">Expected, from risk × amount</div>
            </div>
            <div className="figure">
              <div className="muted small">Estimated loss</div>
              <div className="figure-value num">{fmtMoney(data.expectedLoss)}</div>
              <div className="figure-note">Expected fraud value allowed through</div>
            </div>
            <div className="figure">
              <div className="muted small">Queue</div>
              <div className="figure-value num">{fmtInt(data.queue)}</div>
              <div className="figure-note">Holds and step-ups waiting</div>
            </div>
          </section>

          <div className="grid-2">
            <section className="pane" aria-labelledby="dec-h">
              <div className="pane-head">
                <h2 id="dec-h">Decisions today</h2>
              </div>
              <div className="pane-body">
                <StackedBar parts={ACTIONS.map((a) => ({ label: ACTION_LABEL[a], value: data.byAction[a], fill: DEC_FILL(a), ink: DEC_INK(a) }))} />
              </div>
            </section>
            <section className="pane" aria-labelledby="cost-h">
              <div className="pane-head">
                <h2 id="cost-h">Expected cost breakdown</h2>
                <span className="muted small">
                  Total {fmtMoney(data.cost.missedFraud + data.cost.falsePositives + data.cost.reviews + data.cost.stepUps)}
                </span>
              </div>
              <div className="pane-body">
                <HBars
                  bars={[
                    { label: "Missed fraud", value: data.cost.missedFraud, text: fmtMoney(data.cost.missedFraud) },
                    { label: "False positives", value: data.cost.falsePositives, text: fmtMoney(data.cost.falsePositives) },
                    { label: "Analyst reviews", value: data.cost.reviews, text: fmtMoney(data.cost.reviews) },
                    { label: "Step-ups", value: data.cost.stepUps, text: fmtMoney(data.cost.stepUps) },
                  ]}
                />
                <p className="muted xs">C_FP $10, review $5, OTP $0.50, step-up stops 90% of fraud, $1 friction per legit customer (D43, D45, D46).</p>
              </div>
            </section>
          </div>

          <div className="grid-2">
            <section className="pane" aria-labelledby="hour-h">
              <div className="pane-head">
                <h2 id="hour-h">Transactions by hour, coloured by mean risk</h2>
              </div>
              <div className="pane-body">
                <Columns
                  data={data.riskByHour.map((h) => ({ key: h.hour, value: h.count, color: riskVar(h.meanRisk) }))}
                  format={(v) => fmtInt(v)}
                />
                <p className="muted xs">
                  Highest mean risk: hour {data.riskByHour.reduce((a, b) => (b.meanRisk > a.meanRisk ? b : a)).hour} (risk{" "}
                  {risk(Math.max(...data.riskByHour.map((h) => h.meanRisk)))}).
                </p>
              </div>
            </section>
            <section className="pane" aria-labelledby="avail-h">
              <div className="pane-head">
                <h2 id="avail-h">Evidence available today</h2>
              </div>
              <div className="pane-body">
                <HBars
                  max={1}
                  bars={VIEWS.map((v) => ({ label: VIEW_LABEL[v], value: data.viewAvailability[v], color: VIEW_VAR[v], text: fmtPct(data.viewAvailability[v]) }))}
                />
                <p className="muted xs">A view is missing when it has no evidence for a transaction (no history, no shared entities); it is never filled in.</p>
              </div>
            </section>
          </div>
        </>
      )}
    </div>
  );
}
