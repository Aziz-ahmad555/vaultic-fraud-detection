import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import type { Feedback, ScoreInput, ScoreResult, Transaction } from "../api/types";
import { ACTION_LABEL, VIEWS, VIEW_LABEL } from "../api/types";
import { DecisionBadge, LabelCell, LockIcon, RiskCell, SetBadge } from "../components/Badges";
import { Braid } from "../components/Braid";
import { NetworkGraph } from "../components/NetworkGraph";
import { ReasonList } from "../components/Reasons";
import { RiskGauge } from "../components/RiskGauge";
import { Empty, ErrorState, Loading } from "../components/States";
import { DAY, fmtClock, fmtMoney2, fmtPct, fmtPoints, risk } from "../lib/format";
import { useData } from "../lib/useData";
import { isTyping, useApp } from "../state/app";
import { viewScores } from "./Live";

/** Plain-language case narrative from the same facts the page shows (no free text, D38). */
export function narrative(t: Transaction, cf: string | null): string {
  const d = t.decision;
  const parts = [
    `Transaction ${t.row.TransactionID} (${fmtMoney2(t.amount)}, product ${t.product}) scored a risk of ${risk(d.p)} out of 100 and was routed to ${ACTION_LABEL[d.action].toLowerCase()}.`,
  ];
  if (d.conformal_set === "uncertain" || d.conformal_set === "empty")
    parts.push("The model is uncertain about this case: its conformal set contains both classes.");
  const up = t.reasons.filter((r) => r.points > 0).slice(0, 3);
  if (up.length) parts.push(`The main reasons are: ${up.map((r) => `${r.text.charAt(0).toLowerCase()}${r.text.slice(1)} (${fmtPoints(r.points)} points)`).join("; ")}.`);
  else parts.push("No feature raised the risk above a typical transaction.");
  const missing = VIEWS.filter((v) => t.row[`m_${v}`] === 0).map((v) => VIEW_LABEL[v].toLowerCase());
  if (missing.length) parts.push(`No ${missing.join(" or ")} evidence was available, so ${missing.length > 1 ? "those views carry" : "that view carries"} no weight.`);
  if (t.inputs.sharedEntities > 0)
    parts.push(`Its card or device is shared with ${t.inputs.sharedEntities} other customer${t.inputs.sharedEntities > 1 ? "s" : ""}.`);
  else parts.push("No graph evidence of risk: no shared card or device.");
  parts.push(cf ?? "No counterfactual has been explored yet.");
  return parts.join(" ");
}

function Timeline({ history, current, asOf }: { history: Transaction[]; current: Transaction; asOf: number }) {
  if (history.length <= 1) return <p className="muted small">No earlier transactions for this customer before the as-of time (cold start).</p>;
  const width = 330;
  const height = 120;
  const t0 = Math.min(...history.map((h) => h.row.TransactionDT));
  const t1 = asOf;
  const a = history.map((h) => Math.log10(Math.max(h.amount, 1)));
  const lo = Math.min(...a);
  const hi = Math.max(...a, lo + 0.5);
  const x = (t: number) => 8 + ((t - t0) / Math.max(t1 - t0, 1)) * (width - 16);
  const y = (amt: number) => height - 18 - ((Math.log10(Math.max(amt, 1)) - lo) / (hi - lo)) * (height - 32);
  return (
    <figure className="m0">
      <svg width={width} height={height} style={{ maxWidth: "100%", height: "auto" }} viewBox={`0 0 ${width} ${height}`} className="timeline" role="img" aria-label={`Customer timeline: ${history.length} transactions up to the as-of time`}>
        <line x1={8} x2={width - 8} y1={height - 14} y2={height - 14} stroke="var(--rule)" />
        {history.map((h) => {
          const isCurrent = h.row.TransactionID === current.row.TransactionID;
          const label = h.row.label;
          return (
            <g key={h.row.TransactionID}>
              <circle
                cx={x(h.row.TransactionDT)}
                cy={y(h.amount)}
                r={isCurrent ? 6 : 3.5}
                fill={label === 1 ? "var(--risk-5)" : label === 0 ? "var(--sheet)" : "var(--rule)"}
                stroke={isCurrent ? "var(--ink)" : "var(--graphite)"}
                strokeWidth={isCurrent ? 2 : 1}
              >
                <title>{`${fmtClock(h.row.TransactionDT)} · ${fmtMoney2(h.amount)} · ${label === null ? "label not known yet" : label ? "confirmed fraud" : "confirmed legit"}`}</title>
              </circle>
            </g>
          );
        })}
        <text x={8} y={height - 2} className="chart-tick">{fmtClock(t0)}</text>
        <text x={width - 8} y={height - 2} textAnchor="end" className="chart-tick">as of</text>
      </svg>
      <figcaption className="legend">
        <span><span className="swatch" style={{ background: "var(--risk-5)" }} /> Confirmed fraud</span>
        <span><span className="swatch" style={{ background: "var(--sheet)" }} /> Confirmed legit</span>
        <span><span className="swatch" style={{ background: "var(--rule)" }} /> Label not known yet</span>
      </figcaption>
    </figure>
  );
}

function Counterfactual({ t, onDescribe }: { t: Transaction; onDescribe: (s: string | null) => void }) {
  const [amount, setAmount] = useState(t.amount);
  const [deviceKnown, setDeviceKnown] = useState<boolean | null>(t.inputs.deviceKnown);
  const [result, setResult] = useState<ScoreResult | null>(null);
  useEffect(() => {
    setAmount(t.amount);
    setDeviceKnown(t.inputs.deviceKnown);
  }, [t]);
  useEffect(() => {
    const input: ScoreInput = { ...t.inputs, amount, deviceKnown };
    let live = true;
    api.score(input).then((r) => {
      if (!live) return;
      setResult(r);
      const changes = [];
      if (Math.abs(amount - t.amount) > 0.005) changes.push(`the amount were ${fmtMoney2(amount)} instead of ${fmtMoney2(t.amount)}`);
      if (deviceKnown !== t.inputs.deviceKnown) changes.push(deviceKnown ? "the device were one this customer has used before" : "the device were new to this customer");
      onDescribe(changes.length ? `If ${changes.join(" and ")}, the risk would be ${risk(r.p)} instead of ${risk(t.decision.p)}, and the decision ${ACTION_LABEL[r.decision.action].toLowerCase()}.` : null);
    });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [amount, deviceKnown, t]);

  const max = Math.max(t.amount * 2, 20);
  return (
    <div>
      <div className="cf-row">
        <label htmlFor="cf-amount">Amount</label>
        <input id="cf-amount" type="range" min={1} max={max} step={0.5} value={amount} onChange={(e) => setAmount(Number(e.target.value))} aria-valuetext={fmtMoney2(amount)} />
        <span className="num">{fmtMoney2(amount)}</span>
      </div>
      <div className="cf-row">
        <span>Device</span>
        {t.inputs.deviceKnown === null ? (
          <span className="muted">No device on this transaction</span>
        ) : (
          <div className="segmented" role="group" aria-label="Device">
            <button type="button" aria-pressed={deviceKnown === true} onClick={() => setDeviceKnown(true)}>Known</button>
            <button type="button" aria-pressed={deviceKnown === false} onClick={() => setDeviceKnown(false)}>New</button>
          </div>
        )}
        <span />
      </div>
      <h3 className="small" style={{ marginTop: 12 }}>Locked</h3>
      <ul className="m0 p0 locked-list">
        {[
          ["Card", t.entities.card],
          ["Billing address", t.entities.address ?? "none"],
          ["History", `${t.inputs.historyCount} earlier transactions${t.inputs.usualAmount ? `, usual ${fmtMoney2(t.inputs.usualAmount)}` : ""}`],
          ["Shared entities", String(t.inputs.sharedEntities)],
        ].map(([k, v]) => (
          <li key={k} className="locked">
            <LockIcon title={`${k} is locked and cannot change`} />
            <span>
              {k}: {v}
            </span>
          </li>
        ))}
      </ul>
      {result && (
        <div className="cf-result" aria-live="polite">
          Risk would be <strong className="num">{risk(result.p)}</strong> (now {risk(t.decision.p)}), decision{" "}
          <DecisionBadge action={result.decision.action} />
        </div>
      )}
    </div>
  );
}

export function Case() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { asOf, role, toast } = useApp();
  const alerts = useData(() => api.alerts(asOf), [asOf.time, asOf.labelDelayDays]);
  const caseId = id ? Number(id) : (alerts.data?.[0]?.row.TransactionID ?? null);
  const tx = useData(() => (caseId ? api.transaction(caseId, asOf) : Promise.resolve(null)), [caseId, asOf.time, asOf.labelDelayDays]);
  const t = tx.data;
  const history = useData(() => (t ? api.customerHistory(t.entities.uid, asOf) : Promise.resolve([])), [t?.entities.uid, asOf.time, asOf.labelDelayDays]);
  const net = useData(() => (t ? api.network(asOf, `uid:${t.entities.uid}`) : Promise.resolve(null)), [t?.entities.uid, asOf.time, asOf.labelDelayDays]);
  const [cfText, setCfText] = useState<string | null>(null);
  const [mine, setMine] = useState<Feedback | null>(null);

  const list = alerts.data ?? [];
  const index = useMemo(() => list.findIndex((a) => a.row.TransactionID === caseId), [list, caseId]);

  const mark = async (label: Feedback) => {
    if (!t) return;
    try {
      await api.feedback(t.row.TransactionID, label, role);
      setMine(label);
      toast(`Marked as ${label}`);
    } catch (e) {
      toast(`Couldn't save the label: ${(e as Error).message}`, "error");
    }
  };

  useEffect(() => setMine(null), [caseId]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if ((k === "j" || k === "k") && list.length) {
        const next = k === "j" ? Math.min(index + 1, list.length - 1) : Math.max(index - 1, 0);
        navigate(`/cases/${list[next].row.TransactionID}`);
      } else if (k === "f") mark("fraud");
      else if (k === "l") mark("legit");
      else if (k === "u") mark("unsure");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <div className="console">
      <aside className="console-queue no-print" aria-label="Today's alerts">
        <div className="pane-head">
          <h2>Alerts today</h2>
          <span className="muted xs">J / K</span>
        </div>
        {alerts.loading && !alerts.data && <Loading what="alerts" />}
        {alerts.error && <ErrorState what="alerts" error={alerts.error} onRetry={alerts.reload} />}
        {alerts.data && list.length === 0 && <Empty title="No alerts yet today">Move the as-of time later in the day to see alerts.</Empty>}
        <ul className="queue-list">
          {list.map((a) => (
            <li key={a.row.TransactionID}>
              <Link to={`/cases/${a.row.TransactionID}`} aria-current={a.row.TransactionID === caseId ? "true" : undefined}>
                <span>
                  {a.row.TransactionID}
                  <br />
                  <span className="muted xs">{fmtClock(a.row.TransactionDT, false)} · {fmtMoney2(a.amount)}</span>
                </span>
                <RiskCell p={a.decision.p} />
                <DecisionBadge action={a.decision.action} />
              </Link>
            </li>
          ))}
        </ul>
      </aside>

      <section className="console-case print-region" aria-label="Case">
        {tx.loading && !t && <Loading what="the case" />}
        {tx.error && <ErrorState what="the case" error={tx.error} onRetry={tx.reload} />}
        {!tx.loading && !t && !tx.error && (
          <Empty title={caseId ? `No transaction ${caseId} at this as-of time` : "Pick a case"}>
            {caseId ? "It happened after the as-of time, or the id is wrong. Move the as-of date forward or open another case." : "Choose an alert on the left, or press J."}
          </Empty>
        )}
        {t && (
          <>
            <div className="section case-head">
              <RiskGauge p={t.decision.p} />
              <div>
                <h1>Case {t.row.TransactionID}</h1>
                <p className="muted small">
                  {fmtClock(t.row.TransactionDT)} · customer {t.entities.uid} · {fmtMoney2(t.amount)} · product {t.product}
                </p>
                <div className="actions" style={{ marginTop: 8 }}>
                  <DecisionBadge action={t.decision.action} />
                  <SetBadge set={t.decision.conformal_set} />
                  <span className="badge set-badge">Disagreement {t.decision.disagreement.toFixed(2)}</span>
                  <span className="badge set-badge">Expected loss {fmtMoney2(t.decision.expected_loss)}</span>
                  <LabelCell label={t.row.label} />
                </div>
              </div>
            </div>
            <div className="section">
              <h2>Evidence braid</h2>
              <Braid size="large" weights={t.weights} scores={viewScores(t)} reasons={t.reasons} />
            </div>
            <div className="section">
              <h2>Reasons</h2>
              <ReasonList reasons={t.reasons} />
              <p className="muted xs" style={{ marginTop: 8 }}>
                Points: +20 doubles the fraud odds. Masked features keep their masked names; their meaning is not published.
              </p>
            </div>
            <div className="section">
              <h2>Narrative</h2>
              <p className="narrative">{narrative(t, cfText)}</p>
            </div>
            <div className="section no-print">
              <h2>Your decision</h2>
              <div className="actions">
                <button type="button" className="btn" onClick={() => mark("fraud")} aria-pressed={mine === "fraud"}>Mark as fraud <kbd>F</kbd></button>
                <button type="button" className="btn" onClick={() => mark("legit")} aria-pressed={mine === "legit"}>Mark as legit <kbd>L</kbd></button>
                <button type="button" className="btn" onClick={() => mark("unsure")} aria-pressed={mine === "unsure"}>Mark as unsure <kbd>U</kbd></button>
                <button type="button" className="btn btn-primary" onClick={() => window.print()}>Print or save as PDF</button>
              </div>
              <p className="muted xs" style={{ marginTop: 8 }}>"Unsure" labels are stored but never used for training.</p>
            </div>
          </>
        )}
      </section>

      <aside className="console-evidence" aria-label="Evidence">
        {t && (
          <>
            <div className="section">
              <h2>Customer timeline</h2>
              {history.data ? <Timeline history={history.data} current={t} asOf={asOf.time} /> : <Loading what="the timeline" />}
              {asOf.labelDelayDays > 0 && (
                <p className="curtain xs" style={{ padding: "4px 6px", marginTop: 6 }}>
                  Labels after day {Math.floor((asOf.time - asOf.labelDelayDays * DAY) / DAY)} are not known yet (label delay {asOf.labelDelayDays} days).
                </p>
              )}
            </div>
            <div className="section no-print">
              <h2>Linked entities</h2>
              {net.data ? (
                net.data.nodes.length > 1 ? (
                  <>
                    <NetworkGraph mini network={net.data} until={asOf.time} focus={`uid:${t.entities.uid}`} label={`Entities linked to customer ${t.entities.uid}`} onSelect={() => navigate("/network")} />
                    <p className="muted xs">{net.data.nodes.length} entities within two links. Open the fraud network for details.</p>
                  </>
                ) : (
                  <p className="muted small">No linked entities yet.</p>
                )
              ) : (
                <Loading what="linked entities" />
              )}
            </div>
            <div className="section">
              <h2>Counterfactual</h2>
              <p className="muted xs">What would change the decision. Card, address and history are locked: they describe who the customer is and what already happened.</p>
              <Counterfactual t={t} onDescribe={setCfText} />
            </div>
            <div className="section">
              <h2>View scores</h2>
              <ul className="m0 p0" style={{ listStyle: "none" }}>
                {VIEWS.map((v) => {
                  const p = viewScores(t)[v];
                  return (
                    <li key={v} className="small" style={{ display: "flex", justifyContent: "space-between", padding: "2px 0" }}>
                      <span>{VIEW_LABEL[v]}</span>
                      {p === null ? <span className="muted">not available</span> : (
                        <span className="num">
                          <RiskCell p={p} /> weight {fmtPct(t.weights[v])}
                        </span>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}
