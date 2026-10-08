import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import type { Policy, PolicyQueue, Transaction } from "../api/types";
import { POLICIES, POLICY_LABEL } from "../api/types";
import { DecisionBadge, RiskCell } from "../components/Badges";
import { Braid } from "../components/Braid";
import { HBars } from "../components/Charts";
import { DataTable, type Col } from "../components/DataTable";
import { Empty, ErrorState, Loading } from "../components/States";
import { fmtInt, fmtMoney, fmtMoney2 } from "../lib/format";
import { useData } from "../lib/useData";
import { useApp } from "../state/app";
import { viewScores } from "./Live";

const columns: Col<Transaction>[] = [
  { id: "id", header: "Transaction", accessorFn: (t) => t.row.TransactionID },
  { id: "amount", header: "Amount", accessorFn: (t) => t.amount, cell: (c) => fmtMoney2(c.getValue() as number), meta: { num: true } },
  { id: "risk", header: "Risk", accessorFn: (t) => t.decision.p, cell: (c) => <RiskCell p={c.getValue() as number} />, meta: { num: true } },
  { id: "decision", header: "Decision", accessorFn: (t) => t.decision.action, cell: (c) => <DecisionBadge action={c.row.original.decision.action} /> },
  { id: "braid", header: "Evidence", enableSorting: false, cell: (c) => <Braid weights={c.row.original.weights} scores={viewScores(c.row.original)} /> },
];

export function Queue() {
  const { asOf } = useApp();
  const navigate = useNavigate();
  const [k, setK] = useState(50);
  const [shown, setShown] = useState<Policy[]>(["R1", "R3"]);
  const q = useData(() => api.queues(asOf, k), [asOf.time, asOf.labelDelayDays, k]);

  const toggle = (p: Policy) => setShown((s) => (s.includes(p) ? (s.length > 1 ? s.filter((x) => x !== p) : s) : [...s, p].sort()));
  const byPolicy = (p: Policy) => q.data?.find((x) => x.policy === p) as PolicyQueue;
  const matured = q.data?.[0]?.matured.day ?? null;

  return (
    <div className="page">
      <div className="page-head">
        <h1>Analyst queue</h1>
        <p className="muted small">Today's top K cases under each routing policy. Outcomes are shown for the latest day whose labels are known.</p>
      </div>
      <div className="toolbar">
        <label className="field-inline">
          <span>Cases per day (K)</span>
          <select value={k} onChange={(e) => setK(Number(e.target.value))}>
            {[10, 25, 50, 100, 200].map((x) => (
              <option key={x} value={x}>{x}</option>
            ))}
          </select>
        </label>
        <div className="segmented" role="group" aria-label="Policies to compare">
          {POLICIES.map((p) => (
            <button key={p} type="button" aria-pressed={shown.includes(p)} onClick={() => toggle(p)} title={POLICY_LABEL[p]}>
              {p}
            </button>
          ))}
        </div>
      </div>
      {q.loading && !q.data && <Loading what="the queues" />}
      {q.error && <ErrorState what="the queues" error={q.error} onRetry={q.reload} />}
      {q.data && (
        <>
          <section className="pane">
            <div className="pane-head">
              <h2>Fraud value caught per policy</h2>
              <span className="muted small">{matured === null ? "No day has known labels yet" : `Day ${matured}, top ${k} cases, labels known`}</span>
            </div>
            <div className="pane-body">
              {matured === null ? (
                <p className="muted">Labels arrive {asOf.labelDelayDays} days late, so no day of the mock data has known outcomes yet. Shorten the label delay or move the as-of date forward.</p>
              ) : (
                <HBars
                  labelWidth={300}
                  width={860}
                  bars={POLICIES.map((p) => ({
                    label: `${p} ${POLICY_LABEL[p]}`,
                    value: byPolicy(p).matured.fraudValueCaught,
                    text: `${fmtMoney(byPolicy(p).matured.fraudValueCaught)} · ${fmtInt(byPolicy(p).matured.fraudCaught)} frauds`,
                  }))}
                />
              )}
            </div>
          </section>
          <div className="policies">
            {shown.map((p) => {
              const pq = byPolicy(p);
              return (
                <section key={p} className="pane" aria-labelledby={`pol-${p}`}>
                  <div className="pane-head">
                    <h2 id={`pol-${p}`}>
                      {p} <span className="muted small">{POLICY_LABEL[p]}</span>
                    </h2>
                  </div>
                  <DataTable
                    data={pq.cases}
                    columns={columns}
                    caption={`${p} queue for today`}
                    rowKey={(t) => t.row.TransactionID}
                    onRowClick={(t) => navigate(`/cases/${t.row.TransactionID}`)}
                    empty={<Empty title="No transactions yet today">Move the as-of time later in the day.</Empty>}
                  />
                </section>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
