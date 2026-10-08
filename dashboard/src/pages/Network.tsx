import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import type { NetworkNode } from "../api/types";
import { NetworkGraph } from "../components/NetworkGraph";
import { Empty, ErrorState, Loading } from "../components/States";
import { fmtClock } from "../lib/format";
import { useData } from "../lib/useData";
import { useApp } from "../state/app";

const TYPE_LABEL: Record<NetworkNode["type"], string> = {
  uid: "Customer",
  card: "Card",
  email: "Email domain",
  address: "Billing region",
  device: "Device",
};

export function Network() {
  const { asOf, range } = useApp();
  const [params] = useSearchParams();
  const [focus, setFocus] = useState<string | null>(params.get("focus"));
  const [selected, setSelected] = useState<string | null>(null);
  const [until, setUntil] = useState(asOf.time);
  useEffect(() => setUntil(asOf.time), [asOf.time]);
  const net = useData(() => api.network(asOf, focus ?? undefined), [asOf.time, asOf.labelDelayDays, focus]);
  const node = useMemo(() => net.data?.nodes.find((n) => n.id === selected) ?? null, [net.data, selected]);
  const shownEdges = net.data?.edges.filter((e) => e.time <= until).length ?? 0;

  return (
    <div className="page">
      <div className="page-head">
        <h1>Fraud network</h1>
        <p className="muted small">
          Customers linked through cards, devices, email domains and billing regions. Only links seen by {fmtClock(until)} are drawn; fraud counts use
          labels known at the as-of time.
        </p>
      </div>
      <div className="toolbar">
        {focus ? (
          <>
            <button type="button" className="btn" onClick={() => { setFocus(null); setSelected(null); }}>
              Show all rings
            </button>
            <span className="small">Showing two links around {focus.replace(":", " ")}</span>
          </>
        ) : (
          <span className="small muted">Showing rings: linked customers with at least two confirmed frauds. Click an entity, then expand it.</span>
        )}
      </div>
      <div className="grid-network">
        <section className="pane" aria-label="Entity graph">
          {net.loading && !net.data && <Loading what="the network" />}
          {net.error && <ErrorState what="the network" error={net.error} onRetry={net.reload} />}
          {net.data && net.data.nodes.length === 0 && (
            <Empty title="No rings at this as-of time">
              Rings need at least two confirmed frauds, and labels arrive {asOf.labelDelayDays} days late. Move the as-of date forward or shorten the label
              delay.
            </Empty>
          )}
          {net.data && net.data.nodes.length > 0 && (
            <NetworkGraph network={net.data} until={until} focus={focus} onSelect={setSelected} label={`Entity graph with ${net.data.nodes.length} entities and ${shownEdges} links`} />
          )}
          <div className="pane-body slider-row">
            <label htmlFor="net-time" className="small">Time</label>
            <input
              id="net-time"
              type="range"
              min={range.first}
              max={asOf.time}
              step={3600}
              value={until}
              onChange={(e) => setUntil(Number(e.target.value))}
              aria-valuetext={fmtClock(until)}
            />
            <output className="num small slider-out">
              <span>{fmtClock(until)}</span>
              <span className="muted">{shownEdges} links</span>
            </output>
          </div>
        </section>
        <aside className="pane" aria-label="Details">
          <div className="pane-head">
            <h2>{node ? TYPE_LABEL[node.type] : "Rings"}</h2>
          </div>
          <div className="pane-body">
            {node ? (
              <>
                <dl className="facts">
                  <dt>Entity</dt>
                  <dd>{node.label}</dd>
                  <dt>Transactions</dt>
                  <dd className="num">{node.transactions}</dd>
                  <dt>Confirmed frauds</dt>
                  <dd className="num">{node.knownFraud}</dd>
                  <dt>Ring</dt>
                  <dd>{node.ring ? `Ring ${node.ring}` : "Not part of a ring"}</dd>
                </dl>
                <div className="actions" style={{ marginTop: 12 }}>
                  <button type="button" className="btn btn-primary" onClick={() => setFocus(node.id)}>
                    Expand this entity
                  </button>
                  <button type="button" className="btn" onClick={() => setSelected(null)}>
                    Back to rings
                  </button>
                </div>
              </>
            ) : net.data && net.data.rings.length ? (
              <table className="data">
                <caption className="visually-hidden">Rings with confirmed frauds</caption>
                <thead>
                  <tr>
                    <th scope="col">Ring</th>
                    <th scope="col" className="num">Customers</th>
                    <th scope="col" className="num">Confirmed frauds</th>
                  </tr>
                </thead>
                <tbody>
                  {net.data.rings.map((r) => (
                    <tr key={r.id}>
                      <td>Ring {r.id}</td>
                      <td className="num">{r.customers}</td>
                      <td className="num">{r.knownFrauds}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="muted small">No rings yet at this as-of time.</p>
            )}
            <ul className="legend" style={{ marginTop: 16 }}>
              <li>Circle: customer</li>
              <li>Square: card</li>
              <li>Diamond: device</li>
              <li>Triangle: email domain</li>
              <li>Hexagon: billing region</li>
              <li><span className="swatch" style={{ background: "var(--risk-3)" }} /> Has confirmed fraud</li>
              <li><span className="swatch" style={{ borderColor: "var(--risk-5)", borderWidth: 3 }} /> Ring member</li>
              <li><span className="swatch line" style={{ borderTopColor: "var(--view-relational)" }} /> Link (relational view colour)</li>
            </ul>
          </div>
        </aside>
      </div>
    </div>
  );
}
