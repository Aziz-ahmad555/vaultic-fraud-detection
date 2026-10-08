import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import type { Action, Transaction } from "../api/types";
import { ACTIONS, ACTION_LABEL } from "../api/types";
import { DecisionBadge, RiskCell, setText } from "../components/Badges";
import { Braid } from "../components/Braid";
import { topReason } from "../components/Reasons";
import { Empty } from "../components/States";
import { fmtClock, fmtMoney2 } from "../lib/format";
import { useApp } from "../state/app";

const SPEEDS = [1, 100, 1000] as const;
const TICK_MS = 250;
const KEEP = 250;

/** Replay of the transaction stream from the as-of time, at x1 / x100 / x1000. */
export function Live() {
  const { asOf, range, toast } = useApp();
  const navigate = useNavigate();
  const [clock, setClock] = useState(asOf.time);
  const [rows, setRows] = useState<(Transaction & { fresh?: boolean })[]>([]);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(100);
  const [filter, setFilter] = useState<Action | "alerts" | "all">("all");
  const [search, setSearch] = useState("");
  const clockRef = useRef(clock);
  clockRef.current = clock;

  // a new as-of time restarts the replay there, showing the hour before it as context
  useEffect(() => {
    setPlaying(false);
    setClock(asOf.time);
    api.stream(asOf, asOf.time - 3600, asOf.time).then((txs) => setRows(txs.reverse()));
  }, [asOf]);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(async () => {
      const from = clockRef.current;
      const to = Math.min(from + (TICK_MS / 1000) * speed, range.last);
      if (to >= range.last) {
        setPlaying(false);
        toast("Replay reached the end of the mock data (day 150)");
      }
      const txs = await api.stream({ time: to, labelDelayDays: asOf.labelDelayDays }, from, to);
      setClock(to);
      if (txs.length)
        setRows((prev) => [...txs.reverse().map((t) => ({ ...t, fresh: true })), ...prev.map((t) => ({ ...t, fresh: false }))].slice(0, KEEP));
    }, TICK_MS);
    return () => window.clearInterval(id);
  }, [playing, speed, asOf.labelDelayDays, range.last, toast]);

  const shown = useMemo(() => {
    const q = search.trim().toLowerCase();
    return rows.filter((t) => {
      const a = t.decision.action;
      if (filter === "alerts" && !["step_up", "hold", "block"].includes(a)) return false;
      if (filter !== "all" && filter !== "alerts" && a !== filter) return false;
      if (q && !String(t.row.TransactionID).includes(q) && !t.entities.uid.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [rows, filter, search]);

  return (
    <div className="page">
      <div className="page-head">
        <h1>Live monitor</h1>
        <p className="muted small">
          Replays the mock stream from the as-of time. Replay clock <strong className="num">{fmtClock(clock)}</strong>
        </p>
      </div>
      <div className="toolbar" role="toolbar" aria-label="Replay controls">
        <button type="button" className="btn btn-primary" onClick={() => setPlaying((p) => !p)}>
          {playing ? "Pause replay" : "Start replay"}
        </button>
        <div className="segmented" role="group" aria-label="Replay speed">
          {SPEEDS.map((s) => (
            <button key={s} type="button" aria-pressed={speed === s} onClick={() => setSpeed(s)}>
              ×{s}
            </button>
          ))}
        </div>
        <label className="field-inline">
          <span>Show</span>
          <select value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)}>
            <option value="all">All decisions</option>
            <option value="alerts">Alerts only</option>
            {ACTIONS.map((a) => (
              <option key={a} value={a}>
                {ACTION_LABEL[a]}
              </option>
            ))}
          </select>
        </label>
        <label className="field-inline">
          <span>Search</span>
          <input type="search" placeholder="Transaction or customer id" value={search} onChange={(e) => setSearch(e.target.value)} />
        </label>
        <span className="muted small" aria-live="polite">
          {shown.length} of {rows.length} shown
        </span>
      </div>

      <section className="pane">
        {shown.length === 0 ? (
          <Empty title={rows.length ? "Nothing matches these filters" : "No transactions in the hour before the as-of time"}>
            {rows.length ? "Clear the search or choose All decisions." : "Start the replay, or move the as-of time to a busier hour."}
          </Empty>
        ) : (
          <div className="table-wrap live-wrap">
            <table className="data live">
              <caption className="visually-hidden">Live transaction stream, newest first</caption>
              <thead>
                <tr>
                  <th scope="col">Time</th>
                  <th scope="col">Transaction</th>
                  <th scope="col" className="num">Amount</th>
                  <th scope="col" className="num">Risk</th>
                  <th scope="col">Conformal set</th>
                  <th scope="col">Decision</th>
                  <th scope="col">Top reason</th>
                  <th scope="col">Evidence</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((t) => (
                  <tr
                    key={t.row.TransactionID}
                    className={t.fresh ? "row-new" : undefined}
                    onClick={() => navigate(`/cases/${t.row.TransactionID}`)}
                    style={{ cursor: "pointer" }}
                  >
                    <td className="num">{fmtClock(t.row.TransactionDT, false)}</td>
                    <td>
                      <a href={`/cases/${t.row.TransactionID}`} onClick={(e) => { e.preventDefault(); navigate(`/cases/${t.row.TransactionID}`); }}>
                        {t.row.TransactionID}
                      </a>
                    </td>
                    <td className="num">{fmtMoney2(t.amount)}</td>
                    <td className="num">
                      <RiskCell p={t.decision.p} />
                    </td>
                    <td>
                      {setText(t.decision.conformal_set)}
                    </td>
                    <td>
                      <DecisionBadge action={t.decision.action} />
                    </td>
                    <td className="reason-cell">{topReason(t.reasons)}</td>
                    <td>
                      <Braid weights={t.weights} scores={viewScores(t)} animate={t.fresh} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

export function viewScores(t: Transaction) {
  return {
    tabular: t.row.p_tabular,
    behavioral: t.row.p_behavioral,
    temporal: t.row.p_temporal,
    graph: t.row.p_graph,
    anomaly: t.row.p_anomaly,
  };
}
