import { useEffect, useState } from "react";
import { api } from "../api";
import type { Thresholds } from "../api/types";
import { ACTION_LABEL } from "../api/types";
import { LockIcon } from "../components/Badges";
import { ErrorState, Loading } from "../components/States";
import { fmtClock, fmtInt, fmtMoney } from "../lib/format";
import { useData } from "../lib/useData";
import { useApp } from "../state/app";

const FIELDS: { key: keyof Thresholds; label: string }[] = [
  { key: "t_monitor", label: `${ACTION_LABEL.monitor} from expected loss ($)` },
  { key: "t_step", label: `${ACTION_LABEL.step_up} from expected loss ($)` },
  { key: "t_hold", label: `${ACTION_LABEL.hold} from expected loss ($)` },
  { key: "t_block", label: `${ACTION_LABEL.block} from expected loss ($)` },
];

export function Settings() {
  const { asOf, role, setRole, toast } = useApp();
  const cfg = useData(() => api.thresholds(), []);
  const summary = useData(() => api.daySummary(asOf), [asOf.time, asOf.labelDelayDays]);
  const [nonce, setNonce] = useState(0);
  const log = useData(() => api.auditLog(), [nonce]);
  const [draft, setDraft] = useState<Thresholds | null>(null);
  useEffect(() => {
    if (cfg.data) setDraft(cfg.data.thresholds);
  }, [cfg.data]);
  const admin = role === "admin";

  const saveThresholds = async () => {
    if (!draft) return;
    try {
      await api.saveThresholds(draft, role);
      toast("Thresholds saved");
      setNonce((n) => n + 1);
    } catch (e) {
      toast(`Couldn't save the thresholds. ${(e as Error).message}`, "error");
    }
  };

  return (
    <div className="page">
      <div className="page-head">
        <h1>Reports and settings</h1>
      </div>
      <div className="grid-2">
        <section className="pane">
          <div className="pane-head"><h2>Daily report</h2></div>
          <div className="pane-body print-region">
            {summary.data ? (
              <>
                <p className="small">
                  Day {summary.data.day}, up to {fmtClock(asOf.time, false)}: {fmtInt(summary.data.transactions)} transactions, {fmtInt(summary.data.alerts)} alerts,
                  expected fraud value stopped {fmtMoney(summary.data.expectedFraudStopped)}, estimated loss {fmtMoney(summary.data.expectedLoss)}. Synthetic mock data.
                </p>
                <div className="actions no-print" style={{ marginTop: 12 }}>
                  <button type="button" className="btn btn-primary" onClick={() => window.print()}>Print or save daily report as PDF</button>
                </div>
              </>
            ) : (
              <Loading what="the report" />
            )}
          </div>
        </section>
        <section className="pane">
          <div className="pane-head"><h2>Role</h2></div>
          <div className="pane-body">
            <div className="segmented" role="group" aria-label="Role">
              <button type="button" aria-pressed={role === "analyst"} onClick={() => { setRole("analyst"); toast("Role set to analyst"); }}>Analyst</button>
              <button type="button" aria-pressed={role === "admin"} onClick={() => { setRole("admin"); toast("Role set to admin"); }}>Admin</button>
            </div>
            <p className="muted small" style={{ marginTop: 8 }}>Mock roles for the layout; there is no sign-in yet. Admins can change decision thresholds.</p>
          </div>
        </section>
      </div>
      <section className="pane">
        <div className="pane-head">
          <h2>Decision thresholds</h2>
          {!admin && (
            <span className="locked">
              <LockIcon title="Read-only for analysts" /> Read-only: switch the role to admin to change them
            </span>
          )}
        </div>
        <div className="pane-body">
          {cfg.error && <ErrorState what="thresholds" error={cfg.error} onRetry={cfg.reload} />}
          {draft && (
            <form
              className="threshold-form"
              onSubmit={(e) => {
                e.preventDefault();
                saveThresholds();
              }}
            >
              {FIELDS.map((f) => (
                <label key={f.key} className="field">
                  <span>{f.label}</span>
                  <input
                    type="number"
                    min={0}
                    step={1}
                    value={draft[f.key] ?? 0}
                    disabled={!admin}
                    onChange={(e) => setDraft({ ...draft, [f.key]: Number(e.target.value) })}
                  />
                </label>
              ))}
              <label className="field">
                <span>Hold when view disagreement is at least</span>
                <input type="number" min={0} max={1} step={0.05} value={draft.t_dis ?? ""} disabled={!admin} onChange={(e) => setDraft({ ...draft, t_dis: e.target.value === "" ? null : Number(e.target.value) })} />
              </label>
              <div className="actions" style={{ alignSelf: "end" }}>
                <button type="submit" className="btn btn-primary" disabled={!admin}>Save thresholds</button>
              </div>
            </form>
          )}
          <p className="muted xs" style={{ marginTop: 8 }}>
            In the research pipeline these thresholds are chosen on validation by minimising total cost (D43); changes here are for the mock only and are
            written to the audit log.
          </p>
        </div>
      </section>
      <section className="pane">
        <div className="pane-head"><h2>Audit log</h2></div>
        {log.loading && !log.data && <Loading what="the audit log" />}
        {log.data && log.data.length === 0 && (
          <p className="pane-body muted">Nothing recorded yet. Labels you give on a case and threshold changes appear here.</p>
        )}
        {log.data && log.data.length > 0 && (
          <div className="table-wrap" style={{ maxHeight: 320 }}>
            <table className="data">
              <caption className="visually-hidden">Audit log</caption>
              <thead>
                <tr>
                  <th scope="col">Time</th>
                  <th scope="col">Role</th>
                  <th scope="col">Action</th>
                  <th scope="col">Detail</th>
                </tr>
              </thead>
              <tbody>
                {log.data.map((a, i) => (
                  <tr key={`${a.time}-${i}`}>
                    <td className="num">{new Date(a.time).toLocaleString()}</td>
                    <td>{a.actor}</td>
                    <td>{a.action}</td>
                    <td className="small">{a.detail}</td>
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
