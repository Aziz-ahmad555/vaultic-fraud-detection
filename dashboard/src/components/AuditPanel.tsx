/**
 * The persistent audit log (D208, D209): every analyst action, newest first, with filters, a check
 * of the server's hash chain and anchor, and a CSV export (the export is itself recorded). The log
 * is tamper-evident, not tamper-proof: the check detects edits, removals, reordering and a cut-off
 * tail, but not someone with file access who rewrites the whole chain and its anchor.
 */
import { useState } from "react";
import type { Api, AuditEntry, AuditVerify } from "../api/types";
import { useData } from "../lib/useData";
import { Loading } from "./States";

const show = (x: unknown) => (x === null || x === undefined ? "–" : typeof x === "string" ? x : JSON.stringify(x));

export function AuditPanel({ api, nonce = 0, toast }: { api: Api; nonce?: number; toast: (text: string, tone?: "done" | "error") => void }) {
  const [analyst, setAnalyst] = useState("");
  const [action, setAction] = useState("");
  const [applied, setApplied] = useState({ analyst: "", action: "" });
  const [check, setCheck] = useState<AuditVerify | null>(null);
  const log = useData(() => api.auditLog({ analyst: applied.analyst || undefined, action: applied.action || undefined, limit: 500 }), [nonce, applied.analyst, applied.action]);

  const verify = async () => {
    if (!api.auditVerify) return;
    try {
      setCheck(await api.auditVerify());
    } catch (e) {
      toast(`Couldn't check the audit log. ${(e as Error).message}`, "error");
    }
  };
  const exportCsv = async () => {
    if (!api.auditExport) return;
    try {
      const { csv, audit } = await api.auditExport();
      const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = "vaultic_audit_log.csv";
      a.click();
      URL.revokeObjectURL(url);
      toast(audit === "failed" ? "Audit log exported, but the export could not be recorded in the log" : "Audit log exported (the export is recorded in the log)", audit === "failed" ? "error" : "done");
      setApplied({ ...applied }); // refresh: the export entry appears
    } catch (e) {
      toast(`Couldn't export the audit log. ${(e as Error).message}`, "error");
    }
  };

  return (
    <section className="pane" aria-label="Audit log">
      <div className="pane-head">
        <h2>Audit log</h2>
        <span className="muted xs">Append-only, every entry kept; tamper-evident, not tamper-proof</span>
      </div>
      <div className="pane-body">
        <form
          className="actions"
          aria-label="Filter the audit log"
          onSubmit={(e) => {
            e.preventDefault();
            setApplied({ analyst: analyst.trim(), action: action.trim() });
          }}
        >
          <label className="field-inline small"><span>Analyst</span><input value={analyst} onChange={(e) => setAnalyst(e.target.value)} placeholder="any" /></label>
          <label className="field-inline small"><span>Action contains</span><input value={action} onChange={(e) => setAction(e.target.value)} placeholder="any" /></label>
          <button type="submit" className="btn">Filter</button>
          {api.auditVerify && <button type="button" className="btn" onClick={verify}>Check the chain</button>}
          {api.auditExport && <button type="button" className="btn" onClick={exportCsv}>Export CSV</button>}
        </form>
        <p className="muted xs m0">
          The chain check detects an entry that was edited, removed, reordered or cut off the end. It cannot detect someone with access to the files who
          rewrites the whole chain and its anchor. Analyst names and roles are self-declared by the browser (no sign-in yet): attribution, not proof.
        </p>
        {!api.auditVerify && <p className="muted xs m0" data-testid="audit-check-unavailable">Chain check: not available in mock data.</p>}
        {check && (
          <p className="small" role="status" data-testid="audit-check">
            {check.ok
              ? `Chain intact: ${check.entries.toLocaleString()} ${check.entries === 1 ? "entry" : "entries"}, no edit, removal, reordering or cut-off tail found${check.reason ? ` (${check.reason})` : ""}.`
              : `Chain check failed${check.first_bad ? ` at entry ${check.first_bad}` : ""}: ${check.reason}.`}
          </p>
        )}
        {log.loading && !log.data && <Loading what="the audit log" />}
        {log.data && log.data.length === 0 && <p className="muted">Nothing recorded yet. Labels, Test Lab and Attack Lab actions, report prints and exports appear here.</p>}
        {log.data && log.data.length > 0 && (
          <div className="table-wrap" style={{ maxHeight: 360 }}>
            <table className="data">
              <caption className="visually-hidden">Audit log entries</caption>
              <thead>
                <tr>
                  <th scope="col">Time</th>
                  <th scope="col">Analyst</th>
                  <th scope="col">Role</th>
                  <th scope="col">Action</th>
                  <th scope="col">Target</th>
                  <th scope="col">Before → after</th>
                  <th scope="col">Detail</th>
                </tr>
              </thead>
              <tbody>
                {log.data.map((a: AuditEntry, i) => (
                  <tr key={a.id ?? `${a.time}-${i}`}>
                    <td className="num">{new Date(a.time).toLocaleString()}</td>
                    <td>{a.analyst}</td>
                    <td>{a.role}</td>
                    <td>{a.action}</td>
                    <td className="small">{a.target ?? "–"}</td>
                    <td className="small">{a.before === undefined && a.after === undefined ? "–" : `${show(a.before)} → ${show(a.after)}`}</td>
                    <td className="small">{a.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
