/**
 * Test Lab (D182-D186): the frozen stage-2 system scoring one transaction live, every feature
 * computed point-in-time from the stored history strictly before the transaction. Three modes:
 * a real held-out test transaction (label hidden until revealed, session scoreboard), a custom
 * transaction (no ground truth), and a batch CSV. The page refuses to score until the backend's
 * parity gate has passed, and says why if it failed.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import type { View } from "../api/types";
import { VIEWS, VIEW_LABEL } from "../api/types";
import {
  testLabApi,
  type BatchResult,
  type CustomSpec,
  type LabStatus,
  type LiveResult,
  type Options,
  type Sample,
  type SampleFilter,
  type TestLabApi,
} from "../api/testlab";
import { Braid } from "../components/Braid";
import { emptyScoreboard, outcome, record, type Scoreboard } from "../lib/scoreboard";

const DECISION_LABEL: Record<LiveResult["decision"], string> = {
  allow: "Allow",
  monitor: "Monitor",
  step_up: "Step-up",
  hold: "Hold",
  block: "Block",
};
const FILTERS: { key: SampleFilter; label: string }[] = [
  { key: "any", label: "Any" },
  { key: "fraud", label: "Fraud" },
  { key: "legit", label: "Legitimate" },
  { key: "cold_start", label: "Cold start" },
];
type Mode = "real" | "custom" | "batch";

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const ms = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x)} ms`);
const clock = (s: number) => `${String(Math.floor(s / 3600)).padStart(2, "0")}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}`;

export function TestLab({ api = testLabApi, pollMs = 3000 }: { api?: TestLabApi; pollMs?: number }) {
  const [status, setStatus] = useState<LabStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("real");
  const [board, setBoard] = useState<Scoreboard>(emptyScoreboard);

  const refresh = useCallback(async () => {
    try {
      setStatus(await api.status());
      setStatusError(null);
    } catch (e) {
      setStatusError((e as Error).message);
    }
  }, [api]);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), pollMs);
    return () => window.clearInterval(id);
  }, [refresh, pollMs]);

  const ready = status?.state === "ready";
  return (
    <div className="page">
      <div className="page-head">
        <h1>Test Lab</h1>
        <p className="muted small">
          The frozen final-run system (MVAF, five views, E12 decisions) scoring one transaction live. Every feature is computed from the stored
          history strictly before the transaction. Anonymized e-commerce transactions released by Vesta for the IEEE-CIS competition.
        </p>
      </div>
      <StatusBanner status={status} error={statusError} />
      {ready && (
        <>
          <div className="segmented" role="group" aria-label="Test Lab mode" style={{ margin: "12px 0" }}>
            <button type="button" aria-pressed={mode === "real"} onClick={() => setMode("real")}>Real test transaction</button>
            <button type="button" aria-pressed={mode === "custom"} onClick={() => setMode("custom")}>Build your own transaction</button>
            <button type="button" aria-pressed={mode === "batch"} onClick={() => setMode("batch")}>Batch test (CSV)</button>
          </div>
          {mode === "real" && <RealMode api={api} board={board} setBoard={setBoard} onScored={refresh} />}
          {mode === "custom" && <CustomMode api={api} onScored={refresh} />}
          {mode === "batch" && <BatchMode api={api} onScored={refresh} />}
        </>
      )}
    </div>
  );
}

export function StatusBanner({ status, error }: { status: LabStatus | null; error: string | null }) {
  if (error) {
    return (
      <section className="pane" role="alert">
        <div className="pane-body"><strong>Test Lab unavailable.</strong> {error}</div>
      </section>
    );
  }
  if (!status) return <p className="muted" role="status">Checking the Test Lab…</p>;
  const p = status.parity;
  return (
    <section className="pane" aria-label="Test Lab status" role={status.state === "ready" ? undefined : "alert"}>
      <div className="pane-body small">
        {status.state === "loading" && <p className="m0"><strong>Starting:</strong> {status.message}. The first start builds the history store and runs the parity check (several minutes).</p>}
        {status.state === "refused" && (
          <>
            <p className="m0"><strong>The Test Lab refuses to start.</strong> {status.message}</p>
            {p && Object.keys(p.failing_features).length > 0 && (
              <p className="m0">Failing features: {Object.entries(p.failing_features).slice(0, 8).map(([f, d]) => `${f} (${d.toExponential(1)})`).join(", ")}</p>
            )}
          </>
        )}
        {status.state === "error" && <p className="m0"><strong>The Test Lab failed to start.</strong> {status.message}</p>}
        {status.state === "ready" && p && (
          <p className="m0">
            <strong>Parity passed</strong>: on {p.n.toLocaleString()} real test transactions the live features and scores equal the frozen run
            (largest feature difference {p.max_feature_diff.toExponential(1)}, scores MVAF {p.max_score_diff.MVAF.toExponential(1)} / B5{" "}
            {p.max_score_diff.B5.toExponential(1)}; tolerances: features {p.feature_tolerance}, scores {p.score_tolerance}).{" "}
            Latency this session: p50 {ms(status.latency?.p50_ms)}, p95 {ms(status.latency?.p95_ms)} over {status.latency?.n ?? 0} scores.
          </p>
        )}
      </div>
    </section>
  );
}

export function ResultCard({ result, revealed }: { result: LiveResult; revealed?: 0 | 1 | null }) {
  const weights = Object.fromEntries(VIEWS.map((v) => [v, result.views.find((x) => x.view === v)?.weight ?? 0])) as Record<View, number>;
  const scores = Object.fromEntries(VIEWS.map((v) => [v, result.views.find((x) => x.view === v)?.p ?? null])) as Record<View, number | null>;
  return (
    <section className="pane" aria-label="Live score">
      <div className="pane-head"><h2>Live score</h2></div>
      <div className="pane-body">
        <div className="figures">
          <div className="figure"><div className="figure-value num">{pct(result.risk)}</div><div className="figure-note">MVAF risk</div></div>
          <div className="figure">
            <div className="figure-value"><span className={`badge dec-${result.decision}`}>{DECISION_LABEL[result.decision]}</span></div>
            <div className="figure-note">E12 decision {result.alert ? "(alert)" : "(no alert)"}</div>
          </div>
          {result.expected_loss !== undefined && (
            <div className="figure">
              <div className="figure-value num">${result.expected_loss.toFixed(2)}</div>
              <div className="figure-note">expected loss ({pct(result.calibrated)} calibrated × ${result.amount?.toFixed(2)})</div>
            </div>
          )}
          <div className="figure"><div className="figure-value">{result.conformal_set}</div><div className="figure-note">90% conformal set</div></div>
          <div className="figure">
            <div className="figure-value">{result.flagged_e10 ? "Flagged" : "Not flagged"}</div>
            <div className="figure-note">E10 cost threshold {result.e10_threshold?.toFixed(3)}</div>
          </div>
          <div className="figure"><div className="figure-value num">{ms(result.latency_ms)}</div><div className="figure-note">scoring latency</div></div>
        </div>
        <p className="muted xs">
          E12 decides on the expected loss (calibrated probability × amount) and the conformal set, not on the risk alone: a high risk on a small
          amount can be allowed.
        </p>
        {result.matches_frozen_run !== undefined && (
          <p className="small" data-testid="frozen-match">
            {result.matches_frozen_run ? "Matches frozen run ✓" : "Does NOT match the frozen run ✗"}
            {result.stored_risk !== undefined && ` (stored final-run score ${pct(result.stored_risk)})`}
          </p>
        )}
        {revealed != null && (
          <p className="small" data-testid="truth">
            Truth: <strong>{revealed === 1 ? "fraud" : "legitimate"}</strong> ({outcome(result.alert, revealed)})
          </p>
        )}
        <div style={{ marginTop: 28, clear: "both" }}>
          <Braid weights={weights} scores={scores} risk={result.risk} size="large" label={`Evidence braid: risk ${pct(result.risk)}`} />
        </div>
        {result.reasons && result.reasons.length > 0 && (
          <>
            <h3 className="small">Top reasons</h3>
            <ul className="small">
              {result.reasons.map((r) => (
                <li key={r.feature}>{r.text} <span className="muted">({r.points >= 0 ? "+" : ""}{r.points.toFixed(0)} pts)</span></li>
              ))}
            </ul>
          </>
        )}
        {result.view_reasons && (
          <p className="muted xs">
            View contributions: {result.view_reasons.map((v) => `${VIEW_LABEL[v.view]} ${v.points >= 0 ? "+" : ""}${v.points.toFixed(0)} pts`).join(", ")}
          </p>
        )}
      </div>
    </section>
  );
}

function RealMode({ api, board, setBoard, onScored }: { api: TestLabApi; board: Scoreboard; setBoard: (b: Scoreboard) => void; onScored: () => void }) {
  const [filter, setFilter] = useState<SampleFilter>("any");
  const [sample, setSample] = useState<Sample | null>(null);
  const [result, setResult] = useState<LiveResult | null>(null);
  const [truth, setTruth] = useState<0 | 1 | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pick = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    setTruth(null);
    try {
      setSample(await api.sample(filter));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const score = async () => {
    if (!sample) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.scoreReal(sample.transaction_id));
      onScored();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const reveal = async () => {
    if (!sample || !result) return;
    try {
      const r = await api.reveal(sample.transaction_id);
      setTruth(r.isFraud);
      setBoard(record(board, sample.transaction_id, result.alert, r.isFraud, !!result.matches_frozen_run));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="grid-2">
      <section className="pane">
        <div className="pane-head"><h2>Real test transaction</h2></div>
        <div className="pane-body">
          <div className="segmented" role="group" aria-label="Which transactions">
            {FILTERS.map((f) => (
              <button key={f.key} type="button" aria-pressed={filter === f.key} onClick={() => setFilter(f.key)}>{f.label}</button>
            ))}
          </div>
          <div className="actions" style={{ marginTop: 12 }}>
            <button type="button" className="btn" onClick={pick} disabled={busy}>Pick a random held-out transaction</button>
            <button type="button" className="btn btn-primary" onClick={score} disabled={busy || !sample}>Score live</button>
            <button type="button" className="btn" onClick={reveal} disabled={!result || truth !== null}>Reveal truth</button>
          </div>
          {error && <p role="alert" className="small">{error}</p>}
          {sample && (
            <dl className="small" aria-label="Transaction">
              <dt>Transaction</dt><dd>{sample.transaction_id} · customer {sample.uid} ({sample.history} earlier transactions)</dd>
              <dt>When</dt><dd>Day {sample.day}, {clock(sample.seconds)}</dd>
              <dt>Amount</dt><dd>${sample.amount.toFixed(2)} · product {sample.product ?? "–"}</dd>
              <dt>Card</dt><dd>{sample.card4 ?? "–"} {sample.card6 ?? ""}</dd>
              <dt>Email / device</dt><dd>{sample.email ?? "no email"} · {sample.device_type ?? "no device"} {sample.device_info ?? ""}</dd>
              <dt>Label</dt><dd>{truth === null ? "hidden until you reveal it" : truth === 1 ? "fraud" : "legitimate"}</dd>
            </dl>
          )}
          <h3 className="small">Session scoreboard</h3>
          <table className="small" aria-label="Session scoreboard">
            <tbody>
              <tr><th scope="row">Caught (fraud, alerted)</th><td className="num">{board.caught}</td></tr>
              <tr><th scope="row">Missed (fraud, no alert)</th><td className="num">{board.missed}</td></tr>
              <tr><th scope="row">False alarms (legitimate, alerted)</th><td className="num">{board.falseAlarms}</td></tr>
              <tr><th scope="row">Correct passes</th><td className="num">{board.correctPasses}</td></tr>
              <tr><th scope="row">Matched the frozen run</th><td className="num">{board.matchesFrozen} / {board.seen.length}</td></tr>
            </tbody>
          </table>
          <p className="muted xs">An alert is an E12 decision of step-up or higher.</p>
        </div>
      </section>
      {result && <ResultCard result={result} revealed={truth} />}
    </div>
  );
}

function CustomMode({ api, onScored }: { api: TestLabApi; onScored: () => void }) {
  const [opts, setOpts] = useState<Options | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LiveResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useState({ customerKind: "new", customer: "", amount: "120", product: "W", card4: "visa", card6: "debit", email: "gmail.com", device_type: "", device_info: "", day: "151", time: "12:00" });

  useEffect(() => {
    api.options().then(setOpts, (e: Error) => setError(e.message));
  }, [api]);
  const set = (k: keyof typeof form) => (e: { target: { value: string } }) => setForm({ ...form, [k]: e.target.value });

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const [h, m] = form.time.split(":").map(Number);
    const spec: CustomSpec = {
      customer: form.customerKind === "new" ? "new" : Number(form.customer),
      day: Number(form.day),
      seconds: (h || 0) * 3600 + (m || 0) * 60,
      amount: Number(form.amount),
      product: form.product,
      card4: form.card4 || undefined,
      card6: form.card6 || undefined,
      email: form.email || undefined,
      device_type: form.device_type || undefined,
      device_info: form.device_info || undefined,
    };
    setBusy(true);
    setError(null);
    try {
      setResult(await api.scoreCustom(spec));
      onScored();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const select = (k: keyof typeof form, values: string[] | undefined, label: string, optional = false) => (
    <label className="field">
      <span>{label}</span>
      <select value={form[k]} onChange={set(k)}>
        {optional && <option value="">(none)</option>}
        {(values ?? []).map((v) => <option key={v} value={v}>{v}</option>)}
      </select>
    </label>
  );

  return (
    <div className="grid-2">
      <section className="pane">
        <div className="pane-head"><h2>Build your own transaction</h2></div>
        <div className="pane-body">
          <p className="small" role="note"><strong>Custom input — no ground truth.</strong> The score shows what the frozen system would do; nobody knows whether such a transaction would be fraud.</p>
          <form onSubmit={submit} aria-label="Custom transaction">
            <div className="segmented" role="group" aria-label="Customer">
              <button type="button" aria-pressed={form.customerKind === "new"} onClick={() => setForm({ ...form, customerKind: "new" })}>New customer</button>
              <button type="button" aria-pressed={form.customerKind === "existing"} onClick={() => setForm({ ...form, customerKind: "existing" })}>Existing customer</button>
            </div>
            {form.customerKind === "existing" && (
              <label className="field"><span>Customer ID (uid)</span><input inputMode="numeric" value={form.customer} onChange={set("customer")} required /></label>
            )}
            <label className="field"><span>Amount ($)</span><input type="number" min="0.01" step="0.01" value={form.amount} onChange={set("amount")} required /></label>
            {select("product", opts?.products, "Product")}
            {select("card4", opts?.card4, "Card network", true)}
            {select("card6", opts?.card6, "Card type", true)}
            {select("email", opts?.email_domains, "Purchaser email domain", true)}
            {select("device_type", opts?.device_types, "Device type", true)}
            {select("device_info", opts?.device_infos, "Device", true)}
            <label className="field"><span>Day (test period {opts?.days[0] ?? 151}-{opts?.days[1] ?? 182})</span><input type="number" min={opts?.days[0] ?? 151} max={opts?.days[1] ?? 182} value={form.day} onChange={set("day")} required /></label>
            <label className="field"><span>Time</span><input type="time" value={form.time} onChange={set("time")} required /></label>
            <div className="actions"><button type="submit" className="btn btn-primary" disabled={busy}>Score live</button></div>
          </form>
          <p className="muted xs">
            Existing customer: masked C, M, V and identity fields come from their latest earlier transaction; D1 is set so the transaction belongs to
            them; other time-gap fields are missing. New customer: an unseen card, D1 = 0, masked fields missing.
          </p>
          {error && <p role="alert" className="small">{error}</p>}
        </div>
      </section>
      {result && (
        <div>
          <p className="small" data-testid="custom-label"><strong>{result.kind}</strong>{result.notes?.customer ? ` · ${result.notes.customer}` : ""}</p>
          <ResultCard result={result} />
        </div>
      )}
    </div>
  );
}

function BatchMode({ api, onScored }: { api: TestLabApi; onScored: () => void }) {
  const [result, setResult] = useState<BatchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const download = useMemo(() => (result ? URL.createObjectURL(new Blob([result.csv], { type: "text/csv" })) : null), [result]);

  const upload = async () => {
    const file = input.current?.files?.[0];
    if (!file) {
      setError("Choose a CSV file first.");
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.batch(await file.text()));
      onScored();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const m = result?.metrics;
  return (
    <section className="pane">
      <div className="pane-head"><h2>Batch test (CSV)</h2></div>
      <div className="pane-body">
        <p className="small">
          Download the <a href={api.templateUrl()} download>CSV template</a> (IEEE-CIS transaction and identity columns; isFraud optional), fill
          rows with TransactionDT inside the test period, and upload it. Each row is scored independently against the stored history before its own
          time: rows in the file do not affect each other.
        </p>
        <div className="actions">
          <input ref={input} type="file" accept=".csv,text/csv" aria-label="CSV file" />
          <button type="button" className="btn btn-primary" onClick={upload} disabled={busy}>{busy ? "Scoring…" : "Upload and score"}</button>
        </div>
        {error && <p role="alert" className="small"><strong>File refused:</strong> {error}</p>}
        {result && m && (
          <>
            <p className="small" role="note" data-testid="batch-kind">
              <strong>Your uploaded file only, using your labels. No confidence intervals. Not the research evaluation.</strong>{" "}
              ({result.kind})
            </p>
            <p className="small">{m.rows.toLocaleString()} rows scored. {result.note}.</p>
            {(m.edited_rows ?? 0) > 0 && (
              <p className="small" role="alert" data-testid="batch-edited">
                {m.edited_rows} row{m.edited_rows === 1 ? "" : "s"} reuse a stored TransactionID with different field values: marked
                "edited - not the stored transaction".
              </p>
            )}
            {m.has_label && (
              <div className="figures" aria-label="Batch metrics">
                <div className="figure"><div className="figure-value num">{m.pr_auc == null ? "–" : m.pr_auc.toFixed(3)}</div><div className="figure-note">PR-AUC</div></div>
                <div className="figure"><div className="figure-value num">{m.recall_e12_alert == null ? "–" : pct(m.recall_e12_alert)}</div><div className="figure-note">recall, E12 alert</div></div>
                <div className="figure"><div className="figure-value num">{m.recall_e10_threshold == null ? "–" : pct(m.recall_e10_threshold)}</div><div className="figure-note">recall, E10 threshold</div></div>
                {m.confusion_e12_alert && (
                  <div className="figure">
                    <table className="small" aria-label="Confusion matrix (E12 alert)">
                      <thead><tr><th /><th>Alert</th><th>No alert</th></tr></thead>
                      <tbody>
                        <tr><th scope="row">Fraud</th><td className="num">{m.confusion_e12_alert.tp}</td><td className="num">{m.confusion_e12_alert.fn}</td></tr>
                        <tr><th scope="row">Legitimate</th><td className="num">{m.confusion_e12_alert.fp}</td><td className="num">{m.confusion_e12_alert.tn}</td></tr>
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
            {download && <p className="small"><a href={download} download="testlab_scored.csv">Download the scored CSV</a></p>}
            <div className="table-wrap">
              <table className="small" aria-label="Batch results">
                <thead><tr><th>Transaction</th><th>Customer</th><th>Risk</th><th>Decision</th><th>Alert</th><th>Latency</th>{m.has_label && <th>isFraud</th>}<th>Row</th></tr></thead>
                <tbody>
                  {result.rows.slice(0, 200).map((r) => (
                    <tr key={String(r.TransactionID)}>
                      <td className="num">{String(r.TransactionID)}</td><td className="num">{String(r.uid)}</td>
                      <td className="num">{pct(Number(r.risk))}</td><td>{String(r.decision)}</td><td>{r.alert ? "yes" : "no"}</td>
                      <td className="num">{ms(Number(r.latency_ms))}</td>{m.has_label && <td>{String(r.isFraud)}</td>}<td>{String(r.provenance ?? "")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
