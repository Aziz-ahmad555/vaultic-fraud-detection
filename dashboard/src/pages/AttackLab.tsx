/**
 * Attack Lab (Phase B, D190-D202): seeded synthetic attacks injected into a replay of real
 * legitimate test-period traffic, run live through the Test Lab engine, scored three ways (rules
 * only, model only, rules + model). Synthetic scenario - not the evaluation. The 5-seed suite
 * scorecard and the post-hoc diagnostics are shown below the live run.
 */
import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import {
  attackApi,
  type Action,
  type AttackApi,
  type Intensity,
  type ModeResult,
  type PosthocDay,
  type RateGroup,
  type ReplayRow,
  type RunResult,
  type ScenarioList,
  type SetResult,
  type SuiteResponse,
  type SuiteSetAgg,
} from "../api/attack";
import { testLabApi, type LabStatus, type TestLabApi } from "../api/testlab";
import { StatusBanner } from "./TestLab";

export const LABEL = "synthetic scenario - not the evaluation";
const ACTION_LABEL: Record<Action, string> = { allow: "Allow", monitor: "Monitor", step_up: "Step-up", hold: "Hold", block: "Block" };
const MODES: { key: "rules_only" | "model_only" | "rules_plus_model"; label: string }[] = [
  { key: "rules_only", label: "Rules only" },
  { key: "model_only", label: "Model only" },
  { key: "rules_plus_model", label: "Rules + model" },
];
const pct = (x: number | null | undefined, d = 1) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
const clock = (t: number) => {
  const s = t % 86400;
  return `day ${Math.floor(t / 86400)} ${String(Math.floor(s / 3600)).padStart(2, "0")}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};
const setKind = (name: string) => (name.includes("post hoc") ? "post-hoc sensitivity (chosen after seeing results)" : "");

export function AttackLab({ api = attackApi, lab = testLabApi, pollMs = 3000, replayMs = 350 }: { api?: AttackApi; lab?: TestLabApi; pollMs?: number; replayMs?: number }) {
  const [status, setStatus] = useState<LabStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [list, setList] = useState<ScenarioList | null>(null);
  const [suite, setSuite] = useState<SuiteResponse | null>(null);
  const [scenario, setScenario] = useState("card_testing");

  const refresh = useCallback(async () => {
    try {
      setStatus(await lab.status());
      setStatusError(null);
    } catch (e) {
      setStatusError((e as Error).message);
    }
  }, [lab]);
  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), pollMs);
    return () => window.clearInterval(id);
  }, [refresh, pollMs]);
  useEffect(() => {
    api.scenarios().then(setList).catch(() => setList(null));
    api.suite().then(setSuite).catch(() => setSuite(null));
  }, [api]);

  const info = list?.scenarios.find((s) => s.name === scenario);
  return (
    <div className="page">
      <div className="page-head">
        <h1>Attack Lab</h1>
        <p className="muted small">
          Seeded synthetic attacks injected into a replay of real legitimate test-period traffic (anonymized e-commerce transactions released by Vesta
          for the IEEE-CIS competition), scored live by the rules engine and the frozen system. Every synthetic row is marked and never enters any
          research number.
        </p>
        <p className="small" role="note" data-testid="label"><strong>Synthetic scenario - not the evaluation.</strong> {list?.step_up_note ?? "A step-up is an extra customer check (friction), not a block."}</p>
      </div>
      <StatusBanner status={status} error={statusError} />
      <div className="segmented" role="group" aria-label="Scenario" style={{ margin: "12px 0", flexWrap: "wrap" }}>
        {(list?.scenarios ?? []).map((s) => (
          <button key={s.name} type="button" aria-pressed={scenario === s.name} onClick={() => setScenario(s.name)}>{s.title}</button>
        ))}
      </div>
      {info && (
        <section className="pane" aria-label="Scenario description">
          <div className="pane-body small">
            <p className="m0">{info.description}</p>
            {info.notes.map((n) => <p key={n} className="m0" role="note"><strong>Note:</strong> {n}</p>)}
          </div>
        </section>
      )}
      {status?.state === "ready" && info && <RunPanel key={scenario} api={api} scenario={scenario} meanings={info.intensities} replayMs={replayMs} />}
      <SuitePanel suite={suite} scenario={scenario} />
      <PosthocPanel suite={suite} />
    </div>
  );
}

function RunPanel({ api, scenario, meanings, replayMs }: { api: AttackApi; scenario: string; meanings: Record<Intensity, string>; replayMs: number }) {
  const [intensity, setIntensity] = useState<Intensity>("medium");
  const [seed, setSeed] = useState("0");
  const [model, setModel] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setResult(await api.run({ scenario, intensity, seed: Number(seed), model }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <section className="pane" aria-label="Run a scenario">
        <div className="pane-head"><h2>Run live</h2></div>
        <div className="pane-body">
          <form onSubmit={submit} aria-label="Scenario run">
            <div className="segmented" role="group" aria-label="Intensity">
              {(["low", "medium", "high"] as Intensity[]).map((i) => (
                <button key={i} type="button" aria-pressed={intensity === i} onClick={() => setIntensity(i)}>
                  {i[0].toUpperCase() + i.slice(1)} <span className="muted xs">({meanings[i]})</span>
                </button>
              ))}
            </div>
            <label className="field"><span>Seed</span><input inputMode="numeric" value={seed} onChange={(e) => setSeed(e.target.value)} required pattern="[0-9]{1,4}" /></label>
            <label className="field-inline small">
              <input type="checkbox" checked={!model} onChange={(e) => setModel(!e.target.checked)} /> Model off (degraded mode: the rules alone decide)
            </label>
            <div className="actions"><button type="submit" className="btn btn-primary" disabled={busy}>{busy ? "Running…" : "Run scenario"}</button></div>
          </form>
          {busy && <p className="muted small" role="status">Generating the attack, rebuilding the history with it and scoring live (multi-day scenarios take up to about a minute).</p>}
          {error && <p role="alert" className="small">{error}</p>}
        </div>
      </section>
      {result && <RunResult result={result} replayMs={replayMs} />}
    </>
  );
}

function RunResult({ result, replayMs }: { result: RunResult; replayMs: number }) {
  const degraded = result.model.startsWith("off");
  return (
    <>
      <section className="pane" aria-label="Scorecard">
        <div className="pane-head"><h2>Scorecard: {result.title}, {result.intensity}, seed {result.seed}</h2></div>
        <div className="pane-body small">
          <p className="m0"><strong>Synthetic scenario - not the evaluation.</strong> One seed is an example; the 5-seed results are below.</p>
          <p className="m0">
            Replay: {result.n_legit_replay.toLocaleString()} real legitimate rows on day{result.days.length > 1 ? "s" : ""} {result.days[0]}
            {result.days.length > 1 ? `-${result.days[result.days.length - 1]}` : ""} ({result.n_legit_stored.toLocaleString()} stored (parity-verified),{" "}
            {result.n_legit_live} scored live because they share a customer, card or device with the attack) and {result.n_synthetic} synthetic rows, scored
            in {result.seconds} s. Stored rows reproduce the frozen run's scores (largest difference {result.stored_score_max_diff.toExponential(1)}).
          </p>
          <p className="m0">Model: {result.model}. {result.step_up_note}</p>
          {result.notes.map((n) => <p key={n} className="m0" role="note"><strong>Note:</strong> {n}</p>)}
          {Object.entries(result.results).map(([name, r]) => <SetTable key={name} name={name} r={r} degraded={degraded} />)}
        </div>
      </section>
      <Replay rows={result.replay} replayMs={replayMs} degraded={degraded} />
    </>
  );
}

function SetTable({ name, r, degraded }: { name: string; r: SetResult; degraded: boolean }) {
  const extra = (m: ModeResult) =>
    m.alerted_before_first_cash_out === undefined ? "" : `alerted before cash-out: ${m.alerted_before_first_cash_out ? "yes" : "no"}; build-up alerts ${m.build_up_alerts}/${m.build_up_rows}`;
  return (
    <div className="table-wrap" style={{ marginTop: 12 }}>
      <h3 className="small m0">
        {name} {setKind(name) && <span className="muted">({setKind(name)})</span>}
      </h3>
      <table className="small" aria-label={`Scorecard ${name}`}>
        <thead>
          <tr>
            <th>Mode</th><th>Caught (step-up or higher)</th><th>FPR (step-up or higher)</th><th>FPR step-up only</th><th>FPR hold/block</th><th>Caught (hold/block)</th><th>First alerted row</th><th />
          </tr>
        </thead>
        <tbody>
          {MODES.map(({ key, label }) => {
            const m = r[key];
            return (
              <tr key={key} className={degraded && key === "rules_only" ? "row-new" : undefined}>
                <td>{label}{degraded && key === "rules_only" ? " (deciding)" : ""}</td>
                <td className="num">{m.alerts_synthetic}/{r.n_synthetic_fraud}</td>
                <td className="num">{pct(m.fpr_legit, 2)}</td>
                <td className="num">{pct(m.stepup_only_fpr_legit, 2)}</td>
                <td className="num">{pct(m.hard_fpr_legit, 2)}</td>
                <td className="num">{m.hard_alerts_synthetic}/{r.n_synthetic_fraud}</td>
                <td className="num">{m.first_alerted_synthetic_row ?? "–"}</td>
                <td className="xs">{extra(m)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="xs m0">
        False-positive budget (D191): rules + model adds {(r.added_fpr * 100).toFixed(2)} pp over model only ({r.budget_added_fpr_ok ? "within" : "OVER"} the 0.5 pp
        budget); rules only {r.budget_rules_only_ok ? "within" : "OVER"} the 2% budget.
      </p>
    </div>
  );
}

export function Replay({ rows, replayMs, degraded }: { rows: ReplayRow[]; replayMs: number; degraded: boolean }) {
  const [shown, setShown] = useState(0);
  const sorted = useMemo(() => [...rows].sort((a, b) => a.TransactionDT - b.TransactionDT), [rows]);
  useEffect(() => {
    setShown(0);
    if (replayMs <= 0) {
      setShown(sorted.length);
      return;
    }
    const id = window.setInterval(() => setShown((n) => (n >= sorted.length ? n : n + 1)), replayMs);
    return () => window.clearInterval(id);
  }, [sorted, replayMs]);
  return (
    <section className="pane" aria-label="Live replay">
      <div className="pane-head">
        <h2>Live replay</h2>
        <button type="button" className="btn" onClick={() => setShown(sorted.length)}>Show all</button>
      </div>
      <div className="pane-body">
        <p className="muted xs">
          The synthetic rows and the real legitimate rows scored live, in time order. Decision = {degraded ? "rules only (model off)" : "the more severe of the rule action and the model's E12 action"}.
          Click a synthetic row for its rule trace.
        </p>
        <div className="table-wrap">
          <table className="small" aria-label="Replay">
            <thead>
              <tr><th>Time</th><th>Row</th><th>Amount</th><th>Source</th><th>MVAF risk</th><th>Model</th><th>Rules fired</th><th>Decision</th></tr>
            </thead>
            <tbody>
              {sorted.slice(0, shown).map((r) => (
                <ReplayLine key={r.TransactionID} r={r} />
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted xs">{shown}/{sorted.length} rows</p>
      </div>
    </section>
  );
}

function ReplayLine({ r }: { r: ReplayRow }) {
  const [open, setOpen] = useState(false);
  const kind = r.synthetic ? (r.label === 1 ? "synthetic fraud" : "synthetic legitimate (build-up)") : "real legitimate";
  return (
    <>
      <tr onClick={() => r.trace && setOpen(!open)} style={r.trace ? { cursor: "pointer" } : undefined} aria-expanded={r.trace ? open : undefined}>
        <td>{clock(r.TransactionDT)}</td>
        <td>{kind}</td>
        <td className="num">${r.amount.toFixed(2)}</td>
        <td className="xs">{r.source}</td>
        <td className="num">{pct(r.risk)}</td>
        <td><span className={`badge dec-${r.model_action}`}>{ACTION_LABEL[r.model_action]}</span></td>
        <td>{r.rules_fired || "–"}</td>
        <td><span className={`badge dec-${r.decision}`}>{ACTION_LABEL[r.decision]}</span></td>
      </tr>
      {open && r.trace && (
        <tr>
          <td colSpan={8}>
            <ul className="xs m0" aria-label="Rule trace">
              {r.trace.map((t) => (
                <li key={t.rule}>
                  {t.rule} {t.name}: {t.value == null ? "input missing" : t.value.toFixed(t.value % 1 ? 3 : 0)} vs {t.condition} - {t.fired ? `fired (${ACTION_LABEL[t.action as Action]})` : "not fired"}
                </li>
              ))}
            </ul>
          </td>
        </tr>
      )}
    </>
  );
}

function SuitePanel({ suite, scenario }: { suite: SuiteResponse | null; scenario: string }) {
  const by = suite?.suite?.scenarios?.[scenario] as Record<string, unknown> | undefined;
  if (!by) return null;
  return (
    <section className="pane" aria-label="Five-seed results">
      <div className="pane-head"><h2>Results over 5 seeds × 3 intensities</h2></div>
      <div className="pane-body small">
        <p className="m0"><strong>Synthetic scenario - not the evaluation.</strong> Caught = alerted synthetic fraud rows over seeds 0-4 (per-seed min - max). FPR = real legitimate replay rows, mean (min - max).</p>
        {(["low", "medium", "high"] as const).map((i) => {
          const sets = by[i] as Record<string, SuiteSetAgg> | undefined;
          if (!sets) return null;
          return Object.entries(sets).map(([name, e]) => (
            <div key={`${i}-${name}`} className="table-wrap" style={{ marginTop: 10 }}>
              <h3 className="small m0">{i}, {name} {setKind(name) && <span className="muted">({setKind(name)})</span>}</h3>
              <table className="small" aria-label={`Suite ${i} ${name}`}>
                <thead><tr><th>Mode</th><th>Caught (step-up+)</th><th>FPR (step-up+)</th><th>FPR step-up only</th><th>Caught (hold/block)</th><th>FPR hold/block</th><th /></tr></thead>
                <tbody>
                  {MODES.map(({ key, label }) => {
                    const m = e[key];
                    return (
                      <tr key={key}>
                        <td>{label}</td>
                        <td className="num">{m.alert.recall_count} ({m.alert.recall_per_seed_min_max.join(" - ")})</td>
                        <td className="num">{pct(m.alert.fpr_mean, 2)} ({pct(m.alert.fpr_min_max[0], 2)} - {pct(m.alert.fpr_min_max[1], 2)})</td>
                        <td className="num">{pct(m.stepup_only_fpr_mean, 2)}</td>
                        <td className="num">{m.hard.recall_count}</td>
                        <td className="num">{pct(m.hard.fpr_mean, 2)}</td>
                        <td className="xs">{m.campaigns_alerted_before_cash_out ? `alerted before cash-out ${m.campaigns_alerted_before_cash_out}; build-up alerts ${m.build_up_alerts}` : ""}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="xs m0">Budgets (D191): added FPR within 0.5 pp on {e.budget_added_fpr_ok_seeds} seeds (largest {(e.added_fpr_max * 100).toFixed(2)} pp); rules only within 2% on {e.budget_rules_only_ok_seeds}.</p>
            </div>
          ));
        })}
      </div>
    </section>
  );
}

export function PosthocPanel({ suite }: { suite: SuiteResponse | null }) {
  const p = suite?.posthoc;
  if (!p) return null;
  const acc = p.new_accounts as Record<string, RateGroup>;
  const groups: [string, string][] = [
    ["bust_out_build_up", "Bust-out build-up rows (synthetic, labelled legitimate)"],
    ["all_legit_on_days", "All real legitimate rows, same days"],
    ["young_accounts_d1_0_13", "Real legitimate young accounts (masked time-delta D1 0-13)"],
    ["young_accounts_card_not_present_W", "  of which card-not-present (product W)"],
    ["young_accounts_all_M_V_missing", "  of which every M and V field missing"],
  ];
  return (
    <section className="pane" aria-label="Post-hoc diagnostics">
      <div className="pane-head"><h2>Post-hoc diagnostics</h2></div>
      <div className="pane-body small">
        <p className="m0"><strong>Post-hoc diagnostic (chosen after seeing the scenario results) - not the evaluation.</strong> Real legitimate test-period rows, stored features; thresholds and model fixed.</p>
        <DayChart days={p.per_day} />
        <h3 className="small">Model alerts on real new accounts vs bust-out build-up</h3>
        <div className="table-wrap">
          <table className="small" aria-label="New accounts">
            <thead><tr><th>Group</th><th>Rows</th><th>Alert rate (step-up+)</th><th>Hold/block rate</th></tr></thead>
            <tbody>
              {groups.map(([k, label]) => acc[k] && (
                <tr key={k}><td>{label}</td><td className="num">{acc[k].rows.toLocaleString()}</td><td className="num">{pct(acc[k].alert_rate)}</td><td className="num">{pct(acc[k].hold_block_rate, 2)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export function DayChart({ days }: { days: PosthocDay[] }) {
  const W = 640, H = 220, L = 44, B = 28, T = 10, R = 44;
  const set = (d: PosthocDay, name: string) => (d[name] as { rules_only_fpr: number } | undefined)?.rules_only_fpr ?? 0;
  const maxF = Math.max(0.02, ...days.map((d) => set(d, "D196 (primary)")));
  const maxW = Math.max(800, ...days.map((d) => d.windows_7d_customers_max ?? 0));
  const x = (i: number) => L + (i + 0.5) * ((W - L - R) / days.length);
  const yF = (v: number) => H - B - (v / maxF) * (H - B - T);
  const yW = (v: number) => H - B - (v / maxW) * (H - B - T);
  const bw = ((W - L - R) / days.length) * 0.4;
  const r3 = days[0]?.r3_threshold ?? 778;
  const line = days.map((d, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${yW(d.windows_7d_customers_max ?? 0).toFixed(1)}`).join(" ");
  return (
    <figure className="m0">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Rules-only false-positive rate per test day, with and without R3, and the Windows weekly customer count against R3's threshold">
        <line x1={L} x2={W - R} y1={H - B} y2={H - B} stroke="var(--graphite)" />
        {days.map((d, i) => (
          <g key={d.day}>
            <rect x={x(i) - bw} y={yF(set(d, "D196 (primary)"))} width={bw} height={H - B - yF(set(d, "D196 (primary)"))} fill="var(--risk-5)">
              <title>{`day ${d.day}: rules only ${pct(set(d, "D196 (primary)"), 2)} (D196), ${pct(set(d, "D196 without R3 (post hoc)"), 2)} without R3`}</title>
            </rect>
            <rect x={x(i)} y={yF(set(d, "D196 without R3 (post hoc)"))} width={bw} height={H - B - yF(set(d, "D196 without R3 (post hoc)"))} fill="var(--graphite)" opacity={0.6} />
            {i % 4 === 0 && <text x={x(i)} y={H - 10} fontSize="10" textAnchor="middle" fill="var(--ink)">{d.day}</text>}
          </g>
        ))}
        <line x1={L} x2={W - R} y1={yW(r3)} y2={yW(r3)} stroke="var(--ink)" strokeDasharray="4 3" />
        <text x={W - R + 2} y={yW(r3) + 3} fontSize="10" fill="var(--ink)">R3 {r3}</text>
        <path d={line} fill="none" stroke="var(--ink)" strokeWidth={1.5} />
        <text x={4} y={T + 8} fontSize="10" fill="var(--ink)">{pct(maxF, 0)}</text>
        <text x={W - R + 2} y={T + 8} fontSize="10" fill="var(--ink)">{maxW}</text>
      </svg>
      <figcaption className="xs muted">
        Bars per day: rules-only FPR with the D196 thresholds (left bar) and D196 without R3 (post hoc, right bar); left scale. Line: the largest 7-day distinct-customer
        count of DeviceInfo "Windows" that day, dashed: R3's threshold; right scale. IEEE-CIS has no true device ID; DeviceInfo is a model/OS string.
      </figcaption>
    </figure>
  );
}
