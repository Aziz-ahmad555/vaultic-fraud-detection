/**
 * The copilot panel on the Cases page (D226): a one-click case summary (cached or live), a
 * question box with suggested questions, a streamed answer that is a draft until the checked facts
 * arrive, citation chips that highlight their evidence line, a removed-facts note, and the
 * not-available, refusal, model-not-running, model-loading, Test Lab and error states.
 * It reads only: it cannot change a decision, a label or a setting.
 */
import { useEffect, useRef, useState } from "react";
import {
  copilotApi,
  CopilotError,
  NOT_AVAILABLE,
  type CheckedFact,
  type CopilotApi,
  type CopilotEvent,
  type CopilotFinal,
  type CopilotStatus,
  type EvidenceLine,
} from "../api/copilot";
import type { AsOf, Transaction } from "../api/types";

export const READ_ONLY_NOTE =
  "Read-only assistant: it cannot change decisions, labels or settings. Every fact is checked against the cited evidence; meaning is not checked automatically.";
export const SUGGESTED = ["Why was this flagged?", "What is unusual about this customer?", "Which views drove the score?"];
const FIRST_ANSWER = "a first answer takes about 1-2 minutes on this laptop";

type Run =
  | { kind: "idle" }
  | { kind: "running"; what: "summary" | "question"; evidence: EvidenceLine[]; draft: string }
  | { kind: "done"; final: CopilotFinal }
  | { kind: "failed"; message: string; status: number };

/** The complete facts in a partial JSON draft: shown while streaming, marked "not yet checked". */
export function draftFacts(draft: string): { fact: string; evidence: string }[] {
  const out: { fact: string; evidence: string }[] = [];
  const re = /\{\s*"fact"\s*:\s*("(?:[^"\\]|\\.)*")\s*,\s*"evidence"\s*:\s*("(?:[^"\\]|\\.)*")\s*\}/g;
  for (const m of draft.matchAll(re)) {
    try {
      out.push({ fact: JSON.parse(m[1]) as string, evidence: JSON.parse(m[2]) as string });
    } catch {
      /* an incomplete string: not shown yet */
    }
  }
  return out;
}

/** "Fact [E2]." -> "Fact" (the chips carry the citations). */
const factText = (f: CheckedFact) => f.text.replace(/\s*\[[^\]]*\]\.?\s*$/, "").replace(/\s*\[[^\]]*\]/g, "");

export function removedNote(final: CopilotFinal): string | null {
  const n = final.removed.length;
  if (!n) return null;
  const reasons = [...new Set(final.removed.map((r) => r.why))];
  return `${n} fact${n > 1 ? "s" : ""} removed: ${reasons.join("; ")}`;
}

function testLabNote(state: string): string {
  if (state === "ready") return "Test Lab ready: reasons are live frozen-system reasons (parity-gated).";
  if (state === "loading") return "Test Lab loading: reasons are view-level until it is ready.";
  if (state === "not mounted") return "Test Lab off: reasons are view-level (gate weight × view score).";
  return `Test Lab ${state}: reasons are view-level.`;
}

export function CopilotPanel({ t, asOf, copilot = copilotApi }: { t: Transaction; asOf: AsOf; copilot?: CopilotApi }) {
  const [status, setStatus] = useState<CopilotStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [run, setRun] = useState<Run>({ kind: "idle" });
  const [question, setQuestion] = useState("");
  const [highlight, setHighlight] = useState<string | null>(null);
  const evidenceRef = useRef<HTMLOListElement>(null);
  const caseId = t.row.TransactionID;

  useEffect(() => {
    let live = true;
    setStatusError(null);
    copilot.status().then(
      (s) => live && setStatus(s),
      (e: Error) => live && setStatusError(e.message),
    );
    return () => {
      live = false;
    };
  }, [copilot]);
  useEffect(() => {
    setRun({ kind: "idle" });
    setHighlight(null);
  }, [caseId]);

  const ready = status?.llm === "ready";
  const busy = run.kind === "running";

  const start = async (what: "summary" | "question", q = "") => {
    setHighlight(null);
    setRun({ kind: "running", what, evidence: [], draft: "" });
    const onEvent = (e: CopilotEvent) => {
      if (e.type === "start") setRun((r) => (r.kind === "running" ? { ...r, evidence: e.evidence } : r));
      if (e.type === "token") setRun((r) => (r.kind === "running" ? { ...r, draft: r.draft + e.text } : r));
    };
    try {
      const final = what === "summary" ? await copilot.summary(caseId, asOf, onEvent, t) : await copilot.ask(caseId, q, asOf, onEvent, t);
      setRun({ kind: "done", final });
    } catch (e) {
      setRun({ kind: "failed", message: (e as Error).message, status: e instanceof CopilotError ? e.status : 0 });
    }
    // the answer may have loaded the model (or Ollama may have stopped): read the status again
    copilot.status().then(setStatus, () => {});
  };

  const ask = (q: string) => {
    const text = q.trim();
    if (!text) return;
    setQuestion(text);
    start("question", text);
  };

  const cite = (handle: string) => {
    setHighlight(handle);
    const el = evidenceRef.current?.querySelector(`[data-handle="${handle}"]`) as HTMLElement | null;
    el?.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
  };

  const evidence = run.kind === "done" ? run.final.evidence : run.kind === "running" ? run.evidence : [];

  return (
    <div className="section copilot no-print" data-testid="copilot">
      <h2>
        Copilot <span className="muted xs">{status ? `${status.model}${status.variant ? `, ${status.variant}` : ""}` : ""}</span>
      </h2>
      {status?.mock && <p className="mock-note xs" style={{ display: "inline-block" }}>{status.note}</p>}

      {statusError && <p role="alert" className="small copilot-state" data-testid="copilot-unavailable">{statusError}</p>}
      {status && !ready && (
        <p role="alert" className="small copilot-state" data-testid="copilot-not-running">
          {status.llm} Cached case summaries still work.
        </p>
      )}
      {status && ready && status.loaded === false && (
        <p className="small muted copilot-state" data-testid="copilot-loading-model">
          The model is not loaded yet: the first answer loads it (about 20 s more).
        </p>
      )}
      {status && <p className="xs muted" data-testid="copilot-testlab">{testLabNote(status.testlab)}</p>}

      <div className="actions">
        <button type="button" className="btn btn-primary" onClick={() => start("summary")} disabled={busy || !!statusError}>
          Case summary
        </button>
      </div>
      <form
        className="copilot-ask"
        onSubmit={(e) => {
          e.preventDefault();
          ask(question);
        }}
      >
        <label htmlFor="copilot-q" className="small">Ask about this case</label>
        <div className="copilot-row">
          <input id="copilot-q" type="text" value={question} maxLength={500} onChange={(e) => setQuestion(e.target.value)} disabled={busy || !ready} placeholder="e.g. Why was this flagged?" />
          <button type="submit" className="btn" disabled={busy || !ready || !question.trim()}>Ask</button>
        </div>
        <div className="copilot-suggest" aria-label="Suggested questions">
          {SUGGESTED.map((q) => (
            <button key={q} type="button" className="btn btn-quiet xs" onClick={() => ask(q)} disabled={busy || !ready}>
              {q}
            </button>
          ))}
        </div>
      </form>

      {run.kind === "running" && (
        <div className="copilot-answer" role="status" aria-live="polite" data-testid="copilot-running">
          <p className="small muted">
            {run.draft
              ? "Writing the answer (draft, not yet checked)…"
              : `${run.what === "summary" ? "Preparing the case summary" : "Reading the case evidence"}… ${status?.loaded === false ? "loading the model first; " : ""}${status?.mock ? "" : FIRST_ANSWER}.`}
          </p>
          {draftFacts(run.draft).length > 0 && (
            <ul className="small copilot-draft" data-testid="copilot-draft">
              {draftFacts(run.draft).map((f, i) => (
                <li key={i}>
                  {f.fact} <span className="muted xs">[{f.evidence}] draft, not yet checked</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {run.kind === "failed" && (
        <p role="alert" className="small copilot-state" data-testid="copilot-error">
          {run.status === 503 ? run.message : `The copilot could not answer: ${run.message}`}
        </p>
      )}

      {run.kind === "done" && (
        <div className="copilot-answer" data-testid="copilot-answer">
          {run.final.cached && <p className="mock-note xs" style={{ display: "inline-block" }} data-testid="copilot-cached">{run.final.cached.label}</p>}
          {run.final.mock && <p className="mock-note xs" style={{ display: "inline-block" }}>{run.final.note}</p>}
          {run.final.refused ? (
            <p className="small" data-testid="copilot-refused">{run.final.answer}</p>
          ) : run.final.kept.length === 0 ? (
            <p className="small muted" data-testid="copilot-not-available">{NOT_AVAILABLE}</p>
          ) : (
            <ol className="copilot-facts small">
              {run.final.kept.map((f, i) => (
                <li key={i}>
                  {factText(f)}{" "}
                  {f.citations.map((h) => (
                    <button key={h} type="button" className="cite-chip" aria-pressed={highlight === h} onClick={() => cite(h)} aria-label={`Show evidence ${h}`}>
                      {h}
                    </button>
                  ))}
                </li>
              ))}
            </ol>
          )}
          {removedNote(run.final) && <p className="xs muted" data-testid="copilot-removed">{removedNote(run.final)}</p>}
          {!run.final.refused && (
            <p className="xs muted">
              {run.final.cached
                ? "Checked facts from the cached structured answer, checked again now by the current checker"
                : `Checked facts from the model's structured answer (${run.final.seconds} s)`}
              {run.final.audit === "failed" ? "; the audit entry could not be written." : "."}
            </p>
          )}
        </div>
      )}

      {evidence.length > 0 && (
        <details className="copilot-evidence" open={run.kind === "done"}>
          <summary className="small">Evidence ({evidence.length} lines given to the model)</summary>
          <ol ref={evidenceRef} className="xs m0 p0" data-testid="copilot-evidence">
            {evidence.map((e) => (
              <li key={e.handle} data-handle={e.handle} className={highlight === e.handle ? "ev-line ev-hl" : "ev-line"}>
                <strong>{e.handle}</strong> {e.text} <span className="muted">({e.evidence_id})</span>
              </li>
            ))}
          </ol>
        </details>
      )}

      <p className="xs copilot-note" data-testid="copilot-readonly">{READ_ONLY_NOTE}</p>
      {status && !status.mock && <p className="xs muted">{status.note}</p>}
    </div>
  );
}
