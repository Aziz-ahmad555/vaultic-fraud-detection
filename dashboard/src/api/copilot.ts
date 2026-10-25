/**
 * Client for the read-only copilot (services/api/copilot.py, D224-D226): a local language model
 * that answers questions about one case from its evidence. Answers arrive as an NDJSON stream:
 * "start" (the evidence lines), "token" (draft JSON, not yet checked), then "final" (the checked
 * facts) or "refused" / "error". Nothing here can change a decision, a label or a setting.
 *
 * http mode needs the backend started with VAULTIC_COPILOT=1 (and Ollama on 127.0.0.1 to answer);
 * mock mode (the default) uses a scripted copilot built from the mock case: no Ollama.
 */
import { DATA_SOURCE } from "./index";
import { identityHeaders } from "./identity";
import type { AsOf, Transaction } from "./types";

export interface EvidenceLine {
  handle: string; // "E1"
  evidence_id: string; // "case:3485113/decision"
  tool: string;
  text: string;
}

export interface CheckedFact {
  text: string; // "<fact> [E2]."
  citations: string[];
}

export interface RemovedFact {
  text: string;
  why: string;
}

export interface CopilotFinal {
  type: "final";
  case: number;
  kind: "question" | "summary";
  route: string;
  model: string;
  variant?: string;
  note: string;
  reasons_source: string | null;
  evidence: EvidenceLine[];
  answer: string;
  kept: CheckedFact[];
  removed: RemovedFact[];
  draft?: string | null;
  refused?: string;
  cached?: { label: string; generated_at: string; model: string } | null;
  seconds: number;
  audit: "ok" | "failed";
  mock?: boolean;
}

export type CopilotEvent =
  | { type: "start"; evidence: EvidenceLine[]; model: string; reasons_source: string | null }
  | { type: "token"; text: string; status: string }
  | { type: "refused"; refusal: string; message: string }
  | CopilotFinal
  | { type: "error"; message: string };

export interface CopilotStatus {
  model: string;
  /** "ready", or the reason the model cannot answer (with the start hint) */
  llm: string;
  /** the model is in memory; null when unknown or not running */
  loaded: boolean | null;
  variant?: string;
  testlab: string; // "not mounted" | "loading" | "ready" | "refused" | "error"
  reasons: string;
  cached_summaries: number;
  note: string;
  mock?: boolean;
}

export interface KeyFactsData {
  case: number;
  amount: number;
  product: string;
  decision: Transaction["decision"]["action"];
  p: number;
  expected_loss: number;
  conformal_set: string;
  disagreement: number;
  label: 0 | 1 | null;
  label_delay_days: number;
  reasons_source: string;
  reasons: { text: string; points: number }[];
  testlab: string;
  source: string;
}

export class CopilotError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export const NOT_ENABLED = "The copilot is not enabled on the backend: start it with VAULTIC_COPILOT=1 (see the README).";
export const NOT_AVAILABLE = "Not available from the case evidence.";
export const MOCK_LABEL = "Mock: scripted answers, no language model";

export interface CopilotApi {
  status(): Promise<CopilotStatus>;
  facts(caseId: number, asOf: AsOf, t?: Transaction): Promise<KeyFactsData>;
  ask(caseId: number, question: string, asOf: AsOf, onEvent: (e: CopilotEvent) => void, t?: Transaction): Promise<CopilotFinal>;
  summary(caseId: number, asOf: AsOf, onEvent: (e: CopilotEvent) => void, t?: Transaction): Promise<CopilotFinal>;
}

type Fetch = (input: string, init?: RequestInit) => Promise<Response>;

async function detail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: string };
    if (body.detail) return body.detail;
  } catch {
    /* keep the status */
  }
  return `${res.status} ${res.statusText}`;
}

/** Read an NDJSON body line by line, handing every event over; resolve with the final event. */
export async function readStream(res: Response, onEvent: (e: CopilotEvent) => void): Promise<CopilotFinal> {
  if (!res.body) throw new CopilotError("The copilot sent no answer.", 0);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let final: CopilotFinal | null = null;
  const handle = (line: string) => {
    if (!line.trim()) return;
    const ev = JSON.parse(line) as CopilotEvent;
    onEvent(ev);
    if (ev.type === "final") final = ev;
    if (ev.type === "error") throw new CopilotError(ev.message, 0);
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    lines.forEach(handle);
  }
  handle(buffer);
  if (!final) throw new CopilotError("The copilot stopped before the checked answer.", 0);
  return final;
}

export function createHttpCopilot(baseUrl = "http://127.0.0.1:8000", fetchImpl: Fetch = fetch): CopilotApi {
  const base = baseUrl.replace(/\/$/, "");
  async function send(path: string, init?: RequestInit): Promise<Response> {
    let res: Response;
    try {
      res = await fetchImpl(base + path, { ...init, headers: { "Content-Type": "application/json", ...identityHeaders(), ...(init?.headers ?? {}) } });
    } catch {
      throw new CopilotError(`The backend is not reachable at ${base}.`, 0);
    }
    if (res.status === 404 && path.startsWith("/api/copilot/status")) throw new CopilotError(NOT_ENABLED, 404);
    if (!res.ok) throw new CopilotError(await detail(res), res.status);
    return res;
  }
  const stream = async (path: string, body: object, onEvent: (e: CopilotEvent) => void) =>
    readStream(await send(path, { method: "POST", body: JSON.stringify({ ...body, stream: true }) }), onEvent);
  return {
    status: async () => (await send("/api/copilot/status")).json(),
    facts: async (caseId, asOf) =>
      (await send(`/api/copilot/facts?case_id=${caseId}&time=${asOf.time}&L=${asOf.labelDelayDays}`)).json(),
    ask: (caseId, question, asOf, onEvent) =>
      stream("/api/copilot/ask", { case_id: caseId, question, time: asOf.time, L: asOf.labelDelayDays }, onEvent),
    summary: (caseId, asOf, onEvent) => stream("/api/copilot/summary", { case_id: caseId, time: asOf.time, L: asOf.labelDelayDays }, onEvent),
  };
}

// ---- mock: a scripted copilot built from the mock case (no Ollama) --------------------------------
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function mockEvidence(t: Transaction, asOf: AsOf): EvidenceLine[] {
  const d = t.decision;
  const id = t.row.TransactionID;
  const lines = [
    `transaction ${id}: amount ${t.amount.toFixed(2)}, product ${t.product}; decision ${d.action} (frozen cost rule); calibrated fraud probability ${d.p.toFixed(3)}; expected loss ${d.expected_loss.toFixed(2)}; conformal set ${d.conformal_set}; view disagreement ${d.disagreement.toFixed(3)}.`,
    `customer ${t.entities.uid}: ${t.inputs.historyCount} earlier transactions in total; device seen before ${t.inputs.deviceKnown === null ? "not available" : t.inputs.deviceKnown ? "yes" : "no"}.`,
    t.row.label === null ? `outcome label not known yet (label delay ${asOf.labelDelayDays} days).` : `outcome label: ${t.row.label ? "fraud" : "legit"} (known after the ${asOf.labelDelayDays}-day label delay).`,
    "reasons source: view-level reasons (gate weight x view logit); feature reasons not available: the Test Lab is off.",
    ...t.reasons.slice(0, 3).map((r) => `${r.view} view ${r.points > 0 ? "raises" : "lowers"} the risk by ${Math.abs(r.points).toFixed(1)} points.`),
  ];
  const ids = [`case:${id}/decision`, `case:${id}/customer`, `case:${id}/label`, `case:${id}/reasons`, ...t.reasons.slice(0, 3).map((r) => `case:${id}/view-reason:${r.view}`)];
  return lines.map((text, i) => ({ handle: `E${i + 1}`, evidence_id: ids[i], tool: i < 3 ? "get_case" : "get_reasons", text }));
}

function mockFinal(t: Transaction, kind: "question" | "summary", evidence: EvidenceLine[], facts: [string, string][], removed: RemovedFact[], cached: CopilotFinal["cached"]): CopilotFinal {
  const kept = facts.map(([f, h]) => ({ text: `${f} [${h}].`, citations: [h] }));
  return {
    type: "final", case: t.row.TransactionID, kind, route: kind === "summary" ? "summary" : "why", model: "scripted (mock)", variant: "V3",
    note: `${MOCK_LABEL}. Not the evaluated detection system.`, reasons_source: "view-level", evidence,
    answer: kept.length ? kept.map((k) => k.text).join(" ") : NOT_AVAILABLE, kept, removed, cached, seconds: 0, audit: "ok", mock: true,
  };
}

function scripted(t: Transaction, kind: "question" | "summary", question: string): [string, string][] {
  const d = t.decision;
  const top = t.reasons.find((r) => r.points > 0) ?? t.reasons[0];
  const facts: [string, string][] = [[`The decision is ${d.action} with a calibrated fraud probability of ${d.p.toFixed(3)}`, "E1"]];
  if (/unusual|customer|history/i.test(question) || kind === "summary") facts.push([`Customer ${t.entities.uid} has ${t.inputs.historyCount} earlier transactions in total`, "E2"]);
  if (top && (/why|view|drove|flag/i.test(question) || kind === "summary")) facts.push([`The ${top.view} view ${top.points > 0 ? "raises" : "lowers"} the risk by ${Math.abs(top.points).toFixed(1)} points`, "E5"]);
  if (kind === "summary") facts.push([`The conformal set is ${d.conformal_set}`, "E1"]);
  return facts.slice(0, 4);
}

export function createMockCopilot(delayMs = 25): CopilotApi {
  async function run(t: Transaction | undefined, asOf: AsOf, kind: "question" | "summary", question: string, onEvent: (e: CopilotEvent) => void) {
    if (!t) throw new CopilotError("The mock copilot needs the case.", 0);
    const evidence = mockEvidence(t, asOf);
    onEvent({ type: "start", evidence, model: "scripted (mock)", reasons_source: "view-level" });
    if (/^\s*(change|mark|set|override|approve|release)\b/i.test(question)) {
      const message = "I can't change decisions, labels, thresholds or settings: the copilot is read-only.";
      onEvent({ type: "refused", refusal: "change", message });
      const f = { ...mockFinal(t, "question", [], [], [], null), route: "refused", refused: "change", answer: message };
      onEvent(f);
      return f;
    }
    if (kind === "summary" && t.row.TransactionID % 2 === 0) {
      const f = mockFinal(t, kind, evidence, scripted(t, kind, question), [], { label: "cached, generated at 2026-10-25 04:04 UTC by scripted (mock)", generated_at: "2026-10-25 04:04 UTC", model: "scripted (mock)" });
      onEvent(f);
      return f;
    }
    const facts = scripted(t, kind, question);
    const removed = kind === "summary" ? [{ text: `The customer has 999 earlier transactions [E2].`, why: "number not in the cited evidence: 999" }] : [];
    const draft = JSON.stringify({ facts: facts.map(([fact, evidence]) => ({ fact, evidence })) });
    for (const piece of draft.match(/.{1,12}/g) ?? []) {
      await sleep(delayMs);
      onEvent({ type: "token", text: piece, status: "draft, not yet checked" });
    }
    const f = { ...mockFinal(t, kind, evidence, facts, removed, null), draft };
    onEvent(f);
    return f;
  }
  return {
    status: async () => ({ model: "scripted (mock)", llm: "ready", loaded: true, variant: "V3", testlab: "not mounted", reasons: "view-level reasons", cached_summaries: 0, note: `${MOCK_LABEL}.`, mock: true }),
    facts: async (caseId, asOf, t) => {
      if (!t) throw new CopilotError("The mock key facts need the case.", 0);
      return {
        case: caseId, amount: t.amount, product: t.product, decision: t.decision.action, p: t.decision.p, expected_loss: t.decision.expected_loss,
        conformal_set: t.decision.conformal_set, disagreement: t.decision.disagreement, label: t.row.label, label_delay_days: asOf.labelDelayDays,
        reasons_source: "view-level reasons (gate weight x view logit); feature reasons not available: the Test Lab is off",
        reasons: t.reasons.slice(0, 5).map((r) => ({ text: r.text, points: r.points })), testlab: "not mounted", source: "rendered by code from the mock case; no language model",
      };
    },
    ask: (_id, question, asOf, onEvent, t) => run(t, asOf, "question", question, onEvent),
    summary: (_id, asOf, onEvent, t) => run(t, asOf, "summary", "", onEvent),
  };
}

export const copilotApi: CopilotApi = DATA_SOURCE === "http" ? createHttpCopilot(import.meta.env.VITE_API_URL || undefined) : createMockCopilot();
