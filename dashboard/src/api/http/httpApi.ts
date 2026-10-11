/**
 * Real-data implementation of the Api contract: the demo backend (services/api/demo_server.py)
 * serving the EXP-200-final test period ("IEEE-CIS test period, final run").
 *
 * Every method maps to one endpoint and returns the server's JSON unchanged (the server already
 * follows types.ts). The one exception is score(): the final-run models score recorded
 * transactions only (the server answers 501), so the what-if simulator keeps the mock's
 * illustrative model, and SCORE_SOURCE says so.
 */
import { createMockApi } from "../mock/mockApi";
import type {
  Api,
  AsOf,
  AuditEntry,
  CostModel,
  DaySummary,
  DriftReport,
  Feedback,
  Lab,
  Network,
  PolicyQueue,
  Role,
  ScoreInput,
  ScoreResult,
  Thresholds,
  Transaction,
} from "../types";

export const HTTP_SOURCE_LABEL = "IEEE-CIS test period, final run";
export const SCORE_SOURCE = "Simulator: illustrative model (the final-run models score recorded transactions only)";

type Fetch = (input: string, init?: RequestInit) => Promise<Response>;

export function createHttpApi(baseUrl = "http://127.0.0.1:8000", fetchImpl: Fetch = fetch): Api {
  const base = baseUrl.replace(/\/$/, "");
  const illustrative = createMockApi();

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetchImpl(`${base}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
    if (!res.ok) {
      let detail = `${res.status}`;
      try {
        const body = (await res.json()) as { detail?: string };
        if (body.detail) detail = body.detail;
      } catch {
        /* keep the status code */
      }
      throw new Error(detail);
    }
    return (await res.json()) as T;
  }

  const q = (asOf: AsOf, extra: Record<string, string | number | undefined> = {}) => {
    const params = new URLSearchParams({ time: String(asOf.time), L: String(asOf.labelDelayDays) });
    for (const [k, v] of Object.entries(extra)) if (v !== undefined) params.set(k, String(v));
    return params.toString();
  };

  return {
    range: () => request<{ first: number; last: number }>("/api/range"),
    daySummary: (asOf) => request<DaySummary>(`/api/day-summary?${q(asOf)}`),
    stream: (asOf, from, to) => request<Transaction[]>(`/api/stream?${q(asOf, { start: from, end: to })}`),
    transaction: (id, asOf) => request<Transaction | null>(`/api/transactions/${id}?${q(asOf)}`),
    alerts: (asOf) => request<Transaction[]>(`/api/alerts?${q(asOf)}`),
    customerHistory: (uid, asOf) =>
      request<Transaction[]>(`/api/customers/${encodeURIComponent(uid)}/history?${q(asOf)}`),
    network: (asOf, focus) => request<Network>(`/api/network?${q(asOf, { focus })}`),
    queues: (asOf, k) => request<PolicyQueue[]>(`/api/queues?${q(asOf, { k })}`),
    drift: (asOf) => request<DriftReport>(`/api/drift?${q(asOf)}`),
    lab: () => request<Lab>("/api/lab"),
    score: (input: ScoreInput): Promise<ScoreResult> => illustrative.score(input),
    thresholds: () => request<{ thresholds: Thresholds; cost: CostModel }>("/api/thresholds"),
    saveThresholds: async (t: Thresholds, role: Role) => {
      await request("/api/thresholds", { method: "PUT", body: JSON.stringify({ thresholds: t, role }) });
    },
    feedback: async (id: number, label: Feedback, role: Role) => {
      await request(`/api/feedback/${id}`, { method: "POST", body: JSON.stringify({ label, role }) });
    },
    auditLog: () => request<AuditEntry[]>("/api/audit"),
  };
}
