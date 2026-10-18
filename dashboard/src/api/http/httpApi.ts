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
  AuditEvent,
  AuditFilter,
  AuditVerify,
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
  SourceInfo,
  Thresholds,
  Transaction,
  WhatIf,
} from "../types";
import { flagsIn, noteServed, whatIfKey, whatIfParams } from "../../lib/whatIf";
import { identityHeaders } from "../identity";

export const HTTP_SOURCE_LABEL = "IEEE-CIS test period, final run";
export const SCORE_SOURCE = "Simulator: illustrative model, not the evaluated system (the final-run models score recorded transactions only)";

type Fetch = (input: string, init?: RequestInit) => Promise<Response>;

export function createHttpApi(baseUrl = "http://127.0.0.1:8000", fetchImpl: Fetch = fetch): Api {
  const base = baseUrl.replace(/\/$/, "");
  const illustrative = createMockApi();

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetchImpl(`${base}${path}`, {
      ...init,
      // the acting analyst and role travel with every request (D208)
      headers: { "Content-Type": "application/json", ...identityHeaders(), ...(init?.headers ?? {}) },
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
    // the what-if setting travels with every as-of request (D128); the server keeps none
    for (const [k, v] of Object.entries({ ...whatIfParams(asOf.whatIf), ...extra })) if (v !== undefined) params.set(k, String(v));
    return params.toString();
  };

  /** An as-of request whose response flags are reported for the banner (D128). */
  async function asOfRequest<T>(asOf: AsOf, path: string, extra: Record<string, string | number | undefined> = {}): Promise<T> {
    const body = await request<T>(`${path}?${q(asOf, extra)}`);
    const flags = flagsIn(body);
    if (flags.length) noteServed(whatIfKey(asOf.whatIf), flags.some(Boolean));
    return body;
  }

  return {
    range: () => request<{ first: number; last: number }>("/api/range"),
    daySummary: (asOf) => asOfRequest<DaySummary>(asOf, "/api/day-summary"),
    stream: (asOf, from, to) => asOfRequest<Transaction[]>(asOf, "/api/stream", { start: from, end: to }),
    transaction: (id, asOf) => asOfRequest<Transaction | null>(asOf, `/api/transactions/${id}`),
    alerts: (asOf) => asOfRequest<Transaction[]>(asOf, "/api/alerts"),
    customerHistory: (uid, asOf) => asOfRequest<Transaction[]>(asOf, `/api/customers/${encodeURIComponent(uid)}/history`),
    network: (asOf, focus) => asOfRequest<Network>(asOf, "/api/network", { focus }),
    queues: (asOf, k) => asOfRequest<PolicyQueue[]>(asOf, "/api/queues", { k }),
    drift: (asOf) => request<DriftReport>(`/api/drift?${q(asOf)}`),
    lab: () => request<Lab>("/api/lab"),
    score: (input: ScoreInput): Promise<ScoreResult> => illustrative.score(input),
    thresholds: () => request<{ thresholds: Thresholds; cost: CostModel; frozen?: boolean }>("/api/thresholds"),
    saveThresholds: async (t: Thresholds, role: Role) => {
      await request("/api/thresholds", { method: "PUT", body: JSON.stringify({ thresholds: t, role }) });
    },
    feedback: async (id: number, label: Feedback, role: Role) => {
      await request(`/api/feedback/${id}`, { method: "POST", body: JSON.stringify({ label, role }) });
    },
    auditLog: (filter?: AuditFilter) => {
      const params = new URLSearchParams();
      if (filter?.analyst) params.set("analyst", filter.analyst);
      if (filter?.action) params.set("action", filter.action);
      if (filter?.limit) params.set("limit", String(filter.limit));
      const qs = params.toString();
      return request<AuditEntry[]>(`/api/audit${qs ? `?${qs}` : ""}`);
    },
    auditVerify: () => request<AuditVerify>("/api/audit/verify"),
    auditExport: async () => {
      const res = await fetchImpl(`${base}/api/audit/export.csv`, { headers: identityHeaders() });
      if (!res.ok) throw new Error(`${res.status}`);
      const csv = await res.text();
      return res.headers.get("X-Vaultic-Audit") === "failed" ? { csv, audit: "failed" as const } : { csv };
    },
    auditEvent: async (event: AuditEvent, detail = "") => {
      await request("/api/audit/event", { method: "POST", body: JSON.stringify({ action: event, detail }) });
    },
    info: async (): Promise<SourceInfo> => {
      const m = await request<{ source: string; no_block_note: string | null; whatif_available: boolean }>("/api/meta");
      return { source: m.source, noBlockNote: m.no_block_note, whatIfAvailable: m.whatif_available };
    },
    whatIf: (cost: CostModel | null) => {
      const params = new URLSearchParams(Object.entries(whatIfParams(cost)).map(([k, v]) => [k, String(v)]));
      return request<WhatIf>(`/api/whatif${cost ? `?${params}` : ""}`);
    },
  };
}
