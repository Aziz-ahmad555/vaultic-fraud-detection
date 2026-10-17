/**
 * Client for the Test Lab endpoints (services/api/testlab.py, D182-D186): the frozen stage-2
 * system scoring one transaction live. Needs the backend (VITE_DATA_SOURCE=http) started with
 * VAULTIC_TESTLAB=1; the mock has no Test Lab.
 */
import type { View } from "./types";

export type LabState = "loading" | "ready" | "refused" | "error";

export interface Parity {
  n: number;
  passed: boolean;
  tolerance: number;
  max_feature_diff: number;
  max_score_diff: { MVAF: number; B5: number };
  n_failing_features: number;
  failing_features: Record<string, number>;
  cached?: boolean;
}

export interface Latency {
  n: number;
  p50_ms: number | null;
  p95_ms: number | null;
  max_ms?: number;
}

export interface LabStatus {
  state: LabState;
  message: string;
  parity: Parity | null;
  latency?: Latency;
}

export interface Sample {
  transaction_id: number;
  uid: number;
  day: number;
  seconds: number;
  amount: number;
  product: string | null;
  card4: string | null;
  card6: string | null;
  email: string | null;
  device_type: string | null;
  device_info: string | null;
  history: number;
  filter: SampleFilter;
}

export type SampleFilter = "any" | "fraud" | "legit" | "cold_start";

export interface ViewScore {
  view: View;
  available: boolean;
  raw: number | null;
  p: number | null;
  weight: number | null;
}

export interface LiveResult {
  risk: number;
  calibrated: number;
  conformal_set: string;
  disagreement: number;
  decision: "allow" | "monitor" | "step_up" | "hold" | "block";
  alert: boolean;
  flagged_e10: boolean;
  e10_threshold: number | null;
  b5_score: number;
  views: ViewScore[];
  latency_ms: number;
  explain_ms?: number;
  uid: number;
  view_reasons?: { view: View; points: number }[];
  reasons?: { feature: string; text: string; points: number }[];
  kind: string;
  transaction_id?: number;
  stored_risk?: number;
  matches_frozen_run?: boolean;
  notes?: Record<string, string | number>;
  amount?: number;
  expected_loss?: number;
}

export interface CustomSpec {
  customer: number | "new";
  day: number;
  seconds: number;
  amount: number;
  product: string;
  card4?: string;
  card6?: string;
  email?: string;
  device_type?: string;
  device_info?: string;
}

export interface Options {
  products: string[];
  card4: string[];
  card6: string[];
  email_domains: string[];
  device_types: string[];
  device_infos: string[];
  days: [number, number];
}

export interface BatchResult {
  metrics: {
    rows: number;
    has_label: boolean;
    frauds?: number;
    pr_auc?: number | null;
    recall_e12_alert?: number | null;
    recall_e10_threshold?: number | null;
    confusion_e12_alert?: { tp: number; fp: number; fn: number; tn: number };
  };
  rows: Record<string, string | number | boolean>[];
  csv: string;
  note: string;
}

type Fetch = (input: string, init?: RequestInit) => Promise<Response>;

export class LabError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export function createTestLabApi(baseUrl = "http://127.0.0.1:8000", fetchImpl: Fetch = fetch) {
  const base = baseUrl.replace(/\/$/, "");
  async function call<T>(path: string, init?: RequestInit): Promise<T> {
    let res: Response;
    try {
      res = await fetchImpl(base + path, init);
    } catch {
      throw new LabError(`The Test Lab backend is not reachable at ${base}. Start it with VAULTIC_TESTLAB=1 (see the README).`, 0);
    }
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try {
        const body = (await res.json()) as { detail?: string };
        if (body.detail) detail = body.detail;
      } catch {
        /* keep the status text */
      }
      throw new LabError(detail, res.status);
    }
    return (await res.json()) as T;
  }
  const json = (body: unknown): RequestInit => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return {
    status: () => call<LabStatus>("/api/live/status"),
    sample: (filter: SampleFilter) => call<Sample>(`/api/live/sample?filter=${filter}`),
    scoreReal: (id: number) => call<LiveResult>("/api/score-live", json({ transaction_id: id })),
    scoreCustom: (spec: CustomSpec) => call<LiveResult>("/api/score-live", json({ custom: spec })),
    reveal: (id: number) => call<{ transaction_id: number; isFraud: 0 | 1 }>(`/api/live/reveal/${id}`),
    latency: () => call<Latency>("/api/live/latency"),
    options: () => call<Options>("/api/live/options"),
    templateUrl: () => `${base}/api/live/template.csv`,
    batch: (csv: string) => call<BatchResult>("/api/live/batch", { method: "POST", headers: { "Content-Type": "text/csv" }, body: csv }),
  };
}

export type TestLabApi = ReturnType<typeof createTestLabApi>;

export const testLabApi = createTestLabApi(import.meta.env.VITE_API_URL || undefined);
