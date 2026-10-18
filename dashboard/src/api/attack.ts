/**
 * Client for the Attack Lab endpoints (services/api/attack.py, Phase B): synthetic attack scenarios
 * run live through the Test Lab engine. Synthetic scenario - not the evaluation. Needs the backend
 * started with VAULTIC_TESTLAB=1.
 */
import { LabError } from "./testlab";

export type Action = "allow" | "monitor" | "step_up" | "hold" | "block";
export type Intensity = "low" | "medium" | "high";
export type Generator = "v1" | "v2";

export interface ScenarioInfo {
  name: string;
  title: string;
  description: string;
  intensities: Record<Intensity, string>;
  notes: string[];
}

export interface ScenarioList {
  label: string;
  step_up_note: string;
  intensities: Intensity[];
  threshold_sets: string[];
  scenarios: ScenarioInfo[];
}

export interface ModeResult {
  recall_synthetic: number | null;
  fpr_legit: number | null;
  alerts_synthetic: number;
  alerts_legit: number;
  first_alerted_synthetic_row: number | null;
  hard_recall_synthetic: number | null;
  hard_fpr_legit: number | null;
  hard_alerts_synthetic: number;
  hard_alerts_legit: number;
  stepup_only_fpr_legit: number | null;
  stepup_only_alerts_legit: number;
  build_up_alerts?: number;
  build_up_rows?: number;
  alerted_before_first_cash_out?: boolean;
}

export interface SetResult {
  n_synthetic_fraud: number;
  n_legit: number;
  rules_only: ModeResult;
  model_only: ModeResult;
  rules_plus_model: ModeResult;
  added_fpr: number;
  budget_added_fpr_ok: boolean;
  budget_rules_only_ok: boolean;
}

export interface TraceItem {
  rule: string;
  name: string;
  value: number | null;
  threshold: number;
  fired: boolean;
  action: Action | null;
  condition: string;
}

export interface ReplayRow {
  TransactionID: number;
  TransactionDT: number;
  amount: number;
  synthetic: boolean;
  label: 0 | 1;
  source: string;
  risk: number;
  model_action: Action;
  rule_action: Action;
  rules_fired: string;
  combined_action: Action;
  decision: Action;
  trace: TraceItem[] | null;
}

export interface RunResult {
  label: string;
  scenario: string;
  title: string;
  seed: number;
  intensity: Intensity;
  generator?: Generator;
  model: string;
  days: number[];
  info: Record<string, unknown>;
  notes: string[];
  step_up_note: string;
  n_synthetic: number;
  n_legit_replay: number;
  n_legit_stored: number;
  n_legit_live: number;
  stored_score_max_diff: number;
  seconds: number;
  results: Record<string, SetResult>;
  replay: ReplayRow[];
}

export interface SuiteModeAgg {
  alert: { recall_count: string; recall_per_seed_min_max: string[]; fpr_mean: number; fpr_min_max: [number, number] };
  hard: { recall_count: string; recall_per_seed_min_max: string[]; fpr_mean: number; fpr_min_max: [number, number] };
  stepup_only_fpr_mean?: number;
  stepup_only_fpr_min_max?: [number, number];
  campaigns_alerted_before_cash_out?: string;
  build_up_alerts?: string;
}

export interface SuiteSetAgg {
  n_seeds: number;
  synthetic_fraud_rows: number;
  rules_only: SuiteModeAgg;
  model_only: SuiteModeAgg;
  rules_plus_model: SuiteModeAgg;
  added_fpr_max: number;
  budget_added_fpr_ok_seeds: string;
  budget_rules_only_ok_seeds: string;
}

export interface PosthocDay {
  day: number;
  legit_rows: number;
  windows_7d_customers_max: number | null;
  windows_7d_customers_median: number | null;
  windows_rows_at_or_above_r3: number;
  r3_threshold: number;
  [set: string]: unknown;
}

export interface RateGroup {
  rows: number;
  alert_rate: number | null;
  hold_block_rate: number | null;
}

export interface SuiteSummary {
  label: string;
  generator?: Generator;
  seeds: number[];
  scenarios: Record<string, Record<string, SuiteSetAgg | string | Record<string, SuiteSetAgg>>>;
}

export interface SuiteResponse {
  label: string;
  suite: SuiteSummary | null;
  suite_v1?: SuiteSummary | null;
  posthoc: { label: string; per_day: PosthocDay[]; new_accounts: Record<string, RateGroup | number[]> } | null;
}

export interface RunSpec {
  scenario: string;
  intensity: Intensity;
  seed: number;
  model: boolean;
  generator: Generator;
}

type Fetch = (input: string, init?: RequestInit) => Promise<Response>;

export function createAttackApi(baseUrl = "http://127.0.0.1:8000", fetchImpl: Fetch = fetch) {
  const base = baseUrl.replace(/\/$/, "");
  async function call<T>(path: string, init?: RequestInit): Promise<T> {
    let res: Response;
    try {
      res = await fetchImpl(base + path, init);
    } catch {
      throw new LabError(`The Attack Lab backend is not reachable at ${base}. Start it with VAULTIC_TESTLAB=1 (see the README).`, 0);
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
  return {
    scenarios: () => call<ScenarioList>("/api/attack/scenarios"),
    run: (spec: RunSpec) => call<RunResult>("/api/attack/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(spec) }),
    suite: () => call<SuiteResponse>("/api/attack/suite"),
  };
}

export type AttackApi = ReturnType<typeof createAttackApi>;

export const attackApi = createAttackApi(import.meta.env.VITE_API_URL || undefined);
