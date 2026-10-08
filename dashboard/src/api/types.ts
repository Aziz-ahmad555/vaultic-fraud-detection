/**
 * Typed API contract of the Evidence Room. The mock (api/mock) and the future FastAPI
 * client implement the same `Api` interface; switching is a one-line change in api/index.ts.
 *
 * Row schemas mirror the Python side exactly:
 *   ViewPredictionRow  <- src/vaultic/views/orchestrate.py view_table (decision D42)
 *   DecisionRecord     <- src/vaultic/trust/decision.py decide / CostModel (D43, D45, D46)
 *   Reason             <- src/vaultic/explain/reason_codes.py Reason (D38)
 */

export const VIEWS = ["tabular", "behavioral", "temporal", "graph", "anomaly"] as const;
export type View = (typeof VIEWS)[number];

/** Display names: the roadmap calls the graph view "relational". */
export const VIEW_LABEL: Record<View, string> = {
  tabular: "Tabular",
  behavioral: "Behavioral",
  temporal: "Temporal",
  graph: "Relational",
  anomaly: "Anomaly",
};

/** One row of the out-of-sample view-prediction table (D42). NaN on the Python side is null here. */
export interface ViewPredictionRow {
  TransactionID: number;
  TransactionDT: number; // seconds, relative dataset clock
  day: number;
  fold: string;
  role: "gate_train" | "test";
  split: "train" | "validation" | "test" | "unused";
  /** null when the label is not known at the as-of time (label-delay curtain) */
  label: 0 | 1 | null;
  p_tabular: number | null;
  p_behavioral: number | null;
  p_temporal: number | null;
  p_graph: number | null;
  p_anomaly: number | null;
  m_tabular: 0 | 1;
  m_behavioral: 0 | 1;
  m_temporal: 0 | 1;
  m_graph: 0 | 1;
  m_anomaly: 0 | 1;
  c_tabular: number;
  c_behavioral: number;
  c_temporal: number;
  c_graph: number;
  c_anomaly: number;
  disagreement: number;
  ctx_log_amount: number;
  ctx_hist_n_past: number;
  ctx_has_identity: 0 | 1;
  ctx_product: number;
  ctx_hour: number;
}

export const ACTIONS = ["allow", "monitor", "step_up", "hold", "block"] as const;
export type Action = (typeof ACTIONS)[number];
export const ACTION_LABEL: Record<Action, string> = {
  allow: "Allow",
  monitor: "Monitor",
  step_up: "Step-up",
  hold: "Hold",
  block: "Block",
};

export type ConformalSet = "legit" | "fraud" | "uncertain" | "empty";

export interface CostModel {
  c_fp: number;
  c_rev: number;
  c_step: number;
  step_up_success_rate: number;
  legit_step_up_friction: number;
}

export interface Thresholds {
  t_monitor: number;
  t_step: number;
  t_hold: number;
  t_block: number;
  t_dis: number | null; // null = disagreement rule off (inf on the Python side)
}

/** Decision engine output for one transaction (D43). */
export interface DecisionRecord {
  action: Action;
  p: number; // fused, calibrated fraud probability
  expected_loss: number; // p x amount
  conformal_set: ConformalSet;
  disagreement: number;
}

export interface Reason {
  feature: string;
  text: string;
  points: number; // signed; + raises risk (PDO 20 points, D38)
  view: View;
}

export interface Entities {
  uid: string;
  card: string; // masked card fields combined
  email: string | null;
  address: string | null;
  device: string | null;
}

/** Everything the console shows about one transaction. */
export interface Transaction {
  row: ViewPredictionRow;
  amount: number;
  product: string;
  entities: Entities;
  /** MVAF gate weights; 0 for missing views; available weights sum to 1 */
  weights: Record<View, number>;
  decision: DecisionRecord;
  reasons: Reason[];
  /** inputs of the scoring function, for counterfactuals and the simulator */
  inputs: ScoreInput;
}

/** Inputs of the scoring model (mock: a fixed logistic model; later: POST /score). */
export interface ScoreInput {
  amount: number;
  usualAmount: number | null; // customer's median past amount; null = no history
  deviceKnown: boolean | null; // null = no device on the transaction
  velocity24h: number;
  sharedEntities: number; // other customers sharing card/device
  historyCount: number;
  maskedC13: number;
  available: Record<View, boolean>;
}

export interface ScoreResult {
  p: number;
  views: Record<View, number | null>;
  weights: Record<View, number>;
  reasons: Reason[];
  decision: DecisionRecord;
}

export interface AsOf {
  time: number; // seconds on the dataset clock
  labelDelayDays: number;
}

export interface DaySummary {
  day: number;
  transactions: number;
  alerts: number; // step-up + hold + block
  queue: number; // hold + step-up waiting
  byAction: Record<Action, number>;
  expectedFraudStopped: number; // sum p x amount over held/blocked/stepped-up
  expectedLoss: number; // sum p x amount over allowed/monitored
  cost: { missedFraud: number; falsePositives: number; reviews: number; stepUps: number };
  riskByHour: { hour: number; count: number; meanRisk: number }[];
  viewAvailability: Record<View, number>; // share of transactions with the view
}

export interface NetworkNode {
  id: string;
  type: "uid" | "card" | "email" | "address" | "device";
  label: string;
  knownFraud: number; // frauds known at as-of (matured labels)
  transactions: number;
  ring: number | null; // ring id when part of a highlighted ring
}
export interface NetworkEdge {
  id: string;
  source: string;
  target: string;
  time: number; // first time the link was seen
}
export interface Ring {
  id: number;
  customers: number;
  knownFrauds: number;
}
export interface Network {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  rings: Ring[];
}

export const POLICIES = ["R1", "R2", "R3", "R4"] as const;
export type Policy = (typeof POLICIES)[number];
export const POLICY_LABEL: Record<Policy, string> = {
  R1: "Top-K by risk",
  R2: "Top-K by expected loss",
  R3: "Expected loss + uncertainty",
  R4: "Expected loss + uncertainty + disagreement",
};
export interface PolicyQueue {
  policy: Policy;
  day: number;
  cases: Transaction[];
  /** outcome on the latest day whose labels are known (as-of minus L) */
  matured: { day: number | null; fraudValueCaught: number; fraudCaught: number; reviews: number };
}

export interface DriftSeriesPoint {
  day: number;
  value: number;
}
export interface DriftAlarm {
  day: number;
  detector: "ADWIN" | "Page-Hinkley" | "PSI" | "Disagreement";
}
export interface RetrainEvent {
  day: number;
  trigger: "schedule" | "alarm";
  prChampion: number;
  prChallenger: number;
  gainCiLow: number;
  gainCiHigh: number;
  costChampion: number;
  costChallenger: number;
  accepted: boolean;
}
export interface DriftReport {
  featurePsi: { feature: string; psi: number }[];
  scorePsi: DriftSeriesPoint[];
  disagreement: DriftSeriesPoint[];
  alarms: DriftAlarm[];
  injected: { day: number; kind: string }[];
  retrains: RetrainEvent[];
}

export interface Run {
  id: string;
  name: string;
  kind: "baseline" | "fusion" | "ablation";
  prAuc: number;
  ciLow: number;
  ciHigh: number;
  recallAt1: number;
  cost: number;
  pr: { recall: number; precision: number }[];
  calibration: { predicted: number; observed: number; count: number }[];
}
export interface CostCell {
  c_fp: number;
  step_up_success_rate: number;
  cost: number;
}
export interface Lab {
  runs: Run[];
  ablation: { step: string; runId: string }[];
  sensitivity: CostCell[];
}

export type Role = "analyst" | "admin";
export interface AuditEntry {
  time: string; // wall-clock ISO time of the action
  actor: Role;
  action: string;
  detail: string;
}
export type Feedback = "fraud" | "legit" | "unsure";

export interface Api {
  /** dataset clock range covered by the data */
  range(): Promise<{ first: number; last: number }>;
  daySummary(asOf: AsOf): Promise<DaySummary>;
  /** transactions with TransactionDT in (from, to], never after as-of */
  stream(asOf: AsOf, from: number, to: number): Promise<Transaction[]>;
  transaction(id: number, asOf: AsOf): Promise<Transaction | null>;
  /** today's alerts (step-up, hold, block), highest risk first */
  alerts(asOf: AsOf): Promise<Transaction[]>;
  customerHistory(uid: string, asOf: AsOf): Promise<Transaction[]>;
  network(asOf: AsOf, focus?: string): Promise<Network>;
  queues(asOf: AsOf, k: number): Promise<PolicyQueue[]>;
  drift(asOf: AsOf): Promise<DriftReport>;
  lab(): Promise<Lab>;
  score(input: ScoreInput): Promise<ScoreResult>;
  thresholds(): Promise<{ thresholds: Thresholds; cost: CostModel }>;
  saveThresholds(t: Thresholds, role: Role): Promise<void>;
  feedback(id: number, label: Feedback, role: Role): Promise<void>;
  auditLog(): Promise<AuditEntry[]>;
}
