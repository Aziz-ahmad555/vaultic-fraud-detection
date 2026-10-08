/**
 * Mock scoring model. NOT the research model: a fixed, hand-set logistic model per view with
 * an MVAF-shaped gate, so every screen has coherent synthetic numbers to show.
 *
 * It keeps the real system's rules:
 *  - a view without its evidence is missing (null), never scored (rule 11);
 *  - gate weights are a masked softmax over the available views and sum to 1;
 *  - fused logit = sum_v w_v logit(p_v) + bias (MVAF);
 *  - reason points are an exact decomposition of the fused logit against a typical
 *    transaction, in PDO-20 points (D38), so they add up.
 */

import type {
  Action,
  ConformalSet,
  DecisionRecord,
  Reason,
  ScoreInput,
  ScoreResult,
  Thresholds,
  View,
} from "../types";
import { VIEWS } from "../types";

export const POINTS_PER_LOGODDS = 20 / Math.LN2;
const BIAS = -0.1;

export const sigmoid = (s: number) => 1 / (1 + Math.exp(-s));
export const logit = (p: number) => {
  const q = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return Math.log(q / (1 - q));
};

/** Features of the mock model, each with its baseline ("typical transaction") value. */
type Feature = "amount" | "ratio" | "device" | "velocity" | "shared" | "c13" | "history";

function featureValues(x: ScoreInput): Record<Feature, number> {
  const ratio = x.usualAmount && x.usualAmount > 0 ? x.amount / x.usualAmount : 1;
  return {
    amount: Math.log10(Math.max(x.amount, 1)),
    ratio: Math.log(Math.max(ratio, 0.05)),
    device: x.deviceKnown === false ? 1 : 0,
    velocity: x.velocity24h,
    shared: x.sharedEntities,
    c13: x.maskedC13,
    history: Math.log1p(x.historyCount),
  };
}
const BASELINE: Record<Feature, number> = {
  amount: Math.log10(80),
  ratio: 0,
  device: 0,
  velocity: 1,
  shared: 0,
  c13: 1,
  history: Math.log1p(10),
};

/** Linear coefficients of each view's logit on the features. */
const COEF: Record<View, { intercept: number } & Partial<Record<Feature, number>>> = {
  tabular: { intercept: -3.3, amount: 0.9, c13: 0.09, device: 0.5 },
  behavioral: { intercept: -3.1, ratio: 1.15, velocity: 0.2, device: 0.7, history: -0.15 },
  temporal: { intercept: -3.2, ratio: 0.95, velocity: 0.28 },
  graph: { intercept: -2.9, shared: 0.6 },
  anomaly: { intercept: -2.7, ratio: 0.55, c13: 0.05, amount: 0.3 },
};
const GATE_BASE: Record<View, number> = {
  tabular: 0.2,
  behavioral: 0.1,
  temporal: 0,
  graph: 0.15,
  anomaly: -0.6,
};

function viewLogit(v: View, f: Record<Feature, number>): number {
  const c = COEF[v];
  let s = c.intercept;
  for (const k of Object.keys(f) as Feature[]) s += (c[k] ?? 0) * f[k];
  return s;
}

/** Whether a view has the evidence it needs (on top of the caller's availability switch). */
export function viewAvailable(v: View, x: ScoreInput): boolean {
  if (!x.available[v]) return false;
  if (v === "behavioral" || v === "temporal") return x.historyCount > 0;
  return true;
}

export const MOCK_THRESHOLDS: Thresholds = {
  t_monitor: 4,
  t_step: 15,
  t_hold: 60,
  t_block: 250,
  t_dis: 0.3,
};

/** Mock conformal sets: fixed q-hat cut points on the fused probability. */
export function conformalSet(p: number): ConformalSet {
  if (p >= 0.62) return "fraud";
  if (p <= 0.2) return "legit";
  return "uncertain";
}

/** Port of vaultic.trust.decision.decide for one transaction (D43). */
export function decide(
  p: number,
  amount: number,
  set: ConformalSet,
  disagreement: number,
  th: Thresholds,
): Action {
  const el = p * amount;
  if (el >= th.t_block && set === "fraud") return "block";
  if (el >= th.t_hold || (th.t_dis !== null && disagreement >= th.t_dis)) return "hold";
  if (el >= th.t_step || set === "uncertain" || set === "empty") return "step_up";
  if (el >= th.t_monitor) return "monitor";
  return "allow";
}

function reasonText(feature: Feature, x: ScoreInput, points: number): { name: string; text: string } {
  const ratio = x.usualAmount ? x.amount / x.usualAmount : null;
  switch (feature) {
    case "amount":
      return { name: "TransactionAmt", text: `Amount $${x.amount.toFixed(2)}` };
    case "ratio":
      return ratio === null
        ? { name: "amt_ratio_median", text: "No earlier amounts for this customer" }
        : {
            name: "amt_ratio_median",
            text: `Amount is ${ratio.toFixed(1)}x this customer's usual (median) amount`,
          };
    case "device":
      return x.deviceKnown === null
        ? { name: "new_device", text: "No device on this transaction" }
        : x.deviceKnown
          ? { name: "new_device", text: "This customer has used this device before" }
          : { name: "new_device", text: "First time this customer uses this device" };
    case "velocity":
      return {
        name: "vel_n_24h",
        text: `${x.velocity24h} earlier transactions by this customer in the last 24 hours`,
      };
    case "shared":
      return {
        name: "g_shared_uids_device",
        text: `${x.sharedEntities} other customers share this card or device`,
      };
    case "c13":
      return {
        name: "C13",
        text:
          points > 0
            ? "Unusual value in masked counter C13"
            : "Value in masked counter C13 lowers the risk score",
      };
    case "history":
      return x.historyCount === 0
        ? { name: "hist_n_past", text: "No earlier transactions for this customer (cold start)" }
        : { name: "hist_n_past", text: `Customer has ${x.historyCount} earlier transactions` };
  }
}

export function score(x: ScoreInput, th: Thresholds = MOCK_THRESHOLDS): ScoreResult {
  const f = featureValues(x);
  const views = {} as Record<View, number | null>;
  for (const v of VIEWS) views[v] = viewAvailable(v, x) ? sigmoid(viewLogit(v, f)) : null;

  // masked softmax gate over available views (confidence raises a view's weight)
  const g = {} as Record<View, number>;
  let total = 0;
  for (const v of VIEWS) {
    const p = views[v];
    g[v] = p === null ? 0 : Math.exp(GATE_BASE[v] + 1.2 * Math.abs(2 * p - 1));
    total += g[v];
  }
  const weights = {} as Record<View, number>;
  for (const v of VIEWS) weights[v] = total > 0 ? g[v] / total : 0;

  let fused = BIAS;
  for (const v of VIEWS) if (views[v] !== null) fused += weights[v] * logit(views[v] as number);
  const p = sigmoid(fused);

  // exact decomposition: contribution of feature k = sum_v w_v * coef_vk * (f_k - baseline_k)
  const reasons: Reason[] = [];
  for (const k of Object.keys(f) as Feature[]) {
    let contrib = 0;
    let bestView: View = "tabular";
    let best = 0;
    for (const v of VIEWS) {
      if (views[v] === null) continue;
      const part = weights[v] * (COEF[v][k] ?? 0) * (f[k] - BASELINE[k]);
      contrib += part;
      if (Math.abs(part) > Math.abs(best)) {
        best = part;
        bestView = v;
      }
    }
    if (Math.abs(contrib) < 1e-9) continue;
    const points = contrib * POINTS_PER_LOGODDS;
    const { name, text } = reasonText(k, x, points);
    reasons.push({ feature: name, text, points, view: bestView });
  }
  reasons.sort((a, b) => Math.abs(b.points) - Math.abs(a.points));

  const available = VIEWS.map((v) => views[v]).filter((q): q is number => q !== null);
  const mean = available.reduce((a, b) => a + b, 0) / Math.max(available.length, 1);
  const disagreement =
    available.length >= 2
      ? Math.sqrt(available.reduce((a, b) => a + (b - mean) ** 2, 0) / available.length)
      : 0;
  const set = conformalSet(p);
  const decision: DecisionRecord = {
    action: decide(p, x.amount, set, disagreement, th),
    p,
    expected_loss: p * x.amount,
    conformal_set: set,
    disagreement,
  };
  return { p, views, weights, reasons, decision };
}
