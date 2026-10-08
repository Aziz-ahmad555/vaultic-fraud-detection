/**
 * Mock implementation of the Api interface over the synthetic dataset (generate.ts).
 *
 * The as-of rules live here, in the data layer, so no page can get them wrong:
 *  - nothing after asOf.time is ever returned;
 *  - a label is returned only when its_time + L days <= asOf.time (label-delay curtain);
 *  - known-fraud counts in the network use matured labels only.
 */

import type {
  Action,
  Api,
  AsOf,
  AuditEntry,
  CostModel,
  DaySummary,
  DriftReport,
  Feedback,
  Lab,
  Network,
  NetworkEdge,
  NetworkNode,
  PolicyQueue,
  Ring,
  Role,
  Run,
  ScoreInput,
  Thresholds,
  Transaction,
  View,
} from "../types";
import { ACTIONS, POLICIES, VIEWS } from "../types";
import { DAY, FIRST_DAY, LAST_DAY, generate, type Dataset } from "./generate";
import { MOCK_THRESHOLDS, decide, score } from "./model";

const COST: CostModel = {
  c_fp: 10,
  c_rev: 5,
  c_step: 0.5,
  step_up_success_rate: 0.9,
  legit_step_up_friction: 1,
};
const LAMBDA_U = 0.2;
const LAMBDA_D = 0.5;
const STORE_KEY = "evidence-room-mock-v1";

interface Stored {
  thresholds: Thresholds;
  feedback: Record<number, Feedback>;
  audit: AuditEntry[];
}

function load(): Stored {
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    if (raw) return JSON.parse(raw) as Stored;
  } catch {
    /* storage unavailable: start fresh */
  }
  return { thresholds: { ...MOCK_THRESHOLDS }, feedback: {}, audit: [] };
}

function save(s: Stored) {
  try {
    window.localStorage.setItem(STORE_KEY, JSON.stringify(s));
  } catch {
    /* storage unavailable: keep in memory only */
  }
}

const matured = (t: number, asOf: AsOf) => t + asOf.labelDelayDays * DAY <= asOf.time;

export function createMockApi(seed = 7): Api {
  let data: Dataset | null = null;
  const ds = () => (data ??= generate(seed));
  const stored = load();

  const withLabel = (t: Transaction, asOf: AsOf): Transaction => {
    const known = matured(t.row.TransactionDT, asOf);
    const label = known ? (ds().labels.get(t.row.TransactionID) ?? null) : null;
    const decision = { ...t.decision };
    decision.action = decide(
      decision.p,
      t.amount,
      decision.conformal_set,
      decision.disagreement,
      stored.thresholds,
    );
    return { ...t, row: { ...t.row, label }, decision };
  };

  const upTo = (asOf: AsOf) => {
    const all = ds().transactions;
    let hi = all.length;
    let lo = 0;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (all[mid].row.TransactionDT <= asOf.time) lo = mid + 1;
      else hi = mid;
    }
    return all.slice(0, lo);
  };
  const dayOf = (asOf: AsOf) => Math.floor(asOf.time / DAY);
  const scored = (t: Transaction) => t.row.role !== "history";
  const dayRows = (asOf: AsOf, day = dayOf(asOf)) =>
    upTo(asOf)
      .filter((t) => t.row.day === day && scored(t))
      .map((t) => withLabel(t, asOf));

  const audit = (actor: Role, action: string, detail: string) => {
    stored.audit.unshift({ time: new Date().toISOString(), actor, action, detail });
    stored.audit = stored.audit.slice(0, 500);
    save(stored);
  };

  const priority = (policy: string, t: Transaction) => {
    const { p, expected_loss: el, conformal_set: set, disagreement: d } = t.decision;
    const u = set === "uncertain" || set === "empty" ? 1 : 1 - Math.abs(2 * p - 1);
    if (policy === "R1") return p;
    if (policy === "R2") return el;
    if (policy === "R3") return el + LAMBDA_U * u * t.amount;
    return el + LAMBDA_U * u * t.amount + LAMBDA_D * d * t.amount;
  };

  return {
    async range() {
      return { first: FIRST_DAY * DAY, last: (LAST_DAY + 1) * DAY - 1 };
    },

    async daySummary(asOf) {
      const rows = dayRows(asOf);
      const byAction = Object.fromEntries(ACTIONS.map((a) => [a, 0])) as Record<Action, number>;
      let stopped = 0;
      let loss = 0;
      const cost = { missedFraud: 0, falsePositives: 0, reviews: 0, stepUps: 0 };
      const hours = Array.from({ length: 24 }, (_, hour) => ({ hour, count: 0, risk: 0 }));
      const avail = Object.fromEntries(VIEWS.map((v) => [v, 0])) as Record<View, number>;
      for (const t of rows) {
        const { action, p } = t.decision;
        const el = p * t.amount;
        byAction[action] += 1;
        if (action === "allow" || action === "monitor") {
          loss += el;
          cost.missedFraud += el;
        } else {
          stopped += action === "step_up" ? COST.step_up_success_rate * el : el;
        }
        if (action === "step_up") {
          cost.missedFraud += (1 - COST.step_up_success_rate) * el;
          cost.stepUps += COST.c_step + (1 - p) * COST.legit_step_up_friction;
        }
        if (action === "hold" || action === "block") cost.reviews += COST.c_rev;
        if (action === "block") cost.falsePositives += (1 - p) * COST.c_fp;
        const h = hours[t.row.ctx_hour];
        h.count += 1;
        h.risk += p;
        for (const v of VIEWS) if (t.row[`m_${v}`] === 1) avail[v] += 1;
      }
      for (const v of VIEWS) avail[v] = rows.length ? avail[v] / rows.length : 0;
      const summary: DaySummary = {
        day: dayOf(asOf),
        transactions: rows.length,
        alerts: byAction.step_up + byAction.hold + byAction.block,
        queue: byAction.step_up + byAction.hold,
        byAction,
        expectedFraudStopped: stopped,
        expectedLoss: loss,
        cost,
        riskByHour: hours.map((h) => ({
          hour: h.hour,
          count: h.count,
          meanRisk: h.count ? h.risk / h.count : 0,
        })),
        viewAvailability: avail,
      };
      return summary;
    },

    async stream(asOf, from, to) {
      const end = Math.min(to, asOf.time);
      return upTo({ ...asOf, time: end })
        .filter((t) => t.row.TransactionDT > from && scored(t))
        .map((t) => withLabel(t, { ...asOf, time: end }));
    },

    async transaction(id, asOf) {
      const t = ds().byId.get(id);
      if (!t || t.row.TransactionDT > asOf.time || !scored(t)) return null;
      return withLabel(t, asOf);
    },

    async alerts(asOf) {
      return dayRows(asOf)
        .filter((t) => ["step_up", "hold", "block"].includes(t.decision.action))
        .sort((a, b) => b.decision.p - a.decision.p);
    },

    async customerHistory(uid, asOf) {
      return upTo(asOf)
        .filter((t) => t.entities.uid === uid)
        .map((t) => withLabel(t, asOf));
    },

    async network(asOf, focus) {
      return buildNetwork(upTo(asOf), asOf, ds(), focus);
    },

    async queues(asOf, k) {
      const today = dayRows(asOf);
      const lastKnownDay = Math.floor((asOf.time - asOf.labelDelayDays * DAY + 1) / DAY) - 1;
      const maturedDay = lastKnownDay >= FIRST_DAY ? lastKnownDay : null;
      const past =
        maturedDay === null
          ? []
          : ds()
              .transactions.filter((t) => t.row.day === maturedDay && scored(t))
              .map((t) => withLabel(t, asOf));
      return POLICIES.map((policy): PolicyQueue => {
        const rank = (rows: Transaction[]) =>
          [...rows].sort((a, b) => priority(policy, b) - priority(policy, a)).slice(0, k);
        const reviewed = rank(past);
        const caught = reviewed.filter((t) => t.row.label === 1);
        return {
          policy,
          day: dayOf(asOf),
          cases: rank(today),
          matured: {
            day: maturedDay,
            fraudValueCaught: caught.reduce((s, t) => s + t.amount, 0),
            fraudCaught: caught.length,
            reviews: reviewed.length,
          },
        };
      });
    },

    async drift(asOf) {
      return mockDrift(asOf);
    },

    async lab() {
      return mockLab();
    },

    async score(input: ScoreInput) {
      return score(input, stored.thresholds);
    },

    async thresholds() {
      return { thresholds: { ...stored.thresholds }, cost: { ...COST } };
    },

    async saveThresholds(t, role) {
      if (role !== "admin") {
        throw new Error("Only admins can change thresholds. Switch the role to admin in Settings.");
      }
      if (!(t.t_monitor <= t.t_step && t.t_step <= t.t_hold && t.t_hold <= t.t_block)) {
        throw new Error("Thresholds must rise from Monitor to Block. Fix the order and save again.");
      }
      const before = JSON.stringify(stored.thresholds);
      stored.thresholds = { ...t };
      audit(role, "Changed decision thresholds", `${before} to ${JSON.stringify(t)}`);
      save(stored);
    },

    async feedback(id, label, role) {
      const before = stored.feedback[id];
      stored.feedback[id] = label;
      audit(role, `Marked transaction ${id} as ${label}`, before ? `was ${before}` : "first label");
      save(stored);
    },

    async auditLog() {
      return [...stored.audit];
    },
  };
}

// ---- network ------------------------------------------------------------------------------

function buildNetwork(rows: Transaction[], asOf: AsOf, data: Dataset, focus?: string): Network {
  const nodes = new Map<string, NetworkNode>();
  const edges = new Map<string, NetworkEdge>();
  const node = (id: string, type: NetworkNode["type"], label: string) => {
    let n = nodes.get(id);
    if (!n) {
      n = { id, type, label, knownFraud: 0, transactions: 0, ring: null };
      nodes.set(id, n);
    }
    return n;
  };
  for (const t of rows) {
    const known = matured(t.row.TransactionDT, asOf) && data.labels.get(t.row.TransactionID) === 1;
    const u = node(`uid:${t.entities.uid}`, "uid", t.entities.uid);
    const linked: [string, NetworkNode["type"], string | null][] = [
      [`card:${t.entities.card}`, "card", t.entities.card],
      [`device:${t.entities.device}`, "device", t.entities.device],
      [`email:${t.entities.email}`, "email", t.entities.email],
      [`address:${t.entities.address}`, "address", t.entities.address],
    ];
    u.transactions += 1;
    if (known) u.knownFraud += 1;
    for (const [id, type, label] of linked) {
      if (!label) continue;
      const n = node(id, type, label);
      n.transactions += 1;
      if (known) n.knownFraud += 1;
      const eid = `${u.id}|${id}`;
      if (!edges.has(eid)) edges.set(eid, { id: eid, source: u.id, target: id, time: t.row.TransactionDT });
    }
  }

  // rings: components over uid-card-device links (email domains and regions are hubs)
  const parent = new Map<string, string>();
  const find = (x: string): string => {
    let p = parent.get(x) ?? x;
    if (p !== x) {
      p = find(p);
      parent.set(x, p);
    }
    return p;
  };
  for (const e of edges.values()) {
    if (e.target.startsWith("card:") || e.target.startsWith("device:")) {
      parent.set(find(e.source), find(e.target));
    }
  }
  const comps = new Map<string, { uids: Set<string>; frauds: number }>();
  for (const n of nodes.values()) {
    if (n.type !== "uid") continue;
    const root = find(n.id);
    const c = comps.get(root) ?? { uids: new Set<string>(), frauds: 0 };
    c.uids.add(n.id);
    c.frauds += n.knownFraud;
    comps.set(root, c);
  }
  const ringRoots = [...comps.entries()]
    .filter(([, c]) => c.uids.size >= 2 && c.frauds >= 2)
    .sort((a, b) => b[1].frauds - a[1].frauds)
    .slice(0, 8);
  const rings: Ring[] = ringRoots.map(([, c], i) => ({
    id: i + 1,
    customers: c.uids.size,
    knownFrauds: c.frauds,
  }));
  const ringOfRoot = new Map(ringRoots.map(([root], i) => [root, i + 1]));
  for (const n of nodes.values()) {
    if (n.type === "uid" || n.type === "card" || n.type === "device") {
      n.ring = ringOfRoot.get(find(n.id)) ?? null;
    }
  }

  // which part to show
  let keep: Set<string>;
  if (focus && nodes.has(focus)) {
    keep = new Set([focus]);
    const hubs = (id: string) => id.startsWith("email:") || id.startsWith("address:");
    let frontier = [focus];
    for (let hop = 0; hop < 2; hop++) {
      const next: string[] = [];
      for (const e of edges.values()) {
        for (const [a, b] of [
          [e.source, e.target],
          [e.target, e.source],
        ]) {
          if (frontier.includes(a) && !keep.has(b) && !(hubs(a) && a !== focus)) {
            keep.add(b);
            next.push(b);
          }
        }
      }
      frontier = next;
      if (keep.size > 160) break;
    }
  } else {
    keep = new Set([...nodes.values()].filter((n) => n.ring !== null).map((n) => n.id));
  }
  return {
    nodes: [...nodes.values()].filter((n) => keep.has(n.id)),
    edges: [...edges.values()].filter((e) => keep.has(e.source) && keep.has(e.target)),
    rings,
  };
}

// ---- drift (mock series with one injected drift) -------------------------------------------

const INJECTED_DAY = 139;
const FEATURES = [
  "TransactionAmt",
  "amt_ratio_median",
  "vel_n_24h",
  "new_device",
  "g_shared_uids_device",
  "C13",
  "D1",
  "V258",
  "hist_n_past",
  "card4",
  "P_emaildomain",
  "dist1",
];

function mockDrift(asOf: AsOf): DriftReport {
  const today = Math.floor(asOf.time / DAY);
  const days: number[] = [];
  for (let d = FIRST_DAY; d <= Math.min(today, LAST_DAY); d++) days.push(d);
  const after = (d: number) => Math.max(0, d - INJECTED_DAY + 1);
  const wobble = (d: number, k: number) => 0.012 * Math.sin(d * 1.7 + k * 2.3);
  const shifted = today >= INJECTED_DAY;
  const featurePsi = FEATURES.map((feature, k) => ({
    feature,
    psi:
      0.03 +
      Math.abs(wobble(today, k)) +
      (shifted && (feature === "TransactionAmt" || feature === "amt_ratio_median")
        ? Math.min(0.35, 0.06 * after(today))
        : 0),
  })).sort((a, b) => b.psi - a.psi);
  const scorePsi = days.map((d) => ({ day: d, value: 0.04 + Math.abs(wobble(d, 1)) + Math.min(0.3, 0.05 * after(d)) }));
  const disagreement = days.map((d) => ({ day: d, value: 0.09 + wobble(d, 2) / 2 + Math.min(0.05, 0.008 * after(d)) }));
  const alarms: DriftReport["alarms"] = [];
  const psiDay = INJECTED_DAY + 3;
  if (today >= psiDay) alarms.push({ day: psiDay, detector: "PSI" });
  if (today >= INJECTED_DAY + 5) alarms.push({ day: INJECTED_DAY + 5, detector: "Disagreement" });
  const labelDay = INJECTED_DAY + asOf.labelDelayDays + 4; // errors only arrive L days later
  if (today >= labelDay) alarms.push({ day: labelDay, detector: "ADWIN" });
  if (today >= labelDay + 2) alarms.push({ day: labelDay + 2, detector: "Page-Hinkley" });
  const retrains: DriftReport["retrains"] = [];
  if (today >= psiDay + 1)
    retrains.push({
      day: psiDay + 1,
      trigger: "alarm",
      prChampion: 0.61,
      prChallenger: 0.6,
      gainCiLow: -0.03,
      gainCiHigh: 0.02,
      costChampion: 5120,
      costChallenger: 5210,
      accepted: false,
    });
  if (today >= psiDay + 8)
    retrains.push({
      day: psiDay + 8,
      trigger: "alarm",
      prChampion: 0.55,
      prChallenger: 0.63,
      gainCiLow: 0.02,
      gainCiHigh: 0.13,
      costChampion: 5480,
      costChallenger: 4870,
      accepted: true,
    });
  return {
    featurePsi,
    scorePsi,
    disagreement,
    alarms,
    injected: today >= INJECTED_DAY ? [{ day: INJECTED_DAY, kind: "Amount shift in ProductCD W" }] : [],
    retrains,
  };
}

// ---- model lab (mock runs) --------------------------------------------------------------------

function normCdf(x: number) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp((-x * x) / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - p : p;
}
function normInv(p: number) {
  let lo = -8;
  let hi = 8;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (normCdf(mid) < p) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

function mockRun(id: string, name: string, kind: Run["kind"], dprime: number, cost: number, cal = 1): Run {
  const prevalence = 0.035;
  const pr: Run["pr"] = [];
  for (let i = 1; i <= 50; i++) {
    const recall = i / 50;
    const t = normInv(1 - recall) + dprime; // positives ~ N(d', 1), negatives ~ N(0, 1)
    const fpr = 1 - normCdf(t);
    const precision = (prevalence * recall) / (prevalence * recall + (1 - prevalence) * fpr);
    pr.push({ recall, precision });
  }
  let auc = 0;
  for (let i = 0; i < pr.length; i++) auc += pr[i].precision * (1 / 50);
  const calibration = Array.from({ length: 10 }, (_, i) => {
    const predicted = (i + 0.5) / 10;
    return { predicted, observed: Math.min(1, predicted ** cal), count: Math.round(4000 / (i + 1) ** 1.6) };
  });
  const recallAt1 = 1 - normCdf(normInv(0.99) - dprime);
  return { id, name, kind, prAuc: auc, ciLow: auc - 0.021, ciHigh: auc + 0.019, recallAt1, cost, pr, calibration };
}

function mockLab(): Lab {
  const runs: Run[] = [
    mockRun("B1", "B1 Logistic regression", "baseline", 1.7, 9100, 1.4),
    mockRun("B2", "B2 Random forest", "baseline", 2.0, 8400, 1.2),
    mockRun("B3", "B3 XGBoost raw", "baseline", 2.3, 7600, 1.1),
    mockRun("B4", "B4 LightGBM raw", "baseline", 2.3, 7550, 1.1),
    mockRun("B5", "B5 XGBoost engineered", "baseline", 2.5, 7020, 1.05),
    mockRun("B6", "B6 FYP-1 as built", "baseline", 1.9, 8800, 1.5),
    mockRun("A1", "B5 + behavioral", "ablation", 2.6, 6820, 1.05),
    mockRun("A2", "+ temporal (F4 stacking)", "ablation", 2.65, 6700, 1.05),
    mockRun("A3", "+ relational (F4 stacking)", "ablation", 2.75, 6480, 1.04),
    mockRun("A4", "+ anomaly (F4 stacking)", "ablation", 2.78, 6420, 1.04),
    ...(["F1", "F2", "F3", "F4", "F5", "F6", "F7"] as const).map((f, i) =>
      mockRun(f, `${f} ${["Average", "Fixed weights", "LR stacking", "LightGBM stacking", "Gate, no mask", "Gate, no dropout", "SimMLM-style"][i]}`, "fusion", 2.55 + 0.03 * i, 6900 - 60 * i, 1.03),
    ),
    mockRun("MVAF", "MVAF (full)", "fusion", 2.85, 6250, 1.02),
  ];
  const ablation = [
    { step: "B5 strong tabular", runId: "B5" },
    { step: "+ behavioral", runId: "A1" },
    { step: "+ temporal", runId: "A2" },
    { step: "+ relational", runId: "A3" },
    { step: "+ anomaly", runId: "A4" },
    { step: "MVAF (full)", runId: "MVAF" },
  ];
  const sensitivity = [];
  for (const c_fp of [2, 5, 10, 20, 50])
    for (const rate of [0.7, 0.9, 1.0])
      sensitivity.push({ c_fp, step_up_success_rate: rate, cost: Math.round(5200 + 38 * c_fp + 2400 * (1 - rate)) });
  return { runs, ablation, sensitivity };
}
