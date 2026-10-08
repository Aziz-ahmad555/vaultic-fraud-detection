/**
 * Synthetic mock dataset: customers, a few fraud rings sharing cards and devices, and about
 * 300 transactions a day over the validation period (days 128-150). All inputs of the mock
 * model are computed point-in-time from each customer's own past, like the real features.
 * Nothing here comes from IEEE-CIS data.
 */

import type { Entities, ScoreInput, Transaction, View, ViewPredictionRow } from "../types";
import { VIEWS } from "../types";
import { score } from "./model";
import { rng } from "./rng";

export const DAY = 86_400;
export const FIRST_DAY = 128;
export const LAST_DAY = 150;
const PER_DAY = 300;
const PRODUCTS = ["W", "C", "H", "R", "S"] as const;
const DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", "aol.com"];

interface Customer {
  uid: string;
  card: string;
  email: string | null;
  address: string | null;
  devices: string[];
  base: number;
  fraudProne: number;
  ring: number | null;
}

export interface Dataset {
  transactions: Transaction[]; // sorted by time
  byId: Map<number, Transaction>;
  labels: Map<number, 0 | 1>; // ground truth, released only after the label delay
  ringOf: Map<string, number>; // uid -> generating ring (ground truth, never shown)
}

function makeCustomers(r: ReturnType<typeof rng>): Customer[] {
  const customers: Customer[] = [];
  for (let i = 0; i < 3500; i++) {
    customers.push({
      uid: `u${(10_000 + i).toString(36)}`,
      card: `card ${1000 + r.int(0, 9000)}-${r.pick(["visa", "mastercard", "discover"])}`,
      email: r.chance(0.85) ? r.pick(DOMAINS) : null,
      address: r.chance(0.8) ? `region ${r.int(100, 540)}` : null,
      devices: r.chance(0.7) ? [`device ${r.int(1, 99999)}`] : [],
      base: Math.exp(3.2 + 0.9 * r.normal()),
      fraudProne: r.chance(0.04) ? 0.25 : 0.01,
      ring: null,
    });
  }
  // rings: 3-6 customers sharing one card and one device, high fraud propensity
  for (let ring = 0; ring < 7; ring++) {
    const size = r.int(3, 6);
    const card = `card ${1000 + r.int(0, 9000)}-visa`;
    const device = `device R${ring}-${r.int(100, 999)}`;
    for (let k = 0; k < size; k++) {
      const c = customers[r.int(0, customers.length - 1)];
      c.ring = ring;
      c.card = card;
      c.devices = [device, ...c.devices];
      c.fraudProne = 0.45;
    }
  }
  return customers;
}

export function generate(seed = 7): Dataset {
  const r = rng(seed);
  const customers = makeCustomers(r);
  const weightsCum: number[] = [];
  let acc = 0;
  for (let i = 0; i < customers.length; i++) {
    acc += 0.4 + r.next();
    weightsCum.push(acc);
  }
  const pickCustomer = () => {
    const x = r.next() * acc;
    let lo = 0;
    let hi = weightsCum.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (weightsCum[mid] < x) lo = mid + 1;
      else hi = mid;
    }
    return customers[lo];
  };

  // raw events first (time, customer, amount, device, fraud), then point-in-time inputs
  type Raw = { t: number; c: Customer; amount: number; device: string | null; fraud: 0 | 1; product: string };
  const raws: Raw[] = [];
  for (let day = FIRST_DAY - 10; day <= LAST_DAY; day++) {
    const n = day < FIRST_DAY ? 150 : PER_DAY + r.int(-30, 30);
    for (let i = 0; i < n; i++) {
      const c = pickCustomer();
      const fraud: 0 | 1 = r.chance(c.fraudProne * (day >= 139 && c.ring !== null ? 1.3 : 1)) ? 1 : 0;
      const newDevice = fraud ? r.chance(0.55) : r.chance(0.05);
      let device: string | null = c.devices.length ? r.pick(c.devices) : null;
      if (newDevice) device = `device ${r.int(1, 99999)}`;
      const amount = Math.max(1, c.base * Math.exp(0.35 * r.normal()) * (fraud ? 3 + 7 * r.next() : 1));
      raws.push({ t: day * DAY + r.int(0, DAY - 1), c, amount, device, fraud, product: r.pick(PRODUCTS) });
    }
  }
  raws.sort((a, b) => a.t - b.t);

  const pastAmounts = new Map<string, number[]>();
  const pastTimes = new Map<string, number[]>();
  const seenDevices = new Map<string, Set<string>>();
  const entityUsers = new Map<string, Set<string>>(); // card/device -> uids seen before
  const transactions: Transaction[] = [];
  const labels = new Map<number, 0 | 1>();
  const ringOf = new Map<string, number>();
  let id = 3_100_000;

  for (const e of raws) {
    const uid = e.c.uid;
    if (e.c.ring !== null) ringOf.set(uid, e.c.ring);
    const amounts = pastAmounts.get(uid) ?? [];
    const times = pastTimes.get(uid) ?? [];
    const sorted = [...amounts].sort((a, b) => a - b);
    const usual = sorted.length ? sorted[Math.floor(sorted.length / 2)] : null;
    const velocity = times.filter((t) => t > e.t - DAY && t < e.t).length;
    const devices = seenDevices.get(uid) ?? new Set<string>();
    const others = new Set<string>();
    for (const key of [e.c.card, e.device]) {
      if (!key) continue;
      for (const u of entityUsers.get(key) ?? []) if (u !== uid) others.add(u);
    }
    // relational evidence (D42): an entity of this transaction was seen before; a card seen
    // only by its own customer is not a link, so it needs another customer or a known device
    const entitySeen = others.size > 0 || (e.device !== null && (entityUsers.get(e.device)?.size ?? 0) > 0);
    const input: ScoreInput = {
      amount: e.amount,
      usualAmount: usual,
      deviceKnown: e.device === null ? null : devices.has(e.device),
      velocity24h: velocity,
      sharedEntities: others.size,
      historyCount: amounts.length,
      maskedC13: Math.max(0, Math.round(1 + (e.fraud ? 4 : 1) * Math.abs(r.normal()))),
      available: { tabular: true, behavioral: true, temporal: true, graph: entitySeen, anomaly: true },
    };

    if (e.t >= FIRST_DAY * DAY) {
      const s = score(input);
      const day = Math.floor(e.t / DAY);
      const row = {
        TransactionID: id,
        TransactionDT: e.t,
        day,
        fold: "fixed_validation",
        role: "gate_train",
        split: "validation",
        label: null,
        disagreement: s.decision.disagreement,
        ctx_log_amount: Math.log1p(e.amount),
        ctx_hist_n_past: amounts.length,
        ctx_has_identity: e.device !== null ? 1 : 0,
        ctx_product: PRODUCTS.indexOf(e.product as (typeof PRODUCTS)[number]) + 2,
        ctx_hour: Math.floor(e.t / 3600) % 24,
      } as ViewPredictionRow;
      for (const v of VIEWS) {
        const p = s.views[v];
        (row as unknown as Record<string, unknown>)[`p_${v}`] = p;
        (row as unknown as Record<string, unknown>)[`m_${v}`] = p === null ? 0 : 1;
        (row as unknown as Record<string, unknown>)[`c_${v}`] = p === null ? 0 : Math.abs(2 * p - 1);
      }
      const entities: Entities = {
        uid,
        card: e.c.card,
        email: e.c.email,
        address: e.c.address,
        device: e.device,
      };
      transactions.push({
        row,
        amount: e.amount,
        product: e.product,
        entities,
        weights: s.weights as Record<View, number>,
        decision: s.decision,
        reasons: s.reasons,
        inputs: input,
      });
      labels.set(id, e.fraud);
      id += 1;
    }

    amounts.push(e.amount);
    times.push(e.t);
    pastAmounts.set(uid, amounts);
    pastTimes.set(uid, times);
    if (e.device) devices.add(e.device);
    seenDevices.set(uid, devices);
    for (const key of [e.c.card, e.device]) {
      if (!key) continue;
      const users = entityUsers.get(key) ?? new Set<string>();
      users.add(uid);
      entityUsers.set(key, users);
    }
  }
  return {
    transactions,
    byId: new Map(transactions.map((t) => [t.row.TransactionID, t])),
    labels,
    ringOf,
  };
}
