import { describe, expect, it } from "vitest";
import { VIEWS } from "../types";
import { DAY } from "./generate";
import { createMockApi } from "./mockApi";

const api = createMockApi(7);
const asOf = (day: number, hour: number, L: number) => ({ time: day * DAY + hour * 3600, labelDelayDays: L });

describe("mock API as-of rules", () => {
  it("never returns a transaction after the as-of time", async () => {
    const a = asOf(140, 12, 30);
    const rows = await api.stream(a, 0, 10 ** 9);
    expect(rows.length).toBeGreaterThan(0);
    expect(Math.max(...rows.map((t) => t.row.TransactionDT))).toBeLessThanOrEqual(a.time);
  });

  it("releases a label only after its delay (the label-delay curtain)", async () => {
    for (const L of [0, 7, 30]) {
      const a = asOf(150, 0, L);
      const rows = await api.stream(a, 0, a.time);
      for (const t of rows) {
        const matured = t.row.TransactionDT + L * DAY <= a.time;
        if (matured) expect(t.row.label).not.toBeNull();
        else expect(t.row.label).toBeNull();
      }
    }
  });

  it("a later as-of time never changes an earlier transaction's scores", async () => {
    const early = await api.stream(asOf(135, 0, 30), 0, 135 * DAY);
    const late = await api.stream(asOf(150, 0, 30), 0, 135 * DAY);
    const byId = new Map(late.map((t) => [t.row.TransactionID, t]));
    for (const t of early.slice(0, 200)) {
      const u = byId.get(t.row.TransactionID)!;
      expect(u.decision.p).toBe(t.decision.p);
      for (const v of VIEWS) expect(u.row[`p_${v}`]).toBe(t.row[`p_${v}`]);
    }
  });

  it("missing views are null with mask 0 and weight 0, never a filled-in score", async () => {
    const rows = await api.stream(asOf(150, 0, 30), 0, 150 * DAY);
    const missing = rows.filter((t) => t.row.p_behavioral === null);
    expect(missing.length).toBeGreaterThan(0);
    for (const t of missing) {
      expect(t.row.m_behavioral).toBe(0);
      expect(t.row.c_behavioral).toBe(0);
      expect(t.weights.behavioral).toBe(0);
      expect(t.inputs.historyCount).toBe(0);
    }
    for (const t of rows.slice(0, 300)) {
      const sum = VIEWS.reduce((s, v) => s + t.weights[v], 0);
      expect(sum).toBeCloseTo(1, 10);
    }
  });

  it("network fraud counts use only labels known at the as-of time", async () => {
    const before = await api.network(asOf(141, 14, 30));
    expect(before.rings).toEqual([]); // labels known only up to day 111: no confirmed frauds
    const known = await api.network(asOf(141, 14, 0));
    expect(known.rings.length).toBeGreaterThan(0);
  });

  it("only admins may change thresholds, and thresholds must rise", async () => {
    const { thresholds } = await api.thresholds();
    await expect(api.saveThresholds(thresholds, "analyst")).rejects.toThrow(/Only admins/);
    await expect(api.saveThresholds({ ...thresholds, t_hold: 1 }, "admin")).rejects.toThrow(/must rise/);
  });
});
