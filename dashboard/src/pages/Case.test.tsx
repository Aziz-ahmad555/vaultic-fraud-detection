import { describe, expect, it } from "vitest";
import { createMockApi } from "../api/mock/mockApi";
import type { Transaction } from "../api/types";
import { narrative } from "./Case";

const api = createMockApi(7);

async function someCase(): Promise<Transaction> {
  const rows = await api.stream({ time: 141 * 86400 + 14 * 3600, labelDelayDays: 30 }, 141 * 86400, 141 * 86400 + 14 * 3600);
  return rows[0];
}

describe("case narrative", () => {
  it("never states a fact from a field that is not available (D126)", async () => {
    const t = await someCase();
    // reasons are stored facts of their own; clear them to test only the sentences built from inputs
    const unknown = { ...t, reasons: [], inputs: { ...t.inputs, sharedEntities: null, velocity24h: null, usualAmount: null, deviceKnown: null, maskedC13: null } };
    const text = narrative(unknown, null);
    expect(text).not.toMatch(/card|shared|device|24 hours|usual/i);
  });

  it("states the card fact only as computed", async () => {
    const t = await someCase();
    expect(narrative({ ...t, inputs: { ...t.inputs, sharedEntities: 0 } }, null)).toContain("No other customer used its card in the 7 days before.");
    expect(narrative({ ...t, inputs: { ...t.inputs, sharedEntities: 3 } }, null)).toContain("3 other customers used its card in the 7 days before.");
  });
});
