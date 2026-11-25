import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { createMockApi } from "../api/mock/mockApi";
import type { Transaction } from "../api/types";
import { Counterfactual, narrative, timelineCoverage } from "./Case";

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

  it("leaves counterfactuals out entirely when none is passed (http mode, D127)", async () => {
    const t = await someCase();
    expect(narrative(t)).not.toMatch(/counterfactual|If the amount/i);
    expect(narrative(t, null)).toContain("No counterfactual has been explored yet.");
  });
});

describe("illustrative counterfactual (D127)", () => {
  it("compares the illustrative model only with its own score of the unchanged case", async () => {
    const t = await someCase();
    const real = { ...t, decision: { ...t.decision, p: 0.99 } }; // the evaluated system's risk: 99
    const own = await api.score(t.inputs); // the illustrative model on the unchanged case
    const described: (string | null)[] = [];
    render(<Counterfactual t={real} illustrative onDescribe={(s) => described.push(s)} />);
    await waitFor(() => expect(screen.getByText(/illustrative model on this case/)).toBeTruthy());
    const box = document.querySelector(".cf-result")!.textContent!;
    expect(box).toContain(`illustrative model on this case: ${Math.round(100 * own.p)}`);
    expect(box).toContain("the case's real risk is 99");
    expect(described).toEqual([]); // never handed to the narrative
  });
});

describe("customer timeline coverage (D228)", () => {
  it("never calls a customer with earlier history a cold start", () => {
    expect(timelineCoverage(0, 20)).toBe("No earlier transactions in the test-period data shown; 20 earlier in total, before the test period (point-in-time).");
    expect(timelineCoverage(0, 20)).not.toMatch(/cold start/);
    expect(timelineCoverage(3, 20)).toBe("3 of 20 earlier transactions shown (test-period data); the other 17 are before the test period (point-in-time).");
    expect(timelineCoverage(0, 0)).toMatch(/cold start/);
    expect(timelineCoverage(5, 5)).toBeNull();
  });
});
