import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const FROZEN_COST = { c_fp: 10, c_rev: 5, c_step: 0.5, step_up_success_rate: 0.9, legit_step_up_friction: 1, hold_success_rate: 1 };
// a what-if setting can make cost choose never to hold or block: those thresholds are null
const WHATIF_TH = { t_monitor: 0.5, t_step: 3.25, t_hold: null, t_block: null, t_dis: null };
const whatIf = vi.fn(async (cost: unknown) =>
  cost
    ? { active: true, available: true, cost, thresholds: WHATIF_TH, note: "illustrative" }
    : { active: false, available: true, cost: FROZEN_COST, thresholds: null, note: "illustrative" },
);

vi.mock("../api", () => ({
  DATA_SOURCE: "http",
  api: { range: async () => ({ first: 151 * 86400, last: 183 * 86400 - 1 }), whatIf: (c: unknown) => whatIf(c) },
}));

const { AppProvider } = await import("../state/app");
const { WhatIfPanel } = await import("./WhatIfPanel");
const { fmtThreshold } = await import("../lib/format");
const { decide } = await import("../api/mock/model");

describe("null thresholds render as never (D134)", () => {
  it("formats null as never and numbers with two decimals", () => {
    expect(fmtThreshold(null)).toBe("never");
    expect(fmtThreshold(undefined)).toBe("never");
    expect(fmtThreshold(546)).toBe("546.00");
  });

  it("a null threshold is never reached by the decision rule", () => {
    const never = { t_monitor: null, t_step: null, t_hold: null, t_block: null, t_dis: null };
    expect(decide(0.99, 1e6, "fraud", 1, never)).toBe("allow");
    expect(decide(0.99, 1e6, "uncertain", 0, never)).toBe("step_up"); // the conformal rule stays
    expect(decide(0.5, 100, "legit", 0, { ...never, t_hold: 10 })).toBe("hold");
  });

  it("the what-if panel shows null what-if thresholds as never instead of failing", async () => {
    render(
      <AppProvider>
        <WhatIfPanel />
      </AppProvider>,
    );
    fireEvent.click(await screen.findByRole("button", { name: "Apply what-if costs" }));
    await waitFor(() => expect(screen.getByText(/What-if thresholds/)).toBeTruthy());
    const text = screen.getByText(/What-if thresholds/).textContent!;
    expect(text).toContain("monitor 0.50, step-up 3.25, hold never");
    expect(text).toContain("block never; hold on view disagreement from never");
  });
});
