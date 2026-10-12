import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

// the backend's answer decides the label, not the client's setting (D128, D132)
let served = false;

vi.mock("../api", () => ({
  DATA_SOURCE: "http",
  api: {
    range: async () => ({ first: 151 * 86400, last: 183 * 86400 - 1 }),
    thresholds: async () => ({
      thresholds: { t_monitor: 1, t_step: 2, t_hold: 546, t_block: null, t_dis: null },
      cost: { c_fp: 10, c_rev: 5, c_step: 0.5, step_up_success_rate: 0.9, legit_step_up_friction: 1 },
      frozen: true,
    }),
    daySummary: async () => ({
      day: 151,
      transactions: 10,
      alerts: 2,
      queue: 2,
      byAction: { allow: 8, monitor: 0, step_up: 2, hold: 0, block: 0 },
      expectedFraudStopped: 5,
      expectedLoss: 1,
      cost: { missedFraud: 1, falsePositives: 0, reviews: 0, stepUps: 1 },
      riskByHour: [],
      viewAvailability: { tabular: 1, behavioral: 1, temporal: 1, graph: 1, anomaly: 1 },
      whatIf: served,
      // the costs that produced these figures (D155); the browser holds no what-if setting here
      whatIfCost: served
        ? { c_fp: 3, c_rev: 2, c_step: 0.1, step_up_success_rate: 0.7, legit_step_up_friction: 0, hold_success_rate: 0.8 }
        : null,
    }),
    auditLog: async () => [],
  },
}));

const { AppProvider } = await import("../state/app");
const { Settings } = await import("./Settings");

function report() {
  render(
    <AppProvider>
      <Settings />
    </AppProvider>,
  );
}

describe("printed daily report (D132)", () => {
  it("carries the what-if label inside the printable region when the figures are what-if", async () => {
    served = true;
    report();
    await waitFor(() => expect(screen.getByText(/10 transactions/)).toBeTruthy());
    const region = document.querySelector(".print-region")!;
    expect(region.textContent).toMatch(/What-if cost mode: illustrative, not the evaluated system/);
    expect(region.querySelector(".no-print .whatif-report-note")).toBeNull();
    // D155: the printed costs are the response's, although the client has no what-if setting
    expect(region.textContent).toMatch(/false positive \$3, review \$2/);
    expect(region.textContent).toMatch(/hold stops 80%/);
  });

  it("has no what-if label for the evaluated system", async () => {
    served = false;
    report();
    await waitFor(() => expect(screen.getAllByText(/10 transactions/).length).toBeGreaterThan(0));
    const regions = [...document.querySelectorAll(".print-region")];
    expect(regions.at(-1)!.textContent).not.toMatch(/What-if/);
  });
});
