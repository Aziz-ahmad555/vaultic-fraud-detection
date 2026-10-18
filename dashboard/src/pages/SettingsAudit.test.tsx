import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

const auditEvent = vi.fn(async () => undefined);
vi.mock("../api", () => ({
  DATA_SOURCE: "http",
  api: {
    range: async () => ({ first: 151 * 86400, last: 183 * 86400 - 1 }),
    thresholds: async () => ({
      thresholds: { t_monitor: 1, t_step: 2, t_hold: 3, t_block: null, t_dis: null },
      cost: { c_fp: 10, c_rev: 5, c_step: 0.5, step_up_success_rate: 0.9, legit_step_up_friction: 1 },
      frozen: true,
    }),
    daySummary: async () => ({ day: 151, transactions: 10, alerts: 2, expectedFraudStopped: 5, expectedLoss: 1, whatIf: false }),
    auditLog: async () => [],
    auditEvent,
    saveThresholds: vi.fn(),
  },
}));

const { AppProvider } = await import("../state/app");
const { Settings } = await import("./Settings");
const { getIdentity } = await import("../api/identity");

describe("Settings: analyst name and recorded report prints (D208)", () => {
  it("sets the analyst name sent with requests, remembers it, and records a report print", async () => {
    window.localStorage.clear();
    window.print = vi.fn();
    render(
      <AppProvider>
        <Settings />
      </AppProvider>,
    );
    const name = await screen.findByLabelText(/Your name/);
    await userEvent.type(name, "Aziz");
    await waitFor(() => expect(getIdentity().analyst).toBe("Aziz"));
    expect(JSON.parse(window.localStorage.getItem("evidence-room-prefs-v1") ?? "{}").analyst).toBe("Aziz");
    await userEvent.click(await screen.findByRole("button", { name: "Print or save daily report as PDF" }));
    expect(auditEvent).toHaveBeenCalledWith("report_printed", "day 151");
    expect(window.print).toHaveBeenCalled();
  });
});
