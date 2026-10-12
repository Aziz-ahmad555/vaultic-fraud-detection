import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const FROZEN = { t_monitor: 1.05, t_step: 1.86, t_hold: 546, t_block: null, t_dis: 0.46 };

vi.mock("../api", () => ({
  DATA_SOURCE: "http",
  api: {
    range: async () => ({ first: 151 * 86400, last: 183 * 86400 - 1 }),
    thresholds: async () => ({
      thresholds: FROZEN,
      cost: { c_fp: 10, c_rev: 5, c_step: 0.5, step_up_success_rate: 0.9, legit_step_up_friction: 1 },
      frozen: true,
    }),
    daySummary: async () => null,
    auditLog: async () => [],
    saveThresholds: vi.fn(),
  },
}));

const { AppProvider } = await import("../state/app");
const { Settings } = await import("./Settings");

describe("Settings with the real-data backend (D129)", () => {
  it("shows the frozen thresholds read-only, for an admin too, with no save button", async () => {
    window.localStorage.setItem("evidence-room-prefs-v1", JSON.stringify({ role: "admin" }));
    render(
      <AppProvider>
        <Settings />
      </AppProvider>,
    );
    await waitFor(() => expect(screen.getByText(/frozen thresholds of the final run/)).toBeTruthy());
    const fields = (await screen.findAllByRole("spinbutton")) as HTMLInputElement[];
    const thresholdFields = fields.slice(0, 5);
    expect(thresholdFields.map((f) => f.value)).toEqual(["1.05", "1.86", "546", "", "0.46"]);
    for (const f of thresholdFields) expect(f.disabled).toBe(true);
    expect(screen.queryByRole("button", { name: "Save thresholds" })).toBeNull();
  });
});
