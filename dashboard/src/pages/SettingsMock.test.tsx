import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const saveThresholds = vi.fn(async (_t: unknown, _role: unknown) => undefined);

vi.mock("../api", () => ({
  DATA_SOURCE: "mock",
  api: {
    range: async () => ({ first: 128 * 86400, last: 151 * 86400 - 1 }),
    thresholds: async () => ({
      thresholds: { t_monitor: 4, t_step: 15, t_hold: 60, t_block: 250, t_dis: 0.3 },
      cost: { c_fp: 10, c_rev: 5, c_step: 0.5, step_up_success_rate: 0.9, legit_step_up_friction: 1 },
    }),
    daySummary: async () => null,
    auditLog: async () => [],
    saveThresholds: (t: unknown, role: unknown) => saveThresholds(t, role),
  },
}));

const { AppProvider } = await import("../state/app");
const { Settings } = await import("./Settings");

describe("editable thresholds in the mock (D137)", () => {
  it("ignores an emptied Hold field, and an emptied Block field means never", async () => {
    window.localStorage.setItem("evidence-room-prefs-v1", JSON.stringify({ role: "admin" }));
    render(
      <AppProvider>
        <Settings />
      </AppProvider>,
    );
    const hold = (await screen.findByLabelText(/Hold from expected loss/)) as HTMLInputElement;
    const block = screen.getByLabelText(/Block from expected loss/) as HTMLInputElement;
    fireEvent.change(hold, { target: { value: "" } });
    fireEvent.change(block, { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Save thresholds" }));
    await waitFor(() => expect(saveThresholds).toHaveBeenCalled());
    expect(saveThresholds.mock.calls[0][0]).toEqual({ t_monitor: 4, t_step: 15, t_hold: 60, t_block: null, t_dis: 0.3 });
  });
});
