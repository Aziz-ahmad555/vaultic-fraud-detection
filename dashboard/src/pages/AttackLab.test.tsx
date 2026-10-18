import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { AttackApi, ModeResult, RunResult, ScenarioList, SuiteResponse } from "../api/attack";
import type { LabStatus, TestLabApi } from "../api/testlab";
import { AttackLab } from "./AttackLab";

const PARITY = { n: 2000, passed: true, feature_tolerance: 1e-6, score_tolerance: 1e-12, max_feature_diff: 0, max_score_diff: { MVAF: 0, B5: 0 }, n_failing_features: 0, failing_features: {} };
const READY: LabStatus = { state: "ready", message: "ok", parity: PARITY, latency: { n: 0, p50_ms: null, p95_ms: null } };
const LIST: ScenarioList = {
  label: "synthetic scenario - not the evaluation",
  step_up_note: "A step-up is an extra customer check (friction), not a block.",
  intensities: ["low", "medium", "high"],
  threshold_sets: ["D196 (primary)", "D195 (sensitivity)", "D196 without R3 (post hoc)"],
  scenarios: [
    { name: "card_testing", title: "Card testing", description: "A burst of small charges.", intensities: { low: "5 charges", medium: "12 charges", high: "25 charges" }, notes: [] },
    { name: "mule_ring", title: "Mule ring", description: "New cards on one device.", intensities: { low: "5 cards", medium: "10 cards", high: "20 cards" },
      notes: ["IEEE-CIS has no true device ID; DeviceInfo is a model/OS string, so a rare value stands in for one device."] },
  ],
};
const mode = (alerts: number, fpr: number, hard: number): ModeResult => ({
  recall_synthetic: alerts / 2, fpr_legit: fpr, alerts_synthetic: alerts, alerts_legit: 10, first_alerted_synthetic_row: alerts ? 1 : null,
  hard_recall_synthetic: hard / 2, hard_fpr_legit: 0.0002, hard_alerts_synthetic: hard, hard_alerts_legit: 0, stepup_only_fpr_legit: fpr - 0.0002, stepup_only_alerts_legit: 9,
});
const SET = { n_synthetic_fraud: 2, n_legit: 100, rules_only: mode(1, 0.01, 1), model_only: mode(1, 0.2, 0), rules_plus_model: mode(2, 0.21, 1), added_fpr: 0.01, budget_added_fpr_ok: false, budget_rules_only_ok: true };
const RUN: RunResult = {
  label: "synthetic scenario - not the evaluation", scenario: "card_testing", title: "Card testing", seed: 0, intensity: "medium", model: "on",
  days: [178], info: {}, notes: [], step_up_note: LIST.step_up_note, n_synthetic: 2, n_legit_replay: 100, n_legit_stored: 99, n_legit_live: 1,
  stored_score_max_diff: 1e-19, seconds: 3.1,
  results: { "D196 (primary)": SET, "D195 (sensitivity)": SET, "D196 without R3 (post hoc)": SET },
  replay: [
    { TransactionID: 1000000001, TransactionDT: 178 * 86400 + 70, amount: 8.89, synthetic: true, label: 1, source: "scored live", risk: 0.3, model_action: "step_up", rule_action: "allow", rules_fired: "", combined_action: "step_up", decision: "step_up",
      trace: [{ rule: "R1", name: "customer velocity, 1 hour", value: 1, threshold: 6, fired: false, action: null, condition: "customer's transactions in the previous hour >= 6" }] },
    { TransactionID: 1000000000, TransactionDT: 178 * 86400 + 10, amount: 3.05, synthetic: true, label: 1, source: "scored live", risk: 0.1, model_action: "allow", rule_action: "step_up", rules_fired: "R1", combined_action: "step_up", decision: "step_up", trace: [] },
  ],
};
const SUITE: SuiteResponse = { label: "x", suite: null, posthoc: null };

function fakes(over: Partial<AttackApi> = {}) {
  const api = { scenarios: vi.fn(async () => LIST), run: vi.fn(async () => RUN), suite: vi.fn(async () => SUITE), ...over } as AttackApi;
  const lab = { status: vi.fn(async () => READY) } as unknown as TestLabApi;
  return { api, lab };
}

describe("Attack Lab", () => {
  it("labels every result as synthetic and explains a step-up", async () => {
    const { api, lab } = fakes();
    render(<AttackLab api={api} lab={lab} pollMs={60000} replayMs={0} />);
    expect(await screen.findByTestId("label")).toHaveTextContent("Synthetic scenario - not the evaluation");
    expect(screen.getByTestId("label")).toHaveTextContent("extra customer check (friction), not a block");
  });

  it("shows the mule-ring device note", async () => {
    const { api, lab } = fakes();
    render(<AttackLab api={api} lab={lab} pollMs={60000} replayMs={0} />);
    await userEvent.click(await screen.findByRole("button", { name: "Mule ring" }));
    expect(screen.getByText(/IEEE-CIS has no true device ID/)).toBeInTheDocument();
  });

  it("runs a scenario and shows all three threshold sets, the FPR split, the budget and the replay in time order", async () => {
    const { api, lab } = fakes();
    render(<AttackLab api={api} lab={lab} pollMs={60000} replayMs={0} />);
    await userEvent.click(await screen.findByRole("button", { name: "Run scenario" }));
    await waitFor(() => expect(api.run).toHaveBeenCalledWith({ scenario: "card_testing", intensity: "medium", seed: 0, model: true }));
    expect(await screen.findByRole("table", { name: "Scorecard D196 (primary)" })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Scorecard D196 without R3 (post hoc)" })).toBeInTheDocument();
    expect(screen.getAllByText(/post-hoc sensitivity \(chosen after seeing results\)/).length).toBeGreaterThan(0);
    const primary = screen.getByRole("table", { name: "Scorecard D196 (primary)" });
    expect(within(primary).getByText("FPR step-up only")).toBeInTheDocument();
    expect(within(primary).getByText("FPR hold/block")).toBeInTheDocument();
    expect(screen.getAllByText(/OVER the 0.5 pp/).length).toBe(3);
    const rows = within(screen.getByRole("table", { name: "Replay" })).getAllByRole("row");
    expect(rows[1]).toHaveTextContent("$3.05"); // earlier row first
    await userEvent.click(rows[2]);
    expect(screen.getByRole("list", { name: "Rule trace" })).toHaveTextContent("R1 customer velocity, 1 hour: 1 vs");
  });

  it("sends model off for degraded mode and marks rules only as deciding", async () => {
    const { api, lab } = fakes({ run: vi.fn(async () => ({ ...RUN, model: "off (degraded mode: rules alone decide)" })) });
    render(<AttackLab api={api} lab={lab} pollMs={60000} replayMs={0} />);
    await userEvent.click(await screen.findByLabelText(/Model off/));
    await userEvent.click(screen.getByRole("button", { name: "Run scenario" }));
    await waitFor(() => expect(api.run).toHaveBeenCalledWith(expect.objectContaining({ model: false })));
    expect((await screen.findAllByText("Rules only (deciding)")).length).toBeGreaterThan(0);
  });

  it("shows the backend's refusal", async () => {
    const { api, lab } = fakes({ run: vi.fn(async () => { throw new Error("seed must be between 0 and 9999"); }) });
    render(<AttackLab api={api} lab={lab} pollMs={60000} replayMs={0} />);
    await userEvent.click(await screen.findByRole("button", { name: "Run scenario" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("seed must be between 0 and 9999");
  });
});
