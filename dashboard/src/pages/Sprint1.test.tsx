/**
 * D220 (sprint 1): in http mode the Model lab shows the stage-2 ablation ladder with its source,
 * the Drift monitor shows the EXP-310 simulation labelled as such, Cases say feature reasons are
 * not stored and offer the Test Lab's live reasons, the what-if and the simulator are labelled
 * illustrative, and the fraud network says ring detection is not yet built.
 */
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import type { LabStatus, LiveResult, TestLabApi } from "../api/testlab";
import type { DriftReport, Lab as LabData } from "../api/types";

vi.mock("../components/NetworkGraph", () => ({ NetworkGraph: () => null }));

const LADDER_RUN = (id: string, name: string, pr: number) => ({ id, name, kind: "ablation" as const, prAuc: pr, ciLow: pr - 0.01, ciHigh: pr + 0.01, recallAt1: 0.5, cost: 250000, pr: [], calibration: [] });
const LAB: LabData = {
  runs: [LADDER_RUN("ladder-rung-1", "Rung 1: tabular (F4)", 0.548), LADDER_RUN("ladder-rung-2", "Rung 2: + behavioral (E3)", 0.6146)],
  ablation: [
    { step: "Rung 1: tabular (F4)", runId: "ladder-rung-1", source: "research/tables/stage2/ladder_metrics.json" },
    { step: "Rung 2: + behavioral (E3)", runId: "ladder-rung-2", source: "research/tables/stage2/ladder_metrics.json", stepDiff: 0.0666, stepCiLow: 0.059, stepCiHigh: 0.0756, stepPHolm: 0, stepSupported: true, stepSource: "research/tables/stage2/ladder_steps.json" },
  ],
  sensitivity: [],
};
const DRIFT: DriftReport = {
  reference: "calibration days 147-150 (before the test period)",
  featurePsi: [],
  scorePsi: [],
  disagreement: [],
  injected: [],
  alarms: [{ day: 152, detector: "ADWIN", policy: "P3", seed: 0 }],
  retrains: [
    { day: 156, trigger: "schedule", policy: "P1", seed: 0, accepted: true, reason: "accepted", gain: 0.02, gainCiLow: 0.011, gainCiHigh: 0.03, costChampion: 17539.65, costChallenger: 14024.89, prChampion: null, prChallenger: null },
    { day: 156, trigger: "alarm", policy: "P3", seed: 0, accepted: false, reason: "insufficient evidence", gain: null, gainCiLow: null, gainCiHigh: null, costChampion: null, costChallenger: null, prChampion: null, prChallenger: null },
  ],
  simulation: { label: "EXP-310 simulation on test period", sourceRun: "20261020-174032-191246", sourceFile: "research/tables/feedback/exp310_events.json", seed: 0, delay: 30, champion: "frozen B5 (EXP-310), not the served MVAF", policies: "P1: retrain every week (schedule); P3: retrain after an ADWIN alarm" },
};

vi.mock("../api", async () => {
  const { createMockApi } = await import("../api/mock/mockApi");
  const real = createMockApi(7);
  return { DATA_SOURCE: "http", api: { ...real, lab: async () => LAB, drift: async () => DRIFT } };
});

const { AppProvider } = await import("../state/app");
const { Lab } = await import("./Lab");
const { Drift } = await import("./Drift");
const { Case, WHATIF_LABEL } = await import("./Case");
const { Simulator, SIM_LABEL } = await import("./Simulator");
const { Network, RINGS_NOT_BUILT } = await import("./Network");
const { LiveReasons, NOT_STORED } = await import("../components/LiveReasons");

function at(path: string, route: string, page: ReactNode) {
  window.localStorage.setItem("evidence-room-prefs-v1", JSON.stringify({ time: 141 * 86400 + 20 * 3600, labelDelayDays: 0 }));
  render(
    <AppProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path={route} element={page} />
        </Routes>
      </MemoryRouter>
    </AppProvider>,
  );
}

describe("sprint 1 (D220), http mode", () => {
  it("Model lab: ablation rows from the ladder, step vs previous rung, source cited; ladder rungs not in the curve picker", async () => {
    at("/lab", "/lab", <Lab />);
    const table = await screen.findByRole("table", { name: "Ablation table" });
    expect(within(table).getByText("Rung 2: + behavioral (E3)")).toBeInTheDocument();
    expect(within(table).getByText(/\+0\.0666 \[0\.0590, 0\.0756\] \(0\.000\) H1 supported/)).toBeInTheDocument();
    expect(screen.getByTestId("ablation-source")).toHaveTextContent("research/tables/stage2/ladder_metrics.json, research/tables/stage2/ladder_steps.json");
    expect(within(screen.getByRole("group", { name: "Runs to compare" })).queryByText("Rung 1: tabular (F4)")).toBeNull();
  });

  it("Drift monitor: EXP-310 alarms and retraining, labelled as the simulation, with honest blanks", async () => {
    at("/drift", "/drift", <Drift />);
    const notes = await screen.findAllByTestId("simulation-note");
    expect(notes[0]).toHaveTextContent("EXP-310 simulation on test period: seed 0, label delay 30 days, champion: frozen B5 (EXP-310), not the served MVAF");
    expect(notes[0]).toHaveTextContent("research/tables/feedback/exp310_events.json (run 20261020-174032-191246)");
    expect(screen.getByText("Rejected: insufficient evidence")).toBeInTheDocument();
    expect(screen.getByText("Replaced the champion")).toBeInTheDocument();
    expect(screen.getByText("0.011 to 0.030")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /1 alarms, 2 retraining attempts/ })).toBeInTheDocument();
  });

  it("Cases: what-if labelled 'not the frozen system'; feature reasons not stored, live reasons offered", async () => {
    at("/cases", "/cases", <Case />);
    await waitFor(() => expect(screen.getByRole("heading", { level: 1 }).textContent).toMatch(/^Case /));
    expect(screen.getByTestId("whatif-label")).toHaveTextContent(WHATIF_LABEL);
    expect(WHATIF_LABEL).toBe("Illustrative model, not the frozen system");
    expect(screen.getByTestId("live-reasons")).toHaveTextContent(NOT_STORED);
  });

  it("Scenario simulator: labelled illustrative mock, pointing to the Test Lab", () => {
    at("/simulate", "/simulate", <Simulator />);
    expect(screen.getByTestId("sim-label")).toHaveTextContent(SIM_LABEL);
    expect(screen.getByRole("link", { name: "Open the Test Lab" })).toHaveAttribute("href", "/testlab");
  });

  it("Fraud network: 'ring detection: not yet built' instead of an empty rings panel", async () => {
    at("/network", "/network", <Network />);
    expect(await screen.findByTestId("rings-not-built")).toHaveTextContent(RINGS_NOT_BUILT);
    expect(screen.queryByText(/Showing rings/)).toBeNull();
  });
});

describe("live feature reasons from the Test Lab (D220)", () => {
  const READY: LabStatus = { state: "ready", message: "ok", parity: null };
  const RESULT = { reasons: [{ feature: "C13", text: "Unusual value in masked counter C13", points: 31 }], matches_frozen_run: true } as unknown as LiveResult;
  const lab = (over: Partial<TestLabApi>) => ({ status: vi.fn(async () => READY), scoreReal: vi.fn(async () => RESULT), ...over }) as unknown as TestLabApi;

  it("fetches only on request and labels the reasons live-computed", async () => {
    const api = lab({});
    render(<LiveReasons transactionId={3500000} lab={api} />);
    expect(api.scoreReal).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Compute feature reasons live/ }));
    expect(await screen.findByText(/masked counter C13/)).toBeInTheDocument();
    expect(screen.getByText(/Live-computed by the frozen system in the Test Lab \(matches the frozen run ✓\)/)).toBeInTheDocument();
    expect(api.scoreReal).toHaveBeenCalledWith(3500000);
  });

  it("says so when the Test Lab is not ready or not reachable", async () => {
    render(<LiveReasons transactionId={1} lab={lab({ status: vi.fn(async () => ({ state: "loading", message: "starting up", parity: null }) as LabStatus) })} />);
    fireEvent.click(screen.getByRole("button", { name: /Compute feature reasons live/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("not ready (loading: starting up)");
  });
});
