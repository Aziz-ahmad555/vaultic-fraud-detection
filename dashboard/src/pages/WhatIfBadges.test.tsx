import { render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

// jsdom has no canvas for cytoscape; the drawing is not under test here
vi.mock("../components/NetworkGraph", () => ({ NetworkGraph: () => null }));

// the mock data, with every response flagged what-if by the "backend" (D128)
vi.mock("../api", async () => {
  const { createMockApi } = await import("../api/mock/mockApi");
  const real = createMockApi(7);
  type Tx = Awaited<ReturnType<typeof real.alerts>>[number];
  const flag = (t: Tx): Tx => ({ ...t, decision: { ...t.decision, whatIf: true } });
  return {
    DATA_SOURCE: "http",
    api: {
      ...real,
      stream: async (...a: Parameters<typeof real.stream>) => (await real.stream(...a)).map(flag),
      alerts: async (...a: Parameters<typeof real.alerts>) => (await real.alerts(...a)).map(flag),
      transaction: async (...a: Parameters<typeof real.transaction>) => {
        const t = await real.transaction(...a);
        return t && flag(t);
      },
      queues: async (...a: Parameters<typeof real.queues>) =>
        (await real.queues(...a)).map((q) => ({ ...q, cases: q.cases.map(flag) })),
      network: async (...a: Parameters<typeof real.network>) => ({ ...(await real.network(...a)), whatIf: true }),
    },
  };
});

const { AppProvider } = await import("../state/app");
const { Case } = await import("./Case");
const { Queue } = await import("./Queue");
const { Live } = await import("./Live");
const { Network } = await import("./Network");

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

const badges = () => screen.queryAllByText("What-if", { selector: ".badge-whatif" });

describe("a What-if badge on every what-if decision (D128)", () => {
  it("Case: the alert list and the case header", async () => {
    at("/cases", "/cases", <Case />);
    await waitFor(() => expect(screen.getByRole("heading", { level: 1 }).textContent).toMatch(/^Case /));
    const decisions = document.querySelectorAll(".badge[class*='dec-']").length;
    expect(decisions).toBeGreaterThan(1);
    expect(badges().length).toBe(decisions); // one per decision shown
  });

  it("Queue: every case of every policy", async () => {
    at("/queue", "/queue", <Queue />);
    await waitFor(() => expect(document.querySelectorAll(".badge[class*='dec-']").length).toBeGreaterThan(0));
    expect(badges().length).toBe(document.querySelectorAll(".badge[class*='dec-']").length);
  });

  it("Live: every streamed decision", async () => {
    at("/live", "/live", <Live />);
    await waitFor(() => expect(document.querySelectorAll(".badge[class*='dec-']").length).toBeGreaterThan(0));
    expect(badges().length).toBe(document.querySelectorAll(".badge[class*='dec-']").length);
  });

  it("Network: the what-if network", async () => {
    at("/network", "/network", <Network />);
    await waitFor(() => expect(badges().length).toBe(1));
  });
});
