import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { View } from "../api/types";
import { VIEWS } from "../api/types";
import { score } from "../api/mock/model";
import { Braid, braidWeights, geometry } from "./Braid";

const weights: Record<View, number> = { tabular: 0.3, behavioral: 0.2, temporal: 0.25, graph: 0.15, anomaly: 0.1 };
const scores: Record<View, number | null> = { tabular: 0.8, behavioral: 0.7, temporal: null, graph: 0.9, anomaly: 0.3 };

const strands = (c: HTMLElement) => Array.from(c.querySelectorAll<SVGGElement>(".braid-strand"));
const strand = (c: HTMLElement, v: View) => strands(c).find((g) => g.dataset.view === v)!;

describe("evidence braid (convergence)", () => {
  it("draws five strands in fixed view order", () => {
    const { container } = render(<Braid weights={weights} scores={scores} />);
    expect(strands(container).map((g) => g.dataset.view)).toEqual([...VIEWS]);
  });

  it("shows a missing view as a dashed ghost that stops before the merge", () => {
    for (const size of ["inline", "large"] as const) {
      const { container, unmount } = render(<Braid weights={weights} scores={scores} size={size} risk={0.6} />);
      const ghost = strand(container, "temporal");
      expect(ghost.dataset.missing).toBe("true");
      const line = ghost.querySelector(".braid-ghost")!;
      expect(line.getAttribute("stroke-dasharray")).toBeTruthy();
      expect(Number(line.getAttribute("x2"))).toBeLessThan(geometry(size).xm); // never reaches the cord
      expect(ghost.querySelector("path")).toBeNull(); // no band, no weight
      for (const v of VIEWS.filter((x) => x !== "temporal")) {
        expect(strand(container, v).querySelector(".braid-ghost")).toBeNull();
        expect(strand(container, v).querySelector("path")).not.toBeNull();
      }
      unmount();
    }
  });

  it("weights of the available views sum to 1 and missing views weigh 0", () => {
    const { container } = render(<Braid weights={weights} scores={scores} />);
    const ws = strands(container).map((g) => Number(g.dataset.weight));
    expect(ws.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    expect(Number(strand(container, "temporal").dataset.weight)).toBe(0);
    expect(Number(strand(container, "tabular").dataset.weight)).toBeCloseTo(0.4, 10); // 0.3 / 0.75
  });

  it("strand thickness at the merge is the weight times the cord, and they stack to the full cord", () => {
    const { container } = render(<Braid weights={weights} scores={scores} size="large" risk={0.6} />);
    const g = geometry("large");
    const t = (v: View) => Number(strand(container, v).dataset.thickness);
    expect(t("tabular")).toBeCloseTo(0.4 * g.cord, 6);
    expect(VIEWS.reduce((s, v) => s + t(v), 0)).toBeCloseTo(g.cord, 6);
    // 0.26 vs 0.41 must look different: at least 10 px apart on the large braid
    const { container: c2 } = render(
      <Braid size="large" weights={{ tabular: 0.26, behavioral: 0.41, temporal: 0.13, graph: 0.1, anomaly: 0.1 }} scores={{ tabular: 0.5, behavioral: 0.5, temporal: 0.5, graph: 0.5, anomaly: 0.5 }} />,
    );
    expect(Number(strand(c2, "behavioral").dataset.thickness) - Number(strand(c2, "tabular").dataset.thickness)).toBeGreaterThan(10);
  });

  it("labels each strand with its view score and weight, and ends the cord in the risk number", () => {
    render(<Braid weights={weights} scores={scores} size="large" risk={0.63} />);
    expect(screen.getByText("Tabular")).toBeInTheDocument();
    expect(screen.getByText("80")).toBeInTheDocument(); // tabular score
    expect(screen.getByText("40%")).toBeInTheDocument(); // tabular weight
    expect(screen.getAllByText("not available").length).toBe(1);
    expect(screen.getByText("63")).toBeInTheDocument(); // fused risk at the cord end
  });

  it("braid weights of the mock model always sum to 1 over available views", () => {
    for (const historyCount of [0, 5]) {
      const r = score({
        amount: 120,
        usualAmount: historyCount ? 40 : null,
        deviceKnown: false,
        velocity24h: 2,
        sharedEntities: 1,
        historyCount,
        maskedC13: 3,
        available: { tabular: true, behavioral: true, temporal: true, graph: false, anomaly: true },
      });
      const w = braidWeights(r.weights, r.views);
      expect(VIEWS.reduce((a, v) => a + w[v], 0)).toBeCloseTo(1, 10);
      expect(w.graph).toBe(0);
      if (historyCount === 0) expect(w.behavioral + w.temporal).toBe(0);
    }
  });

  it("large braid shows the hovered view's reasons and says when a view is missing", () => {
    const reasons = [
      { feature: "amt_ratio_median", text: "Amount is 25.0x this customer's usual (median) amount", points: 20, view: "behavioral" as View },
      { feature: "C13", text: "Unusual value in masked counter C13", points: 9, view: "tabular" as View },
    ];
    const { container } = render(<Braid weights={weights} scores={scores} size="large" reasons={reasons} />);
    fireEvent.focus(strand(container, "behavioral"));
    expect(screen.getByText(/25.0x this customer's usual/)).toBeInTheDocument();
    expect(screen.queryByText(/masked counter C13/)).toBeNull();
    fireEvent.focus(strand(container, "temporal"));
    expect(screen.getByText(/could not score this transaction/)).toBeInTheDocument();
  });
});
