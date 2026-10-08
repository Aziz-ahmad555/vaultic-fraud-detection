import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { View } from "../api/types";
import { VIEWS } from "../api/types";
import { score } from "../api/mock/model";
import { Braid, braidWeights } from "./Braid";

const weights: Record<View, number> = { tabular: 0.3, behavioral: 0.2, temporal: 0.25, graph: 0.15, anomaly: 0.1 };
const scores: Record<View, number | null> = { tabular: 0.8, behavioral: 0.7, temporal: null, graph: 0.9, anomaly: 0.3 };

const strands = (c: HTMLElement) => Array.from(c.querySelectorAll<SVGGElement>(".braid-strand"));

describe("evidence braid", () => {
  it("draws five strands in fixed view order", () => {
    const { container } = render(<Braid weights={weights} scores={scores} />);
    expect(strands(container).map((g) => g.dataset.view)).toEqual([...VIEWS]);
  });

  it("shows a missing view as a dashed gap with no end cap", () => {
    const { container } = render(<Braid weights={weights} scores={scores} />);
    const temporal = strands(container).find((g) => g.dataset.view === "temporal")!;
    expect(temporal.dataset.missing).toBe("true");
    expect(temporal.querySelector("path")!.getAttribute("stroke-dasharray")).toBeTruthy();
    expect(temporal.querySelector(".braid-cap")).toBeNull();
    for (const g of strands(container).filter((s) => s.dataset.view !== "temporal")) {
      expect(g.dataset.missing).toBe("false");
      expect(g.querySelector("path")!.getAttribute("stroke-dasharray")).toBeNull();
    }
  });

  it("weights of the available views sum to 1 and missing views weigh 0", () => {
    const { container } = render(<Braid weights={weights} scores={scores} />);
    const ws = strands(container).map((g) => Number(g.dataset.weight));
    expect(ws.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    expect(Number(strands(container)[2].dataset.weight)).toBe(0);
    // renormalised: tabular 0.3 / 0.75
    expect(Number(strands(container)[0].dataset.weight)).toBeCloseTo(0.4, 10);
  });

  it("strand thickness follows the weight", () => {
    const { container } = render(<Braid weights={weights} scores={scores} />);
    const width = (v: View) =>
      Number(strands(container).find((g) => g.dataset.view === v)!.querySelector("path")!.getAttribute("stroke-width"));
    expect(width("tabular")).toBeGreaterThan(width("behavioral"));
    expect(width("behavioral")).toBeGreaterThan(width("anomaly"));
  });

  it("shades each strand end by that view's score", () => {
    const { container } = render(<Braid weights={weights} scores={scores} />);
    const cap = (v: View) =>
      Number(strands(container).find((g) => g.dataset.view === v)!.querySelector(".braid-cap")!.getAttribute("fill-opacity"));
    expect(cap("graph")).toBeGreaterThan(cap("anomaly"));
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
      const sum = VIEWS.reduce((a, v) => a + w[v], 0);
      expect(sum).toBeCloseTo(1, 10);
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
    fireEvent.focus(strands(container)[1]);
    expect(screen.getByText(/25.0x this customer's usual/)).toBeInTheDocument();
    expect(screen.queryByText(/masked counter C13/)).toBeNull();
    fireEvent.focus(strands(container)[2]);
    expect(screen.getByText(/could not score this transaction/)).toBeInTheDocument();
  });
});
