import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CopilotError, createMockCopilot, type KeyFactsData } from "../api/copilot";
import { createMockApi } from "../api/mock/mockApi";
import type { AsOf, Transaction } from "../api/types";
import { nextChecks } from "../lib/nextChecks";
import { KeyFacts, KeyFactsView } from "./KeyFacts";
import { NextChecks } from "./NextChecks";

const asOf: AsOf = { time: 141 * 86400 + 14 * 3600, labelDelayDays: 30 };
const F: KeyFactsData = {
  case: 1, amount: 104.5, product: "W", decision: "step_up", p: 0.912, expected_loss: 95.3, conformal_set: "uncertain", disagreement: 0.35,
  label: null, label_delay_days: 30, reasons_source: "view-level reasons (gate weight x view logit); feature reasons not available: the Test Lab is off",
  reasons: [{ text: "Tabular view raises risk", points: 41.2 }], testlab: "not mounted", source: "rendered by code from the case evidence; no language model",
};

async function someCase(): Promise<Transaction> {
  return (await createMockApi(7).stream(asOf, 141 * 86400, asOf.time))[0];
}

describe("key facts card (D226)", () => {
  it("shows the decision, probability, expected loss, conformal set and the view-level reasons, labelled", () => {
    render(<KeyFactsView f={F} />);
    const card = screen.getByTestId("key-facts");
    expect(card).toHaveTextContent("Step-up");
    expect(card).toHaveTextContent("0.912");
    expect(card).toHaveTextContent("$95.30");
    expect(screen.getByTestId("reasons-source")).toHaveTextContent("View-level reasons: feature reasons not available: the Test Lab is off");
    expect(card).toHaveTextContent("Tabular view raises risk");
    expect(card).toHaveTextContent("no language model");
  });

  it("shows the label only once it has matured", () => {
    const { unmount } = render(<KeyFactsView f={F} />);
    expect(screen.getByTestId("key-label")).toHaveTextContent("Not known yet");
    unmount();
    render(<KeyFactsView f={{ ...F, label: 1 }} />);
    expect(screen.getByTestId("key-label")).toHaveTextContent("Confirmed fraud");
  });

  it("labels live frozen-system reasons and a loading Test Lab", () => {
    const { unmount } = render(<KeyFactsView f={{ ...F, reasons_source: "live frozen-system reasons (Test Lab, parity-gated)" }} />);
    expect(screen.getByTestId("reasons-source")).toHaveTextContent("Live frozen-system reasons (Test Lab, parity-gated)");
    unmount();
    render(<KeyFactsView f={{ ...F, reasons_source: "view-level reasons (gate weight x view logit); feature reasons not available: the Test Lab is loading" }} />);
    expect(screen.getByTestId("reasons-source")).toHaveTextContent("the Test Lab is loading");
  });

  it("falls back to the page's own data, labelled view-level, when the copilot is not mounted (D226a)", async () => {
    const t = await someCase();
    const notMounted = { ...createMockCopilot(0), facts: () => Promise.reject(new CopilotError("Not Found", 404)) };
    render(<KeyFacts t={t} asOf={asOf} copilot={notMounted} />);
    const card = await screen.findByTestId("key-facts");
    expect(card).toHaveTextContent(t.decision.p.toFixed(3));
    expect(screen.getByTestId("reasons-source")).toHaveTextContent("View-level reasons: page data; the copilot is not enabled on the backend");
    expect(card).toHaveTextContent("page's transaction data");
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("shows other errors as errors", async () => {
    const t = await someCase();
    const broken = { ...createMockCopilot(0), facts: () => Promise.reject(new CopilotError("backend error", 500)) };
    render(<KeyFacts t={t} asOf={asOf} copilot={broken} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("backend error");
  });

  it("is built from the mock case in mock mode", async () => {
    const t = await someCase();
    render(<KeyFacts t={t} asOf={asOf} copilot={createMockCopilot(0)} />);
    expect(await screen.findByTestId("key-facts")).toHaveTextContent(t.decision.p.toFixed(3));
  });
});

describe("suggested next checks (D226)", () => {
  it("are rules on the case's fields, labelled 'rules, not AI'", async () => {
    const t = await someCase();
    const odd: Transaction = {
      ...t,
      row: { ...t.row, m_graph: 0, label: null },
      decision: { ...t.decision, conformal_set: "uncertain", disagreement: 0.4 },
      inputs: { ...t.inputs, historyCount: 0, deviceKnown: false, sharedEntities: 2 },
    };
    expect(nextChecks(odd, asOf).map((c) => c.id)).toEqual(expect.arrayContaining(["graph", "uncertain", "disagree", "cold", "device", "shared", "label"]));
    render(<NextChecks t={odd} asOf={asOf} />);
    expect(screen.getByTestId("next-checks")).toHaveTextContent("rules, not AI");
    expect(screen.getByTestId("next-checks")).toHaveTextContent("2 other customers used the card in the 7 days before");
  });

  it("say so when no rule applies", async () => {
    const t = await someCase();
    const calm: Transaction = {
      ...t,
      row: { ...t.row, m_tabular: 1, m_behavioral: 1, m_temporal: 1, m_graph: 1, m_anomaly: 1, label: 0 },
      decision: { ...t.decision, conformal_set: "legit", disagreement: 0.05 },
      inputs: { ...t.inputs, historyCount: 5, deviceKnown: true, sharedEntities: 0 },
    };
    expect(nextChecks(calm, asOf)).toEqual([]);
    render(<NextChecks t={calm} asOf={asOf} />);
    expect(screen.getByTestId("next-checks")).toHaveTextContent("No rule applies to this case.");
  });
});
