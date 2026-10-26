import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CopilotError, NOT_AVAILABLE, createMockCopilot, type CopilotApi, type CopilotEvent, type CopilotFinal, type CopilotStatus, type EvidenceLine } from "../api/copilot";
import { createMockApi } from "../api/mock/mockApi";
import type { AsOf, Transaction } from "../api/types";
import { CopilotPanel, READ_ONLY_NOTE, SUGGESTED, progressText, removedNote } from "./CopilotPanel";

const asOf: AsOf = { time: 141 * 86400 + 14 * 3600, labelDelayDays: 30 };

async function someCase(): Promise<Transaction> {
  const rows = await createMockApi(7).stream(asOf, 141 * 86400, asOf.time);
  return rows[0];
}

const READY: CopilotStatus = { model: "qwen2.5:7b-instruct-q4_K_M", llm: "ready", loaded: true, variant: "V3", testlab: "not mounted", reasons: "view-level reasons", cached_summaries: 5, note: "Local language model (qwen), answers only from the case evidence; not the evaluated detection system." };
const EVIDENCE: EvidenceLine[] = [
  { handle: "E1", evidence_id: "case:1/decision", tool: "get_case", text: "transaction 1: decision hold; calibrated fraud probability 0.912." },
  { handle: "E2", evidence_id: "case:1/customer", tool: "get_case", text: "customer 7: 6 earlier transactions in total." },
];

function final(over: Partial<CopilotFinal> = {}): CopilotFinal {
  return {
    type: "final", case: 1, kind: "question", route: "why", model: READY.model, variant: "V3", note: READY.note, reasons_source: "view-level",
    evidence: EVIDENCE, answer: "The decision is hold [E1].", kept: [{ text: "The decision is hold [E1].", citations: ["E1"] }], removed: [],
    cached: null, seconds: 42, audit: "ok", ...over,
  };
}

/** A fake copilot: status and answers as given; events are replayed before the final. */
function fake(status: CopilotStatus | Error, answer: CopilotFinal | Error = final(), events: CopilotEvent[] = []): CopilotApi & { asked: string[] } {
  const asked: string[] = [];
  const reply = async (onEvent: (e: CopilotEvent) => void) => {
    events.forEach(onEvent);
    if (answer instanceof Error) throw answer;
    onEvent(answer);
    return answer;
  };
  return {
    asked,
    status: () => (status instanceof Error ? Promise.reject(status) : Promise.resolve(status)),
    facts: () => Promise.reject(new Error("unused")),
    ask: (_id, q, _a, onEvent) => {
      asked.push(q);
      return reply(onEvent);
    },
    summary: (_id, _a, onEvent) => {
      asked.push("<summary>");
      return reply(onEvent);
    },
  };
}

describe("copilot panel (D226)", () => {
  it("always shows the read-only note and the suggested questions", async () => {
    render(<CopilotPanel t={await someCase()} asOf={asOf} copilot={fake(READY)} />);
    expect(screen.getByTestId("copilot-readonly")).toHaveTextContent(READ_ONLY_NOTE);
    for (const q of SUGGESTED) expect(await screen.findByRole("button", { name: q })).toBeEnabled();
    expect(screen.getByTestId("copilot-testlab")).toHaveTextContent("Test Lab off: reasons are view-level");
  });

  it("asks a suggested question and shows checked facts with citation chips that highlight the evidence", async () => {
    const cop = fake(READY);
    render(<CopilotPanel t={await someCase()} asOf={asOf} copilot={cop} />);
    fireEvent.click(await screen.findByRole("button", { name: "Why was this flagged?" }));
    await screen.findByTestId("copilot-answer");
    expect(cop.asked).toEqual(["Why was this flagged?"]);
    expect(screen.getByTestId("copilot-answer")).toHaveTextContent("The decision is hold");
    const chip = screen.getByRole("button", { name: "Show evidence E1" });
    const line = () => screen.getByTestId("copilot-evidence").querySelector('[data-handle="E1"]')!;
    expect(line()).not.toHaveClass("ev-hl");
    fireEvent.click(chip);
    expect(chip).toHaveAttribute("aria-pressed", "true");
    expect(line()).toHaveClass("ev-hl");
  });

  it("shows the removed-facts note with the reasons", async () => {
    const f = final({ removed: [{ why: "decision count not in the customer-history line: hold" }] });
    expect(removedNote(f)).toBe("1 fact removed: decision count not in the customer-history line: hold");
    render(<CopilotPanel t={await someCase()} asOf={asOf} copilot={fake(READY, f)} />);
    fireEvent.click(await screen.findByRole("button", { name: "Case summary" }));
    expect(await screen.findByTestId("copilot-removed")).toHaveTextContent("1 fact removed: decision count not in the customer-history line");
  });

  it("shows 'Not available from the case evidence' when no fact is kept", async () => {
    const f = final({ kept: [], answer: NOT_AVAILABLE, removed: [{ why: "no citation" }, { why: "no citation" }] });
    render(<CopilotPanel t={await someCase()} asOf={asOf} copilot={fake(READY, f)} />);
    fireEvent.click(await screen.findByRole("button", { name: "Which views drove the score?" }));
    expect(await screen.findByTestId("copilot-not-available")).toHaveTextContent(NOT_AVAILABLE);
    expect(screen.getByTestId("copilot-removed")).toHaveTextContent("2 facts removed: no citation");
  });

  it("labels a cached summary with its time and model", async () => {
    const f = final({ kind: "summary", cached: { label: "cached, generated at 2026-10-25 04:04 UTC by qwen2.5:7b-instruct-q4_K_M", generated_at: "2026-10-25 04:04 UTC", model: READY.model } });
    render(<CopilotPanel t={await someCase()} asOf={asOf} copilot={fake(READY, f)} />);
    fireEvent.click(await screen.findByRole("button", { name: "Case summary" }));
    expect(await screen.findByTestId("copilot-cached")).toHaveTextContent("cached, generated at 2026-10-25 04:04 UTC by qwen2.5:7b-instruct-q4_K_M");
  });

  it("streams only checked facts with progress counts between them, never the model's text (D226a)", async () => {
    let resolve: (f: CopilotFinal) => void = () => {};
    let push: (e: CopilotEvent) => void = () => {};
    const cop: CopilotApi = {
      ...fake(READY),
      summary: (_id, _a, onEvent) => {
        push = onEvent;
        return new Promise((r) => (resolve = r));
      },
    };
    render(<CopilotPanel t={await someCase()} asOf={asOf} copilot={cop} />);
    fireEvent.click(await screen.findByRole("button", { name: "Case summary" }));
    expect(await screen.findByTestId("copilot-running")).toHaveTextContent("Preparing the case summary");
    expect(screen.getByRole("button", { name: "Case summary" })).toBeDisabled();
    act(() => push({ type: "start", evidence: EVIDENCE, model: READY.model, reasons_source: "view-level" }));
    act(() => push({ type: "progress", phase: "writing", drafted: 0, shown: 0, removed: 0 }));
    expect(screen.getByTestId("copilot-progress")).toHaveTextContent("each fact is checked as soon as it is complete");
    act(() => push({ type: "fact", fact: { text: "The decision is hold [E1].", citations: ["E1"] } }));
    act(() => push({ type: "progress", phase: "writing", drafted: 2, shown: 1, removed: 1 }));
    expect(screen.getByTestId("copilot-progress")).toHaveTextContent("2 facts drafted, 1 shown, 1 removed; checking");
    expect(screen.getByTestId("copilot-streamed")).toHaveTextContent("The decision is hold");
    expect(screen.queryByText(/not yet checked/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Show evidence E1" }));
    expect(screen.getByTestId("copilot-evidence").querySelector('[data-handle="E1"]')).toHaveClass("ev-hl");
    act(() => resolve(final({ kind: "summary" })));
    await screen.findByTestId("copilot-answer");
    expect(screen.queryByTestId("copilot-running")).toBeNull();
  });

  it("shows a refusal with its fixed message", async () => {
    const f = final({ route: "refused", refused: "change", kept: [], evidence: [], answer: "I can't change decisions, labels, thresholds or settings: the copilot is read-only." });
    render(<CopilotPanel t={await someCase()} asOf={asOf} copilot={fake(READY, f)} />);
    const box = await screen.findByLabelText("Ask about this case");
    fireEvent.change(box, { target: { value: "Change the decision to allow" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByTestId("copilot-refused")).toHaveTextContent("the copilot is read-only");
  });

  it("says when Ollama is not running, with the start hint, and keeps the cached summary available", async () => {
    const down = { ...READY, llm: "The local language model is not running: start it with E:\\ollama\\start_ollama.cmd (Ollama listening on 127.0.0.1:11434).", loaded: null };
    render(<CopilotPanel t={await someCase()} asOf={asOf} copilot={fake(down)} />);
    expect(await screen.findByTestId("copilot-not-running")).toHaveTextContent("start_ollama.cmd");
    expect(screen.getByRole("button", { name: SUGGESTED[0] })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Case summary" })).toBeEnabled();
  });

  it("shows a 503 from the model as its message", async () => {
    render(<CopilotPanel t={await someCase()} asOf={asOf} copilot={fake(READY, new CopilotError("The local language model is not running: start it with E:\\ollama\\start_ollama.cmd.", 503))} />);
    fireEvent.click(await screen.findByRole("button", { name: "Case summary" }));
    expect(await screen.findByTestId("copilot-error")).toHaveTextContent("start_ollama.cmd");
  });

  it("says when the model is not loaded yet, and reads the status again after an answer", async () => {
    let loaded = false;
    const cop = { ...fake(READY), status: () => Promise.resolve({ ...READY, loaded }) };
    render(<CopilotPanel t={await someCase()} asOf={asOf} copilot={cop} />);
    expect(await screen.findByTestId("copilot-loading-model")).toHaveTextContent("the first answer loads it");
    loaded = true; // the first answer loads the model
    fireEvent.click(screen.getByRole("button", { name: SUGGESTED[0] }));
    await screen.findByTestId("copilot-answer");
    await waitFor(() => expect(screen.queryByTestId("copilot-loading-model")).toBeNull());
  });

  it("labels the Test Lab state: loading, ready, refused", async () => {
    const t = await someCase();
    for (const [state, text] of [["loading", "Test Lab loading: reasons are view-level until it is ready"], ["ready", "live frozen-system reasons"], ["refused", "Test Lab refused: reasons are view-level"]]) {
      const { unmount } = render(<CopilotPanel t={t} asOf={asOf} copilot={fake({ ...READY, testlab: state })} />);
      expect(await screen.findByTestId("copilot-testlab")).toHaveTextContent(text);
      unmount();
    }
  });

  it("shows when the copilot is not enabled on the backend, and errors while answering", async () => {
    const t = await someCase();
    const { unmount } = render(<CopilotPanel t={t} asOf={asOf} copilot={fake(new CopilotError("The copilot is not enabled on the backend: start it with VAULTIC_COPILOT=1 (see the README).", 404))} />);
    expect(await screen.findByTestId("copilot-unavailable")).toHaveTextContent("VAULTIC_COPILOT=1");
    unmount();
    render(<CopilotPanel t={t} asOf={asOf} copilot={fake(READY, new Error("socket reset"))} />);
    fireEvent.click(await screen.findByRole("button", { name: SUGGESTED[1] }));
    expect(await screen.findByTestId("copilot-error")).toHaveTextContent("The copilot could not answer: socket reset");
  });

  it("works in mock mode with scripted answers, labelled as mock", async () => {
    const t = await someCase();
    render(<CopilotPanel t={t} asOf={asOf} copilot={createMockCopilot(0)} />);
    expect(await screen.findByText(/Mock: scripted answers, no language model/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Why was this flagged?" }));
    await waitFor(() => expect(screen.getByTestId("copilot-answer")).toHaveTextContent(`The decision is ${t.decision.action}`));
    expect(screen.getAllByRole("button", { name: /Show evidence E/ }).length).toBeGreaterThan(0);
  });
});

describe("progress text", () => {
  it("shows counts only", () => {
    expect(progressText({ drafted: 1, shown: 1, removed: 0 })).toBe("1 fact drafted, 1 shown; checking…");
    expect(progressText({ drafted: 3, shown: 2, removed: 1 })).toBe("3 facts drafted, 2 shown, 1 removed; checking…");
  });
});
