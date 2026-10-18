import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { createTestLabApi, LabError, type LabStatus, type LiveResult, type TestLabApi } from "../api/testlab";
import { emptyScoreboard, outcome, record } from "../lib/scoreboard";
import { TestLab } from "./TestLab";

const PARITY = { n: 2000, passed: true, feature_tolerance: 1e-6, score_tolerance: 1e-12, max_feature_diff: 0, max_score_diff: { MVAF: 0, B5: 0 }, n_failing_features: 0, failing_features: {} };
const READY: LabStatus = { state: "ready", message: "ok", parity: PARITY, latency: { n: 3, p50_ms: 410, p95_ms: 620 } };
const RESULT: LiveResult = {
  risk: 0.93, calibrated: 0.4, conformal_set: "fraud", disagreement: 0.1, decision: "hold", alert: true, flagged_e10: true, e10_threshold: 0.881,
  b5_score: 0.5, latency_ms: 450, uid: 7, kind: "real test transaction (label hidden until revealed)", transaction_id: 3500000,
  stored_risk: 0.93, matches_frozen_run: true,
  views: ["tabular", "behavioral", "temporal", "graph", "anomaly"].map((v, i) => ({ view: v as never, available: i !== 3, raw: i !== 3 ? 0.5 : null, p: i !== 3 ? 0.6 : null, weight: i !== 3 ? 0.25 : 0 })),
  reasons: [{ feature: "C13", text: "Unusual value in masked counter C13", points: 31 }],
  view_reasons: [{ view: "tabular", points: 20 }],
};

function fakeApi(over: Partial<TestLabApi> = {}): TestLabApi {
  return {
    status: vi.fn(async () => READY),
    sample: vi.fn(async (filter) => ({ transaction_id: 3500000, uid: 7, day: 160, seconds: 3600, amount: 59, product: "W", card4: "visa", card6: "debit", email: "gmail.com", device_type: null, device_info: null, history: 4, filter })),
    scoreReal: vi.fn(async () => RESULT),
    scoreCustom: vi.fn(async () => ({ ...RESULT, kind: "custom input - no ground truth", matches_frozen_run: undefined, stored_risk: undefined, notes: { customer: "new customer: unseen card, D1 = 0" } })),
    reveal: vi.fn(async () => ({ transaction_id: 3500000, isFraud: 1 as const })),
    latency: vi.fn(),
    options: vi.fn(async () => ({ products: ["W", "C"], card4: ["visa"], card6: ["debit"], email_domains: ["gmail.com"], device_types: ["mobile"], device_infos: ["Windows"], days: [151, 182] as [number, number] })),
    templateUrl: () => "http://x/api/live/template.csv",
    batch: vi.fn(async () => ({
      kind: "user file - not the evaluation",
      metrics: { rows: 2, has_label: true, edited_rows: 1, frauds: 1, pr_auc: 1, recall_e12_alert: 1, recall_e10_threshold: 0.5, confusion_e12_alert: { tp: 1, fp: 0, fn: 0, tn: 1 } },
      rows: [{ TransactionID: 1, uid: 2, risk: 0.9, decision: "hold", alert: true, latency_ms: 400, isFraud: 1, provenance: "edited - not the stored transaction" }],
      csv: "TransactionID,risk\n1,0.9\n",
      note: "each row was scored independently against the stored history before its own time",
    })),
    ...over,
  } as TestLabApi;
}

describe("session scoreboard (D186)", () => {
  it("classifies alerts against the revealed label and counts each transaction once", () => {
    expect(outcome(true, 1)).toBe("caught");
    expect(outcome(false, 1)).toBe("missed");
    expect(outcome(true, 0)).toBe("false alarm");
    expect(outcome(false, 0)).toBe("correct pass");
    let b = record(emptyScoreboard, 1, true, 1, true);
    b = record(b, 2, true, 0, false);
    b = record(b, 1, true, 1, true); // revealed again: not counted twice
    expect(b).toMatchObject({ caught: 1, falseAlarms: 1, missed: 0, matchesFrozen: 1, seen: [1, 2] });
  });
});

describe("Test Lab client", () => {
  it("surfaces the server's reason and a clear message when the backend is down", async () => {
    const refused = createTestLabApi("http://x", async () => new Response(JSON.stringify({ detail: "Test Lab not ready: parity failed" }), { status: 503 }));
    await expect(refused.sample("any")).rejects.toThrow("parity failed");
    const down = createTestLabApi("http://x", async () => {
      throw new TypeError("network");
    });
    await expect(down.status()).rejects.toBeInstanceOf(LabError);
    await expect(down.status()).rejects.toThrow(/not reachable.*VAULTIC_TESTLAB=1/);
  });
});

describe("Test Lab page", () => {
  it("refuses to score when parity failed and says why", async () => {
    const api = fakeApi({
      status: vi.fn(async () => ({ state: "refused", message: "parity failed: 2 feature columns differ", parity: { ...PARITY, passed: false, n_failing_features: 2, failing_features: { g_deg_tx_card: 3 } } }) as LabStatus),
    });
    render(<TestLab api={api} pollMs={60000} />);
    expect(await screen.findByText(/refuses to start/)).toBeInTheDocument();
    expect(screen.getByText(/g_deg_tx_card/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Real test transaction" })).toBeNull();
  });

  it("real mode: label hidden, live score matches the frozen run, reveal updates the scoreboard", async () => {
    const api = fakeApi();
    render(<TestLab api={api} pollMs={60000} />);
    expect((await screen.findByText(/Parity passed/)).closest("p")).toHaveTextContent("2,000");
    expect(screen.getByText(/p50 410 ms, p95 620 ms/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Fraud" }));
    fireEvent.click(screen.getByRole("button", { name: /Pick a random/ }));
    expect(await screen.findByText("hidden until you reveal it")).toBeInTheDocument();
    expect(api.sample).toHaveBeenCalledWith("fraud");
    fireEvent.click(screen.getByRole("button", { name: "Score live" }));
    expect(await screen.findByTestId("frozen-match")).toHaveTextContent("Matches frozen run ✓");
    expect(screen.getByText("Hold")).toBeInTheDocument();
    expect(screen.getByText(/masked counter C13/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reveal truth" }));
    expect(await screen.findByTestId("truth")).toHaveTextContent("fraud (caught)");
    const board = screen.getByRole("table", { name: "Session scoreboard" });
    expect(within(board).getByText("Caught (fraud, alerted)").nextSibling).toHaveTextContent("1");
    expect(screen.getByRole("button", { name: "Reveal truth" })).toBeDisabled();
  });

  it("custom mode is labelled as having no ground truth", async () => {
    const api = fakeApi();
    render(<TestLab api={api} pollMs={60000} />);
    fireEvent.click(await screen.findByRole("button", { name: "Build your own transaction" }));
    expect(screen.getByText(/Custom input — no ground truth/)).toBeInTheDocument();
    await waitFor(() => expect(api.options).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("button", { name: "Score live" }));
    expect(await screen.findByTestId("custom-label")).toHaveTextContent("custom input - no ground truth");
    expect(api.scoreCustom).toHaveBeenCalledWith(expect.objectContaining({ customer: "new", day: 151, amount: 120, seconds: 12 * 3600 }));
  });

  it("batch mode shows metrics and the confusion matrix, and a refused file's reason", async () => {
    const api = fakeApi();
    const user = userEvent.setup();
    render(<TestLab api={api} pollMs={60000} />);
    fireEvent.click(await screen.findByRole("button", { name: "Batch test (CSV)" }));
    expect(screen.getByText(/scored independently/)).toBeInTheDocument();
    const file = new File(["TransactionID,TransactionDT\n1,13046400\n"], "rows.csv", { type: "text/csv" });
    await user.upload(screen.getByLabelText("CSV file"), file);
    fireEvent.click(screen.getByRole("button", { name: "Upload and score" }));
    expect(await screen.findByRole("table", { name: "Confusion matrix (E12 alert)" })).toBeInTheDocument();
    expect(screen.getByText("recall, E10 threshold").previousSibling).toHaveTextContent("50.0%");
    expect(screen.getByText("Download the scored CSV")).toBeInTheDocument();
    expect(screen.getByTestId("batch-kind")).toHaveTextContent("Your uploaded file only, using your labels. No confidence intervals. Not the research evaluation.");
    expect(screen.getByTestId("batch-kind")).toHaveTextContent("user file - not the evaluation");
    expect(screen.getByTestId("batch-edited")).toHaveTextContent("1 row reuse");
    expect(screen.getByText("edited - not the stored transaction")).toBeInTheDocument();

    (api.batch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new LabError("unknown columns: ['bogus']", 400));
    fireEvent.click(screen.getByRole("button", { name: "Upload and score" }));
    expect((await screen.findByText(/File refused:/)).closest("p")).toHaveTextContent("unknown columns");
  });
});
