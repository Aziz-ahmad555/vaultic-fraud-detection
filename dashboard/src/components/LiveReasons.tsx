/**
 * D220: feature-level reasons for a case. No per-transaction SHAP reasons are committed (E15
 * stores aggregates only), so the case shows its real view-level reasons and, on request, the
 * frozen system's live top-5 feature reasons from the Test Lab (parity-gated, D182-D189). They are
 * labelled live-computed; nothing is fetched until the analyst asks.
 */
import { useState } from "react";
import { testLabApi, type LiveResult, type TestLabApi } from "../api/testlab";

export const NOT_STORED = "Feature-level SHAP reasons are not stored for this transaction; the reasons above are view-level (MVAF gate weight × view score).";

type State = { kind: "idle" } | { kind: "busy" } | { kind: "error"; text: string } | { kind: "done"; result: LiveResult };

export function LiveReasons({ transactionId, lab = testLabApi }: { transactionId: number; lab?: TestLabApi }) {
  const [state, setState] = useState<State>({ kind: "idle" });

  const fetchReasons = async () => {
    setState({ kind: "busy" });
    try {
      const status = await lab.status();
      if (status.state !== "ready") {
        setState({ kind: "error", text: `The Test Lab is not ready (${status.state}: ${status.message}).` });
        return;
      }
      setState({ kind: "done", result: await lab.scoreReal(transactionId) });
    } catch (e) {
      setState({ kind: "error", text: `The Test Lab is not available: ${(e as Error).message}` });
    }
  };

  return (
    <div data-testid="live-reasons">
      <p className="muted xs">{NOT_STORED}</p>
      {state.kind !== "done" && (
        <button type="button" className="btn" onClick={fetchReasons} disabled={state.kind === "busy"}>
          {state.kind === "busy" ? "Scoring in the Test Lab…" : "Compute feature reasons live (Test Lab, frozen system)"}
        </button>
      )}
      {state.kind === "error" && <p role="alert" className="small">{state.text}</p>}
      {state.kind === "done" && (
        <>
          <p className="mock-note xs" style={{ display: "inline-block" }}>
            Live-computed by the frozen system in the Test Lab
            {state.result.matches_frozen_run === true ? " (matches the frozen run ✓)" : state.result.matches_frozen_run === false ? " (does not match the frozen run)" : ""}
          </p>
          <ol className="small">
            {(state.result.reasons ?? []).map((r) => (
              <li key={r.feature}>
                {r.text} <span className="muted">({r.points >= 0 ? "+" : ""}{r.points.toFixed(0)} pts)</span>
              </li>
            ))}
          </ol>
        </>
      )}
    </div>
  );
}
