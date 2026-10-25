/**
 * Key facts card (D226): always visible, rendered by code from the case evidence, never by the
 * language model. http mode reads GET /api/copilot/facts (the copilot's own tools: same as-of
 * cut, label curtain and live / view-level reasons); mock mode builds the same from the mock case.
 * The label is shown only once it has matured.
 */
import { copilotApi, type CopilotApi, type KeyFactsData } from "../api/copilot";
import type { AsOf, Transaction } from "../api/types";
import { ACTION_LABEL } from "../api/types";
import { fmtMoney2, fmtPoints } from "../lib/format";
import { useData } from "../lib/useData";
import { asOfKey } from "../lib/whatIf";
import { labelText, setText } from "./Badges";
import { ErrorState, Loading } from "./States";

export function KeyFactsView({ f }: { f: KeyFactsData }) {
  const live = f.reasons_source.startsWith("live");
  return (
    <div data-testid="key-facts">
      <dl className="facts facts-grid">
        <div>
          <dt>Decision</dt>
          <dd>{ACTION_LABEL[f.decision]}</dd>
        </div>
        <div>
          <dt>Calibrated fraud probability</dt>
          <dd className="num">{f.p.toFixed(3)}</dd>
        </div>
        <div>
          <dt>Expected loss</dt>
          <dd className="num">{fmtMoney2(f.expected_loss)}</dd>
        </div>
        <div>
          <dt>Conformal set</dt>
          <dd>{setText(f.conformal_set as Transaction["decision"]["conformal_set"])}</dd>
        </div>
        <div>
          <dt>Outcome label</dt>
          <dd className={f.label === null ? "muted" : undefined} data-testid="key-label">
            {labelText(f.label, f.label_delay_days)}
          </dd>
        </div>
      </dl>
      <h3 className="small" style={{ marginTop: 10 }}>Top reasons</h3>
      <p className={live ? "mock-note xs" : "muted xs"} data-testid="reasons-source" style={{ display: "inline-block" }}>
        {live ? "Live frozen-system reasons (Test Lab, parity-gated)" : `View-level reasons: ${f.reasons_source.replace(/^view-level reasons \(gate weight x view logit\);?\s*/, "") || "gate weight × view score"}`}
      </p>
      <ol className="small m0">
        {f.reasons.map((r, i) => (
          <li key={i}>
            {r.text} <span className="muted">({fmtPoints(r.points)} pts)</span>
          </li>
        ))}
      </ol>
      <p className="muted xs" style={{ marginTop: 6 }}>Rendered by code from the case evidence; no language model.</p>
    </div>
  );
}

export function KeyFacts({ t, asOf, copilot = copilotApi }: { t: Transaction; asOf: AsOf; copilot?: CopilotApi }) {
  const facts = useData(() => copilot.facts(t.row.TransactionID, asOf, t), [t.row.TransactionID, asOfKey(asOf)]);
  return (
    <div className="section key-facts">
      <h2>Key facts</h2>
      {facts.loading && !facts.data && <Loading what="the key facts" />}
      {facts.error && <ErrorState what="the key facts" error={facts.error} onRetry={facts.reload} />}
      {facts.data && <KeyFactsView f={facts.data} />}
    </div>
  );
}
