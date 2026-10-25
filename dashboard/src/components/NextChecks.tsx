import type { AsOf, Transaction } from "../api/types";
import { nextChecks } from "../lib/nextChecks";

/** Suggested next checks (D226): fixed rules on the case's fields, labelled "rules, not AI". */
export function NextChecks({ t, asOf }: { t: Transaction; asOf: AsOf }) {
  const checks = nextChecks(t, asOf);
  return (
    <div className="section" data-testid="next-checks">
      <h2>
        Suggested next checks <span className="rules-tag xs">rules, not AI</span>
      </h2>
      {checks.length === 0 ? (
        <p className="muted small">No rule applies to this case.</p>
      ) : (
        <ul className="small m0">
          {checks.map((c) => (
            <li key={c.id}>
              {c.text} <span className="muted xs">(because {c.because})</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
