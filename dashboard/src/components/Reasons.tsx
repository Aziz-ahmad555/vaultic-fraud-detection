import type { Reason } from "../api/types";
import { VIEW_LABEL } from "../api/types";
import { VIEW_VAR } from "../lib/colors";
import { fmtPoints } from "../lib/format";

/** Reason codes with signed points (PDO 20: +20 doubles the fraud odds) and the view colour. */
export function ReasonList({ reasons, limit = 5 }: { reasons: Reason[]; limit?: number }) {
  if (!reasons.length) return <p className="muted small">No feature-level reasons are available for this transaction.</p>;
  const shown = reasons.filter((r) => Math.round(r.points) !== 0).slice(0, limit);
  const max = Math.max(...shown.map((r) => Math.abs(r.points)), 1);
  return (
    <ol className="reasons">
      {shown.map((r) => (
        <li key={r.feature} className="reason">
          <span className="num pts" aria-label={`${Math.round(r.points)} points`}>
            {fmtPoints(r.points)}
          </span>
          <span className="reason-bar" aria-hidden="true">
            <span
              style={{
                width: `${(Math.abs(r.points) / max) * 100}%`,
                background: VIEW_VAR[r.view],
                opacity: r.points < 0 ? 0.45 : 1,
              }}
            />
          </span>
          <span className="reason-text">
            {r.text}
            <span className="muted xs"> · {VIEW_LABEL[r.view]}</span>
          </span>
        </li>
      ))}
    </ol>
  );
}

export function topReason(reasons: Reason[]): string {
  const up = reasons.find((r) => r.points > 0);
  return up ? `${up.text} (${fmtPoints(up.points)})` : "No risk-raising reason";
}
