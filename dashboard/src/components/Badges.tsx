import type { Action, ConformalSet, View } from "../api/types";
import { ACTION_LABEL, VIEW_LABEL } from "../api/types";
import { VIEW_VAR, riskInk, riskVar } from "../lib/colors";
import { risk } from "../lib/format";

export function DecisionBadge({ action }: { action: Action }) {
  return <span className={`badge dec-${action}`}>{ACTION_LABEL[action]}</span>;
}

const SET_TEXT: Record<ConformalSet, string> = {
  legit: "Legit",
  fraud: "Fraud",
  uncertain: "Uncertain",
  empty: "Empty set",
};
export function SetBadge({ set }: { set: ConformalSet }) {
  return (
    <span className="badge set-badge" data-set={set} title="Conformal prediction set">
      {SET_TEXT[set]}
    </span>
  );
}

export function RiskCell({ p }: { p: number }) {
  return (
    <span className="risk-cell" style={{ background: riskVar(p), color: riskInk(p) }} aria-label={`Risk ${risk(p)} of 100`}>
      {risk(p)}
    </span>
  );
}

export function ViewKey({ view }: { view: View }) {
  return (
    <span className="view-key" style={{ ["--c" as string]: VIEW_VAR[view] }}>
      {VIEW_LABEL[view]}
    </span>
  );
}

export function LabelCell({ label }: { label: 0 | 1 | null }) {
  if (label === null)
    return (
      <span className="badge curtain" title="The label is not known yet at the as-of time (label delay)">
        Not known yet
      </span>
    );
  return <span className="badge set-badge">{label === 1 ? "Confirmed fraud" : "Confirmed legit"}</span>;
}

export function LockIcon({ title = "Locked" }: { title?: string }) {
  return (
    <svg width="12" height="14" viewBox="0 0 12 14" role="img" aria-label={title} className="lock-icon">
      <rect x="1" y="6" width="10" height="7.5" rx="1" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M3.5 6V4.2a2.5 2.5 0 0 1 5 0V6" fill="none" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}
