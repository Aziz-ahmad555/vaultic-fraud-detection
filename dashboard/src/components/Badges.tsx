import type { Action, ConformalSet, CostModel, View } from "../api/types";
import { ACTION_LABEL, VIEW_LABEL } from "../api/types";
import { VIEW_VAR, riskInk, riskStep, riskVar } from "../lib/colors";
import { risk } from "../lib/format";

/** A decision; `whatIf` (from the response, D128) adds a what-if badge: not the evaluated system. */
export function DecisionBadge({ action, whatIf = false }: { action: Action; whatIf?: boolean }) {
  return (
    <>
      <span className={`badge dec-${action}`}>{ACTION_LABEL[action]}</span>
      {whatIf && <WhatIfBadge />}
    </>
  );
}

/** Marks figures made under the what-if cost mode: illustrative, not the evaluated system. */
export function WhatIfBadge() {
  return (
    <span className="badge badge-whatif" title="What-if cost mode: illustrative, not the evaluated system">
      What-if
    </span>
  );
}

const SET_TEXT: Record<ConformalSet, string> = {
  legit: "Legit",
  fraud: "Fraud",
  uncertain: "Uncertain (both classes)",
  empty: "Empty (neither class)",
};
/** Conformal prediction set as plain text: only the decision is a badge (DESIGN.md 5). */
export const setText = (set: ConformalSet) => SET_TEXT[set];

/** Outcome label as plain text, honest about the label delay. */
export function labelText(label: 0 | 1 | null, delayDays: number): string {
  if (label === null) return `Not known yet (labels arrive after ${delayDays} days)`;
  return label === 1 ? "Confirmed fraud" : "Confirmed legit";
}

export function RiskCell({ p }: { p: number }) {
  return (
    <span className="risk-cell" style={{ background: riskVar(p), color: riskInk(p) }} aria-label={`Risk ${risk(p)} of 100`} data-step={riskStep(p)}>
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

export function LockIcon({ title = "Locked" }: { title?: string }) {
  return (
    <svg width="12" height="14" viewBox="0 0 12 14" role="img" aria-label={title} className="lock-icon">
      <rect x="1" y="6" width="10" height="7.5" rx="1" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M3.5 6V4.2a2.5 2.5 0 0 1 5 0V6" fill="none" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

/**
 * The what-if label inside a printable report (D132): shown, and printed, whenever the report's
 * figures came back with whatIf = true, so a printout can never pass for the evaluated system.
 */
export function WhatIfReportNote({ cost }: { cost?: CostModel | null }) {
  return (
    <p className="whatif-report-note small" role="note">
      <strong>What-if cost mode: illustrative, not the evaluated system.</strong> Decisions and costs use thresholds re-chosen on the calibration days
      for other costs
      {cost
        ? ` (false positive $${cost.c_fp}, review $${cost.c_rev}, OTP $${cost.c_step}, step-up stops ${Math.round(100 * cost.step_up_success_rate)}%, friction $${cost.legit_step_up_friction}, hold stops ${Math.round(100 * (cost.hold_success_rate ?? 1))}%)`
        : ""}
      .
    </p>
  );
}
