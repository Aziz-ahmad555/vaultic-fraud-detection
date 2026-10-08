import type { View } from "../api/types";

/** CSS variable of each view's fixed colour (DESIGN.md: the graph view is "relational"). */
export const VIEW_VAR: Record<View, string> = {
  tabular: "var(--view-tabular)",
  behavioral: "var(--view-behavioral)",
  temporal: "var(--view-temporal)",
  graph: "var(--view-relational)",
  anomaly: "var(--view-anomaly)",
};

export const RISK_STEPS = 8;

/**
 * Single-hue risk scale step (1-8), deeper with higher probability. p^0.6 spreads the many
 * low risks over the light steps so 14 and 27 do not look the same (DESIGN.md 1).
 */
export function riskStep(p: number): number {
  const t = Math.pow(Math.min(Math.max(p, 0), 1), 0.6);
  return Math.min(RISK_STEPS, 1 + Math.floor(t * RISK_STEPS));
}
export const riskVar = (p: number) => `var(--risk-${riskStep(p)})`;
/** AA text colour for each risk step (checked per step in DESIGN.md). */
export const riskInk = (p: number) => `var(--risk-${riskStep(p)}-fg)`;
