import type { View } from "../api/types";

/** CSS variable of each view's fixed colour (DESIGN.md: the graph view is "relational"). */
export const VIEW_VAR: Record<View, string> = {
  tabular: "var(--view-tabular)",
  behavioral: "var(--view-behavioral)",
  temporal: "var(--view-temporal)",
  graph: "var(--view-relational)",
  anomaly: "var(--view-anomaly)",
};

/** Single-hue risk scale step (1-5) for a probability. */
export function riskStep(p: number): 1 | 2 | 3 | 4 | 5 {
  if (p < 0.1) return 1;
  if (p < 0.3) return 2;
  if (p < 0.55) return 3;
  if (p < 0.8) return 4;
  return 5;
}
export const riskVar = (p: number) => `var(--risk-${riskStep(p)})`;
/** Text colour that stays readable on each risk step. */
export const riskInk = (p: number) => (riskStep(p) >= 4 ? "var(--sheet)" : "var(--ink)");
