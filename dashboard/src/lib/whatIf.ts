/**
 * What-if cost mode on the client (D128). The what-if parameters live in the client and travel
 * with every request (AsOf.whatIf); the server keeps nothing. Whether figures ARE what-if is never
 * assumed from the client's setting: every response says so (whatIf flags), the data layer reports
 * those flags here, and the banner shows only when what-if figures were actually served for the
 * current setting.
 */
import { useSyncExternalStore } from "react";
import type { AsOf, CostModel } from "../api/types";

/** Stable key of a what-if setting ("" = the evaluated system). */
export function whatIfKey(cost: CostModel | null | undefined): string {
  if (!cost) return "";
  const c = { hold_success_rate: 1, ...cost };
  return [c.c_fp, c.c_rev, c.c_step, c.step_up_success_rate, c.legit_step_up_friction, c.hold_success_rate].join("|");
}

/** Dependency key of a request context: refetch when the time, the delay or what-if changes. */
export const asOfKey = (asOf: AsOf) => `${asOf.time}|${asOf.labelDelayDays}|${whatIfKey(asOf.whatIf)}`;

/** Query parameters of a what-if setting (none for the evaluated system). */
export function whatIfParams(cost: CostModel | null | undefined): Record<string, number> {
  if (!cost) return {};
  return {
    c_fp: cost.c_fp,
    c_rev: cost.c_rev,
    c_step: cost.c_step,
    step_up_success_rate: cost.step_up_success_rate,
    legit_step_up_friction: cost.legit_step_up_friction,
    hold_success_rate: cost.hold_success_rate ?? 1,
  };
}

// ---- flags served by the backend --------------------------------------------------------------

const served = new Map<string, boolean>(); // what-if key -> last flag served for it
const listeners = new Set<() => void>();
let version = 0;

/** Record the whatIf flag of a response to a request made with `key`. */
export function noteServed(key: string, flag: boolean) {
  if (served.get(key) === flag) return;
  served.set(key, flag);
  version += 1;
  for (const l of listeners) l();
}

/** The whatIf flags carried by a response body (decisions, summaries, networks, queues). */
export function flagsIn(body: unknown): boolean[] {
  const out: boolean[] = [];
  const visit = (x: unknown, depth: number) => {
    if (!x || typeof x !== "object" || depth > 3) return;
    if (Array.isArray(x)) {
      for (const y of x) visit(y, depth + 1);
      return;
    }
    const o = x as Record<string, unknown>;
    if (typeof o.whatIf === "boolean") out.push(o.whatIf);
    if (o.decision) visit(o.decision, depth + 1);
    if (Array.isArray(o.cases)) visit(o.cases, depth + 1);
  };
  visit(body, 0);
  return out;
}

/** True when the backend served what-if figures for this setting (drives the banner). */
export function useServedWhatIf(cost: CostModel | null): boolean {
  const key = whatIfKey(cost);
  useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => version,
  );
  return key !== "" && served.get(key) === true;
}

/** For tests. */
export function resetServed() {
  served.clear();
  version += 1;
  for (const l of listeners) l();
}
