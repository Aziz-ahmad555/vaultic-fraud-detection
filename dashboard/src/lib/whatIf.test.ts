import { beforeEach, describe, expect, it } from "vitest";
import { flagsIn, noteServed, resetServed, whatIfKey } from "./whatIf";

const cost = { c_fp: 2, c_rev: 1, c_step: 0.1, step_up_success_rate: 0.9, legit_step_up_friction: 0 };
const key = whatIfKey(cost);
const tx = (flag: boolean) => ({ row: {}, decision: { action: "hold", whatIf: flag } });

/** What the banner shows after a sequence of responses, as the HTTP adapter reports them. */
function bannerAfter(responses: unknown[]): boolean | undefined {
  let shown: boolean | undefined;
  for (const body of responses) {
    const flags = flagsIn(body);
    if (flags.length) {
      noteServed(key, flags.some(Boolean));
      shown = flags.some(Boolean);
    }
  }
  return shown;
}

describe("flagsIn (D151): only decisions drive the what-if banner", () => {
  beforeEach(() => resetServed());

  it("reads flags from decisions, queue cases and day summaries only", () => {
    expect(flagsIn([tx(true), tx(true)])).toEqual([true, true]);
    expect(flagsIn([{ policy: "R1", cases: [tx(false)] }])).toEqual([false]);
    expect(flagsIn({ byAction: { allow: 1 }, whatIf: true })).toEqual([true]);
    expect(flagsIn({ nodes: [], edges: [], rings: [], whatIf: false })).toEqual([]); // a network
    expect(flagsIn({ featurePsi: [], whatIf: false })).toEqual([]); // a drift report
    expect(flagsIn([])).toEqual([]);
  });

  it("a response without decisions never clears the banner; decisions do", () => {
    const network = { nodes: [{ id: "uid:u1" }], edges: [], rings: [], whatIf: false };
    expect(bannerAfter([[tx(true)], network])).toBe(true); // network after what-if decisions
    expect(bannerAfter([{ byAction: {}, whatIf: true }, network, []])).toBe(true);
    expect(bannerAfter([[tx(true)], network, [tx(false)]])).toBe(false); // evaluated decisions clear it
    expect(bannerAfter([[tx(false), tx(true)]])).toBe(true); // mixed within one response
  });
});
