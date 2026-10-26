import { describe, expect, it } from "vitest";
import type { Transaction } from "../api/types";
import { narrative } from "./Case";
import fixture from "./__fixtures__/narrative_e20.json";

/**
 * E20's system C is this narrative, rebuilt in Python by the runner (src/vaultic/copilot/e20.py,
 * D227). The fixture's expected texts are written by the Python port; this test pins them to the
 * TypeScript original (tests/test_e20_runner.py checks the Python side against the same file).
 */
describe("E20 narrative parity (D227)", () => {
  it.each(fixture.map((f, i) => [i, f] as const))("case %i matches the Python port", (_i, f) => {
    expect(narrative(f.transaction as unknown as Transaction)).toBe(f.expected);
  });
});
