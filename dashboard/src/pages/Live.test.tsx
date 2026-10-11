import { describe, expect, it } from "vitest";
import { replayEndText, streamName } from "./Live";

describe("replay labels follow the data source (D138)", () => {
  it("names the replayed data and its real last time", () => {
    expect(streamName("http")).toBe("IEEE-CIS test period (final run)");
    expect(streamName("mock")).toBe("synthetic mock data");
    expect(replayEndText("http", 183 * 86400 - 1)).toBe("Replay reached the end of the IEEE-CIS test period (final run) (Day 182, 23:59)");
    expect(replayEndText("mock", 151 * 86400 - 1)).toBe("Replay reached the end of the synthetic mock data (Day 150, 23:59)");
    expect(replayEndText("http", 183 * 86400 - 1)).not.toMatch(/mock|day 150/i);
  });
});
