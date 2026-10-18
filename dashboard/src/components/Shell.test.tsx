import { describe, expect, it } from "vitest";
import { ICON_PATHS, PAGES } from "./Shell";

describe("collapsed rail icons", () => {
  it("has an icon for every page, so no page shows a blank icon below 1280px", () => {
    for (const p of PAGES) expect(ICON_PATHS[p.path]?.length ?? 0, p.path).toBeGreaterThan(0);
  });

  it("gives each page a distinct icon", () => {
    const shapes = PAGES.map((p) => ICON_PATHS[p.path].join("|"));
    expect(new Set(shapes).size).toBe(PAGES.length);
  });
});
