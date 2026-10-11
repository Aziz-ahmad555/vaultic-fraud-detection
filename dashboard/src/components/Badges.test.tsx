import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DecisionBadge } from "./Badges";

describe("decision badge (D128)", () => {
  it("adds the What-if badge only when the response flagged the decision", () => {
    const { container: plain } = render(<DecisionBadge action="hold" />);
    const { container: off } = render(<DecisionBadge action="hold" whatIf={false} />);
    const { container: on } = render(<DecisionBadge action="hold" whatIf />);
    expect(plain.querySelector(".badge-whatif")).toBeNull();
    expect(off.querySelector(".badge-whatif")).toBeNull();
    expect(on.querySelector(".badge-whatif")?.textContent).toBe("What-if");
    expect(on.querySelector(".dec-hold")?.textContent).toBe("Hold");
  });
});
