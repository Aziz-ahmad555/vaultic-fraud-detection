import { describe, expect, it } from "vitest";
import { parseField, parseNullableField, toField } from "./fields";

describe("number fields (D137)", () => {
  it("an empty or unreadable field keeps the current value instead of becoming 0", () => {
    expect(parseField("", 10)).toBe(10);
    expect(parseField("   ", 10)).toBe(10);
    expect(parseField("abc", 10)).toBe(10);
    expect(parseField("0", 10)).toBe(0); // a typed 0 is a 0
    expect(parseField(" 2.5 ", 10)).toBe(2.5);
  });

  it("an empty never-field means never; an unreadable one keeps its value", () => {
    expect(parseNullableField("", 5)).toBeNull();
    expect(parseNullableField("x", 5)).toBe(5);
    expect(parseNullableField("7", null)).toBe(7);
    expect(toField(null)).toBe("");
    expect(toField(0)).toBe("0");
  });
});
