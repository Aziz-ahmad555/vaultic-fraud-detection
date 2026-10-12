/**
 * Number fields kept as text while editing (D137): an emptied field is ignored on submit (it keeps
 * its current value) instead of turning into 0, which `Number("")` would silently give.
 */

export const toField = (x: number | null | undefined) => (x === null || x === undefined ? "" : String(x));

/** The number typed in a field, or `fallback` when the field is empty or not a number. */
export function parseField(text: string, fallback: number): number {
  const t = text.trim();
  if (t === "") return fallback;
  const n = Number(t);
  return Number.isFinite(n) ? n : fallback;
}

/** As parseField, for a field whose empty value explicitly means "never" (null). */
export function parseNullableField(text: string, fallback: number | null): number | null {
  const t = text.trim();
  if (t === "") return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : fallback;
}
