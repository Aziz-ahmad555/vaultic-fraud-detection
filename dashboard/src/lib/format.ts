export const DAY = 86_400;

const money0 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const money2 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 });
const int = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export const fmtMoney = (x: number) => money0.format(x);
export const fmtMoney2 = (x: number) => money2.format(x);
export const fmtInt = (x: number) => int.format(x);
export const fmtPct = (x: number, digits = 0) => `${(100 * x).toFixed(digits)}%`;
/** Fraud probability as an integer risk score 0-100 (no false precision). */
export const risk = (p: number) => Math.round(100 * p);
export const fmtPoints = (pts: number) => {
  const r = Math.round(pts);
  return `${r > 0 ? "+" : r < 0 ? "−" : ""}${Math.abs(r)}`;
};

/** Dataset clock (relative seconds) as "day 141 · 14:05". */
export function fmtClock(t: number, withDay = true) {
  const day = Math.floor(t / DAY);
  const sec = t - day * DAY;
  const hh = String(Math.floor(sec / 3600)).padStart(2, "0");
  const mm = String(Math.floor((sec % 3600) / 60)).padStart(2, "0");
  return withDay ? `day ${day} · ${hh}:${mm}` : `${hh}:${mm}`;
}
export const dayOf = (t: number) => Math.floor(t / DAY);
