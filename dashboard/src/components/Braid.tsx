import { useId, useRef, useState, type KeyboardEvent } from "react";
import type { Reason, View } from "../api/types";
import { VIEWS, VIEW_LABEL } from "../api/types";
import { VIEW_VAR, riskInk, riskVar } from "../lib/colors";
import { fmtPct, fmtPoints, risk } from "../lib/format";

/**
 * The evidence braid (DESIGN.md section 2): a convergence. Five strands start apart on the left,
 * one lane per view in fixed order, and flow (Sankey-style bands) into one cord on the right.
 * A strand's thickness is its MVAF gate weight, constant along its length, so at the merge the
 * cord is exactly the stack of the available weights (they sum to 1). The cord ends in the
 * fused risk number on the risk scale. A missing view is a thin dashed ghost that stops well
 * before the merge: missing evidence is visibly missing, never zero (rule 11).
 */
export interface BraidProps {
  weights: Record<View, number>;
  scores: Record<View, number | null>;
  /** fused probability; shown at the end of the cord (large braid) */
  risk?: number;
  size?: "inline" | "large";
  reasons?: Reason[];
  /** draw-in animation (the live stream only); disabled by prefers-reduced-motion in CSS */
  animate?: boolean;
  label?: string;
}

/** Normalise weights over the available views so they sum to 1 (missing views get 0). */
export function braidWeights(weights: Record<View, number>, scores: Record<View, number | null>) {
  const out = {} as Record<View, number>;
  let total = 0;
  for (const v of VIEWS) total += scores[v] === null ? 0 : Math.max(weights[v], 0);
  for (const v of VIEWS) out[v] = scores[v] === null || total === 0 ? 0 : Math.max(weights[v], 0) / total;
  return out;
}

export interface BraidGeometry {
  width: number;
  height: number;
  x0: number; // strands start
  xm: number; // merge
  xc: number; // cord end
  cord: number; // cord thickness (sum of available strand thicknesses)
  lane: (i: number) => number; // lane centre at the start
  ghostEnd: number;
}

export function geometry(size: "inline" | "large"): BraidGeometry {
  if (size === "inline") {
    return { width: 80, height: 22, x0: 0, xm: 50, xc: 80, cord: 12, lane: (i) => 2.5 + 4.25 * i, ghostEnd: 22 };
  }
  return { width: 460, height: 180, x0: 136, xm: 360, xc: 410, cord: 72, lane: (i) => 18 + 36 * i, ghostEnd: 136 + (360 - 136) * 0.5 };
}

/** Filled band from (x0, lane centre) to (xm, its slot in the cord) and on to xc; constant thickness t. */
function bandPath(g: BraidGeometry, yStart: number, yMerge: number, t: number) {
  const mid = (g.x0 + g.xm) / 2;
  const a0 = yStart - t / 2;
  const b0 = yStart + t / 2;
  const a1 = yMerge - t / 2;
  const b1 = yMerge + t / 2;
  return `M${g.x0},${a0} C${mid},${a0} ${mid},${a1} ${g.xm},${a1} L${g.xc},${a1} L${g.xc},${b1} L${g.xm},${b1} C${mid},${b1} ${mid},${b0} ${g.x0},${b0} Z`;
}

export function Braid({ weights, scores, risk: fused, size = "inline", reasons = [], animate = false, label }: BraidProps) {
  const large = size === "large";
  const g = geometry(size);
  const w = braidWeights(weights, scores);
  const [active, setActive] = useState<View | null>(null);
  const refs = useRef<(SVGGElement | null)[]>([]);
  const titleId = useId();
  const yc = g.height / 2;

  const summary =
    label ??
    VIEWS.map((v) =>
      scores[v] === null ? `${VIEW_LABEL[v]} not available` : `${VIEW_LABEL[v]} weight ${fmtPct(w[v])}, score ${risk(scores[v] as number)}`,
    ).join("; ") + (fused !== undefined ? `; fused risk ${risk(fused)}` : "");

  const onKey = (i: number) => (e: KeyboardEvent<SVGGElement>) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const j = (i + (e.key === "ArrowDown" ? 1 : VIEWS.length - 1)) % VIEWS.length;
      refs.current[j]?.focus();
    }
  };

  // slots in the cord: stacked in view order, centred on the cord
  let cursor = yc - g.cord / 2;
  const slot = {} as Record<View, number>;
  for (const v of VIEWS) {
    const t = w[v] * g.cord;
    slot[v] = cursor + t / 2;
    cursor += t;
  }

  const strands = VIEWS.map((v, i) => {
    const p = scores[v];
    const missing = p === null;
    const t = w[v] * g.cord;
    const y = g.lane(i);
    const isActive = active === v;
    return (
      <g
        key={v}
        ref={(el) => {
          refs.current[i] = el;
        }}
        className="braid-strand"
        data-view={v}
        data-missing={missing ? "true" : "false"}
        data-weight={w[v]}
        data-thickness={missing ? 0 : t}
        tabIndex={large ? 0 : -1}
        role={large ? "button" : undefined}
        aria-label={
          large
            ? missing
              ? `${VIEW_LABEL[v]}: not available for this transaction`
              : `${VIEW_LABEL[v]}: score ${risk(p as number)}, weight ${fmtPct(w[v])}. Show reasons`
            : undefined
        }
        onMouseEnter={() => large && setActive(v)}
        onFocus={() => large && setActive(v)}
        onKeyDown={large ? onKey(i) : undefined}
        style={{ opacity: active && !isActive ? 0.4 : 1 }}
      >
        {missing ? (
          <line
            className="braid-ghost"
            x1={g.x0}
            x2={g.ghostEnd}
            y1={y}
            y2={y}
            stroke={VIEW_VAR[v]}
            strokeWidth={large ? 1.5 : 1}
            strokeDasharray={large ? "5 5" : "2 2"}
          />
        ) : (
          <path
            d={bandPath(g, y, slot[v], t)}
            fill={VIEW_VAR[v]}
            className={animate ? "braid-draw" : undefined}
            style={animate ? { animationDelay: `${i * 40}ms` } : undefined}
          />
        )}
        {large && (
          <>
            <text x={0} y={y - 3} className="braid-label">
              {VIEW_LABEL[v]}
            </text>
            <text x={0} y={y + 13} className="braid-meta">
              {missing ? (
                "not available"
              ) : (
                <>
                  <tspan>score </tspan>
                  <tspan className="braid-val">{risk(p as number)}</tspan>
                  <tspan dx="10">weight </tspan>
                  <tspan className="braid-val">{fmtPct(w[v])}</tspan>
                </>
              )}
            </text>
            <rect x={0} y={y - 18} width={g.xm} height={36} fill="transparent" />
          </>
        )}
      </g>
    );
  });

  if (!large) {
    return (
      <svg width={g.width} height={g.height} viewBox={`0 0 ${g.width} ${g.height}`} role="img" aria-label={summary} className="braid braid-inline">
        {strands}
      </svg>
    );
  }

  const activeReasons = active ? reasons.filter((r) => r.view === active) : [];
  const boxW = 76;
  return (
    <div className="braid-large" onMouseLeave={() => setActive(null)}>
      <svg
        width={g.width + boxW}
        height={g.height}
        viewBox={`0 0 ${g.width + boxW} ${g.height}`}
        aria-labelledby={titleId}
        className="braid"
        role="group"
        style={{ maxWidth: "100%", height: "auto" }}
      >
        <title id={titleId}>Evidence braid. {summary}</title>
        {strands}
        {fused !== undefined && (
          <g className="braid-risk" data-risk={risk(fused)}>
            <text x={g.xc + 8} y={yc - 30} className="braid-meta">
              Risk
            </text>
            <rect x={g.xc + 8} y={yc - 22} width={boxW - 8} height={44} rx={2} fill={riskVar(fused)} />
            <text x={g.xc + 8 + (boxW - 8) / 2} y={yc + 8} textAnchor="middle" className="braid-risk-num" fill={riskInk(fused)}>
              {risk(fused)}
            </text>
          </g>
        )}
      </svg>
      <div className="braid-detail" aria-live="polite">
        {active === null ? (
          <p className="muted small">Hover or focus a strand for that view's reasons. Arrow keys move between strands.</p>
        ) : scores[active] === null ? (
          <p className="small">
            <strong>{VIEW_LABEL[active]}</strong> could not score this transaction, so it carries no weight. Missing evidence stays missing.
          </p>
        ) : activeReasons.length === 0 ? (
          <p className="small">
            <strong>{VIEW_LABEL[active]}</strong> has weight {fmtPct(w[active])} and score {risk(scores[active] as number)}. No feature-level reasons
            were attributed to this view.
          </p>
        ) : (
          <ul className="braid-reasons small">
            {activeReasons.map((r) => (
              <li key={r.feature}>
                <span className="num pts">{fmtPoints(r.points)}</span> {r.text}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
