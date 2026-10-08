import { useId, useRef, useState, type KeyboardEvent } from "react";
import type { Reason, View } from "../api/types";
import { VIEWS, VIEW_LABEL } from "../api/types";
import { VIEW_VAR } from "../lib/colors";
import { fmtPct, fmtPoints, risk } from "../lib/format";

/**
 * The evidence braid (DESIGN.md section 2). Five strands in fixed order; thickness = MVAF gate
 * weight (available weights sum to 1, so the braid's total thickness is constant); a missing
 * view keeps its lane as a thin dashed gap; the strand's end cap is shaded by the view's score.
 */
export interface BraidProps {
  weights: Record<View, number>;
  scores: Record<View, number | null>;
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

export function Braid({ weights, scores, size = "inline", reasons = [], animate = false, label }: BraidProps) {
  const large = size === "large";
  const width = large ? 640 : 72;
  const lane = large ? 30 : 4;
  const height = lane * VIEWS.length;
  const capW = large ? 14 : 5;
  const maxThick = large ? lane * 2.2 : lane * 2.4; // a weight of 1 may spill into neighbouring lanes
  const w = braidWeights(weights, scores);
  const [active, setActive] = useState<View | null>(null);
  const refs = useRef<(SVGGElement | null)[]>([]);
  const titleId = useId();

  const summary =
    label ??
    VIEWS.map((v) =>
      scores[v] === null
        ? `${VIEW_LABEL[v]} not available`
        : `${VIEW_LABEL[v]} weight ${fmtPct(w[v])}, score ${risk(scores[v] as number)}`,
    ).join("; ");

  const onKey = (i: number) => (e: KeyboardEvent<SVGGElement>) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const j = (i + (e.key === "ArrowDown" ? 1 : VIEWS.length - 1)) % VIEWS.length;
      refs.current[j]?.focus();
    }
  };

  const strands = VIEWS.map((v, i) => {
    const y = lane * i + lane / 2;
    const p = scores[v];
    const missing = p === null;
    const thick = missing ? 1 : Math.max(w[v] * maxThick, large ? 1.5 : 1);
    const x1 = large ? 120 : 0;
    const x2 = width - capW - (large ? 90 : 1);
    // a gentle weave in the large version; straight inline
    const amp = large && !missing ? Math.min(lane * 0.18, 4) : 0;
    const d = large
      ? `M${x1},${y} C${x1 + (x2 - x1) * 0.33},${y - amp} ${x1 + (x2 - x1) * 0.66},${y + amp} ${x2},${y}`
      : `M${x1},${y} L${x2},${y}`;
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
        tabIndex={large ? 0 : -1}
        role={large ? "button" : undefined}
        aria-label={
          large
            ? missing
              ? `${VIEW_LABEL[v]}: not available for this transaction`
              : `${VIEW_LABEL[v]}: weight ${fmtPct(w[v])}, score ${risk(p as number)}. Show reasons`
            : undefined
        }
        onMouseEnter={() => large && setActive(v)}
        onFocus={() => large && setActive(v)}
        onKeyDown={large ? onKey(i) : undefined}
        style={{ opacity: active && !isActive ? 0.45 : 1 }}
      >
        {large && (
          <text x={0} y={y + 4} className="braid-label" fill="currentColor">
            {VIEW_LABEL[v]}
          </text>
        )}
        <path
          d={d}
          fill="none"
          stroke={VIEW_VAR[v]}
          strokeWidth={thick}
          strokeLinecap="butt"
          strokeDasharray={missing ? (large ? "6 6" : "2 2") : undefined}
          strokeOpacity={missing ? 0.7 : 1}
          className={animate && !missing ? "braid-draw" : undefined}
          style={animate ? { animationDelay: `${i * 40}ms` } : undefined}
        />
        {!missing && (
          <rect
            x={x2}
            y={y - Math.max(thick, large ? 8 : 3) / 2}
            width={capW}
            height={Math.max(thick, large ? 8 : 3)}
            fill={VIEW_VAR[v]}
            fillOpacity={0.18 + 0.82 * (p as number)}
            stroke={VIEW_VAR[v]}
            strokeWidth={large ? 1 : 0.5}
            className="braid-cap"
          />
        )}
        {large && (
          <text x={width - 82} y={y + 4} className="braid-meta" fill="currentColor">
            {missing ? "not available" : `w ${w[v].toFixed(2)} · ${risk(p as number)}`}
          </text>
        )}
        {/* wide invisible hit area for hover */}
        {large && <rect x={0} y={lane * i} width={width} height={lane} fill="transparent" />}
      </g>
    );
  });

  if (!large) {
    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={summary} className="braid braid-inline">
        {strands}
      </svg>
    );
  }

  const activeReasons = active ? reasons.filter((r) => r.view === active) : [];
  return (
    <div className="braid-large" onMouseLeave={() => setActive(null)}>
      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        aria-labelledby={titleId}
        className="braid"
        role="group"
      >
        <title id={titleId}>Evidence braid. {summary}</title>
        {strands}
      </svg>
      <div className="braid-detail" aria-live="polite">
        {active === null ? (
          <p className="muted small">Hover or focus a strand to see that view's reasons. Arrow keys move between strands.</p>
        ) : scores[active] === null ? (
          <p className="small">
            <strong>{VIEW_LABEL[active]}</strong> could not score this transaction, so it carries no weight. Missing evidence stays missing.
          </p>
        ) : activeReasons.length === 0 ? (
          <p className="small">
            <strong>{VIEW_LABEL[active]}</strong>: weight {fmtPct(w[active])}, score {risk(scores[active] as number)}. No feature-level reasons
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
