/**
 * Small custom SVG charts on d3 scales. Rules: values are printed, not only encoded; axes in
 * Graphite; marks use the view colours, the risk hue or Ink, never decorative colour.
 */
import { scaleBand, scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import type { CSSProperties, ReactNode } from "react";

const FIT: CSSProperties = { maxWidth: "100%", height: "auto" };

export interface Bar {
  label: string;
  value: number;
  color?: string;
  text?: string; // printed value
}

/** Horizontal labelled bars. */
export function HBars({ bars, max, width = 420, rowHeight = 24, labelWidth = 150, rule }: {
  bars: Bar[];
  max?: number;
  width?: number;
  rowHeight?: number;
  labelWidth?: number;
  rule?: { value: number; label: string };
}) {
  const top = Math.max(max ?? 0, ...bars.map((b) => b.value), rule?.value ?? 0, 1e-9);
  const reserve = Math.max(64, ...bars.map((b) => String(b.text ?? b.value).length * 7 + 12));
  const x = scaleLinear().domain([0, top]).range([0, Math.max(width - labelWidth - reserve, 40)]);
  const height = bars.length * rowHeight + 8;
  return (
    <svg width={width} height={height} style={FIT} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={bars.map((b) => `${b.label} ${b.text ?? b.value}`).join(", ")} className="chart">
      {bars.map((b, i) => (
        <g key={b.label} transform={`translate(0, ${i * rowHeight + 4})`}>
          <text x={labelWidth - 8} y={rowHeight / 2 + 4} textAnchor="end" className="chart-label">
            {b.label}
          </text>
          <rect x={labelWidth} y={4} width={Math.max(x(b.value), 1)} height={rowHeight - 10} fill={b.color ?? "var(--ink)"} />
          <text x={labelWidth + x(b.value) + 6} y={rowHeight / 2 + 4} className="chart-value">
            {b.text ?? b.value}
          </text>
        </g>
      ))}
      {rule && (
        <g>
          <line x1={labelWidth + x(rule.value)} x2={labelWidth + x(rule.value)} y1={0} y2={height} stroke="var(--graphite)" strokeDasharray="3 3" />
          <text x={labelWidth + x(rule.value) + 4} y={height - 2} className="chart-tick">
            {rule.label}
          </text>
        </g>
      )}
    </svg>
  );
}

/** One horizontal stacked bar (e.g. decisions of the day), with a legend underneath. */
export function StackedBar({ parts, width = 520 }: { parts: { label: string; value: number; fill: string; ink: string }[]; width?: number }) {
  const total = parts.reduce((a, p) => a + p.value, 0) || 1;
  let x0 = 0;
  return (
    <div>
      <svg width={width} height={34} style={FIT} viewBox={`0 0 ${width} 34`} role="img" aria-label={parts.map((p) => `${p.label} ${p.value}`).join(", ")} className="chart">
        {parts.map((p) => {
          const w = (p.value / total) * width;
          const g = (
            <g key={p.label}>
              <rect x={x0} y={2} width={Math.max(w, 0)} height={30} fill={p.fill} stroke="var(--graphite)" strokeWidth={0.5} />
              {w > 46 && (
                <text x={x0 + 6} y={22} fill={p.ink} className="chart-value">
                  {p.value}
                </text>
              )}
            </g>
          );
          x0 += w;
          return g;
        })}
      </svg>
      <ul className="legend">
        {parts.map((p) => (
          <li key={p.label}>
            <span className="swatch" style={{ background: p.fill }} /> {p.label} <span className="num">{p.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Vertical bars over a category axis (e.g. hours). */
export function Columns({ data, width = 520, height = 150, color = "var(--ink)", format = (v: number) => String(v) }: {
  data: { key: string | number; value: number; color?: string }[];
  width?: number;
  height?: number;
  color?: string;
  format?: (v: number) => string;
}) {
  const pad = { l: 28, r: 4, t: 8, b: 20 };
  const x = scaleBand<string>().domain(data.map((d) => String(d.key))).range([pad.l, width - pad.r]).padding(0.15);
  const top = Math.max(...data.map((d) => d.value), 1e-9);
  const y = scaleLinear().domain([0, top]).nice().range([height - pad.b, pad.t]);
  return (
    <svg width={width} height={height} style={FIT} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={data.map((d) => `${d.key}: ${format(d.value)}`).join(", ")} className="chart">
      {y.ticks(3).map((t) => (
        <g key={t}>
          <line x1={pad.l} x2={width - pad.r} y1={y(t)} y2={y(t)} stroke="var(--rule)" />
          <text x={pad.l - 4} y={y(t) + 4} textAnchor="end" className="chart-tick">
            {format(t)}
          </text>
        </g>
      ))}
      {data.map((d, i) => (
        <g key={d.key}>
          <rect x={x(String(d.key))} y={y(d.value)} width={x.bandwidth()} height={y(0) - y(d.value)} fill={d.color ?? color} stroke="var(--graphite)" strokeWidth={0.75}>
            <title>{`${d.key}: ${format(d.value)}`}</title>
          </rect>
          {(i % Math.ceil(data.length / 12) === 0) && (
            <text x={(x(String(d.key)) ?? 0) + x.bandwidth() / 2} y={height - 6} textAnchor="middle" className="chart-tick">
              {d.key}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}

export interface Series {
  name: string;
  color: string;
  points: { x: number; y: number }[];
  dashed?: boolean;
}

/** Line chart with optional vertical markers (alarms, drift starts). */
export function Lines({ series, width = 560, height = 200, xDomain, yDomain, xLabel, yLabel, markers = [], diagonal, format = (v: number) => v.toFixed(2) }: {
  series: Series[];
  width?: number;
  height?: number;
  xDomain?: [number, number];
  yDomain?: [number, number];
  xLabel?: string;
  yLabel?: string;
  markers?: { x: number; label: string; kind?: "solid" | "dashed" }[];
  diagonal?: boolean;
  format?: (v: number) => string;
}) {
  const pad = { l: 40, r: 10, t: 12, b: 32 };
  const all = series.flatMap((s) => s.points);
  const xd = xDomain ?? [Math.min(...all.map((p) => p.x)), Math.max(...all.map((p) => p.x))];
  const yd = yDomain ?? [0, Math.max(...all.map((p) => p.y), 1e-9)];
  const x = scaleLinear().domain(xd).range([pad.l, width - pad.r]);
  const y = scaleLinear().domain(yd).nice().range([height - pad.b, pad.t]);
  const path = line<{ x: number; y: number }>()
    .x((p) => x(p.x))
    .y((p) => y(p.y));
  return (
    <svg width={width} height={height} style={FIT} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${yLabel ?? "value"} by ${xLabel ?? "x"}: ${series.map((s) => s.name).join(", ")}`} className="chart">
      {y.ticks(4).map((t) => (
        <g key={`y${t}`}>
          <line x1={pad.l} x2={width - pad.r} y1={y(t)} y2={y(t)} stroke="var(--rule)" />
          <text x={pad.l - 4} y={y(t) + 4} textAnchor="end" className="chart-tick">
            {format(t)}
          </text>
        </g>
      ))}
      {x.ticks(6).map((t) => (
        <text key={`x${t}`} x={x(t)} y={height - pad.b + 14} textAnchor="middle" className="chart-tick">
          {Number.isInteger(t) ? t : t.toFixed(1)}
        </text>
      ))}
      {xLabel && (
        <text x={width - pad.r} y={height - 2} textAnchor="end" className="chart-tick">
          {xLabel}
        </text>
      )}
      {diagonal && <line x1={x(xd[0])} y1={y(yd[0])} x2={x(xd[1])} y2={y(yd[1])} stroke="var(--graphite)" strokeDasharray="4 4" />}
      {markers.map((m, i) => (
        <g key={`${m.x}-${m.label}-${i}`}>
          <line x1={x(m.x)} x2={x(m.x)} y1={pad.t} y2={height - pad.b} stroke="var(--graphite)" strokeDasharray={m.kind === "dashed" ? "4 3" : undefined} />
          <text x={x(m.x) + 3} y={pad.t + 10 + (i % 3) * 12} className="chart-tick">
            {m.label}
          </text>
        </g>
      ))}
      {series.map((s) => (
        <path key={s.name} d={path(s.points) ?? ""} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? "5 4" : undefined} />
      ))}
    </svg>
  );
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <ul className="legend">
      {items.map((it) => (
        <li key={it.label}>
          <span className="swatch line" style={{ borderTopColor: it.color, borderTopStyle: it.dashed ? "dashed" : "solid" }} />
          {it.label}
        </li>
      ))}
    </ul>
  );
}

/** Heat-map in the single risk hue with values printed in every cell. */
export function HeatMap({ rows, cols, value, format, rowTitle, colTitle }: {
  rows: number[];
  cols: number[];
  value: (r: number, c: number) => number;
  format: (v: number) => string;
  rowTitle: string;
  colTitle: string;
}): ReactNode {
  const vals = rows.flatMap((r) => cols.map((c) => value(r, c)));
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const step = (v: number) => 1 + Math.min(4, Math.floor(((v - lo) / Math.max(hi - lo, 1e-9)) * 5));
  return (
    <table className="heat data">
      <caption className="visually-hidden">
        {rowTitle} by {colTitle}
      </caption>
      <thead>
        <tr>
          <th scope="col">
            {rowTitle} \ {colTitle}
          </th>
          {cols.map((c) => (
            <th key={c} scope="col" className="num">
              {c}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r}>
            <th scope="row">{r}</th>
            {cols.map((c) => {
              const s = step(value(r, c));
              return (
                <td key={c} className="num" style={{ background: `var(--risk-${s})`, color: s >= 4 ? "var(--sheet)" : "var(--ink)" }}>
                  {format(value(r, c))}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
