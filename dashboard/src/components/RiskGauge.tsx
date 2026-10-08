import { riskVar } from "../lib/colors";
import { risk } from "../lib/format";

/** Half-ring gauge in the single risk hue; the number is always printed. */
export function RiskGauge({ p, size = 140 }: { p: number; size?: number }) {
  const r = size / 2 - 10;
  const cx = size / 2;
  const cy = size / 2 + 4;
  const angle = Math.PI * (1 - Math.min(Math.max(p, 0), 1));
  const x = cx + r * Math.cos(angle);
  const y = cy - r * Math.sin(angle);
  const large = 0;
  return (
    <figure className="gauge" aria-label={`Risk ${risk(p)} of 100`}>
      <svg width={size} height={size / 2 + 16} viewBox={`0 0 ${size} ${size / 2 + 16}`} aria-hidden="true">
        <path d={`M${cx - r},${cy} A${r},${r} 0 0 1 ${cx + r},${cy}`} fill="none" stroke="var(--rule)" strokeWidth={10} />
        <path d={`M${cx - r},${cy} A${r},${r} 0 ${large} 1 ${x},${y}`} fill="none" stroke={riskVar(p)} strokeWidth={10} />
        <text x={cx} y={cy - 6} textAnchor="middle" className="gauge-num" fill="currentColor">
          {risk(p)}
        </text>
      </svg>
      <figcaption className="muted xs">Risk out of 100</figcaption>
    </figure>
  );
}
