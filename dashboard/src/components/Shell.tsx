import { useEffect, useRef, type ReactNode } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { DATA_SOURCE } from "../api";
import { DAY, fmtClock } from "../lib/format";
import { isTyping, useApp } from "../state/app";
import { CommandPalette } from "./CommandPalette";

export const PAGES = [
  { path: "/", label: "Overview", key: "o" },
  { path: "/live", label: "Live monitor", key: "m" },
  { path: "/queue", label: "Analyst queue", key: "q" },
  { path: "/cases", label: "Cases", key: "c" },
  { path: "/network", label: "Fraud network", key: "n" },
  { path: "/drift", label: "Drift monitor", key: "d" },
  { path: "/lab", label: "Model lab", key: "b" },
  { path: "/simulate", label: "Scenario simulator", key: "s" },
  { path: "/settings", label: "Reports and settings", key: "r" },
] as const;

function AsOfControl() {
  const { asOf, setAsOfTime, range, setLabelDelay } = useApp();
  return (
    <div className="asof" role="group" aria-label="As-of time machine">
      <span className="asof-title">As of</span>
      <button type="button" className="btn btn-icon" onClick={() => setAsOfTime(asOf.time - 3600)} aria-label="One hour earlier ([)">
        <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M7 1 3 5l4 4" fill="none" stroke="currentColor" strokeWidth="1.6" /></svg>
      </button>
      <label className="asof-range">
        <span className="visually-hidden">As-of time</span>
        <input
          type="range"
          min={range.first}
          max={range.last}
          step={3600}
          value={asOf.time}
          onChange={(e) => setAsOfTime(Number(e.target.value))}
          aria-valuetext={fmtClock(asOf.time)}
        />
      </label>
      <button type="button" className="btn btn-icon" onClick={() => setAsOfTime(asOf.time + 3600)} aria-label="One hour later (])">
        <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="m3 1 4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.6" /></svg>
      </button>
      <output className="asof-value num" aria-live="polite">
        {fmtClock(asOf.time)}
      </output>
      <label className="asof-delay">
        <span>Label delay</span>
        <select value={asOf.labelDelayDays} onChange={(e) => setLabelDelay(Number(e.target.value))}>
          {[0, 7, 30, 60].map((d) => (
            <option key={d} value={d}>
              {d} days
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

function Toasts() {
  const { toasts, dismissToast } = useApp();
  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={`toast toast-${t.tone}`}>
          <span>{t.text}</span>
          <button type="button" className="btn-link" onClick={() => dismissToast(t.id)}>
            Dismiss
          </button>
        </div>
      ))}
    </div>
  );
}

export const SHORTCUTS: [string, string][] = [
  ["Ctrl+K", "Open the command palette"],
  ["J / K", "Next / previous case"],
  ["F / L / U", "Mark the open case as fraud / legit / unsure"],
  ["G then O, M, Q, C, N, D, B, S, R", "Go to a page"],
  ["[ / ]", "Move the as-of time back / forward one hour"],
  ["?", "Show this help"],
  ["Esc", "Close a dialog"],
];

function Help() {
  const { helpOpen, setHelpOpen } = useApp();
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (helpOpen && !d.open) d.showModal?.();
    if (!helpOpen && d.open) d.close?.();
  }, [helpOpen]);
  return (
    <dialog ref={ref} className="dialog" onClose={() => setHelpOpen(false)} aria-labelledby="help-title">
      <h2 id="help-title">Keyboard shortcuts</h2>
      <table className="data">
        <tbody>
          {SHORTCUTS.map(([k, what]) => (
            <tr key={k}>
              <td>
                <kbd>{k}</kbd>
              </td>
              <td>{what}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">Single-key shortcuts are ignored while you type in a field.</p>
      <button type="button" className="btn" onClick={() => setHelpOpen(false)}>
        Close
      </button>
    </dialog>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const { theme, setTheme, setPaletteOpen, setHelpOpen, asOf, setAsOfTime } = useApp();
  const navigate = useNavigate();
  const pendingG = useRef(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen(true);
        return;
      }
      if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return;
      if (pendingG.current) {
        pendingG.current = false;
        const page = PAGES.find((p) => p.key === e.key.toLowerCase());
        if (page) {
          e.preventDefault();
          navigate(page.path);
        }
        return;
      }
      if (e.key === "g") pendingG.current = true;
      else if (e.key === "?") setHelpOpen(true);
      else if (e.key === "[") setAsOfTime(asOf.time - 3600);
      else if (e.key === "]") setAsOfTime(asOf.time + 3600);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate, setPaletteOpen, setHelpOpen, setAsOfTime, asOf.time]);

  return (
    <div className="shell">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <header className="topbar no-print">
        <div className="brand">
          <strong>Vaultic</strong> <span className="muted">Evidence Room</span>
        </div>
        <AsOfControl />
        {DATA_SOURCE === "mock" && (
          <p className="mock-note small" title="Every number on these screens is synthetic until the research pipeline produces real scores">
            Synthetic mock data
          </p>
        )}
        <div className="topbar-actions">
          <button type="button" className="btn" onClick={() => setPaletteOpen(true)}>
            Search or jump <kbd>Ctrl K</kbd>
          </button>
          <button type="button" className="btn" onClick={() => setTheme(theme === "light" ? "dark" : "light")} aria-pressed={theme === "dark"}>
            {theme === "light" ? "Dark theme" : "Light theme"}
          </button>
          <button type="button" className="btn btn-icon" onClick={() => setHelpOpen(true)} aria-label="Keyboard shortcuts (?)">
            ?
          </button>
        </div>
      </header>
      <nav className="rail no-print" aria-label="Pages">
        <ul>
          {PAGES.map((p) => (
            <li key={p.path}>
              <NavLink to={p.path} end={p.path === "/"} title={`${p.label} (G then ${p.key.toUpperCase()})`}>
                <span className="rail-label">{p.label}</span>
                <span className="rail-short" aria-hidden="true">
                  {p.label.slice(0, 2)}
                </span>
              </NavLink>
            </li>
          ))}
        </ul>
        <p className="rail-foot xs muted">As of {fmtClock(asOf.time)} · labels known up to day {Math.floor((asOf.time - asOf.labelDelayDays * DAY) / DAY)}</p>
      </nav>
      <main id="main" className="main" tabIndex={-1}>
        {children}
      </main>
      <Toasts />
      <Help />
      <CommandPalette />
    </div>
  );
}
