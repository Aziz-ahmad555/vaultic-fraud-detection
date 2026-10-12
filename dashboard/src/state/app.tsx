import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api";
import type { AsOf, CostModel, Role } from "../api/types";
import { LABEL_DELAYS } from "../api/types";
import { DAY } from "../lib/format";

// the mock covers the validation period; the real-data backend reports its own range (test period)
const FIRST = 128 * DAY;
const LAST = 151 * DAY - 1;
const PREFS_KEY = "evidence-room-prefs-v1";

type Theme = "light" | "dark";
export interface Toast {
  id: number;
  text: string;
  tone: "done" | "error";
}

interface AppState {
  asOf: AsOf;
  setAsOfTime: (t: number) => void;
  setLabelDelay: (days: number) => void;
  range: { first: number; last: number };
  theme: Theme;
  setTheme: (t: Theme) => void;
  role: Role;
  setRole: (r: Role) => void;
  toasts: Toast[];
  toast: (text: string, tone?: Toast["tone"]) => void;
  dismissToast: (id: number) => void;
  paletteOpen: boolean;
  setPaletteOpen: (open: boolean) => void;
  helpOpen: boolean;
  setHelpOpen: (open: boolean) => void;
  /**
   * The what-if cost setting sent with every request (real-data backend, D128); null = the
   * evaluated system. Kept in memory only: a reload always starts on the evaluated system.
   */
  whatIf: CostModel | null;
  setWhatIf: (cost: CostModel | null) => void;
}

const Ctx = createContext<AppState | null>(null);

interface Prefs {
  time: number;
  labelDelayDays: number;
  theme: Theme;
  role: Role;
}

/** A remembered label delay outside LABEL_DELAYS (e.g. from an older version) falls back to 30 days. */
export function validDelay(days: unknown): number {
  return typeof days === "number" && (LABEL_DELAYS as readonly number[]).includes(days) ? days : 30;
}

function loadPrefs(): Prefs {
  const fallback: Prefs = {
    time: 141 * DAY + 14 * 3600,
    labelDelayDays: 30,
    theme: window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light",
    role: "analyst",
  };
  try {
    const raw = window.localStorage.getItem(PREFS_KEY);
    const prefs = raw ? { ...fallback, ...(JSON.parse(raw) as Partial<Prefs>) } : fallback;
    return { ...prefs, labelDelayDays: validDelay(prefs.labelDelayDays) };
  } catch {
    return fallback;
  }
}

let toastId = 0;

export function AppProvider({ children }: { children: ReactNode }) {
  const initial = useMemo(loadPrefs, []);
  const [time, setTime] = useState(initial.time);
  const [labelDelayDays, setLabelDelayDays] = useState(initial.labelDelayDays);
  const [theme, setTheme] = useState<Theme>(initial.theme);
  const [role, setRole] = useState<Role>(initial.role);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [range, setRange] = useState({ first: FIRST, last: LAST });
  const [whatIf, setWhatIf] = useState<CostModel | null>(null);
  // one object per (time, delay, what-if): pages refetch when any of them changes, and only then
  const asOf = useMemo<AsOf>(() => ({ time, labelDelayDays, whatIf }), [time, labelDelayDays, whatIf]);

  useEffect(() => {
    let live = true;
    api
      .range()
      .then((r) => {
        if (!live) return;
        setRange(r);
        // a remembered time outside the data's range moves to the first day, 14:00
        setTime((t) => (t < r.first || t > r.last ? Math.floor(r.first / DAY) * DAY + 14 * 3600 : t));
      })
      .catch(() => {
        /* keep the default range */
      });
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  useEffect(() => {
    try {
      window.localStorage.setItem(PREFS_KEY, JSON.stringify({ time, labelDelayDays, theme, role }));
    } catch {
      /* storage unavailable: preferences last for this session only */
    }
  }, [time, labelDelayDays, theme, role]);

  const dismissToast = useCallback((id: number) => setToasts((ts) => ts.filter((t) => t.id !== id)), []);
  const toast = useCallback(
    (text: string, tone: Toast["tone"] = "done") => {
      const id = ++toastId;
      setToasts((ts) => [...ts.slice(-3), { id, text, tone }]);
      window.setTimeout(() => dismissToast(id), tone === "error" ? 8000 : 4000);
    },
    [dismissToast],
  );

  const value: AppState = {
    asOf,
    setAsOfTime: (t) => setTime(Math.min(range.last, Math.max(range.first, Math.round(t)))),
    setLabelDelay: (days) => setLabelDelayDays(validDelay(days)),
    range,
    theme,
    setTheme,
    role,
    setRole,
    toasts,
    toast,
    dismissToast,
    paletteOpen,
    setPaletteOpen,
    helpOpen,
    setHelpOpen,
    whatIf,
    setWhatIf,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp must be used inside AppProvider");
  return v;
}

/** True while the user is typing, so single-key shortcuts stay out of the way. */
export function isTyping(e: KeyboardEvent): boolean {
  const el = e.target as HTMLElement | null;
  if (!el) return false;
  return el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName);
}
