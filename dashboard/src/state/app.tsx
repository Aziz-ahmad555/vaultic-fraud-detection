import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { AsOf, Role } from "../api/types";
import { DAY } from "../lib/format";

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
}

const Ctx = createContext<AppState | null>(null);

interface Prefs {
  time: number;
  labelDelayDays: number;
  theme: Theme;
  role: Role;
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
    return raw ? { ...fallback, ...(JSON.parse(raw) as Partial<Prefs>) } : fallback;
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
    asOf: { time, labelDelayDays },
    setAsOfTime: (t) => setTime(Math.min(LAST, Math.max(FIRST, Math.round(t)))),
    setLabelDelay: setLabelDelayDays,
    range: { first: FIRST, last: LAST },
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
